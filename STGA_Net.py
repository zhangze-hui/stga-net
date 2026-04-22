import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable  # VariableyTorch 0.4.0Tensor
import numpy as np
from tgcn import ConvTemporalGraphical
from graph import Graph
import math

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A

# ---  Model STGCN_FeatureExtractor ---
class STGCN_FeatureExtractor(nn.Module):
    def __init__(self, in_channels, graph_args, edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)  #  (K)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * self.graph.num_node)  # V_in * C_in (22 * 3)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 16, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(16, 16, kernel_size, 1, **kwargs),
            st_gcn(16, 16, kernel_size, 1, **kwargs),
            st_gcn(16, 16, kernel_size, 1, **kwargs),
            st_gcn(16, 32, kernel_size, 2, **kwargs),  # 
            st_gcn(32, 32, kernel_size, 1, **kwargs),
            st_gcn(32, 32, kernel_size, 1, **kwargs),
            st_gcn(32, 64, kernel_size, 2, **kwargs),  # 
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

    def forward(self, x):
        # x input shape: (N, C, T, V, M)
        N, C, T, V, M = x.size()

        # 
        # (N, C, T, V, M) -> (N*M, V*C, T)
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # (N, M, V, C, T)
        x = x.view(N * M, V * C, T)  # (N*M, V*C, T)
        x = self.data_bn(x)

        # (N*M, V*C, T) -> (N*M, C, T, V)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # (N, M, C, T, V)
        x = x.view(N * M, C, T, V)  # (N*M, C, T, V)

        # forwad through ST-GCN blocks
        # (N*M, out_channels, T_out, V)
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # x  (N*M, final_out_channels, final_T, V)
        return x, N, M  # N, MTransformer


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()

        # pe shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # position shape: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term shape: (d_model/2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # argument shape: (max_len, d_model/2)
        argument = position * div_term

        #  PE 
        pe[:, 0::2] = torch.sin(argument)  # 
        pe[:, 1::2] = torch.cos(argument)  # 

        # PE (max_len, 1, d_model)  (B, S, E) 
        self.register_buffer('pe', pe.unsqueeze(1).permute(1, 0, 2))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, E)
        Returns:
            x + pe[:S]
        """
        # x : (B, S, E)
        seq_len = x.size(1)

        # self.pe : (1, max_len, E) -> (1, S, E)
        # x  (B, S, E) + pe  (1, S, E) =  (B, S, E) ()
        x = x + self.pe[:, :seq_len]
        return x


class TransformerEncoderLayerWithWeights(nn.TransformerEncoderLayer):
    """Transformer encoder layer that can optionally cache attention weights."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture_attention = False
        self.latest_attn_weights = None

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        attn_output, attn_weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=self.capture_attention,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        if self.capture_attention and attn_weights is not None:
            # Shape (N, num_heads, S, S) when batch_first=True
            self.latest_attn_weights = attn_weights.detach()
        else:
            self.latest_attn_weights = None
        return self.dropout1(attn_output)


# ---  STGA-Net_Model  ---
class STGA_Net_Model(nn.Module):
    def __init__(self, in_channels, num_class, graph_args, edge_importance_weighting,
                 transformer_hidden_dim=256, nhead=8, num_encoder_layers=3, dropout_rate=0.1, **kwargs):
        super().__init__()

        self.stgcn_extractor = STGCN_FeatureExtractor(in_channels, graph_args, edge_importance_weighting, **kwargs)


        final_stgcn_out_channels = 64
        final_T = 50  
        num_nodes = self.stgcn_extractor.graph.num_node  # 22
        max_seq_len = final_T * num_nodes

        # Transformer 
        transformer_input_dim = final_stgcn_out_channels  # 64

        self.positional_encoder = SinusoidalPositionalEncoding(
            d_model=transformer_input_dim,
            max_len=max_seq_len
        )

        encoder_layer = TransformerEncoderLayerWithWeights(
            d_model=transformer_input_dim,  # 
            nhead=nhead,
            dim_feedforward=transformer_hidden_dim,
            dropout=dropout_rate,
            batch_first=True  # (N, S, E)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        self.fcn = nn.Linear(transformer_input_dim, num_class)

    def _set_attention_capture(self, enabled: bool):
        for layer in self.transformer_encoder.layers:
            layer.capture_attention = enabled
            if not enabled:
                layer.latest_attn_weights = None

    def forward(self, x, return_attention=False):

        stgcn_out, N_orig, M_orig = self.stgcn_extractor(x)

        N_batch_transformer = stgcn_out.size(0)  
        E_transformer = stgcn_out.size(1)  
        S_transformer = stgcn_out.size(2) * stgcn_out.size(3)  

        x_transformer_input = stgcn_out.view(N_batch_transformer, E_transformer, S_transformer)
        x_transformer_input = x_transformer_input.permute(0, 2, 1)  

        x_with_pe = self.positional_encoder(x_transformer_input)

        self._set_attention_capture(return_attention)
        prev_fastpath = torch.backends.mha.get_fastpath_enabled()
        if return_attention:
            torch.backends.mha.set_fastpath_enabled(False)
        try:
            transformer_output = self.transformer_encoder(x_with_pe)
        finally:
            if return_attention:
                torch.backends.mha.set_fastpath_enabled(prev_fastpath)

        last_layer_attention = None
        if return_attention:
            for layer in reversed(self.transformer_encoder.layers):
                if getattr(layer, "latest_attn_weights", None) is not None:
                    last_layer_attention = layer.latest_attn_weights
                    break
        self._set_attention_capture(False)


        pooled_output = transformer_output.mean(dim=1)  

 
        x = self.fcn(pooled_output)

        x = x.view(N_orig, M_orig, -1).mean(dim=1)  

        if return_attention:
            return x, last_layer_attention
        return x