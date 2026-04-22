import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import random
import time
import seaborn as sns
import re

from explainability_plot_utils import (
    add_group_map,
    action_quality_to_compact_label,
    attention_weights_to_t2t,
    class_id_to_compact_label,
    compute_kinematic_metrics,
    format_kinematic_metrics,
    parse_action_quality_from_name,
    plot_gradcam_time_joint,
    plot_transformer_t2t,
    save_attention_correct_incorrect_comparisons,
    save_attention_long_range_metrics,
    save_attention_phase_matrix_comparisons,
    save_attention_target_curve_comparisons,
    save_attention_t2t_comparison_with_skeleton,
    save_gradcam_action_representative_skeletons,
    save_gradcam_cross_action_quality_comparison,
    save_gradcam_phase_aligned_skeleton_comparisons,
    save_gradcam_comparison_with_skeleton,
    save_gradcam_correct_incorrect_comparisons,
    save_kinematic_records,
)

# Kinect skeleton edges (0-indexed, without self-links)
KINECT_NEIGHBOR_LINKS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (3, 6), (6, 7), (7, 8), (8, 9),
    (3, 10), (10, 11), (11, 12), (12, 13),
    (0, 18), (18, 19), (19, 20), (20, 21),
    (0, 14), (14, 15), (15, 16), (16, 17),
]

ACTION_SIDE_BY_SUBJECT = {
    "m02": {"s01": "R", "s02": "L", "s03": "R", "s04": "R", "s05": "R", "s06": "R", "s07": "L", "s08": "R", "s09": "R", "s10": "L"},
    "m03": {"s01": "L", "s02": "R", "s03": "L", "s04": "L", "s05": "L", "s06": "L", "s07": "R", "s08": "L", "s09": "L", "s10": "R"},
    "m04": {"s01": "R", "s02": "R", "s03": "R", "s04": "R", "s05": "R", "s06": "R", "s07": "L", "s08": "R", "s09": "R", "s10": "L"},
    "m06": {"s01": "R", "s02": "R", "s03": "R", "s04": "R", "s05": "R", "s06": "R", "s07": "L", "s08": "R", "s09": "R", "s10": "L"},
    "m07": {"s01": "R", "s02": "R", "s03": "R", "s04": "R", "s05": "R", "s06": "R", "s07": "L", "s08": "R", "s09": "R", "s10": "L"},
    "m08": {"s01": "R", "s02": "R", "s03": "R", "s04": "R", "s05": "R", "s06": "R", "s07": "L", "s08": "R", "s09": "R", "s10": "L"},
    "m09": {"s01": "R", "s02": "R", "s03": "R", "s04": "R", "s05": "R", "s06": "R", "s07": "L", "s08": "R", "s09": "R", "s10": "L"},
    "m10": {"s01": "R", "s02": "R", "s03": "R", "s04": "R", "s05": "R", "s06": "R", "s07": "L", "s08": "R", "s09": "R", "s10": "L"},
}
BILATERAL_ACTIONS = {"m01", "m05"}


def _extract_subject_id_from_prefix(original_pos_filename_prefix):
    if original_pos_filename_prefix is None:
        return None
    text = str(original_pos_filename_prefix).lower()
    m = re.search(r"(s\d+)", text)
    return m.group(1) if m else None


def infer_movement_side(action_type, original_pos_filename_prefix):
    """
    Infer side label for side-dependent actions.
    Returns one of: 'L', 'R', 'B' (bilateral), 'U' (unknown).
    """
    action = str(action_type).lower() if action_type is not None else ""
    if action in BILATERAL_ACTIONS:
        return "B"

    subject = _extract_subject_id_from_prefix(original_pos_filename_prefix)
    if action in ACTION_SIDE_BY_SUBJECT and subject in ACTION_SIDE_BY_SUBJECT[action]:
        return ACTION_SIDE_BY_SUBJECT[action][subject]
    return "U"

# --- Grad-CAM ---
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fwd_hook = target_layer.register_forward_hook(self._forward_hook)
        self._bwd_hook = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        if isinstance(output, (tuple, list)):
            self.activations = output[0]
        else:
            self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        if isinstance(grad_output, (tuple, list)):
            self.gradients = grad_output[0]
        else:
            self.gradients = grad_output

    def compute_cam(self, input_tensor: torch.Tensor, target_class=None):
        self.model.zero_grad(set_to_none=True)
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1)
        if isinstance(target_class, int):
            target_class = torch.tensor([target_class] * output.size(0), device=output.device)
        score = output.gather(1, target_class.view(-1, 1)).sum()
        score.backward()

        activations = self.activations
        gradients = self.gradients
        if activations is None or gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1)
        cam = torch.relu(cam)
        cam_min = cam.view(cam.size(0), -1).min(dim=1)[0].view(-1, 1, 1)
        cam_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(-1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.detach(), output.detach(), target_class.detach()

    def close(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def generate_gradcam_visualizations(model, data_loader, device, save_dir,
                                    max_samples=5, target_class_mode="pred",
                                    class_names=None, save_comparison=True,
                                    only_correct_predictions=True,
                                    save_heatmap_comparison=True,
                                    save_skeleton_comparison=True,
                                    save_action_representative=True,
                                    save_phase_aligned=True,
                                    force_flip_incorrect_actions=(),
                                    representative_rank_overrides=None,
                                    side_filter="all"):
    def _safe_label_name(label_id: int):
        if class_names is not None and 0 <= label_id < len(class_names):
            return str(class_names[label_id])
        return str(label_id)

    def _normalize_quality_label(quality_label):
        q = str(quality_label).lower() if quality_label is not None else ""
        if "incorrect" in q:
            return "incorrect"
        if "correct" in q:
            return "correct"
        return None

    def _safe_token(value):
        text = str(value) if value is not None else "unknown"
        for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|', ' ']:
            text = text.replace(ch, '-')
        return text

    os.makedirs(save_dir, exist_ok=True)
    target_layer = model.stgcn_extractor.st_gcn_networks[-1]
    grad_cam = GradCAM(model, target_layer)
    side_filter_norm = str(side_filter).strip().upper()
    if side_filter_norm not in {"ALL", "L", "R", "B", "U"}:
        side_filter_norm = "ALL"

    model.eval()
    samples_done = 0
    grouped_maps = {}
    grouped_examples = {}

    try:
        for batch in data_loader:
            batch_data, batch_labels = batch[0], batch[1]
            batch_size = batch_data.size(0)
            batch_action_types = list(batch[2]) if len(batch) > 2 else [None] * batch_size
            batch_quality_labels = list(batch[3]) if len(batch) > 3 else [None] * batch_size
            batch_original_prefixes = list(batch[4]) if len(batch) > 4 else [None] * batch_size
            batch_original_skeletons = (
                batch[5] if len(batch) > 5 and isinstance(batch[5], torch.Tensor) and batch[5].dim() == 4 else None
            )

            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            model_input = batch_data.unsqueeze(4)

            if target_class_mode == "true":
                target_class = batch_labels
            else:
                target_class = None

            cam_maps, outputs, used_target = grad_cam.compute_cam(model_input, target_class=target_class)
            preds = outputs.argmax(dim=1)

            for i in range(batch_size):
                cam_i = cam_maps[i].detach().cpu().numpy()  # (T_reduced, V)
                if batch_original_skeletons is not None:
                    skel_i = batch_original_skeletons[i].detach().cpu().numpy()  # (C, T_full, V), uncentered
                else:
                    skel_i = batch_data[i].detach().cpu().numpy()  # (C, T_full, V), centered fallback

                true_label = int(batch_labels[i].item())
                pred_label = int(preds[i].item())
                target_label = int(used_target[i].item())
                if only_correct_predictions and pred_label != true_label:
                    continue
                true_name = _safe_label_name(true_label)
                pred_name = _safe_label_name(pred_label)

                action_type = str(batch_action_types[i]) if i < len(batch_action_types) else None
                quality_label = str(batch_quality_labels[i]) if i < len(batch_quality_labels) else None
                inferred_action, inferred_quality = parse_action_quality_from_name(true_name)
                if not action_type or action_type == "None":
                    action_type = inferred_action if inferred_action else "unknown_action"
                quality_norm = _normalize_quality_label(quality_label)
                if quality_norm is None:
                    quality_norm = _normalize_quality_label(inferred_quality)
                original_prefix = batch_original_prefixes[i] if i < len(batch_original_prefixes) else None
                movement_side = infer_movement_side(action_type, original_prefix)
                if side_filter_norm != "ALL" and movement_side != side_filter_norm:
                    continue

                add_group_map(grouped_maps, action_type, quality_norm, cam_i)
                if quality_norm in {"correct", "incorrect"}:
                    if action_type not in grouped_examples:
                        grouped_examples[action_type] = {"correct": [], "incorrect": []}
                    grouped_examples[action_type][quality_norm].append({
                        "heatmap": cam_i,
                        "skeleton": skel_i,
                        "side": movement_side,
                        "source_prefix": str(original_prefix) if original_prefix is not None else "",
                    })

                if samples_done < max_samples:
                    aq_label = action_quality_to_compact_label(action_type, quality_norm)
                    title = (
                        f"Grad-CAM ({aq_label})\n"
                        f"side={movement_side}, "
                        f"target={target_label} ({class_id_to_compact_label(target_label)}), "
                        f"true={true_label} ({class_id_to_compact_label(true_label)}), "
                        f"pred={pred_label} ({class_id_to_compact_label(pred_label)})"
                    )
                    save_path = os.path.join(
                        save_dir,
                        f"gradcam_{samples_done + 1:03d}_{_safe_token(action_type)}_{_safe_token(quality_norm)}"
                        f"_side_{_safe_token(movement_side)}_true_{true_label}_pred_{pred_label}_target_{target_label}.png"
                    )
                    plot_gradcam_time_joint(cam_i, save_path=save_path, title=title)
                    samples_done += 1

            if not save_comparison and samples_done >= max_samples:
                break
    finally:
        grad_cam.close()

    if save_comparison:
        if save_heatmap_comparison:
            save_gradcam_correct_incorrect_comparisons(
                grouped_maps,
                save_dir=os.path.join(save_dir, "compare_correct_incorrect_heatmap"),
            )
            # Cross-action comparison requested for incorrect-quality samples:
            # m07 vs m10 with identical heatmap bounds and shared color scale.
            save_gradcam_cross_action_quality_comparison(
                grouped_maps,
                save_dir=os.path.join(save_dir, "compare_cross_action_heatmap"),
                action_a="m07",
                action_b="m10",
                quality="incorrect",
            )
        if save_skeleton_comparison:
            save_gradcam_comparison_with_skeleton(
                grouped_examples,
                save_dir=os.path.join(save_dir, "compare_correct_incorrect_skeleton_heatmap"),
                representative_rank_overrides=representative_rank_overrides,
            )
        if save_action_representative:
            save_gradcam_action_representative_skeletons(
                grouped_examples,
                save_dir=os.path.join(save_dir, "action_representative_skeletons"),
                quality="correct",
                top_k_joints=0,
                representative_rank_overrides=representative_rank_overrides,
            )
            save_gradcam_action_representative_skeletons(
                grouped_examples,
                save_dir=os.path.join(save_dir, "action_representative_skeletons"),
                quality="incorrect",
                top_k_joints=0,
                representative_rank_overrides=representative_rank_overrides,
            )
        if save_phase_aligned:
            save_gradcam_phase_aligned_skeleton_comparisons(
                grouped_examples,
                save_dir=os.path.join(save_dir, "phase_aligned_skeleton_comparisons"),
                phase_ratios=(0.0, 0.25, 0.5, 0.75, 1.0),
                top_k_joints=0,
                force_flip_incorrect_actions=force_flip_incorrect_actions,
                representative_rank_overrides=representative_rank_overrides,
            )


def generate_transformer_attention_visualizations(model, data_loader, device, save_dir, max_samples=5,
                                                  class_names=None, save_comparison=True,
                                                  remap_attention_to_full_time=False,
                                                  only_correct_predictions=True,
                                                  save_heatmap_comparison=True,
                                                  save_skeleton_comparison=True,
                                                  save_target_curve=True,
                                                  save_phase_matrix=True,
                                                  save_long_range_metrics=True,
                                                  save_kinematic_csv=True,
                                                  side_filter="all"):
    """
    Visualize Transformer attention as T x T maps and export kinematic indicators.
    """
    def _safe_label_name(label_id: int):
        if class_names is not None and 0 <= label_id < len(class_names):
            return str(class_names[label_id])
        return str(label_id)

    def _normalize_quality_label(quality_label):
        q = str(quality_label).lower() if quality_label is not None else ""
        if "incorrect" in q:
            return "incorrect"
        if "correct" in q:
            return "correct"
        return None

    def _safe_token(value):
        text = str(value) if value is not None else "unknown"
        for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|', ' ']:
            text = text.replace(ch, '-')
        return text

    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    samples_done = 0
    grouped_maps = {}
    grouped_examples = {}
    kinematic_records = []
    side_filter_norm = str(side_filter).strip().upper()
    if side_filter_norm not in {"ALL", "L", "R", "B", "U"}:
        side_filter_norm = "ALL"

    for batch in data_loader:
        batch_data, batch_labels = batch[0], batch[1]
        batch_size = batch_data.size(0)
        batch_action_types = list(batch[2]) if len(batch) > 2 else [None] * batch_size
        batch_quality_labels = list(batch[3]) if len(batch) > 3 else [None] * batch_size
        batch_original_prefixes = list(batch[4]) if len(batch) > 4 else [None] * batch_size
        batch_original_skeletons = (
            batch[5] if len(batch) > 5 and isinstance(batch[5], torch.Tensor) and batch[5].dim() == 4 else None
        )

        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        model_input = batch_data.unsqueeze(4)

        with torch.no_grad():
            outputs = model(model_input, return_attention=True)

        if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
            raise RuntimeError(
                "Model did not return (logits, attention_weights). "
                "Expected STGCN_Transformer_Classifier with return_attention=True support."
            )
        logits, attn_weights = outputs
        preds = logits.argmax(dim=1)

        if attn_weights is None or attn_weights.dim() != 4:
            continue

        # Token map for joint-level keyframe skeleton visualization: (N_batch, T_reduced, V)
        token_importance = attn_weights.mean(dim=(1, 2))
        s_tokens = token_importance.size(1)
        v_joints = batch_data.size(3)
        if s_tokens % v_joints != 0:
            continue
        t_reduced = s_tokens // v_joints
        token_map = token_importance.reshape(-1, t_reduced, v_joints).detach().cpu().numpy()

        # T' x T' map by default (no remapping to full T), optional full-T remap for visualization alignment.
        t_full = int(batch_data.size(2)) if remap_attention_to_full_time else None
        t2t_maps, _ = attention_weights_to_t2t(attn_weights, v_joints=v_joints, t_full=t_full)
        if t2t_maps is None:
            continue

        for i in range(batch_size):
            t2t_i = t2t_maps[i]
            token_i = token_map[i]
            if batch_original_skeletons is not None:
                skel_i = batch_original_skeletons[i].detach().cpu().numpy()  # (C, T_full, V), uncentered
            else:
                skel_i = batch_data[i].detach().cpu().numpy()  # (C, T_full, V), centered fallback

            true_label = int(batch_labels[i].item())
            pred_label = int(preds[i].item())
            if only_correct_predictions and pred_label != true_label:
                continue
            true_name = _safe_label_name(true_label)
            pred_name = _safe_label_name(pred_label)

            action_type = str(batch_action_types[i]) if i < len(batch_action_types) else None
            quality_label = str(batch_quality_labels[i]) if i < len(batch_quality_labels) else None
            inferred_action, inferred_quality = parse_action_quality_from_name(true_name)
            if not action_type or action_type == "None":
                action_type = inferred_action if inferred_action else "unknown_action"
            quality_norm = _normalize_quality_label(quality_label)
            if quality_norm is None:
                quality_norm = _normalize_quality_label(inferred_quality)
            original_prefix = batch_original_prefixes[i] if i < len(batch_original_prefixes) else None
            movement_side = infer_movement_side(action_type, original_prefix)
            if side_filter_norm != "ALL" and movement_side != side_filter_norm:
                continue

            kinematic = compute_kinematic_metrics(skel_i)
            kinematic_records.append({
                "sample_index": len(kinematic_records) + 1,
                "action_type": action_type,
                "movement_side": movement_side,
                "quality_true": quality_norm if quality_norm else "unknown",
                "true_label": true_label,
                "true_name": true_name,
                "pred_label": pred_label,
                "pred_name": pred_name,
                "avg_speed": kinematic["avg_speed"],
                "peak_speed": kinematic["peak_speed"],
                "speed_cv": kinematic["speed_cv"],
                "pause_ratio": kinematic["pause_ratio"],
                "jerk_rms": kinematic["jerk_rms"],
                "stutter_score": kinematic["stutter_score"],
                "stutter_hint": kinematic.get("stutter_hint", "unknown"),
            })

            add_group_map(grouped_maps, action_type, quality_norm, t2t_i)
            if quality_norm in {"correct", "incorrect"}:
                if action_type not in grouped_examples:
                    grouped_examples[action_type] = {"correct": [], "incorrect": []}
                grouped_examples[action_type][quality_norm].append({
                    "t2t_map": t2t_i,
                    "token_map": token_i,
                    "skeleton": skel_i,
                    "side": movement_side,
                })

            if samples_done < max_samples:
                aq_label = action_quality_to_compact_label(action_type, quality_norm)
                title = (
                    f"Transformer T x T ({aq_label})\n"
                    f"side={movement_side}, "
                    f"true={true_label} ({class_id_to_compact_label(true_label)}), "
                    f"pred={pred_label} ({class_id_to_compact_label(pred_label)})"
                )
                save_path = os.path.join(
                    save_dir,
                    f"attention_t2t_{samples_done + 1:03d}_{_safe_token(action_type)}_{_safe_token(quality_norm)}"
                    f"_side_{_safe_token(movement_side)}_true_{true_label}_pred_{pred_label}.png"
                )
                plot_transformer_t2t(
                    t2t_i,
                    save_path=save_path,
                    title=title,
                    metrics_text=format_kinematic_metrics(kinematic),
                )
                samples_done += 1

        if not save_comparison and samples_done >= max_samples:
            break

    if save_kinematic_csv:
        save_kinematic_records(
            kinematic_records,
            save_path=os.path.join(save_dir, "transformer_attention_kinematic_metrics.csv"),
        )

    if save_comparison:
        if save_heatmap_comparison:
            save_attention_correct_incorrect_comparisons(
                grouped_maps,
                save_dir=os.path.join(save_dir, "compare_correct_incorrect_heatmap"),
            )
        if save_skeleton_comparison:
            save_attention_t2t_comparison_with_skeleton(
                grouped_examples,
                save_dir=os.path.join(save_dir, "compare_correct_incorrect_skeleton_heatmap"),
            )
        if save_target_curve:
            save_attention_target_curve_comparisons(
                grouped_maps,
                save_dir=os.path.join(save_dir, "global_logic_target_curve"),
            )
        if save_phase_matrix:
            save_attention_phase_matrix_comparisons(
                grouped_maps,
                save_dir=os.path.join(save_dir, "global_logic_phase_matrix"),
                n_phase=5,
            )
        if save_long_range_metrics:
            save_attention_long_range_metrics(
                grouped_maps,
                save_dir=os.path.join(save_dir, "global_logic_long_range_metrics"),
                long_ratio_threshold=0.25,
            )


def _draw_colored_skeleton_frame(
    skeleton_ctv: np.ndarray,
    frame_idx: int,
    joint_scores: np.ndarray,
    save_path: str,
    title: str,
):
    """
    Draw one skeleton frame with joints colored by importance score.
    skeleton_ctv shape: (C, T, V), joint_scores shape: (V,)
    """
    x = skeleton_ctv[0, frame_idx, :]
    y = skeleton_ctv[1, frame_idx, :]

    scores = np.asarray(joint_scores, dtype=np.float32)
    s_min = float(scores.min())
    s_max = float(scores.max())
    scores_norm = (scores - s_min) / (s_max - s_min + 1e-8)

    fig, ax = plt.subplots(figsize=(5, 5))

    # Draw bones first
    for a, b in KINECT_NEIGHBOR_LINKS:
        ax.plot([x[a], x[b]], [y[a], y[b]], color='lightgray', linewidth=1.5, zorder=1)

    # Draw joints with importance color
    sc = ax.scatter(
        x, y,
        c=scores_norm,
        cmap='jet',
        s=110,
        edgecolors='black',
        linewidths=0.3,
        zorder=2
    )

    # Stable axis range with small margins
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    x_margin = max((x_max - x_min) * 0.15, 1e-3)
    y_margin = max((y_max - y_min) * 0.15, 1e-3)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.15)

    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Joint Importance")

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close(fig)


def _map_reduced_t_to_full_t(t_idx_reduced: int, t_reduced: int, t_full: int) -> int:
    if t_reduced <= 1:
        return 0
    mapped = int(round((t_idx_reduced + 0.5) * (t_full / t_reduced) - 0.5))
    return max(0, min(t_full - 1, mapped))


def generate_gradcam_keyframe_skeleton_visualizations(
    model,
    data_loader,
    device,
    save_dir,
    max_samples=5,
    top_k_frames=3,
    target_class_mode="pred",
):
    """
    Keep Grad-CAM heatmaps and also export top-K important skeleton frames with colored joints.
    """
    os.makedirs(save_dir, exist_ok=True)
    target_layer = model.stgcn_extractor.st_gcn_networks[-1]
    grad_cam = GradCAM(model, target_layer)
    model.eval()

    samples_done = 0
    for batch in data_loader:
        batch_data, batch_labels = batch[0], batch[1]
        batch_original_skeletons = (
            batch[5] if len(batch) > 5 and isinstance(batch[5], torch.Tensor) and batch[5].dim() == 4 else None
        )
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        model_input = batch_data.unsqueeze(4)

        target_class = batch_labels if target_class_mode == "true" else None
        cam_maps, outputs, used_target = grad_cam.compute_cam(model_input, target_class=target_class)
        preds = outputs.argmax(dim=1)

        for i in range(batch_data.size(0)):
            if samples_done >= max_samples:
                break

            cam_i = cam_maps[i].detach().cpu().numpy()  # (T_reduced, V)
            frame_scores = cam_i.mean(axis=1)
            top_k = min(top_k_frames, cam_i.shape[0])
            top_reduced_indices = np.argsort(frame_scores)[::-1][:top_k]

            if batch_original_skeletons is not None:
                skel_i = batch_original_skeletons[i].detach().cpu().numpy()  # (C, T_full, V), uncentered
            else:
                skel_i = batch_data[i].detach().cpu().numpy()  # (C, T_full, V), centered fallback
            t_full = skel_i.shape[1]
            t_reduced = cam_i.shape[0]
            true_label = int(batch_labels[i].item())
            pred_label = int(preds[i].item())
            target_label = int(used_target[i].item())

            for rank, t_red in enumerate(top_reduced_indices, start=1):
                t_full_idx = _map_reduced_t_to_full_t(int(t_red), t_reduced, t_full)
                joint_scores = cam_i[int(t_red)]
                save_path = os.path.join(
                    save_dir,
                    (
                        f"gradcam_skeleton_{samples_done + 1:03d}_rank{rank}_"
                        f"fullT{t_full_idx:03d}_redT{int(t_red):03d}_"
                        f"true_{true_label}_pred_{pred_label}_target_{target_label}.png"
                    )
                )
                title = (
                    f"Grad-CAM Keyframe rank={rank} "
                    f"(fullT={t_full_idx}, redT={int(t_red)})\n"
                    f"true={true_label} ({class_id_to_compact_label(true_label)}), "
                    f"pred={pred_label} ({class_id_to_compact_label(pred_label)}), "
                    f"target={target_label} ({class_id_to_compact_label(target_label)})"
                )
                _draw_colored_skeleton_frame(skel_i, t_full_idx, joint_scores, save_path, title)

            samples_done += 1

        if samples_done >= max_samples:
            break

    grad_cam.close()


def generate_attention_keyframe_skeleton_visualizations(
    model,
    data_loader,
    device,
    save_dir,
    max_samples=5,
    top_k_frames=3,
):
    """
    Export top-K important skeleton frames from Transformer attention token importance.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    samples_done = 0

    for batch in data_loader:
        batch_data, batch_labels = batch[0], batch[1]
        batch_original_skeletons = (
            batch[5] if len(batch) > 5 and isinstance(batch[5], torch.Tensor) and batch[5].dim() == 4 else None
        )
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        model_input = batch_data.unsqueeze(4)

        with torch.no_grad():
            outputs = model(model_input, return_attention=True)

        if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
            raise RuntimeError(
                "Model did not return (logits, attention_weights). "
                "Expected STGCN_Transformer_Classifier with return_attention=True support."
            )

        logits, attn_weights = outputs
        if attn_weights is None or attn_weights.dim() != 4:
            continue

        preds = logits.argmax(dim=1)

        # (N_batch, num_heads, S, S) -> (N_batch, S)
        token_importance = attn_weights.mean(dim=(1, 2))
        s_tokens = token_importance.size(1)
        v_joints = batch_data.size(3)
        if s_tokens % v_joints != 0:
            continue

        t_reduced = s_tokens // v_joints
        token_map = token_importance.reshape(-1, t_reduced, v_joints).detach().cpu().numpy()

        for i in range(batch_data.size(0)):
            if samples_done >= max_samples:
                break

            map_i = token_map[i]  # (T_reduced, V)
            frame_scores = map_i.mean(axis=1)
            top_k = min(top_k_frames, map_i.shape[0])
            top_reduced_indices = np.argsort(frame_scores)[::-1][:top_k]

            if batch_original_skeletons is not None:
                skel_i = batch_original_skeletons[i].detach().cpu().numpy()  # (C, T_full, V), uncentered
            else:
                skel_i = batch_data[i].detach().cpu().numpy()  # (C, T_full, V), centered fallback
            t_full = skel_i.shape[1]
            true_label = int(batch_labels[i].item())
            pred_label = int(preds[i].item())

            for rank, t_red in enumerate(top_reduced_indices, start=1):
                t_full_idx = _map_reduced_t_to_full_t(int(t_red), t_reduced, t_full)
                joint_scores = map_i[int(t_red)]
                save_path = os.path.join(
                    save_dir,
                    (
                        f"attention_skeleton_{samples_done + 1:03d}_rank{rank}_"
                        f"fullT{t_full_idx:03d}_redT{int(t_red):03d}_"
                        f"true_{true_label}_pred_{pred_label}.png"
                    )
                )
                title = (
                    f"Attention Keyframe rank={rank} "
                    f"(fullT={t_full_idx}, redT={int(t_red)})\n"
                    f"true={true_label} ({class_id_to_compact_label(true_label)}), "
                    f"pred={pred_label} ({class_id_to_compact_label(pred_label)})"
                )
                _draw_colored_skeleton_frame(skel_i, t_full_idx, joint_scores, save_path, title)

            samples_done += 1

        if samples_done >= max_samples:
            break

# --- Dataset and model imports ---
from com_verify_cached_dataset import ActionQualityDataset
# from verify_cached_dataset import ActionQualityDataset
# --- Model definitions ---
from STGA_Net import STGA_Net_Model

TCN_CHANNELS = [128, 128, 128]
KERNEL_SIZE = 5
DROPOUT_RATE = 0.1


# --- Confusion matrix plotting ---
def plot_confusion_matrix(cm, class_names=None, save_path=None, title=None, final_size_mm=None, dpi=600):
    """
    Plot a confusion matrix heatmap with fixed typography/layout for paper figures.
    - If class_names is None, labels default to m01..mNN based on cm.shape[0].
    - Default figure width is 128 mm, font is Arial 9 pt, x labels rotate 45 deg.
    """
    fontsize_pt = 9  # 9 pt

    # Use Arial consistently
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': fontsize_pt
    })

    # Build default class labels if not provided
    if class_names is None:
        n = np.array(cm).shape[0]
        class_names = [f"m{idx:02d}" for idx in range(1, n + 1)]

    # Default paper-friendly size around 128 mm width
    if final_size_mm is None:
        width_mm = 128.0
        per_label_height_mm = 6.0
        min_height_mm = 40.0
        height_mm = max(min_height_mm, len(class_names) * per_label_height_mm)
        final_size_mm = (width_mm, height_mm)

    # mm -> inch
    width_in = max(final_size_mm[0] / 25.4, 0.5)
    height_in = max(final_size_mm[1] / 25.4, 0.5)
    figsize = (width_in, height_in)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(np.array(cm), annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                annot_kws={'fontsize': fontsize_pt}, cbar=False)

    # Axis labels
    ax.set_xlabel('Predicted Label', fontsize=fontsize_pt)
    ax.set_ylabel('True Label', fontsize=fontsize_pt)

    # Keep x labels readable
    ax.set_xticklabels(class_names, rotation=45, ha='right', rotation_mode='anchor', fontsize=fontsize_pt)
    ax.set_yticklabels(class_names, rotation=0, fontsize=fontsize_pt)
    ax.tick_params(axis='both', which='major', labelsize=fontsize_pt)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# --- Data augmentation for skeletons ---
class RandomSkeletonTransform:
    def __init__(self,
                 scale_range=(0.95, 1.05),
                 rotation_z_deg=5,
                 noise_std=0.002,
                 p_joint_noise=0.05):
        self.scale_range = scale_range
        self.rotation_z_rad = np.deg2rad(rotation_z_deg)
        self.noise_std = noise_std
        self.p_joint_noise = p_joint_noise

    def __call__(self, skel_data):
        augmented_skel_data = skel_data.clone()

        # 1) Random global scaling
        scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
        augmented_skel_data = augmented_skel_data * scale_factor

        # 2) Small random rotation around Z axis
        if self.rotation_z_rad > 0:
            angle = random.uniform(-self.rotation_z_rad, self.rotation_z_rad)
            cos_theta = np.cos(angle)
            sin_theta = np.sin(angle)

            rotation_matrix = torch.tensor([
                [cos_theta, -sin_theta, 0],
                [sin_theta, cos_theta, 0],
                [0, 0, 1]
            ], dtype=augmented_skel_data.dtype, device=augmented_skel_data.device)

            N, C, T, V = augmented_skel_data.shape

            reshaped_skel = augmented_skel_data.permute(0, 2, 3, 1).reshape(-1, C)
            rotated_reshaped_skel = torch.matmul(reshaped_skel, rotation_matrix.T)
            augmented_skel_data = rotated_reshaped_skel.reshape(N, T, V, C).permute(0, 3, 1, 2)

        # 3) Sparse per-joint Gaussian noise
        if self.noise_std > 0:
            noise_mask = (torch.rand(augmented_skel_data.shape[0], augmented_skel_data.shape[2],
                                     augmented_skel_data.shape[3],
                                     device=augmented_skel_data.device) < self.p_joint_noise).float()
            noise_mask = noise_mask.unsqueeze(1)

            noise = torch.randn_like(augmented_skel_data) * self.noise_std

            augmented_skel_data = augmented_skel_data + (noise * noise_mask)

        return augmented_skel_data


# --- Global config ---
# DEVICE is initialized in __main__
CACHED_DATASET_DIR = r"G:\paper[20260408]\UI-PRMD\data\CachedDataset_Kinect_resample_named"
model_path = None  # deprecated: final eval now loads best model from this run

# Model config
IN_CHANNELS = 3
NUM_CLASSES = 20
GRAPH_ARGS = {'strategy': 'spatial', 'layout': 'kinect'}
EDGE_IMPORTANCE_WEIGHTING = True

# Transformer config
TRANSFORMER_HIDDEN_DIM = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 3
DROPOUT_RATE = 0.1

# Training config
BATCH_SIZE = 16
NUM_EPOCHS = 500
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
#WEIGHT_DECAY = 5e-5
PATIENCE = 20
ENABLE_GRADCAM = True
GRADCAM_MAX_SAMPLES = 8
GRADCAM_TARGET_MODE = "pred"  # "pred" or "true"
ENABLE_ATTENTION_MAP = True
ATTENTION_MAX_SAMPLES = 8
ENABLE_CORRECT_INCORRECT_COMPARISON = True
EXPLAIN_ONLY_CORRECT_PREDICTIONS = True
FORCE_FLIP_INCORRECT_ACTIONS = ()  # e.g. ("m01",)
# Optional manual representative-sample rank overrides (1-based) for skeleton drawing.
# Here we intentionally switch m01-correct / m01-incorrect away from the default first representative.
GRADCAM_REPRESENTATIVE_RANK_OVERRIDES = {
    ("m01", "correct"): 9,
    ("m01", "incorrect"): 2,
}
ENABLE_KEYFRAME_SKELETON = True
KEYFRAME_TOPK_FRAMES = 3
KEYFRAME_MAX_SAMPLES = 8
# Runtime compatibility (keeps CUDA training, avoids missing cuDNN DLL crashes on some Windows setups)
DISABLE_CUDNN_RUNTIME = True
NUM_WORKERS = 0
SEED = 42
CUDNN_DETERMINISTIC = True
CUDNN_BENCHMARK = False


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- Split strategy ---
N_SPLITS = 5  # 5-fold cross validation
# First split out final hold-out test set, then run KFold on remaining data
# Within each fold, split train/val by VAL_SPLIT_RATIO_IN_FOLD
INITIAL_TEST_SPLIT_RATIO = 0.2
VAL_SPLIT_RATIO_IN_FOLD = 0.20  # validation ratio inside each KFold train_val subset


# --- Main training / evaluation entry ---
def train_and_evaluate_cross_validation():
    current_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir_base = os.path.join(os.getcwd(), f"stga-net_cross_val_experiment_results_{current_run_timestamp}")
    os.makedirs(results_dir_base, exist_ok=True)
    print(f"Results directory: '{results_dir_base}'")

    if not os.path.exists(CACHED_DATASET_DIR):
        print(f"Error: cached dataset directory not found: {CACHED_DATASET_DIR}. Run cache_dataset.py first.")
        return

    print("Loading cached dataset...")
    full_dataset_no_transform = ActionQualityDataset(CACHED_DATASET_DIR, transform=None)

    class_labels_map = {v: k for k, v in full_dataset_no_transform.combined_class_mapping.items()}
    CLASS_NAMES = [class_labels_map[i] for i in sorted(class_labels_map.keys())]
    SHORT_CLASS_NAMES = [f"m{i:02d}" for i in range(1, NUM_CLASSES + 1)]
    print(f"Class names: {CLASS_NAMES}")
    print(f"Total samples: {len(full_dataset_no_transform)}")
    all_labels = [s[1] for s in full_dataset_no_transform.samples_info]

    primary_indices, final_test_indices, primary_labels, final_test_labels = train_test_split(
        range(len(full_dataset_no_transform)), all_labels,
        test_size=INITIAL_TEST_SPLIT_RATIO,
        random_state=42,
        stratify=all_labels
    )
    final_test_dataset = Subset(
        ActionQualityDataset(CACHED_DATASET_DIR, transform=None, return_original_skeleton=True),
        final_test_indices,
    )
    final_test_loader = DataLoader(final_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                                   pin_memory=True)
    print(f"\nFinal hold-out test samples: {len(final_test_dataset)}")
    print(f"Samples for KFold train/val/test: {len(primary_indices)}")

    train_data_augmentor = RandomSkeletonTransform(
        scale_range=(0.95, 1.05),
        rotation_z_deg=3,
        noise_std=0.002,
        p_joint_noise=0.05,
    )

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    fold_best_val_accuracies_at_best_f1 = []
    fold_best_val_f1_macros = []
    fold_test_accuracies = []
    fold_test_f1_macros = []
    fold_test_f1_weighteds = []
    fold_test_precisions = []
    fold_test_recalls = []
    fold_test_times = []
    fold_epochs_trained = []

    # Print model size / parameter count for reporting
    def get_model_complexity(model, fold_results_dir):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Theoretical float32 size (4 bytes per parameter)
        theoretical_size_bytes = total_params * 4
        theoretical_size_MB = theoretical_size_bytes / (1024 * 1024)

        # Actual saved checkpoint size (if already saved)
        best_model_path = os.path.join(fold_results_dir, 'best_model_val_acc.pth')
        actual_file_size_MB = None
        if os.path.exists(best_model_path):
            actual_file_size_bytes = os.path.getsize(best_model_path)
            actual_file_size_MB = actual_file_size_bytes / (1024 * 1024)

        print("\n" + "=" * 80)
        print("--- STGCN-Transformer Model Complexity ---")
        print("=" * 80)
        print(f"1. Total Parameters: {total_params:,} ({total_params / 1_000_000:.2f} M)")
        print(f"2. Trainable Parameters: {trainable_params:,} ({trainable_params / 1_000_000:.2f} M)")
        print(f"3. Theoretical Model Size (float32): {theoretical_size_MB:.2f} MB")

        if actual_file_size_MB is not None:
            print(f"4. Actual Saved File Size: {actual_file_size_MB:.2f} MB")
            if abs(theoretical_size_MB - actual_file_size_MB) > 1.0 and theoretical_size_MB > 5.0:
                print("\n*** Note: theoretical and actual sizes differ. This is usually expected due to storage format and metadata. ***")
        else:
            print("4. Actual Saved File Size: unavailable (model file not saved yet).")
        print("-" * 80)

        return total_params, theoretical_size_MB

    model_complexity_printed = False  # only print once

    for fold, (train_val_sub_indices, test_sub_indices) in enumerate(kf.split(primary_indices, primary_labels)):
        print(f"\n--- Fold {fold + 1}/{N_SPLITS} ---")
        fold_results_dir = os.path.join(results_dir_base, f"fold_{fold + 1}")
        os.makedirs(fold_results_dir, exist_ok=True)

        current_fold_train_val_global_indices = [primary_indices[i] for i in train_val_sub_indices]
        current_fold_test_global_indices = [primary_indices[i] for i in test_sub_indices]

        current_fold_train_val_labels = [all_labels[i] for i in current_fold_train_val_global_indices]

        train_sub_indices_in_fold, val_sub_indices_in_fold, _, _ = train_test_split(
            range(len(current_fold_train_val_global_indices)), current_fold_train_val_labels,
            test_size=VAL_SPLIT_RATIO_IN_FOLD,
            random_state=42,
            stratify=current_fold_train_val_labels
        )

        actual_train_indices = [current_fold_train_val_global_indices[i] for i in train_sub_indices_in_fold]
        actual_val_indices = [current_fold_train_val_global_indices[i] for i in val_sub_indices_in_fold]

        train_dataset = Subset(ActionQualityDataset(CACHED_DATASET_DIR, transform=None), actual_train_indices)
        val_dataset = Subset(ActionQualityDataset(CACHED_DATASET_DIR, transform=None), actual_val_indices)
        fold_test_dataset = Subset(ActionQualityDataset(CACHED_DATASET_DIR, transform=None),
                                   current_fold_test_global_indices)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        fold_test_loader = DataLoader(fold_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                                      pin_memory=True)

        print(f"  Fold {fold + 1}: train samples = {len(train_dataset)}")
        print(f"  Fold {fold + 1}: val samples = {len(val_dataset)}")
        print(f"  Fold {fold + 1}: fold-test samples = {len(fold_test_dataset)}")

        model = STGA_Net_Model(
            in_channels=IN_CHANNELS,
            num_class=NUM_CLASSES,
            graph_args=GRAPH_ARGS,
            edge_importance_weighting=EDGE_IMPORTANCE_WEIGHTING,
            nhead=NHEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            dropout_rate=DROPOUT_RATE,
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        best_val_accuracy_at_best_f1_this_fold = 0.0
        best_val_f1_macro_this_fold = -1.0
        epochs_no_improve = 0
        best_epoch = 0
        best_val_loss_at_best_f1 = float('inf')

        current_fold_epochs_trained = 0
        current_fold_best_model_path = os.path.join(fold_results_dir, 'best_model_val_acc.pth')

        for epoch in range(NUM_EPOCHS):
            current_fold_epochs_trained = epoch + 1
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_data_original, batch_labels, _, _, _ in tqdm(train_loader,
                                                                   desc=f"  Fold {fold + 1} Epoch {epoch + 1} (Train)"):
                batch_data_original = batch_data_original.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)

                batch_data_augmented = train_data_augmentor(batch_data_original.clone())

                combined_batch_data = torch.cat((batch_data_original, batch_data_augmented), dim=0)
                combined_batch_labels = torch.cat((batch_labels, batch_labels), dim=0)

                model_input = combined_batch_data.unsqueeze(4)

                optimizer.zero_grad()
                outputs = model(model_input)
                loss = criterion(outputs, combined_batch_labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += combined_batch_labels.size(0)
                train_correct += (predicted == combined_batch_labels).sum().item()

            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_preds_val = []
            all_labels_val = []

            with torch.no_grad():
                for batch_data, batch_labels, _, _, _ in tqdm(val_loader,
                                                              desc=f"  Fold {fold + 1} Epoch {epoch + 1} (Val)"):
                    batch_data = batch_data.to(DEVICE)
                    batch_labels = batch_labels.to(DEVICE)
                    model_input = batch_data.unsqueeze(4)
                    outputs = model(model_input)
                    loss = criterion(outputs, batch_labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()

                    all_preds_val.extend(predicted.cpu().numpy())
                    all_labels_val.extend(batch_labels.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total
            val_f1_macro = f1_score(all_labels_val, all_preds_val, average='macro', zero_division=0)

            print(f"  Fold {fold + 1} Epoch {epoch + 1}: "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, Val F1 (Macro): {val_f1_macro:.4f}")

            scheduler.step(val_f1_macro)

            is_better_f1 = val_f1_macro > best_val_f1_macro_this_fold
            is_tie_better_acc = (
                val_f1_macro == best_val_f1_macro_this_fold and
                val_accuracy > best_val_accuracy_at_best_f1_this_fold
            )
            if is_better_f1 or is_tie_better_acc:
                best_val_f1_macro_this_fold = val_f1_macro
                best_val_accuracy_at_best_f1_this_fold = val_accuracy
                epochs_no_improve = 0
                best_epoch = epoch + 1
                best_val_loss_at_best_f1 = avg_val_loss

                torch.save(model.state_dict(), current_fold_best_model_path)
                print(
                    f"  Saved new best model: {current_fold_best_model_path} | "
                    f"Val F1 (Macro): {best_val_f1_macro_this_fold:.4f}, "
                    f"Val Acc: {best_val_accuracy_at_best_f1_this_fold:.2f}%")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(f"  Early stopping triggered at epoch {epoch + 1}; best epoch was {best_epoch}.")
                    break

        fold_best_val_f1_macros.append(best_val_f1_macro_this_fold)
        fold_best_val_accuracies_at_best_f1.append(best_val_accuracy_at_best_f1_this_fold)
        fold_epochs_trained.append(current_fold_epochs_trained)

        print(f"\n--- Fold {fold + 1} training complete ---")
        print(f"  Best validation F1 (Macro): {best_val_f1_macro_this_fold:.4f} (Epoch {best_epoch})")
        print(f"  Validation accuracy at best F1: {best_val_accuracy_at_best_f1_this_fold:.2f}%")

        # Print model complexity once (after fold 1 checkpoint exists)
        if fold == 0 and not model_complexity_printed:
            get_model_complexity(model, fold_results_dir)
            model_complexity_printed = True

        print(f"  Evaluating fold {fold + 1} on fold-test split...")
        model.load_state_dict(torch.load(current_fold_best_model_path))
        model.eval()
        fold_test_correct = 0
        fold_test_total = 0
        all_preds_fold_test = []
        all_labels_fold_test = []

        start_test_time = time.time()

        with torch.no_grad():
            for batch_data, batch_labels, _, _, _ in tqdm(fold_test_loader, desc=f"  Fold {fold + 1} test"):
                batch_data = batch_data.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                model_input = batch_data.unsqueeze(4)
                outputs = model(model_input)
                _, predicted = torch.max(outputs.data, 1)

                fold_test_total += batch_labels.size(0)
                fold_test_correct += (predicted == batch_labels).sum().item()

                all_preds_fold_test.extend(predicted.cpu().numpy())
                all_labels_fold_test.extend(batch_labels.cpu().numpy())

        end_test_time = time.time()
        current_fold_test_time = end_test_time - start_test_time
        fold_test_times.append(current_fold_test_time)

        fold_test_accuracy = 100 * fold_test_correct / fold_test_total
        fold_test_f1_macro = f1_score(all_labels_fold_test, all_preds_fold_test, average='macro', zero_division=0)
        fold_test_f1_weighted = f1_score(all_labels_fold_test, all_preds_fold_test, average='weighted', zero_division=0)

        fold_test_precision_macro = precision_score(all_labels_fold_test, all_preds_fold_test, average='macro',
                                                    zero_division=0)
        fold_test_recall_macro = recall_score(all_labels_fold_test, all_preds_fold_test, average='macro',
                                              zero_division=0)

        fold_test_precisions.append(fold_test_precision_macro)
        fold_test_recalls.append(fold_test_recall_macro)

        print(f"  Fold {fold + 1} fold-test accuracy: {fold_test_accuracy:.2f}%")
        print(f"  Fold {fold + 1} fold-test precision (macro): {fold_test_precision_macro:.4f}")
        print(f"  Fold {fold + 1} fold-test recall (macro): {fold_test_recall_macro:.4f}")
        print(f"  Fold {fold + 1} fold-test F1-score (macro): {fold_test_f1_macro:.4f}")
        print(f"  Fold {fold + 1} fold-test F1-score (weighted): {fold_test_f1_weighted:.4f}")
        print(f"  Fold {fold + 1} test time: {current_fold_test_time:.2f} sec")
        print(f"  Fold {fold + 1} epochs trained: {current_fold_epochs_trained}")

        fold_test_accuracies.append(fold_test_accuracy)
        fold_test_f1_macros.append(fold_test_f1_macro)
        fold_test_f1_weighteds.append(fold_test_f1_weighted)

        cm_fold_test = confusion_matrix(all_labels_fold_test, all_preds_fold_test, labels=range(NUM_CLASSES))
        cm_fold_test_save_path = os.path.join(fold_results_dir, f"confusion_matrix_fold_{fold + 1}_test_set.png")
        plot_confusion_matrix(cm_fold_test, CLASS_NAMES, save_path=cm_fold_test_save_path,
                              title=f"Fold {fold + 1} Test Set Confusion Matrix")
        print(f"  Fold {fold + 1} confusion matrix saved to '{cm_fold_test_save_path}'")

    # Cross-fold summary
    print("\n\n--- Cross-fold Raw Metrics ---")
    print(f"Best val F1 (macro) per fold: {np.array(fold_best_val_f1_macros).round(4)}")
    print(f"Val accuracies at best-F1 per fold: {np.array(fold_best_val_accuracies_at_best_f1).round(2)}")
    print(f"Fold-test accuracies per fold: {np.array(fold_test_accuracies).round(2)}")
    print(f"Fold-test precision (macro): {np.array(fold_test_precisions).round(4)}")
    print(f"Fold-test recall (macro): {np.array(fold_test_recalls).round(4)}")
    print(f"Fold-test F1 (macro): {np.array(fold_test_f1_macros).round(4)}")
    print(f"Fold-test F1 (weighted): {np.array(fold_test_f1_weighteds).round(4)}")
    print(f"Fold-test time (sec): {np.array(fold_test_times).round(2)}")
    print(f"Epochs trained per fold: {np.array(fold_epochs_trained)}")

    print("\n--- Cross-fold Aggregate Statistics ---")

    # Accuracy
    print(
        f"Accuracy: mean={np.mean(fold_test_accuracies):.2f}% | med={np.median(fold_test_accuracies):.2f}% | max={np.max(fold_test_accuracies):.2f}%")
    # Precision
    print(
        f"Precision (Macro): mean={np.mean(fold_test_precisions):.4f} | med={np.median(fold_test_precisions):.4f} | max={np.max(fold_test_precisions):.4f}")
    # Recall
    print(
        f"Recall (Macro): mean={np.mean(fold_test_recalls):.4f} | med={np.median(fold_test_recalls):.4f} | max={np.max(fold_test_recalls):.4f}")
    # F1-Score (Macro)
    print(
        f"F1-Score (Macro): mean={np.mean(fold_test_f1_macros):.4f} | med={np.median(fold_test_f1_macros):.4f} | max={np.max(fold_test_f1_macros):.4f}")
    # F1-Score (Weighted)
    print(
        f"F1-Score (Weighted): mean={np.mean(fold_test_f1_weighteds):.4f} | med={np.median(fold_test_f1_weighteds):.4f} | max={np.max(fold_test_f1_weighteds):.4f}")
    # Testing Time
    print(
        f"Testing Time (sec): mean={np.mean(fold_test_times):.2f} | med={np.median(fold_test_times):.2f} | max={np.max(fold_test_times):.2f}")
    # Epochs
    print(
        f"Epochs Trained: mean={np.mean(fold_epochs_trained):.2f} | med={np.median(fold_epochs_trained):.2f} | max={np.max(fold_epochs_trained):.2f}")

    # Select the fold with the best validation macro-F1
    best_overall_fold_index = np.argmax(fold_best_val_f1_macros)
    best_overall_val_f1_macro = fold_best_val_f1_macros[best_overall_fold_index]
    best_overall_fold_number = best_overall_fold_index + 1  # convert 0-index -> 1-index

    # Load best checkpoint from this run for final hold-out test evaluation
    best_overall_fold_results_dir = os.path.join(results_dir_base, f"fold_{best_overall_fold_number}")
    best_overall_fold_model_path = os.path.join(best_overall_fold_results_dir, 'best_model_val_acc.pth')

    print(
        f"\n\n--- Final hold-out test using best fold model (Fold {best_overall_fold_number}, "
        f"Val F1 (Macro) {best_overall_val_f1_macro:.4f}) ---")

    final_model = STGA_Net_Model(
        in_channels=IN_CHANNELS,
        num_class=NUM_CLASSES,
        graph_args=GRAPH_ARGS,
        edge_importance_weighting=EDGE_IMPORTANCE_WEIGHTING,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dropout_rate=DROPOUT_RATE,
    ).to(DEVICE)

    final_model.load_state_dict(torch.load(best_overall_fold_model_path))
    print(f"Loaded best model from this run: {best_overall_fold_model_path}")

    final_model.eval()
    final_test_correct = 0
    final_test_total = 0
    all_preds_final_test = []
    all_labels_final_test = []

    start_final_test_time = time.time()

    with torch.no_grad():
        for batch in tqdm(final_test_loader, desc="Final hold-out test"):
            batch_data = batch[0].to(DEVICE)
            batch_labels = batch[1].to(DEVICE)
            model_input = batch_data.unsqueeze(4)
            outputs = final_model(model_input)
            _, predicted = torch.max(outputs.data, 1)

            final_test_total += batch_labels.size(0)
            final_test_correct += (predicted == batch_labels).sum().item()

            all_preds_final_test.extend(predicted.cpu().numpy())
            all_labels_final_test.extend(batch_labels.cpu().numpy())

    end_final_test_time = time.time()
    final_test_time = end_final_test_time - start_final_test_time

    final_test_accuracy = 100 * final_test_correct / final_test_total
    final_test_f1_macro = f1_score(all_labels_final_test, all_preds_final_test, average='macro', zero_division=0)
    final_test_f1_weighted = f1_score(all_labels_final_test, all_preds_final_test, average='weighted', zero_division=0)

    final_test_precision_macro = precision_score(all_labels_final_test, all_preds_final_test, average='macro',
                                                 zero_division=0)
    final_test_recall_macro = recall_score(all_labels_final_test, all_preds_final_test, average='macro',
                                           zero_division=0)

    cm_final_test = confusion_matrix(all_labels_final_test, all_preds_final_test, labels=range(NUM_CLASSES))

    print("\n--- Final Hold-out Test Results ---")
    print(f"Final hold-out test accuracy: {final_test_accuracy:.2f}%")
    print(f"Final hold-out precision (macro): {final_test_precision_macro:.4f}")
    print(f"Final hold-out recall (macro): {final_test_recall_macro:.4f}")
    print(f"Final hold-out F1-score (macro): {final_test_f1_macro:.4f}")
    print(f"Final hold-out F1-score (weighted): {final_test_f1_weighted:.4f}")
    print(f"Final hold-out test time: {final_test_time:.2f} sec")
    print("\nFinal hold-out confusion matrix:\n", cm_final_test)

    cm_final_test_save_path = os.path.join(results_dir_base, "confusion_matrix_final_test_set_best_fold.png")
    plot_confusion_matrix(cm_final_test, SHORT_CLASS_NAMES, save_path=cm_final_test_save_path,
                          title="Final Test Set Confusion Matrix (Best Fold)")
    if ENABLE_GRADCAM:
        gradcam_dir = os.path.join(results_dir_base, "gradcam_visualizations")
        print(f"\ngradcam_visualizations: {gradcam_dir}")
        generate_gradcam_visualizations(
            final_model,
            final_test_loader,
            DEVICE,
            gradcam_dir,
            max_samples=GRADCAM_MAX_SAMPLES,
            target_class_mode=GRADCAM_TARGET_MODE,
            class_names=CLASS_NAMES,
            save_comparison=ENABLE_CORRECT_INCORRECT_COMPARISON,
            only_correct_predictions=EXPLAIN_ONLY_CORRECT_PREDICTIONS,
            force_flip_incorrect_actions=FORCE_FLIP_INCORRECT_ACTIONS,
            representative_rank_overrides=GRADCAM_REPRESENTATIVE_RANK_OVERRIDES,
        )
    if ENABLE_ATTENTION_MAP:
        attention_dir = os.path.join(results_dir_base, "transformer_attention_visualizations")
        print(f"\ntransformer_attention_visualizations: {attention_dir}")
        generate_transformer_attention_visualizations(
            final_model,
            final_test_loader,
            DEVICE,
            attention_dir,
            max_samples=ATTENTION_MAX_SAMPLES,
            class_names=CLASS_NAMES,
            save_comparison=ENABLE_CORRECT_INCORRECT_COMPARISON,
            only_correct_predictions=EXPLAIN_ONLY_CORRECT_PREDICTIONS,
        )
    if ENABLE_KEYFRAME_SKELETON:
        if ENABLE_GRADCAM:
            gradcam_keyframe_dir = os.path.join(results_dir_base, "gradcam_keyframe_skeletons")
            print(f"\ngradcam_keyframe_skeletons: {gradcam_keyframe_dir}")
            generate_gradcam_keyframe_skeleton_visualizations(
                final_model,
                final_test_loader,
                DEVICE,
                gradcam_keyframe_dir,
                max_samples=KEYFRAME_MAX_SAMPLES,
                top_k_frames=KEYFRAME_TOPK_FRAMES,
                target_class_mode=GRADCAM_TARGET_MODE
            )
        if ENABLE_ATTENTION_MAP:
            attention_keyframe_dir = os.path.join(results_dir_base, "attention_keyframe_skeletons")
            print(f"\nattention_keyframe_skeletons: {attention_keyframe_dir}")
            generate_attention_keyframe_skeleton_visualizations(
                final_model,
                final_test_loader,
                DEVICE,
                attention_keyframe_dir,
                max_samples=KEYFRAME_MAX_SAMPLES,
                top_k_frames=KEYFRAME_TOPK_FRAMES
            )
    print(f"Final hold-out confusion matrix saved to '{cm_final_test_save_path}'")


# --- Script entry ---
if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by current settings, but no GPU is available.")
    set_global_seed(SEED)
    print(f"Global seed set to: {SEED}")
    if DISABLE_CUDNN_RUNTIME:
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled for compatibility; training still runs on CUDA.")
    elif torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = CUDNN_DETERMINISTIC
        torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
        print(
            f"cuDNN deterministic={torch.backends.cudnn.deterministic}, "
            f"benchmark={torch.backends.cudnn.benchmark}"
        )
    DEVICE = torch.device("cuda")
    print(f"Using device: {DEVICE}")

    train_and_evaluate_cross_validation()
