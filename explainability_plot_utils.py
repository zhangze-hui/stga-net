import csv
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

KINECT_NEIGHBOR_LINKS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (3, 6), (6, 7), (7, 8), (8, 9),
    (3, 10), (10, 11), (11, 12), (12, 13),
    (0, 18), (18, 19), (19, 20), (20, 21),
    (0, 14), (14, 15), (15, 16), (16, 17),
]

ACTION_ORDER = ["m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09", "m10"]
ACTION_DISPLAY_NAMES = {
    "m01": "Deep squat",
    "m02": "Hurdle step",
    "m03": "Inline lunge",
    "m04": "Side lunge",
    "m05": "Sit to stand",
    "m06": "Standing active straight leg raise",
    "m07": "Standing shoulder abduction",
    "m08": "Standing shoulder extension",
    "m09": "Standing shoulder internal-external rotation",
    "m10": "Standing shoulder scaption",
}
# Prior center (in reduced-time ratio) for representative key pose per action.
ACTION_PHASE_PRIOR_CENTER = {
    "m01": 0.52,
    "m02": 0.56,
    "m03": 0.56,
    "m04": 0.52,
    "m05": 0.46,
    "m06": 0.58,
    "m07": 0.62,
    "m08": 0.62,
    "m09": 0.50,
    "m10": 0.62,
}
# Optional manual skeleton-frame overrides for phase-aligned plotting.
# Key: (action_id, quality, phase_percent), value: full-time frame index.
# Example: use an earlier frame for m01-correct at phase=50%.
PHASE_ALIGNED_FULL_FRAME_OVERRIDES = {
    ("m01", "correct", 50): 55,
}

LEFT_ARM_JOINTS = [6, 7, 8, 9]
RIGHT_ARM_JOINTS = [10, 11, 12, 13]
LEFT_LEG_JOINTS = [18, 19, 20, 21]
RIGHT_LEG_JOINTS = [14, 15, 16, 17]
LOWER_SIDE_ACTIONS = {"m02", "m03", "m04", "m06"}
UPPER_SIDE_ACTIONS = {"m07", "m08", "m09", "m10"}


def class_id_to_compact_label(class_id: int) -> str:
    return f"m{int(class_id) + 1:02d}"


def action_quality_to_compact_label(action: Optional[str], quality: Optional[str]) -> str:
    action_text = str(action).strip().lower() if action is not None else ""
    quality_text = str(quality).strip().lower() if quality is not None else ""
    if action_text.startswith("m") and action_text[1:].isdigit():
        action_idx = int(action_text[1:])
        if action_idx >= 1:
            if quality_text == "correct":
                return f"m{(action_idx - 1) * 2 + 1:02d}"
            if quality_text == "incorrect":
                return f"m{(action_idx - 1) * 2 + 2:02d}"
    if action_text and quality_text:
        return f"{action_text}_{quality_text}"
    return action_text if action_text else "unknown"


def normalize_map01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    a_min = float(arr.min())
    a_max = float(arr.max())
    return (arr - a_min) / (a_max - a_min + 1e-8)


def _tick_step(length: int, target_ticks: int = 10) -> int:
    if length <= 1:
        return 1
    return max(1, int(np.ceil(length / float(target_ticks))))


def _joint_index_ticks(v_len: int) -> np.ndarray:
    """
    Tick positions for every joint row (for Kinect V=22 => positions 0..21).
    """
    if v_len <= 0:
        return np.array([0], dtype=int)
    return np.arange(0, v_len, 1, dtype=int)


def _joint_index_tick_labels_1based(v_len: int) -> np.ndarray:
    """
    Display labels for joints using 1-based indexing (1..V).
    """
    if v_len <= 0:
        return np.array([1], dtype=int)
    return np.arange(1, v_len + 1, 1, dtype=int)


def parse_action_quality_from_name(label_name: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not label_name:
        return None, None
    parts = str(label_name).split("_")
    if len(parts) >= 2 and parts[-1] in {"correct", "incorrect"}:
        return "_".join(parts[:-1]), parts[-1]
    return None, None


def compute_kinematic_metrics(skeleton_ctv: np.ndarray) -> Dict[str, float]:
    """
    skeleton_ctv shape: (C, T, V)
    """
    skel = np.asarray(skeleton_ctv, dtype=np.float32)
    metrics = {
        "avg_speed": 0.0,
        "peak_speed": 0.0,
        "speed_cv": 0.0,
        "pause_ratio": 0.0,
        "jerk_rms": 0.0,
        "stutter_score": 0.0,
    }

    if skel.ndim != 3 or skel.shape[1] < 4:
        metrics["stutter_hint"] = "insufficient_frames"
        return metrics

    # Frame-wise displacement speed over all joints.
    displacement = np.diff(skel, axis=1)  # (C, T-1, V)
    joint_speed = np.linalg.norm(displacement, axis=0)  # (T-1, V)
    speed_t = joint_speed.mean(axis=1)  # (T-1,)

    avg_speed = float(np.mean(speed_t))
    peak_speed = float(np.max(speed_t))
    speed_std = float(np.std(speed_t))
    speed_cv = speed_std / (avg_speed + 1e-8)

    pause_threshold = max(1e-8, 0.15 * peak_speed)
    pause_ratio = float(np.mean(speed_t < pause_threshold))

    # Third-order temporal change proxy via jerk on average speed curve.
    accel_t = np.diff(speed_t)  # (T-2,)
    jerk_t = np.diff(accel_t)   # (T-3,)
    jerk_rms = float(np.sqrt(np.mean(np.square(jerk_t)))) if jerk_t.size > 0 else 0.0

    normalized_jerk = jerk_rms / (avg_speed + 1e-8)
    stutter_score = float(0.6 * pause_ratio + 0.4 * min(1.0, normalized_jerk))

    metrics.update({
        "avg_speed": avg_speed,
        "peak_speed": peak_speed,
        "speed_cv": float(speed_cv),
        "pause_ratio": pause_ratio,
        "jerk_rms": jerk_rms,
        "stutter_score": stutter_score,
    })
    metrics["stutter_hint"] = "possible_stutter" if stutter_score >= 0.45 else "smooth_like"
    return metrics


def format_kinematic_metrics(metrics: Dict[str, float]) -> str:
    return (
        f"avg_speed: {metrics['avg_speed']:.4f}\n"
        f"peak_speed: {metrics['peak_speed']:.4f}\n"
        f"speed_cv: {metrics['speed_cv']:.4f}\n"
        f"pause_ratio: {metrics['pause_ratio']:.2%}\n"
        f"jerk_rms: {metrics['jerk_rms']:.4f}\n"
        f"stutter_score: {metrics['stutter_score']:.3f}\n"
        f"stutter_hint: {metrics.get('stutter_hint', 'unknown')}"
    )


def plot_gradcam_time_joint(
    cam_t_v: np.ndarray,
    save_path: str,
    title: str,
):
    """
    cam_t_v shape: (T_reduced, V)
    x-axis: time, y-axis: joint
    """
    cam_t_v = normalize_map01(cam_t_v)
    t_len, v_len = cam_t_v.shape

    fig, ax = plt.subplots(figsize=(10, 4.6))
    im = ax.imshow(cam_t_v.T, aspect='auto', cmap='jet', origin='lower')
    plt.colorbar(im, ax=ax)

    ax.set_xlabel("Reduced Time (T')")
    ax.set_ylabel("Joint Index (V)")
    ax.set_title(title)

    ax.set_xticks(np.arange(0, t_len, _tick_step(t_len, target_ticks=12), dtype=int))
    y_ticks = _joint_index_ticks(v_len)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(_joint_index_tick_labels_1based(v_len))

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close(fig)


def attention_weights_to_t2t(
    attn_weights: torch.Tensor,
    v_joints: int,
    t_full: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Convert attention from (N, H, S, S) to time-time matrix.
    Steps:
    1) average over heads -> (N, S, S)
    2) reshape S=(T_reduced*V), aggregate joints on query and key -> (N, T_reduced, T_reduced)
    3) optional nearest-neighbor expansion to full T -> (N, T_full, T_full)
    """
    if attn_weights is None or attn_weights.dim() != 4:
        return None, None

    n_batch, _, s_q, s_k = attn_weights.shape
    if s_q != s_k or v_joints <= 0 or (s_q % v_joints) != 0:
        return None, None

    t_reduced = s_q // v_joints
    attn = attn_weights.mean(dim=1)  # (N, S, S)
    attn = attn.reshape(n_batch, t_reduced, v_joints, t_reduced, v_joints).mean(dim=(2, 4))
    # (N, T_reduced, T_reduced)

    if t_full is not None and t_full > 0 and t_full != t_reduced:
        # map full-time index -> reduced-time index (nearest center mapping)
        idx = np.round((np.arange(t_full) + 0.5) * (t_reduced / float(t_full)) - 0.5).astype(np.int64)
        idx = np.clip(idx, 0, t_reduced - 1)
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=attn.device)
        attn = attn.index_select(1, idx_t).index_select(2, idx_t)

    return attn.detach().cpu().numpy(), t_reduced


def plot_transformer_t2t(
    attn_t_t: np.ndarray,
    save_path: str,
    title: str,
    metrics_text: Optional[str] = None,
):
    """
    attn_t_t shape: (T, T) or (T_reduced, T_reduced)
    x-axis: target time frame (key)
    y-axis: source time frame (query)
    """
    attn_t_t = normalize_map01(attn_t_t)
    t_size = attn_t_t.shape[0]

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    im = ax.imshow(attn_t_t, aspect='auto', cmap='magma', origin='lower')
    plt.colorbar(im, ax=ax)

    ax.set_xlabel("Target Reduced Time Frame (Key, T')")
    ax.set_ylabel("Source Reduced Time Frame (Query, T')")
    ax.set_title(title)

    tick_step = _tick_step(t_size, target_ticks=12)
    ticks = np.arange(0, t_size, tick_step, dtype=int)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    if metrics_text:
        ax.text(
            1.02,
            0.02,
            metrics_text,
            transform=ax.transAxes,
            ha='left',
            va='bottom',
            fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close(fig)


def add_group_map(
    group_maps: Dict[str, Dict[str, List[np.ndarray]]],
    action_type: Optional[str],
    quality_label: Optional[str],
    map_array: np.ndarray,
):
    action = action_type if action_type else "unknown_action"
    quality_raw = (quality_label or "").lower()
    if "correct" in quality_raw and "incorrect" not in quality_raw:
        quality = "correct"
    elif "incorrect" in quality_raw:
        quality = "incorrect"
    else:
        return

    if action not in group_maps:
        group_maps[action] = {"correct": [], "incorrect": []}
    group_maps[action][quality].append(normalize_map01(map_array))


def _safe_stack_mean(map_list: List[np.ndarray]) -> Optional[np.ndarray]:
    if not map_list:
        return None
    ref_shape = map_list[0].shape
    same_shape = [m for m in map_list if m.shape == ref_shape]
    if not same_shape:
        return None
    return np.stack(same_shape, axis=0).mean(axis=0)


def _map_reduced_t_to_full_t(t_idx_reduced: int, t_reduced: int, t_full: int) -> int:
    if t_reduced <= 1:
        return 0
    mapped = int(round((t_idx_reduced + 0.5) * (t_full / float(t_reduced)) - 0.5))
    return max(0, min(t_full - 1, mapped))


def _draw_colored_skeleton_on_axis(
    ax: plt.Axes,
    skeleton_ctv: np.ndarray,
    frame_idx: int,
    joint_scores: np.ndarray,
    title: str,
    annotate_top_k: int = 0,
    mirror_x: bool = False,
    fixed_xlim: Optional[Tuple[float, float]] = None,
    fixed_ylim: Optional[Tuple[float, float]] = None,
):
    x = np.asarray(skeleton_ctv[0, frame_idx, :], dtype=np.float32)
    y = np.asarray(skeleton_ctv[1, frame_idx, :], dtype=np.float32)

    if mirror_x:
        x = -x
    scores = normalize_map01(np.asarray(joint_scores, dtype=np.float32))

    for a, b in KINECT_NEIGHBOR_LINKS:
        ax.plot([x[a], x[b]], [y[a], y[b]], color='lightgray', linewidth=1.5, zorder=1)

    sc = ax.scatter(
        x,
        y,
        c=scores,
        cmap='jet',
        s=95,
        edgecolors='black',
        linewidths=0.3,
        zorder=2,
    )

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    if fixed_xlim is None:
        x_margin = max((x_max - x_min) * 0.15, 1e-3)
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
    else:
        ax.set_xlim(float(fixed_xlim[0]), float(fixed_xlim[1]))
    if fixed_ylim is None:
        y_margin = max((y_max - y_min) * 0.15, 1e-3)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
    else:
        ax.set_ylim(float(fixed_ylim[0]), float(fixed_ylim[1]))
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.15)

    if annotate_top_k > 0:
        top_k = min(int(annotate_top_k), scores.size)
        top_idx = np.argsort(scores)[::-1][:top_k]
        for rank, j in enumerate(top_idx, start=1):
            dx = 0.008 * (x_max - x_min + 1e-6) * (1 if rank % 2 else -1)
            dy = 0.01 * (y_max - y_min + 1e-6) * (1 if rank <= 2 else -1)
            ax.text(
                x[j] + dx,
                y[j] + dy,
                f"v{int(j)}",
                fontsize=8,
                color="black",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="gray", alpha=0.82),
                zorder=3,
            )
    return sc


def _infer_should_mirror_by_action(action: str, mean_joint_scores: np.ndarray, eps_ratio: float = 0.08) -> bool:
    """
    Decide whether to mirror a skeleton on X axis for side-consistent visualization.
    Canonical rule: dominant limb side appears on the right.
    """
    action = str(action).lower()
    scores = np.asarray(mean_joint_scores, dtype=np.float32)
    if scores.ndim != 1 or scores.size < 22:
        return False

    if action in LOWER_SIDE_ACTIONS:
        left_score = float(scores[LEFT_LEG_JOINTS].mean())
        right_score = float(scores[RIGHT_LEG_JOINTS].mean())
    elif action in UPPER_SIDE_ACTIONS:
        left_score = float(scores[LEFT_ARM_JOINTS].mean())
        right_score = float(scores[RIGHT_ARM_JOINTS].mean())
    else:
        # Bilateral or unknown action: keep original orientation.
        return False

    total = left_score + right_score + 1e-8
    if abs(left_score - right_score) / total < eps_ratio:
        return False
    return left_score > right_score


def _pose_xy_for_alignment(
    skeleton_ctv: np.ndarray,
    frame_idx: int,
    mirror_x: bool = False,
) -> np.ndarray:
    x = np.asarray(skeleton_ctv[0, frame_idx, :], dtype=np.float32)
    y = np.asarray(skeleton_ctv[1, frame_idx, :], dtype=np.float32)
    if mirror_x:
        x = -x
    xy = np.stack([x, y], axis=1)  # (V,2)
    # translation invariance by root joint
    xy = xy - xy[0:1]
    scale = float(np.sqrt(np.mean(np.sum(np.square(xy), axis=1))) + 1e-8)
    xy = xy / scale
    return xy


def _infer_inc_mirror_to_match_corr(
    corr_skel: np.ndarray,
    corr_frame_idx: int,
    corr_mirror_x: bool,
    inc_skel: np.ndarray,
    inc_frame_idx: int,
    inc_mirror_x_base: bool,
    improve_tol: float = 1e-6,
) -> bool:
    """
    Choose incorrect mirror flag that better aligns to correct sample orientation.
    """
    corr_xy = _pose_xy_for_alignment(corr_skel, corr_frame_idx, mirror_x=corr_mirror_x)

    inc_xy_base = _pose_xy_for_alignment(inc_skel, inc_frame_idx, mirror_x=inc_mirror_x_base)
    inc_xy_flip = _pose_xy_for_alignment(inc_skel, inc_frame_idx, mirror_x=(not inc_mirror_x_base))

    err_base = float(np.mean(np.square(corr_xy - inc_xy_base)))
    err_flip = float(np.mean(np.square(corr_xy - inc_xy_flip)))
    if err_flip + improve_tol < err_base:
        return not inc_mirror_x_base
    return inc_mirror_x_base


def save_gradcam_correct_incorrect_comparisons(
    group_maps: Dict[str, Dict[str, List[np.ndarray]]],
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)
    for action in sorted(group_maps.keys()):
        corr = _safe_stack_mean(group_maps[action]["correct"])
        inc = _safe_stack_mean(group_maps[action]["incorrect"])
        if corr is None or inc is None:
            continue
        corr_label = action_quality_to_compact_label(action, "correct")
        inc_label = action_quality_to_compact_label(action, "incorrect")

        diff = corr - inc
        v_len = corr.shape[1]
        t_len = corr.shape[0]

        fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.6), constrained_layout=True)
        im0 = axes[0].imshow(corr.T, aspect='auto', cmap='jet', origin='lower', vmin=0.0, vmax=1.0)
        im1 = axes[1].imshow(inc.T, aspect='auto', cmap='jet', origin='lower', vmin=0.0, vmax=1.0)
        im2 = axes[2].imshow(diff.T, aspect='auto', cmap='bwr', origin='lower', vmin=-1.0, vmax=1.0)
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        axes[0].set_title(f"{corr_label} (mean)")
        axes[1].set_title(f"{inc_label} (mean)")
        axes[2].set_title(f"Diff ({corr_label} - {inc_label})")

        for ax in axes:
            ax.set_xlabel("Reduced Time (T')")
            ax.set_ylabel("Joint Index (V)")
            ax.set_xticks(np.arange(0, t_len, _tick_step(t_len, target_ticks=12), dtype=int))
            y_ticks = _joint_index_ticks(v_len)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(_joint_index_tick_labels_1based(v_len))

        save_path = os.path.join(save_dir, f"gradcam_compare_{action}.png")
        plt.savefig(save_path, dpi=600)
        plt.close(fig)


def _pad_t_v_map(map_t_v: np.ndarray, t_len: int, v_len: int) -> np.ndarray:
    arr = np.asarray(map_t_v, dtype=np.float32)
    out = np.full((t_len, v_len), np.nan, dtype=np.float32)
    t_take = min(t_len, arr.shape[0])
    v_take = min(v_len, arr.shape[1])
    out[:t_take, :v_take] = arr[:t_take, :v_take]
    return out


def save_gradcam_cross_action_quality_comparison(
    group_maps: Dict[str, Dict[str, List[np.ndarray]]],
    save_dir: str,
    action_a: str = "m07",
    action_b: str = "m10",
    quality: str = "incorrect",
):
    """
    Compare two actions under the same quality label with:
    1) same heatmap bounds (same T/V edges)
    2) same color scale range (shared vmin/vmax)
    """
    os.makedirs(save_dir, exist_ok=True)
    q = str(quality).lower()
    if q not in {"correct", "incorrect"}:
        return

    a = str(action_a).lower()
    b = str(action_b).lower()
    if a not in group_maps or b not in group_maps:
        return

    map_a = _safe_stack_mean(group_maps[a].get(q, []))
    map_b = _safe_stack_mean(group_maps[b].get(q, []))
    if map_a is None or map_b is None:
        return

    t_max = max(int(map_a.shape[0]), int(map_b.shape[0]))
    v_max = max(int(map_a.shape[1]), int(map_b.shape[1]))
    map_a_pad = _pad_t_v_map(map_a, t_max, v_max)
    map_b_pad = _pad_t_v_map(map_b, t_max, v_max)

    merged = np.concatenate(
        [
            np.nan_to_num(map_a_pad, nan=np.nan),
            np.nan_to_num(map_b_pad, nan=np.nan),
        ],
        axis=None,
    )
    merged = merged[np.isfinite(merged)]
    if merged.size <= 0:
        return
    vmin = float(np.min(merged))
    vmax = float(np.max(merged))
    if vmax - vmin < 1e-8:
        vmax = vmin + 1e-8

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), constrained_layout=True)
    im0 = axes[0].imshow(map_a_pad.T, aspect='auto', cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
    im1 = axes[1].imshow(map_b_pad.T, aspect='auto', cmap='jet', origin='lower', vmin=vmin, vmax=vmax)

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_label("Grad-CAM Importance")

    axes[0].set_title(f"{action_quality_to_compact_label(a, q)} (mean)")
    axes[1].set_title(f"{action_quality_to_compact_label(b, q)} (mean)")

    tick_step = _tick_step(t_max, target_ticks=12)
    xticks = np.arange(0, t_max, tick_step, dtype=int)
    yticks = _joint_index_ticks(v_max)
    for ax in axes:
        ax.set_xlabel("Reduced Time (T')")
        ax.set_ylabel("Joint Index (V)")
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_yticklabels(_joint_index_tick_labels_1based(v_max))
        ax.set_xlim(-0.5, t_max - 0.5)
        ax.set_ylim(-0.5, v_max - 0.5)

    save_path = os.path.join(save_dir, f"gradcam_cross_action_{a}_{b}_{q}_same_bounds.png")
    plt.savefig(save_path, dpi=600)
    plt.close(fig)


def _sample_motion_energy_to_reduced_time(skeleton_ctv: np.ndarray, t_reduced: int) -> np.ndarray:
    skel = np.asarray(skeleton_ctv, dtype=np.float32)
    if skel.ndim != 3 or skel.shape[1] < 2 or t_reduced <= 0:
        return np.zeros((max(t_reduced, 1),), dtype=np.float32)
    displacement = np.diff(skel, axis=1)  # (C, T-1, V)
    speed_t = np.linalg.norm(displacement, axis=0).mean(axis=1)  # (T-1,)
    t_full = skel.shape[1]
    sampled = np.zeros((t_reduced,), dtype=np.float32)
    for t_red in range(t_reduced):
        t_full_idx = _map_reduced_t_to_full_t(t_red, t_reduced, t_full)
        speed_idx = min(max(0, t_full_idx), speed_t.shape[0] - 1)
        sampled[t_red] = speed_t[speed_idx]
    return normalize_map01(sampled)


def _select_representative_t_reduced(
    action: str,
    heatmap_t_v: np.ndarray,
    skeleton_ctv: np.ndarray,
) -> int:
    t_reduced = heatmap_t_v.shape[0]
    if t_reduced <= 1:
        return 0

    frame_importance = normalize_map01(heatmap_t_v.mean(axis=1))
    motion_score = _sample_motion_energy_to_reduced_time(skeleton_ctv, t_reduced)

    center = ACTION_PHASE_PRIOR_CENTER.get(action, 0.5) * (t_reduced - 1)
    sigma = max(1.5, 0.18 * (t_reduced - 1))
    t_axis = np.arange(t_reduced, dtype=np.float32)
    prior = np.exp(-0.5 * np.square((t_axis - center) / sigma)).astype(np.float32)
    prior = normalize_map01(prior)

    combined = 0.55 * frame_importance + 0.25 * motion_score + 0.20 * prior
    return int(np.argmax(combined))


def _select_representative_item(
    items: List[Dict[str, np.ndarray]],
    mean_heatmap: np.ndarray,
    rank: int = 1,
) -> Optional[Dict[str, np.ndarray]]:
    if not items:
        return None
    mean_norm = normalize_map01(mean_heatmap)
    ranked: List[Tuple[float, Dict[str, np.ndarray]]] = []
    for item in items:
        h = item.get("heatmap")
        if h is None or np.asarray(h).shape != mean_norm.shape:
            continue
        dist = float(np.mean(np.square(normalize_map01(h) - mean_norm)))
        ranked.append((dist, item))
    if not ranked:
        return items[0]
    ranked.sort(key=lambda x: x[0])
    rank_1based = max(1, int(rank))
    pick_idx = min(rank_1based - 1, len(ranked) - 1)
    return ranked[pick_idx][1]


def save_gradcam_action_representative_skeletons(
    group_examples: Dict[str, Dict[str, List[Dict[str, np.ndarray]]]],
    save_dir: str,
    quality: str = "correct",
    top_k_joints: int = 0,
    representative_rank_overrides: Optional[Dict[Tuple[str, str], int]] = None,
):
    """
    Export representative Grad-CAM skeletons for m01..m10.
    For each action, one representative frame is selected using:
      Grad-CAM frame importance + motion energy + action-specific phase prior.
    """
    os.makedirs(save_dir, exist_ok=True)
    quality_key = "correct" if quality != "incorrect" else "incorrect"
    rep_rank_map: Dict[Tuple[str, str], int] = {}
    if representative_rank_overrides is not None:
        for k, v in representative_rank_overrides.items():
            if isinstance(k, tuple) and len(k) == 2:
                a = str(k[0]).strip().lower()
                q = str(k[1]).strip().lower()
                rep_rank_map[(a, q)] = max(1, int(v))

    fig, axes = plt.subplots(2, 5, figsize=(20, 8.5), constrained_layout=True)
    subplot_tags = list("abcdefghij")
    color_handle = None
    records: List[Dict[str, object]] = []

    for idx, action in enumerate(ACTION_ORDER):
        ax = axes.flat[idx]
        label_compact = action_quality_to_compact_label(action, quality_key)
        title_prefix = f"({subplot_tags[idx]}) {label_compact}"

        items = group_examples.get(action, {}).get(quality_key, [])
        maps = [item["heatmap"] for item in items if "heatmap" in item]
        mean_map = _safe_stack_mean(maps)
        if mean_map is None:
            ax.set_title(f"{title_prefix}\nno {quality_key} sample", fontsize=10)
            ax.axis("off")
            continue

        rep_rank = rep_rank_map.get((str(action).lower(), str(quality_key).lower()), 1)
        rep_item = _select_representative_item(items, mean_map, rank=rep_rank)
        if rep_item is None or rep_item.get("skeleton") is None:
            ax.set_title(f"{title_prefix}\nmissing skeleton", fontsize=10)
            ax.axis("off")
            continue

        skeleton = rep_item["skeleton"]
        t_red = _select_representative_t_reduced(action, mean_map, skeleton)
        t_full = _map_reduced_t_to_full_t(t_red, mean_map.shape[0], skeleton.shape[1])
        joint_scores = mean_map[t_red]
        mirror_x = _infer_should_mirror_by_action(action, mean_map.mean(axis=0))

        color_handle = _draw_colored_skeleton_on_axis(
            ax=ax,
            skeleton_ctv=skeleton,
            frame_idx=t_full,
            joint_scores=joint_scores,
            title=f"{title_prefix}\nredT={t_red}, fullT={t_full}",
            annotate_top_k=top_k_joints,
            mirror_x=mirror_x,
        )

        top_joints = np.argsort(normalize_map01(joint_scores))[::-1][:min(top_k_joints, joint_scores.size)]
        records.append({
            "action": action,
            "display_name": label_compact,
            "quality": quality_key,
            "selected_reduced_t": int(t_red),
            "selected_full_t": int(t_full),
            "selected_item_rank": int(rep_rank),
            "selected_source_prefix": str(rep_item.get("source_prefix", "")),
            "mirrored_x": int(mirror_x),
            "top_joint_indices": "|".join([f"v{int(j)}" for j in top_joints]),
        })

    if color_handle is not None:
        fig.colorbar(color_handle, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01, label="Joint Importance")

    panel_path = os.path.join(save_dir, f"gradcam_action_representative_skeletons_{quality_key}.png")
    plt.savefig(panel_path, dpi=600)
    plt.close(fig)

    if records:
        csv_path = os.path.join(save_dir, f"gradcam_action_representative_selection_{quality_key}.csv")
        save_kinematic_records(records, csv_path)


def save_gradcam_phase_aligned_skeleton_comparisons(
    group_examples: Dict[str, Dict[str, List[Dict[str, np.ndarray]]]],
    save_dir: str,
    phase_ratios: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    top_k_joints: int = 0,
    force_flip_incorrect_actions: Tuple[str, ...] = (),
    representative_rank_overrides: Optional[Dict[Tuple[str, str], int]] = None,
):
    """
    For each action m01..m10, compare correct vs incorrect at the same normalized motion phases.
    This avoids comparing arbitrary different frames and improves interpretability.
    """
    os.makedirs(save_dir, exist_ok=True)
    phase_ratios = tuple(float(p) for p in phase_ratios)

    records: List[Dict[str, object]] = []

    force_flip_incorrect_actions = tuple(str(a).lower() for a in force_flip_incorrect_actions)
    rep_rank_map: Dict[Tuple[str, str], int] = {}
    if representative_rank_overrides is not None:
        for k, v in representative_rank_overrides.items():
            if isinstance(k, tuple) and len(k) == 2:
                a = str(k[0]).strip().lower()
                q = str(k[1]).strip().lower()
                rep_rank_map[(a, q)] = max(1, int(v))

    def _phase_windows_by_ratios(t_len: int, ratios: Tuple[float, ...]) -> List[Tuple[int, int]]:
        """
        Build inclusive reduced-time windows for each phase ratio by midpoint splitting.
        Example for 5 phases: [0%-12.5%], [12.5%-37.5%], ..., [87.5%-100%].
        """
        if t_len <= 1:
            return [(0, 0) for _ in ratios]
        centers = [max(0.0, min(1.0, float(r))) for r in ratios]
        center_idx = [int(round(r * (t_len - 1))) for r in centers]
        windows: List[Tuple[int, int]] = []
        for i, c in enumerate(center_idx):
            if i == 0:
                left = 0
            else:
                left = int(round(0.5 * (center_idx[i - 1] + c)))
            if i == len(center_idx) - 1:
                right = t_len - 1
            else:
                right = int(round(0.5 * (c + center_idx[i + 1])))
            left = max(0, min(t_len - 1, left))
            right = max(left, min(t_len - 1, right))
            windows.append((left, right))
        return windows

    for action in ACTION_ORDER:
        corr_items = group_examples.get(action, {}).get("correct", [])
        inc_items = group_examples.get(action, {}).get("incorrect", [])
        if not corr_items or not inc_items:
            continue

        corr_maps = [item["heatmap"] for item in corr_items if "heatmap" in item]
        inc_maps = [item["heatmap"] for item in inc_items if "heatmap" in item]
        corr_mean = _safe_stack_mean(corr_maps)
        inc_mean = _safe_stack_mean(inc_maps)
        if corr_mean is None or inc_mean is None:
            continue

        corr_rank = rep_rank_map.get((str(action).lower(), "correct"), 1)
        inc_rank = rep_rank_map.get((str(action).lower(), "incorrect"), 1)
        corr_rep = _select_representative_item(corr_items, corr_mean, rank=corr_rank)
        inc_rep = _select_representative_item(inc_items, inc_mean, rank=inc_rank)
        if corr_rep is None or inc_rep is None:
            continue

        corr_skel = corr_rep.get("skeleton")
        inc_skel = inc_rep.get("skeleton")
        if corr_skel is None or inc_skel is None:
            continue
        corr_mirror_x = _infer_should_mirror_by_action(action, corr_mean.mean(axis=0))
        inc_mirror_x = _infer_should_mirror_by_action(action, inc_mean.mean(axis=0))
        # Enforce pairwise orientation consistency (incorrect follows correct orientation when helpful).
        ref_t_red_corr = int(round(0.5 * (corr_mean.shape[0] - 1)))
        ref_t_red_inc = int(round(0.5 * (inc_mean.shape[0] - 1)))
        ref_t_full_corr = _map_reduced_t_to_full_t(ref_t_red_corr, corr_mean.shape[0], corr_skel.shape[1])
        ref_t_full_inc = _map_reduced_t_to_full_t(ref_t_red_inc, inc_mean.shape[0], inc_skel.shape[1])
        inc_mirror_x = _infer_inc_mirror_to_match_corr(
            corr_skel=corr_skel,
            corr_frame_idx=ref_t_full_corr,
            corr_mirror_x=corr_mirror_x,
            inc_skel=inc_skel,
            inc_frame_idx=ref_t_full_inc,
            inc_mirror_x_base=inc_mirror_x,
        )
        if action.lower() in force_flip_incorrect_actions:
            inc_mirror_x = not inc_mirror_x

        n_phase = len(phase_ratios)
        fig, axes = plt.subplots(2, n_phase, figsize=(3.8 * n_phase, 7.1), constrained_layout=True)
        corr_label = action_quality_to_compact_label(action, "correct")
        inc_label = action_quality_to_compact_label(action, "incorrect")
        color_handle = None
        corr_windows = _phase_windows_by_ratios(corr_mean.shape[0], phase_ratios)
        inc_windows = _phase_windows_by_ratios(inc_mean.shape[0], phase_ratios)
        phase_infos: List[Dict[str, int]] = []

        for phase in phase_ratios:
            phase_percent = int(round(phase * 100))
            t_red_corr = int(round(phase * (corr_mean.shape[0] - 1)))
            t_red_inc = int(round(phase * (inc_mean.shape[0] - 1)))
            t_red_corr = max(0, min(corr_mean.shape[0] - 1, t_red_corr))
            t_red_inc = max(0, min(inc_mean.shape[0] - 1, t_red_inc))
            t_full_corr = _map_reduced_t_to_full_t(t_red_corr, corr_mean.shape[0], corr_skel.shape[1])
            t_full_inc = _map_reduced_t_to_full_t(t_red_inc, inc_mean.shape[0], inc_skel.shape[1])
            corr_override = PHASE_ALIGNED_FULL_FRAME_OVERRIDES.get((str(action).lower(), "correct", phase_percent))
            if corr_override is not None:
                t_full_corr = int(max(0, min(corr_skel.shape[1] - 1, int(corr_override))))
            phase_infos.append({
                "t_red_corr": int(t_red_corr),
                "t_red_inc": int(t_red_inc),
                "t_full_corr": int(t_full_corr),
                "t_full_inc": int(t_full_inc),
            })

        # Shared limits so all boxes in this panel are consistent.
        x_all: List[np.ndarray] = []
        y_all: List[np.ndarray] = []
        for info in phase_infos:
            x_corr = np.asarray(corr_skel[0, info["t_full_corr"], :], dtype=np.float32)
            y_corr = np.asarray(corr_skel[1, info["t_full_corr"], :], dtype=np.float32)
            if corr_mirror_x:
                x_corr = -x_corr
            x_inc = np.asarray(inc_skel[0, info["t_full_inc"], :], dtype=np.float32)
            y_inc = np.asarray(inc_skel[1, info["t_full_inc"], :], dtype=np.float32)
            if inc_mirror_x:
                x_inc = -x_inc
            x_all.extend([x_corr, x_inc])
            y_all.extend([y_corr, y_inc])
        x_concat = np.concatenate(x_all, axis=0)
        y_concat = np.concatenate(y_all, axis=0)
        x_min, x_max = float(np.min(x_concat)), float(np.max(x_concat))
        y_min, y_max = float(np.min(y_concat)), float(np.max(y_concat))
        x_margin = max((x_max - x_min) * 0.15, 1e-3)
        y_margin = max((y_max - y_min) * 0.15, 1e-3)
        shared_xlim = (x_min - x_margin, x_max + x_margin)
        shared_ylim = (y_min - y_margin, y_max + y_margin)

        for phase_idx, phase in enumerate(phase_ratios):
            t_red_corr = phase_infos[phase_idx]["t_red_corr"]
            t_red_inc = phase_infos[phase_idx]["t_red_inc"]
            t_full_corr = phase_infos[phase_idx]["t_full_corr"]
            t_full_inc = phase_infos[phase_idx]["t_full_inc"]
            corr_l, corr_r = corr_windows[phase_idx]
            inc_l, inc_r = inc_windows[phase_idx]
            corr_joint_scores = corr_mean[corr_l:corr_r + 1].mean(axis=0)
            inc_joint_scores = inc_mean[inc_l:inc_r + 1].mean(axis=0)

            color_handle = _draw_colored_skeleton_on_axis(
                axes[0, phase_idx],
                corr_skel,
                t_full_corr,
                corr_joint_scores,
                title=f"{corr_label} phase={int(round(phase * 100))}%",
                annotate_top_k=top_k_joints,
                mirror_x=corr_mirror_x,
                fixed_xlim=shared_xlim,
                fixed_ylim=shared_ylim,
            )
            _draw_colored_skeleton_on_axis(
                axes[1, phase_idx],
                inc_skel,
                t_full_inc,
                inc_joint_scores,
                title=f"{inc_label} phase={int(round(phase * 100))}%",
                annotate_top_k=top_k_joints,
                mirror_x=inc_mirror_x,
                fixed_xlim=shared_xlim,
                fixed_ylim=shared_ylim,
            )
            for row in range(2):
                axes[row, phase_idx].set_xlabel("")
                axes[row, phase_idx].set_ylabel("")
                axes[row, phase_idx].set_xticks([])
                axes[row, phase_idx].set_yticks([])
                axes[row, phase_idx].grid(False)

            corr_top = np.argsort(normalize_map01(corr_joint_scores))[::-1][:min(top_k_joints, corr_joint_scores.size)]
            inc_top = np.argsort(normalize_map01(inc_joint_scores))[::-1][:min(top_k_joints, inc_joint_scores.size)]
            records.append({
                "action": action,
                "display_name": ACTION_DISPLAY_NAMES.get(action, action),
                "phase_percent": int(round(phase * 100)),
                "corr_reduced_t": int(t_red_corr),
                "corr_full_t": int(t_full_corr),
                "corr_window_start_t": int(corr_l),
                "corr_window_end_t": int(corr_r),
                "corr_mirrored_x": int(corr_mirror_x),
                "corr_item_rank": int(corr_rank),
                "corr_source_prefix": str(corr_rep.get("source_prefix", "")),
                "inc_reduced_t": int(t_red_inc),
                "inc_full_t": int(t_full_inc),
                "inc_window_start_t": int(inc_l),
                "inc_window_end_t": int(inc_r),
                "inc_mirrored_x": int(inc_mirror_x),
                "inc_item_rank": int(inc_rank),
                "inc_source_prefix": str(inc_rep.get("source_prefix", "")),
                "corr_top_joints": "|".join([f"v{int(j)}" for j in corr_top]),
                "inc_top_joints": "|".join([f"v{int(j)}" for j in inc_top]),
            })

        if color_handle is not None:
            fig.colorbar(color_handle, ax=axes.ravel().tolist(), fraction=0.016, pad=0.01, label="Joint Importance")

        save_path = os.path.join(save_dir, f"gradcam_phase_aligned_compare_{action}.png")
        plt.savefig(save_path, dpi=600)
        plt.close(fig)

    if records:
        csv_path = os.path.join(save_dir, "gradcam_phase_aligned_selection.csv")
        save_kinematic_records(records, csv_path)


def save_gradcam_comparison_with_skeleton(
    group_examples: Dict[str, Dict[str, List[Dict[str, np.ndarray]]]],
    save_dir: str,
    representative_rank_overrides: Optional[Dict[Tuple[str, str], int]] = None,
):
    """
    group_examples[action][quality] each item:
      {"heatmap": (T_reduced, V), "skeleton": (C, T_full, V)}
    """
    os.makedirs(save_dir, exist_ok=True)
    rep_rank_map: Dict[Tuple[str, str], int] = {}
    if representative_rank_overrides is not None:
        for k, v in representative_rank_overrides.items():
            if isinstance(k, tuple) and len(k) == 2:
                a = str(k[0]).strip().lower()
                q = str(k[1]).strip().lower()
                rep_rank_map[(a, q)] = max(1, int(v))

    for action in sorted(group_examples.keys()):
        corr_label = action_quality_to_compact_label(action, "correct")
        inc_label = action_quality_to_compact_label(action, "incorrect")
        corr_items = group_examples[action].get("correct", [])
        inc_items = group_examples[action].get("incorrect", [])
        if not corr_items or not inc_items:
            continue

        corr_maps = [item["heatmap"] for item in corr_items if "heatmap" in item]
        inc_maps = [item["heatmap"] for item in inc_items if "heatmap" in item]
        corr_mean = _safe_stack_mean(corr_maps)
        inc_mean = _safe_stack_mean(inc_maps)
        if corr_mean is None or inc_mean is None:
            continue

        corr_rank = rep_rank_map.get((str(action).lower(), "correct"), 1)
        inc_rank = rep_rank_map.get((str(action).lower(), "incorrect"), 1)
        corr_rep = _select_representative_item(corr_items, corr_mean, rank=corr_rank)
        inc_rep = _select_representative_item(inc_items, inc_mean, rank=inc_rank)
        if corr_rep is None or inc_rep is None:
            continue
        corr_skel = corr_rep.get("skeleton")
        inc_skel = inc_rep.get("skeleton")
        if corr_skel is None or inc_skel is None:
            continue

        corr_t_red = int(np.argmax(corr_mean.mean(axis=1)))
        inc_t_red = int(np.argmax(inc_mean.mean(axis=1)))
        corr_t_full = _map_reduced_t_to_full_t(corr_t_red, corr_mean.shape[0], corr_skel.shape[1])
        inc_t_full = _map_reduced_t_to_full_t(inc_t_red, inc_mean.shape[0], inc_skel.shape[1])
        corr_joint_scores = corr_mean[corr_t_red]
        inc_joint_scores = inc_mean[inc_t_red]
        corr_mirror_x = _infer_should_mirror_by_action(action, corr_mean.mean(axis=0))
        inc_mirror_x = _infer_should_mirror_by_action(action, inc_mean.mean(axis=0))
        inc_mirror_x = _infer_inc_mirror_to_match_corr(
            corr_skel=corr_skel,
            corr_frame_idx=corr_t_full,
            corr_mirror_x=corr_mirror_x,
            inc_skel=inc_skel,
            inc_frame_idx=inc_t_full,
            inc_mirror_x_base=inc_mirror_x,
        )

        fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), constrained_layout=True)

        im0 = axes[0, 0].imshow(corr_mean.T, aspect='auto', cmap='jet', origin='lower', vmin=0.0, vmax=1.0)
        axes[0, 0].set_title(f"{corr_label} - Mean Grad-CAM")
        axes[0, 0].set_xlabel("Reduced Time (T')")
        axes[0, 0].set_ylabel("Joint Index (V)")
        corr_y_ticks = _joint_index_ticks(corr_mean.shape[1])
        axes[0, 0].set_yticks(corr_y_ticks)
        axes[0, 0].set_yticklabels(_joint_index_tick_labels_1based(corr_mean.shape[1]))
        axes[0, 0].set_xticks(np.arange(0, corr_mean.shape[0], _tick_step(corr_mean.shape[0], 12), dtype=int))
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        sc0 = _draw_colored_skeleton_on_axis(
            axes[0, 1],
            corr_skel,
            corr_t_full,
            corr_joint_scores,
            f"{corr_label} Skeleton @ fullT={corr_t_full}",
            mirror_x=corr_mirror_x,
        )
        plt.colorbar(sc0, ax=axes[0, 1], fraction=0.046, pad=0.04, label="Joint Importance")

        im1 = axes[1, 0].imshow(inc_mean.T, aspect='auto', cmap='jet', origin='lower', vmin=0.0, vmax=1.0)
        axes[1, 0].set_title(f"{inc_label} - Mean Grad-CAM")
        axes[1, 0].set_xlabel("Reduced Time (T')")
        axes[1, 0].set_ylabel("Joint Index (V)")
        inc_y_ticks = _joint_index_ticks(inc_mean.shape[1])
        axes[1, 0].set_yticks(inc_y_ticks)
        axes[1, 0].set_yticklabels(_joint_index_tick_labels_1based(inc_mean.shape[1]))
        axes[1, 0].set_xticks(np.arange(0, inc_mean.shape[0], _tick_step(inc_mean.shape[0], 12), dtype=int))
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

        sc1 = _draw_colored_skeleton_on_axis(
            axes[1, 1],
            inc_skel,
            inc_t_full,
            inc_joint_scores,
            f"{inc_label} Skeleton @ fullT={inc_t_full}",
            mirror_x=inc_mirror_x,
        )
        plt.colorbar(sc1, ax=axes[1, 1], fraction=0.046, pad=0.04, label="Joint Importance")

        save_path = os.path.join(save_dir, f"gradcam_skeleton_heatmap_compare_{action}.png")
        plt.savefig(save_path, dpi=600)
        plt.close(fig)


def save_attention_correct_incorrect_comparisons(
    group_maps: Dict[str, Dict[str, List[np.ndarray]]],
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)
    for action in sorted(group_maps.keys()):
        corr_label = action_quality_to_compact_label(action, "correct")
        inc_label = action_quality_to_compact_label(action, "incorrect")
        corr = _safe_stack_mean(group_maps[action]["correct"])
        inc = _safe_stack_mean(group_maps[action]["incorrect"])
        if corr is None or inc is None:
            continue

        corr = normalize_map01(corr)
        inc = normalize_map01(inc)
        diff = corr - inc
        t_size = corr.shape[0]

        fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.0), constrained_layout=True)
        im0 = axes[0].imshow(corr, aspect='auto', cmap='magma', origin='lower', vmin=0.0, vmax=1.0)
        im1 = axes[1].imshow(inc, aspect='auto', cmap='magma', origin='lower', vmin=0.0, vmax=1.0)
        im2 = axes[2].imshow(diff, aspect='auto', cmap='bwr', origin='lower', vmin=-1.0, vmax=1.0)
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        axes[0].set_title(f"{corr_label} (mean)")
        axes[1].set_title(f"{inc_label} (mean)")
        axes[2].set_title(f"Diff ({corr_label} - {inc_label})")

        tick_step = _tick_step(t_size, target_ticks=12)
        ticks = np.arange(0, t_size, tick_step, dtype=int)
        for ax in axes:
            ax.set_xlabel("Target Reduced Time Frame (Key, T')")
            ax.set_ylabel("Source Reduced Time Frame (Query, T')")
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)

        save_path = os.path.join(save_dir, f"attention_t2t_compare_{action}.png")
        plt.savefig(save_path, dpi=600)
        plt.close(fig)


def save_attention_t2t_comparison_with_skeleton(
    group_examples: Dict[str, Dict[str, List[Dict[str, np.ndarray]]]],
    save_dir: str,
):
    """
    group_examples[action][quality] each item:
      {"t2t_map": (T, T), "token_map": (T_reduced, V), "skeleton": (C, T_full, V)}
    """
    os.makedirs(save_dir, exist_ok=True)
    for action in sorted(group_examples.keys()):
        corr_label = action_quality_to_compact_label(action, "correct")
        inc_label = action_quality_to_compact_label(action, "incorrect")
        corr_items = group_examples[action].get("correct", [])
        inc_items = group_examples[action].get("incorrect", [])
        if not corr_items or not inc_items:
            continue

        corr_t2t = _safe_stack_mean([i["t2t_map"] for i in corr_items if "t2t_map" in i])
        inc_t2t = _safe_stack_mean([i["t2t_map"] for i in inc_items if "t2t_map" in i])
        corr_token = _safe_stack_mean([i["token_map"] for i in corr_items if "token_map" in i])
        inc_token = _safe_stack_mean([i["token_map"] for i in inc_items if "token_map" in i])
        if corr_t2t is None or inc_t2t is None or corr_token is None or inc_token is None:
            continue

        corr_skel = corr_items[0].get("skeleton")
        inc_skel = inc_items[0].get("skeleton")
        if corr_skel is None or inc_skel is None:
            continue

        corr_t_red = int(np.argmax(corr_token.mean(axis=1)))
        inc_t_red = int(np.argmax(inc_token.mean(axis=1)))
        corr_t_full = _map_reduced_t_to_full_t(corr_t_red, corr_token.shape[0], corr_skel.shape[1])
        inc_t_full = _map_reduced_t_to_full_t(inc_t_red, inc_token.shape[0], inc_skel.shape[1])
        corr_joint_scores = corr_token[corr_t_red]
        inc_joint_scores = inc_token[inc_t_red]
        corr_mirror_x = _infer_should_mirror_by_action(action, corr_token.mean(axis=0))
        inc_mirror_x = _infer_should_mirror_by_action(action, inc_token.mean(axis=0))

        fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.0), constrained_layout=True)

        corr_t2t = normalize_map01(corr_t2t)
        inc_t2t = normalize_map01(inc_t2t)
        im0 = axes[0, 0].imshow(corr_t2t, aspect='auto', cmap='magma', origin='lower', vmin=0.0, vmax=1.0)
        axes[0, 0].set_title(f"{corr_label} - Mean Transformer T x T")
        axes[0, 0].set_xlabel("Target Reduced Time Frame (Key, T')")
        axes[0, 0].set_ylabel("Source Reduced Time Frame (Query, T')")
        corr_ticks = np.arange(0, corr_t2t.shape[0], _tick_step(corr_t2t.shape[0], 12), dtype=int)
        axes[0, 0].set_xticks(corr_ticks)
        axes[0, 0].set_yticks(corr_ticks)
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        sc0 = _draw_colored_skeleton_on_axis(
            axes[0, 1],
            corr_skel,
            corr_t_full,
            corr_joint_scores,
            f"{corr_label} Skeleton @ fullT={corr_t_full}",
            mirror_x=corr_mirror_x,
        )
        plt.colorbar(sc0, ax=axes[0, 1], fraction=0.046, pad=0.04, label="Joint Importance")

        im1 = axes[1, 0].imshow(inc_t2t, aspect='auto', cmap='magma', origin='lower', vmin=0.0, vmax=1.0)
        axes[1, 0].set_title(f"{inc_label} - Mean Transformer T x T")
        axes[1, 0].set_xlabel("Target Reduced Time Frame (Key, T')")
        axes[1, 0].set_ylabel("Source Reduced Time Frame (Query, T')")
        inc_ticks = np.arange(0, inc_t2t.shape[0], _tick_step(inc_t2t.shape[0], 12), dtype=int)
        axes[1, 0].set_xticks(inc_ticks)
        axes[1, 0].set_yticks(inc_ticks)
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

        sc1 = _draw_colored_skeleton_on_axis(
            axes[1, 1],
            inc_skel,
            inc_t_full,
            inc_joint_scores,
            f"{inc_label} Skeleton @ fullT={inc_t_full}",
            mirror_x=inc_mirror_x,
        )
        plt.colorbar(sc1, ax=axes[1, 1], fraction=0.046, pad=0.04, label="Joint Importance")

        save_path = os.path.join(save_dir, f"attention_t2t_skeleton_heatmap_compare_{action}.png")
        plt.savefig(save_path, dpi=600)
        plt.close(fig)


def _attention_target_curve(attn_t_t: np.ndarray) -> np.ndarray:
    """
    attn_t_t shape: (source/query, target/key)
    Returns key-target importance curve by averaging over source axis.
    """
    return normalize_map01(np.asarray(attn_t_t, dtype=np.float32).mean(axis=0))


def _attention_long_range_metrics(attn_t_t: np.ndarray, long_ratio_threshold: float = 0.25) -> Dict[str, float]:
    """
    Compute simple global-dependency metrics from T' x T' attention.
    - normalized_mean_span: expected |i-j| / (T-1)
    - long_range_ratio: mass where |i-j| > threshold*(T-1)
    """
    attn = np.asarray(attn_t_t, dtype=np.float32)
    if attn.ndim != 2 or attn.shape[0] != attn.shape[1] or attn.size == 0:
        return {"normalized_mean_span": 0.0, "long_range_ratio": 0.0}

    t_size = attn.shape[0]
    if t_size <= 1:
        return {"normalized_mean_span": 0.0, "long_range_ratio": 0.0}

    weights = attn / (float(attn.sum()) + 1e-8)
    idx = np.arange(t_size, dtype=np.float32)
    dist = np.abs(idx[:, None] - idx[None, :])  # (T, T)

    max_dist = float(t_size - 1)
    normalized_mean_span = float((weights * dist).sum() / (max_dist + 1e-8))
    long_range_mask = dist > (long_ratio_threshold * max_dist)
    long_range_ratio = float(weights[long_range_mask].sum())
    return {
        "normalized_mean_span": normalized_mean_span,
        "long_range_ratio": long_range_ratio,
    }


def _attention_phase_matrix(attn_t_t: np.ndarray, n_phase: int = 5) -> np.ndarray:
    """
    Reduce T' x T' to n_phase x n_phase block-mean matrix.
    """
    attn = np.asarray(attn_t_t, dtype=np.float32)
    t_size = attn.shape[0]
    n_phase = max(2, int(n_phase))
    bins = np.linspace(0, t_size, n_phase + 1, dtype=int)
    phase_mat = np.zeros((n_phase, n_phase), dtype=np.float32)

    for i in range(n_phase):
        rs, re = bins[i], bins[i + 1]
        if re <= rs:
            re = min(t_size, rs + 1)
        for j in range(n_phase):
            cs, ce = bins[j], bins[j + 1]
            if ce <= cs:
                ce = min(t_size, cs + 1)
            block = attn[rs:re, cs:ce]
            phase_mat[i, j] = float(block.mean()) if block.size > 0 else 0.0
    return phase_mat


def save_attention_target_curve_comparisons(
    group_maps: Dict[str, Dict[str, List[np.ndarray]]],
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)
    for action in sorted(group_maps.keys()):
        corr_label = action_quality_to_compact_label(action, "correct")
        inc_label = action_quality_to_compact_label(action, "incorrect")
        corr = _safe_stack_mean(group_maps[action]["correct"])
        inc = _safe_stack_mean(group_maps[action]["incorrect"])
        if corr is None or inc is None:
            continue

        corr_curve = _attention_target_curve(corr)
        inc_curve = _attention_target_curve(inc)
        diff_curve = corr_curve - inc_curve

        x = np.arange(corr_curve.shape[0], dtype=int)
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        ax.plot(x, corr_curve, color="#1f77b4", linewidth=2.0, label="Correct")
        ax.plot(x, inc_curve, color="#ff7f0e", linewidth=2.0, label="Incorrect")
        ax.plot(x, diff_curve, color="#2ca02c", linewidth=1.8, linestyle="--", label="Diff (C-I)")
        ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.6)
        ax.set_title(f"Target-Time Attention: {corr_label} vs {inc_label}")
        ax.set_xlabel("Target Reduced Time Frame (Key, T')")
        ax.set_ylabel("Normalized Attention")
        ax.set_xticks(np.arange(0, corr_curve.shape[0], _tick_step(corr_curve.shape[0], target_ticks=12), dtype=int))
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=9)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"attention_target_curve_compare_{action}.png")
        plt.savefig(save_path, dpi=600)
        plt.close(fig)


def save_attention_phase_matrix_comparisons(
    group_maps: Dict[str, Dict[str, List[np.ndarray]]],
    save_dir: str,
    n_phase: int = 5,
):
    os.makedirs(save_dir, exist_ok=True)
    phase_labels = [f"P{i + 1}" for i in range(max(2, int(n_phase)))]

    for action in sorted(group_maps.keys()):
        corr_label = action_quality_to_compact_label(action, "correct")
        inc_label = action_quality_to_compact_label(action, "incorrect")
        corr = _safe_stack_mean(group_maps[action]["correct"])
        inc = _safe_stack_mean(group_maps[action]["incorrect"])
        if corr is None or inc is None:
            continue

        corr_phase = normalize_map01(_attention_phase_matrix(corr, n_phase=n_phase))
        inc_phase = normalize_map01(_attention_phase_matrix(inc, n_phase=n_phase))
        diff_phase = corr_phase - inc_phase

        fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4), constrained_layout=True)
        im0 = axes[0].imshow(corr_phase, cmap="magma", origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
        im1 = axes[1].imshow(inc_phase, cmap="magma", origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
        im2 = axes[2].imshow(diff_phase, cmap="bwr", origin="lower", aspect="auto", vmin=-1.0, vmax=1.0)
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        axes[0].set_title(f"{corr_label} (phase)")
        axes[1].set_title(f"{inc_label} (phase)")
        axes[2].set_title(f"Diff ({corr_label} - {inc_label})")

        for ax in axes:
            ax.set_xlabel("Target Phase (Key)")
            ax.set_ylabel("Source Phase (Query)")
            ax.set_xticks(np.arange(len(phase_labels), dtype=int))
            ax.set_yticks(np.arange(len(phase_labels), dtype=int))
            ax.set_xticklabels(phase_labels)
            ax.set_yticklabels(phase_labels)

        save_path = os.path.join(save_dir, f"attention_phase_matrix_compare_{action}.png")
        plt.savefig(save_path, dpi=600)
        plt.close(fig)


def save_attention_long_range_metrics(
    group_maps: Dict[str, Dict[str, List[np.ndarray]]],
    save_dir: str,
    long_ratio_threshold: float = 0.25,
):
    os.makedirs(save_dir, exist_ok=True)
    records: List[Dict[str, object]] = []

    for action in sorted(group_maps.keys()):
        corr = _safe_stack_mean(group_maps[action]["correct"])
        inc = _safe_stack_mean(group_maps[action]["incorrect"])
        if corr is None or inc is None:
            continue

        corr_m = _attention_long_range_metrics(corr, long_ratio_threshold=long_ratio_threshold)
        inc_m = _attention_long_range_metrics(inc, long_ratio_threshold=long_ratio_threshold)
        records.append({
            "action": action,
            "corr_normalized_mean_span": corr_m["normalized_mean_span"],
            "inc_normalized_mean_span": inc_m["normalized_mean_span"],
            "diff_normalized_mean_span": corr_m["normalized_mean_span"] - inc_m["normalized_mean_span"],
            "corr_long_range_ratio": corr_m["long_range_ratio"],
            "inc_long_range_ratio": inc_m["long_range_ratio"],
            "diff_long_range_ratio": corr_m["long_range_ratio"] - inc_m["long_range_ratio"],
            "long_ratio_threshold": float(long_ratio_threshold),
        })

    if not records:
        return

    csv_path = os.path.join(save_dir, "attention_long_range_metrics.csv")
    save_kinematic_records(records, csv_path)

    actions = [r["action"] for r in records]
    x = np.arange(len(actions))
    width = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), constrained_layout=True)
    corr_span = [r["corr_normalized_mean_span"] for r in records]
    inc_span = [r["inc_normalized_mean_span"] for r in records]
    corr_long = [r["corr_long_range_ratio"] for r in records]
    inc_long = [r["inc_long_range_ratio"] for r in records]

    axes[0].bar(x - width / 2, corr_span, width=width, color="#1f77b4", label="Correct")
    axes[0].bar(x + width / 2, inc_span, width=width, color="#ff7f0e", label="Incorrect")
    axes[0].set_title("Normalized Mean Attention Span by Action")
    axes[0].set_ylabel("Span (0-1)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(actions, rotation=0)
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(loc="best")

    axes[1].bar(x - width / 2, corr_long, width=width, color="#1f77b4", label="Correct")
    axes[1].bar(x + width / 2, inc_long, width=width, color="#ff7f0e", label="Incorrect")
    axes[1].set_title(f"Long-Range Attention Ratio by Action (|i-j| > {long_ratio_threshold:.2f}*T')")
    axes[1].set_ylabel("Ratio (0-1)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(actions, rotation=0)
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(loc="best")

    bar_path = os.path.join(save_dir, "attention_long_range_metrics_barplot.png")
    plt.savefig(bar_path, dpi=600)
    plt.close(fig)


def save_kinematic_records(records: List[Dict[str, object]], save_path: str):
    if not records:
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fieldnames = list(records[0].keys())
    with open(save_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)
