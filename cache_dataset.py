import math
import os
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def e2r(x):
    g = x[0]
    b = x[1]
    a = x[2]
    r = rotz(a).dot(roty(b)).dot(rotx(g))
    return np.array(r)


def rotx(t):
    ct = math.cos(t)
    st = math.sin(t)
    r = [[1, 0, 0], [0, ct, -st], [0, st, ct]]
    return np.array(r)


def roty(t):
    ct = math.cos(t)
    st = math.sin(t)
    r = [[ct, 0, st], [0, 1, 0], [-st, 0, ct]]
    return np.array(r)


def rotz(t):
    ct = math.cos(t)
    st = math.sin(t)
    r = [[ct, -st, 0], [st, ct, 0], [0, 0, 1]]
    return np.array(r)


def smooth(a, wsz):
    out0 = np.convolve(a, np.ones(wsz, dtype=int), "valid") / wsz
    r = np.arange(1, wsz - 1, 2)
    start = np.cumsum(a[: wsz - 1])[::2] / r
    stop = (np.cumsum(a[: -wsz : -1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


def cartesian_frames(pos_file, ang_file):
    dfp = pd.read_csv(pos_file, sep=",")
    dfa = pd.read_csv(ang_file, sep=",")
    readpos = dfp.to_numpy()
    readang = dfa.to_numpy()

    for i in range(len(readpos[1, :])):
        readpos[:, i] = smooth(readpos[:, i], 5)

    for i in range(len(readang[1, :])):
        readang[:, i] = smooth(readang[:, i], 23)

    frames = np.shape(readpos)[0]
    skeleton_pos = np.zeros((22, 3, frames))
    skeleton_ang = np.zeros((22, 3, frames))

    tp = np.transpose(readpos)
    for i in range(frames):
        skeleton_pos[:, :, i] = tp[:, i].reshape((22, 3))

    ta = np.transpose(readang)
    for i in range(frames):
        skeleton_ang[:, :, i] = ta[:, i].reshape((22, 3))

    skel = np.zeros((22, 3, frames))
    for i in range(frames):
        joint_pos = skeleton_pos[:, :, i]
        joint_ang = skeleton_ang[:, :, i]

        rot_1 = e2r(joint_ang[0, :].dot(np.pi / 180))
        joint_pos[1, :] = np.transpose(rot_1.dot(np.transpose(joint_pos[1, :]))) + joint_pos[0, :]
        rot_2 = rot_1.dot(e2r(joint_ang[1, :].dot(np.pi / 180)))
        joint_pos[2, :] = np.transpose(rot_2.dot(np.transpose(joint_pos[2, :]))) + joint_pos[1, :]
        rot_3 = rot_2.dot(e2r(joint_ang[2, :].dot(np.pi / 180)))
        joint_pos[3, :] = np.transpose(rot_3.dot(np.transpose(joint_pos[3, :]))) + joint_pos[2, :]
        rot_4 = rot_3.dot(e2r(joint_ang[3, :].dot(np.pi / 180)))
        joint_pos[4, :] = np.transpose(rot_4.dot(np.transpose(joint_pos[4, :]))) + joint_pos[3, :]
        rot_5 = rot_4.dot(e2r(joint_ang[4, :].dot(np.pi / 180)))
        joint_pos[5, :] = np.transpose(rot_5.dot(np.transpose(joint_pos[5, :]))) + joint_pos[4, :]
        rot_6 = e2r(joint_ang[2, :].dot(np.pi / 180))
        joint_pos[6, :] = np.transpose(rot_6.dot(np.transpose(joint_pos[6, :]))) + joint_pos[2, :]
        rot_7 = rot_6.dot(e2r(joint_ang[6, :].dot(np.pi / 180)))
        joint_pos[7, :] = np.transpose(rot_7.dot(np.transpose(joint_pos[7, :]))) + joint_pos[6, :]
        rot_8 = rot_7.dot(e2r(joint_ang[7, :].dot(np.pi / 180)))
        joint_pos[8, :] = np.transpose(rot_8.dot(np.transpose(joint_pos[8, :]))) + joint_pos[7, :]
        rot_9 = rot_8.dot(e2r(joint_ang[8, :].dot(np.pi / 180)))
        joint_pos[9, :] = np.transpose(rot_9.dot(np.transpose(joint_pos[9, :]))) + joint_pos[8, :]
        rot_10 = e2r(joint_ang[2, :].dot(np.pi / 180))
        joint_pos[10, :] = np.transpose(rot_10.dot(np.transpose(joint_pos[10, :]))) + joint_pos[2, :]
        rot_11 = rot_10.dot(e2r(joint_ang[10, :].dot(np.pi / 180)))
        joint_pos[11, :] = np.transpose(rot_11.dot(np.transpose(joint_pos[11, :]))) + joint_pos[10, :]
        rot_12 = rot_11.dot(e2r(joint_ang[11, :].dot(np.pi / 180)))
        joint_pos[12, :] = np.transpose(rot_12.dot(np.transpose(joint_pos[12, :]))) + joint_pos[11, :]
        rot_13 = rot_12.dot(e2r(joint_ang[12, :].dot(np.pi / 180)))
        joint_pos[13, :] = np.transpose(rot_13.dot(np.transpose(joint_pos[13, :]))) + joint_pos[12, :]
        rot_14 = e2r(joint_ang[0, :].dot(np.pi / 180))
        joint_pos[14, :] = np.transpose(rot_14.dot(np.transpose(joint_pos[14, :]))) + joint_pos[0, :]
        rot_15 = rot_14.dot(e2r(joint_ang[14, :].dot(np.pi / 180)))
        joint_pos[15, :] = np.transpose(rot_15.dot(np.transpose(joint_pos[15, :]))) + joint_pos[14, :]
        rot_16 = rot_15.dot(e2r(joint_ang[15, :].dot(np.pi / 180)))
        joint_pos[16, :] = np.transpose(rot_16.dot(np.transpose(joint_pos[16, :]))) + joint_pos[15, :]
        rot_17 = rot_16.dot(e2r(joint_ang[16, :].dot(np.pi / 180)))
        joint_pos[17, :] = np.transpose(rot_17.dot(np.transpose(joint_pos[17, :]))) + joint_pos[16, :]
        rot_18 = e2r(joint_ang[0, :].dot(np.pi / 180))
        joint_pos[18, :] = np.transpose(rot_18.dot(np.transpose(joint_pos[18, :]))) + joint_pos[0, :]
        rot_19 = rot_18.dot(e2r(joint_ang[18, :].dot(np.pi / 180)))
        joint_pos[19, :] = np.transpose(rot_19.dot(np.transpose(joint_pos[19, :]))) + joint_pos[18, :]
        rot_20 = rot_19.dot(e2r(joint_ang[19, :].dot(np.pi / 180)))
        joint_pos[20, :] = np.transpose(rot_20.dot(np.transpose(joint_pos[20, :]))) + joint_pos[19, :]
        rot_21 = rot_20.dot(e2r(joint_ang[20, :].dot(np.pi / 180)))
        joint_pos[21, :] = np.transpose(rot_21.dot(np.transpose(joint_pos[21, :]))) + joint_pos[20, :]

        skel[:, :, i] = joint_pos

    return frames, skel.transpose(2, 1, 0)  # (T, C, V)


class OriginalDatasetProcessor:
    def __init__(self, root_dirs, cache_dir, max_frames=None, length_normalization="resample"):
        self.root_dirs = root_dirs
        self.cache_dir = cache_dir
        self.max_frames = max_frames
        self.length_normalization = length_normalization

        self.samples_info = []  # (pos_path, ang_path, action_type_str, quality_label_str, unique_id)
        self.action_type_mapping = {}
        self.quality_mapping = {"correct": 0, "incorrect": 1}
        self.combined_class_mapping = {}

        if self.length_normalization not in {"resample", "pad_truncate"}:
            raise ValueError(
                f"Unsupported length_normalization={self.length_normalization}. "
                "Use 'resample' or 'pad_truncate'."
            )

        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_and_process_info()

    def _load_and_process_info(self):
        self._collect_files(self.root_dirs["correct"], quality_label="correct")
        self._collect_files(self.root_dirs["incorrect"], quality_label="incorrect")

        if self.max_frames is None:
            self.max_frames = self._get_max_frames_from_files()
            print(f"Detected maximum frames across all datasets: {self.max_frames}.")

        self.class_labels = sorted(self.action_type_mapping.keys())
        for action_type in self.class_labels:
            action_idx = self.action_type_mapping[action_type]
            self.combined_class_mapping[f"{action_type}_correct"] = action_idx * 2 + self.quality_mapping["correct"]
            self.combined_class_mapping[f"{action_type}_incorrect"] = action_idx * 2 + self.quality_mapping["incorrect"]

        print(f"Total original file pairs found: {len(self.samples_info)}")
        print(f"Action Type Mapping: {self.action_type_mapping}")
        print(f"Combined Class Mapping: {self.combined_class_mapping}")
        print(f"Length normalization mode: {self.length_normalization}")
        print(f"Target frames: {self.max_frames}")

    def _collect_files(self, base_dir, quality_label):
        positions_dir = os.path.join(base_dir, "Positions")
        angles_dir = os.path.join(base_dir, "Angles")

        if not os.path.exists(positions_dir) or not os.path.exists(angles_dir):
            print(f"Warning: Directory structure not found in {base_dir}. Skipping.")
            return

        all_pos_files = sorted(os.listdir(positions_dir))
        for pos_file_name in all_pos_files:
            if not pos_file_name.endswith(".txt"):
                continue

            try:
                action_type_str = pos_file_name.split("_")[0]
                if action_type_str not in self.action_type_mapping:
                    self.action_type_mapping[action_type_str] = len(self.action_type_mapping)
            except IndexError:
                print(f"Warning: Could not parse action type from {pos_file_name}. Skipping.")
                continue

            pos_path = os.path.join(positions_dir, pos_file_name)
            if quality_label == "correct":
                ang_file_name = pos_file_name.replace("_positions.txt", "_angles.txt")
            elif quality_label == "incorrect":
                ang_file_name = pos_file_name.replace("_positions_inc.txt", "_angles_inc.txt")
            else:
                raise ValueError(f"Unknown quality_label: {quality_label}")

            ang_path = os.path.join(angles_dir, ang_file_name)
            if not os.path.exists(ang_path):
                print(
                    f"Warning: Corresponding angle file not found for {pos_file_name} "
                    f"(expected: {ang_file_name}). Skipping."
                )
                continue

            # Concise naming using semantic tokens:
            # m = action, s = subject, e = repetition.
            # Example: m01_correct_s01_e01.pt
            unique_id = self._build_unique_id(action_type_str, quality_label, pos_file_name)
            self.samples_info.append((pos_path, ang_path, action_type_str, quality_label, unique_id))

    def _build_unique_id(self, action_type_str, quality_label, pos_file_name):
        base = os.path.splitext(pos_file_name)[0]
        # Supports:
        #   m01_s01_e01_positions
        #   m01_s01_e01_positions_inc
        match = re.match(r"^(m\d+)_s(\d+)_e(\d+)_positions(?:_inc)?$", base)
        if match:
            m_token = match.group(1).lower()
            s_token = f"s{int(match.group(2)):02d}"
            e_token = f"e{int(match.group(3)):02d}"
            return f"{m_token}_{quality_label}_{s_token}_{e_token}"

        # Fallback for unexpected patterns.
        return f"{action_type_str}_{quality_label}_{base}"

    def _get_max_frames_from_files(self):
        max_f = 0
        print("Calculating max frames from all files...")
        for pos_path, _, _, _, _ in tqdm(self.samples_info, desc="Finding max frames"):
            dfp = pd.read_csv(pos_path, sep=",")
            current_frames = np.shape(dfp.to_numpy())[0]
            if current_frames > max_f:
                max_f = current_frames
        return max_f

    def _temporal_resample(self, skel_data, target_frames):
        """
        skel_data shape: (T, C, V)
        returns: (target_frames, C, V)
        """
        current_frames = skel_data.shape[0]
        if current_frames == target_frames:
            return skel_data
        if current_frames <= 1:
            return np.repeat(skel_data, target_frames, axis=0)

        src_idx = np.linspace(0, current_frames - 1, current_frames, dtype=np.float32)
        dst_idx = np.linspace(0, current_frames - 1, target_frames, dtype=np.float32)

        out = np.empty((target_frames, skel_data.shape[1], skel_data.shape[2]), dtype=np.float32)
        for c in range(skel_data.shape[1]):
            for v in range(skel_data.shape[2]):
                out[:, c, v] = np.interp(dst_idx, src_idx, skel_data[:, c, v])
        return out

    def _normalize_temporal_length(self, skel_data):
        current_frames = skel_data.shape[0]
        if self.length_normalization == "resample":
            return self._temporal_resample(skel_data, self.max_frames), current_frames

        # old baseline behavior
        if current_frames > self.max_frames:
            skel_data = skel_data[: self.max_frames, :, :]
        elif current_frames < self.max_frames:
            padding = np.zeros((self.max_frames - current_frames, skel_data.shape[1], skel_data.shape[2]))
            skel_data = np.concatenate((skel_data, padding), axis=0)
        return skel_data, current_frames

    def generate_and_cache(self, overwrite=False, max_samples=None):
        print(f"\n--- Generating and caching dataset to {self.cache_dir} ---")
        processed = 0
        for pos_path, ang_path, action_type_str, quality_label_str, unique_id in tqdm(
            self.samples_info, desc="Processing files"
        ):
            cache_file_path = os.path.join(self.cache_dir, f"{unique_id}.pt")
            if os.path.exists(cache_file_path) and not overwrite:
                continue

            try:
                combined_label = self.combined_class_mapping[f"{action_type_str}_{quality_label_str}"]
                _, skel_data = cartesian_frames(pos_path, ang_path)
                skel_data, original_frames = self._normalize_temporal_length(skel_data)

                skel_tensor = torch.from_numpy(skel_data).float().permute(1, 0, 2)  # (C, T, V)
                torch.save(
                    {
                        "data": skel_tensor,
                        "label": torch.tensor(combined_label, dtype=torch.long),
                        "meta": {
                            "original_frames": int(original_frames),
                            "target_frames": int(self.max_frames),
                            "length_normalization": self.length_normalization,
                            "action_type": action_type_str,
                            "quality_label": quality_label_str,
                            "sample_id": unique_id,
                        },
                    },
                    cache_file_path,
                )
                processed += 1
                if max_samples is not None and processed >= max_samples:
                    break
            except Exception as e:
                print(f"\nError processing {pos_path}: {e}. Skipping.")
                if os.path.exists(cache_file_path):
                    os.remove(cache_file_path)

        print(f"Dataset caching complete! Newly written files: {processed}")


def export_skeleton_point_preview(
    cache_dir,
    output_csv_path,
    sample_index=0,
    frame_indices=(0, 50, 100, 150, 199),
    joint_indices=(0, 1, 2, 3, 4),
):
    cache_files = sorted([f for f in os.listdir(cache_dir) if f.endswith(".pt")])
    if not cache_files:
        raise RuntimeError(f"No .pt files found in {cache_dir}")
    if sample_index < 0 or sample_index >= len(cache_files):
        raise IndexError(f"sample_index={sample_index} out of range [0, {len(cache_files) - 1}]")

    sample_name = cache_files[sample_index]
    sample_path = os.path.join(cache_dir, sample_name)
    payload = torch.load(sample_path, map_location="cpu")
    data = payload["data"].cpu().numpy()  # (C, T, V)

    _, total_t, total_v = data.shape
    valid_frames = [int(t) for t in frame_indices if 0 <= int(t) < total_t]
    valid_joints = [int(j) for j in joint_indices if 0 <= int(j) < total_v]

    rows = []
    for t in valid_frames:
        for j in valid_joints:
            rows.append(
                {
                    "sample_file": sample_name,
                    "frame": t,
                    "joint": j,
                    "x": float(data[0, t, j]),
                    "y": float(data[1, t, j]),
                    "z": float(data[2, t, j]),
                }
            )

    preview_df = pd.DataFrame(rows)
    preview_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"Skeleton point preview saved to: {output_csv_path}")
    print(preview_df.to_string(index=False))


if __name__ == "__main__":
    original_data_root_dirs = {
        "correct": r"G:\paper[20260408]\UI-PRMD\data\Segmented Movements\Kinect",
        "incorrect": r"G:\paper[20260408]\UI-PRMD\data\Incorrect Segmented Movements\Kinect",
    }
    cached_dataset_dir = r"G:\paper[20260408]\UI-PRMD\data\CachedDataset_Kinect"

    for _, path in original_data_root_dirs.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Original data root not found: {path}")

    max_frames = 200
    length_normalization = "resample"  # recommended for rigorous research
    overwrite_existing = True  # set True when preprocessing changes

    processor = OriginalDatasetProcessor(
        original_data_root_dirs,
        cached_dataset_dir,
        max_frames=max_frames,
        length_normalization=length_normalization,
    )
    processor.generate_and_cache(overwrite=overwrite_existing)

    preview_csv = os.path.join(cached_dataset_dir, "skeleton_point_preview.csv")
    export_skeleton_point_preview(
        cache_dir=cached_dataset_dir,
        output_csv_path=preview_csv,
        sample_index=0,
        frame_indices=(0, 50, 100, 150, 199),
        joint_indices=(0, 1, 2, 3, 4),
    )

    print(f"\nCached dataset directory: {cached_dataset_dir}")
