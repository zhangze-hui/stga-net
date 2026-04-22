import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import numpy as np
import math
import matplotlib.pyplot as plt  # ?
import re

def _restore_original_pos_prefix(action_type_str, quality_label_str, original_pos_filename_prefix):
    """
    Support both cache naming styles:
    - old: m01_correct_m01_s01_e01_positions.pt
    - new: m01_correct_s01_e01.pt
    """
    prefix = original_pos_filename_prefix
    if prefix.startswith(f"{action_type_str}_"):
        return prefix

    if prefix.startswith("s") and "_e" in prefix and "positions" not in prefix:
        tail = "positions_inc" if quality_label_str == "incorrect" else "positions"
        return f"{action_type_str}_{prefix}_{tail}"

    if prefix.startswith("s") and "_e" in prefix:
        return f"{action_type_str}_{prefix}"

    return prefix

# --- 2r, rotx, roty, rotz, smooth, cartesian_frames () ---
# ?

def e2r(x):
    g = x[0]
    b = x[1]
    a = x[2]
    R = rotz(a).dot(roty(b)).dot(rotx(g))
    return np.array(R)


def rotx(t):
    ct = math.cos(t)
    st = math.sin(t)
    r = [[1, 0, 0],
         [0, ct, -st],
         [0, st, ct]]
    return np.array(r)


def roty(t):
    ct = math.cos(t)
    st = math.sin(t)
    r = [[ct, 0, st],
         [0, 1, 0],
         [-st, 0, ct]]
    return np.array(r)


def rotz(t):
    ct = math.cos(t)
    st = math.sin(t)
    r = [[ct, -st, 0],
         [st, ct, 0],
         [0, 0, 1]]
    return np.array(r)


def smooth(a, WSZ):
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid') / WSZ
    r = np.arange(1, WSZ - 1, 2)
    start = np.cumsum(a[:WSZ - 1])[::2] / r
    stop = (np.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


def cartesian_frames(pos_file, ang_file):
    dfp = pd.read_csv(pos_file, sep=',')
    dfa = pd.read_csv(ang_file, sep=',')
    readpos = dfp.to_numpy()
    readang = dfa.to_numpy()

    for i in range(len(readpos[1, :])):
        readpos[:, i] = smooth(readpos[:, i], 5)

    for i in range(len(readang[1, :])):
        readang[:, i] = smooth(readang[:, i], 23)

    frames = np.shape(readpos)[0]
    skeleton_pos = np.zeros((22, 3, frames));
    skeleton_ang = np.zeros((22, 3, frames));

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
        joint_pos[2, :] = (np.transpose(rot_2.dot(np.transpose(joint_pos[2, :])))) + joint_pos[1, :]
        rot_3 = rot_2.dot(e2r(joint_ang[2, :].dot(np.pi / 180)))
        joint_pos[3, :] = (np.transpose(rot_3.dot(np.transpose(joint_pos[3, :])))) + joint_pos[2, :]
        rot_4 = rot_3.dot(e2r(joint_ang[3, :].dot(np.pi / 180)))
        joint_pos[4, :] = (np.transpose(rot_4.dot(np.transpose(joint_pos[4, :])))) + joint_pos[3, :]
        rot_5 = rot_4.dot(e2r(joint_ang[4, :].dot(np.pi / 180)))
        joint_pos[5, :] = (np.transpose(rot_5.dot(np.transpose(joint_pos[5, :])))) + joint_pos[4, :]
        rot_6 = e2r(joint_ang[2, :].dot(np.pi / 180))
        joint_pos[6, :] = (np.transpose(rot_6.dot(np.transpose(joint_pos[6, :])))) + joint_pos[2, :]
        rot_7 = rot_6.dot(e2r(joint_ang[6, :].dot(np.pi / 180)))
        joint_pos[7, :] = (np.transpose(rot_7.dot(np.transpose(joint_pos[7, :])))) + joint_pos[6, :]
        rot_8 = rot_7.dot(e2r(joint_ang[7, :].dot(np.pi / 180)))
        joint_pos[8, :] = (np.transpose(rot_8.dot(np.transpose(joint_pos[8, :])))) + joint_pos[7, :]
        rot_9 = rot_8.dot(e2r(joint_ang[8, :].dot(np.pi / 180)))
        joint_pos[9, :] = (np.transpose(rot_9.dot(np.transpose(joint_pos[9, :])))) + joint_pos[8, :]
        rot_10 = e2r(joint_ang[2, :].dot(np.pi / 180))
        joint_pos[10, :] = (np.transpose(rot_10.dot(np.transpose(joint_pos[10, :])))) + joint_pos[2, :]
        rot_11 = rot_10.dot(e2r(joint_ang[10, :].dot(np.pi / 180)))
        joint_pos[11, :] = (np.transpose(rot_11.dot(np.transpose(joint_pos[11, :])))) + joint_pos[10, :]
        rot_12 = rot_11.dot(e2r(joint_ang[11, :].dot(np.pi / 180)))
        joint_pos[12, :] = (np.transpose(rot_12.dot(np.transpose(joint_pos[12, :])))) + joint_pos[11, :]
        rot_13 = rot_12.dot(e2r(joint_ang[12, :].dot(np.pi / 180)))
        joint_pos[13, :] = (np.transpose(rot_13.dot(np.transpose(joint_pos[13, :])))) + joint_pos[12, :]
        rot_14 = e2r(joint_ang[0, :].dot(np.pi / 180))
        joint_pos[14, :] = (np.transpose(rot_14.dot(np.transpose(joint_pos[14, :])))) + joint_pos[0, :]
        rot_15 = rot_14.dot(e2r(joint_ang[14, :].dot(np.pi / 180)))
        joint_pos[15, :] = (np.transpose(rot_15.dot(np.transpose(joint_pos[15, :])))) + joint_pos[14, :]
        rot_16 = rot_15.dot(e2r(joint_ang[15, :].dot(np.pi / 180)))
        joint_pos[16, :] = (np.transpose(rot_16.dot(np.transpose(joint_pos[16, :])))) + joint_pos[15, :]
        rot_17 = rot_16.dot(e2r(joint_ang[16, :].dot(np.pi / 180)))
        joint_pos[17, :] = (np.transpose(rot_17.dot(np.transpose(joint_pos[17, :])))) + joint_pos[16, :]
        rot_18 = e2r(joint_ang[0, :].dot(np.pi / 180))
        joint_pos[18, :] = (np.transpose(rot_18.dot(np.transpose(joint_pos[18, :])))) + joint_pos[0, :]
        rot_19 = rot_18.dot(e2r(joint_ang[18, :].dot(np.pi / 180)))
        joint_pos[19, :] = (np.transpose(rot_19.dot(np.transpose(joint_pos[19, :])))) + joint_pos[18, :]
        rot_20 = rot_19.dot(e2r(joint_ang[19, :].dot(np.pi / 180)))
        joint_pos[20, :] = (np.transpose(rot_20.dot(np.transpose(joint_pos[20, :])))) + joint_pos[19, :]
        rot_21 = rot_20.dot(e2r(joint_ang[20, :].dot(np.pi / 180)))
        joint_pos[21, :] = (np.transpose(rot_21.dot(np.transpose(joint_pos[21, :])))) + joint_pos[20, :]

        skel[:, :, i] = joint_pos
    return frames, skel.transpose(2, 1, 0)  # returns (frames, 3, 22)


class ActionQualityDataset(Dataset):
    def __init__(self, cached_data_dir, transform=None, return_original_skeleton=False):
        self.cached_data_dir = cached_data_dir
        self.transform = transform
        self.return_original_skeleton = bool(return_original_skeleton)
        #  (cache_file_path, combined_label_id, original_action_type_str, original_quality_label_str, original_pos_filename_prefix)
        self.samples_info = []

        # ? 'correct' ?'incorrect'  ID
        self.quality_mapping = {'correct': 0, 'incorrect': 1}

        self.combined_class_mapping = {}  #  combined_class_mapping  20 ?
        self.action_type_mapping = {}  # m01 -> 0, m02 -> 1, ... m10 -> 9

        self.root_joint_idx = 0  # ?

        self._load_cached_data_info()

        if len(self.samples_info) > 0:
            try:
                first_sample_data = torch.load(self.samples_info[0][0])['data']
                self.max_frames = first_sample_data.shape[1]
                expected_num_joints = 22  # ?
                if first_sample_data.shape[2] != expected_num_joints:
                    print(
                        f"Warning: joint count in cached data ({first_sample_data.shape[2]}) "
                        f"does not match expected ({expected_num_joints})."
                    )
                print(f"Inferred max_frames from cached data: {self.max_frames}")
            except Exception as e:
                print(f"Warning: Could not infer max_frames from first cached sample: {e}. Setting max_frames to 0.")
                self.max_frames = 0
        else:
            self.max_frames = 0

        print(f"Loaded {len(self.samples_info)} cached samples.")
        print(f"Combined Class Mapping ({len(self.combined_class_mapping)} classes): {self.combined_class_mapping}")

    def _load_cached_data_info(self):
        all_cache_files = sorted(os.listdir(self.cached_data_dir))

        temp_action_types = set()
        # ? 'correct' ?'incorrect' ?
        quality_regex = r'_(correct|incorrect)_'  #  'i_correct'/'i_incorrect'

        raw_samples_info_temp = []  # 

        for file_name in tqdm(all_cache_files, desc="Scanning cached files"):
            if not file_name.endswith('.pt'):
                continue

            match = re.search(quality_regex, file_name)
            if not match:
                #  'correct' ?'incorrect'
                print(
                    f"Warning: Cached file name '{file_name}' does not contain 'correct' or 'incorrect' quality tag. Skipping.")
                continue

            try:
                raw_quality_label_str = match.group(1)  # ?('correct' ?'incorrect')

                # ?action_type_str
                start_idx_of_quality_tag = file_name.find(raw_quality_label_str)
                action_type_str = file_name[:start_idx_of_quality_tag - 1]

                # original_pos_filename_prefix  .pt
                original_pos_filename_prefix = file_name[
                                               start_idx_of_quality_tag + len(raw_quality_label_str) + 1:].replace(
                    '.pt', '')

                temp_action_types.add(action_type_str)

                raw_samples_info_temp.append((os.path.join(self.cached_data_dir, file_name),
                                              action_type_str,
                                              raw_quality_label_str,
                                              original_pos_filename_prefix))
            except Exception as e:
                print(f"Warning: Could not parse info from cached file {file_name}: {e}. Skipping.")
                continue

        # ??'correct' ?'incorrect'  combined_class_mapping
        self.class_labels = sorted(list(temp_action_types))  # ?m01, m02...m10 (?0?
        for idx, action_type in enumerate(self.class_labels):
            self.action_type_mapping[action_type] = idx

        #  combined_class_mapping
        for action_type in self.class_labels:
            action_idx = self.action_type_mapping[action_type]
            #  'm01_correct'  ID 0*2 + 0 = 0
            self.combined_class_mapping[f"{action_type}_correct"] = action_idx * 2 + self.quality_mapping['correct']
            #  'm01_incorrect'  ID 0*2 + 1 = 1
            self.combined_class_mapping[f"{action_type}_incorrect"] = action_idx * 2 + self.quality_mapping['incorrect']

        # ? samples_info
        final_samples_info = []
        for cache_path, action_type_str, raw_quality_label_str, original_pos_filename_prefix in raw_samples_info_temp:
            # ?raw_quality_label_str  'correct' ?'incorrect'
            # ?
            combined_id = self.combined_class_mapping.get(f"{action_type_str}_{raw_quality_label_str}")

            if combined_id is not None:
                final_samples_info.append(
                    (cache_path, combined_id, action_type_str, raw_quality_label_str, original_pos_filename_prefix))
            else:
                # ?
                print(
                    f"Warning: Could not find combined_id for {action_type_str}_{raw_quality_label_str}. This suggests an issue with file parsing or mapping. Skipping sample: {cache_path}")

        self.samples_info = final_samples_info  # 

        self.inverse_combined_class_mapping = {v: k for k, v in self.combined_class_mapping.items()}

    def __len__(self):
        return len(self.samples_info)

    def __getitem__(self, idx):
        cache_file_path, combined_label_id, action_type_str, quality_label_str, original_pos_filename_prefix = \
            self.samples_info[idx]

        cached_data = torch.load(cache_file_path)
        skel_tensor = cached_data['data']
        original_skel_tensor = skel_tensor.clone()

        root_coords = skel_tensor[:, :, self.root_joint_idx]
        root_coords_expanded = root_coords.unsqueeze(2).expand_as(skel_tensor)
        relative_skel_tensor = skel_tensor - root_coords_expanded

        if self.transform:
            relative_skel_tensor = self.transform(relative_skel_tensor)

        if self.return_original_skeleton:
            return (
                relative_skel_tensor,
                combined_label_id,
                action_type_str,
                quality_label_str,
                original_pos_filename_prefix,
                original_skel_tensor,
            )
        return relative_skel_tensor, combined_label_id, action_type_str, quality_label_str, original_pos_filename_prefix


# ---  () ---
if __name__ == "__main__":
    print("ActionQualityDataset module loaded. Run training/visualization scripts to use it.")
