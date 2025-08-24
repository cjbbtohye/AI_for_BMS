# src/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob

# --- Data Loading (Single File) ---
def load_data(file_path):
    """Loads data from a single enhanced .npz file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: NPZ file not found at {file_path}")
    try:
        data = np.load(file_path)
        timeseries_data = data.get('timeseries_data_enhanced')
        soh_per_cycle = data.get('soh_per_cycle')
        eol = data.get('eol')

        if timeseries_data is None or soh_per_cycle is None or eol is None:
             missing_keys = [k for k, v in {'timeseries_data_enhanced': timeseries_data, 'soh_per_cycle': soh_per_cycle, 'eol': eol}.items() if v is None]
             raise KeyError(f"Missing required keys in {file_path}: {missing_keys}")

        eol = eol.item() # Extract scalar from 0-d array if necessary

        if timeseries_data.size == 0 or soh_per_cycle.size == 0:
             print(f"Warning: Loaded data arrays are empty in {file_path}. Skipping file.")
             return None

        soh_map = {int(cycle): soh for cycle, soh in soh_per_cycle}
        num_total_cycles = 0
        if timeseries_data.size > 0:
             cycle_nums = timeseries_data[:, 0]
             if not np.all(np.isfinite(cycle_nums)):
                  print(f"Warning: Non-finite cycle numbers found in {file_path}. Skipping file.")
                  return None
             max_cycle_num = cycle_nums.max()
             num_total_cycles = int(max_cycle_num) + 1
        elif soh_per_cycle.size > 0:
             soh_cycle_nums = soh_per_cycle[:, 0]
             if np.all(np.isfinite(soh_cycle_nums)):
                  max_soh_cycle = soh_cycle_nums.max()
                  num_total_cycles = int(max_soh_cycle) + 1

        return timeseries_data, soh_map, eol, num_total_cycles
    except Exception as e:
        print(f"Error loading or validating data from {file_path}: {e}")
        return None

# --- Data Loading (All Files) ---
def load_all_battery_data(dir_path):
    """Loads data from all .npz files in the specified directory."""
    all_data = {}
    npz_files = glob.glob(os.path.join(dir_path, '*.npz'))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in directory: {dir_path}")

    print(f"Found {len(npz_files)} NPZ files in {dir_path}. Loading...")
    for file_path in npz_files:
        battery_id = os.path.basename(file_path).replace('.npz', '')
        loaded = load_data(file_path)
        if loaded is not None:
            timeseries_data, soh_map, eol, num_total_cycles = loaded
            if timeseries_data is not None and soh_map is not None and timeseries_data.size > 0:
                all_data[battery_id] = {
                    'timeseries': timeseries_data,
                    'soh_map': soh_map,
                    'eol': eol,
                    'num_cycles': num_total_cycles
                }
            else:
                 print(f"Skipping {battery_id} due to empty or invalid data.")
        else:
            print(f"Skipping {battery_id} due to loading errors.")
    if not all_data:
        raise ValueError("No valid battery data could be loaded from the directory.")
    print(f"\nSuccessfully loaded data for {len(all_data)} batteries.")
    return all_data

# --- PyTorch Dataset for Multiple Batteries ---
class MultiBatteryCycleDataset(Dataset):
    """Dataset for handling timeseries data and SOH from multiple batteries."""
    def __init__(self, all_battery_data, feature_indices, num_points_per_cycle, look_back_cycles):
        self.all_battery_data = all_battery_data
        self.feature_indices = feature_indices
        self.num_points_per_cycle = num_points_per_cycle
        self.look_back_cycles = look_back_cycles
        self.samples = []

        print("\n--- Creating Multi-Battery Dataset ---")
        for battery_id, data in self.all_battery_data.items():
            soh_map = data['soh_map']
            num_cycles = data['num_cycles']
            timeseries_shape = data['timeseries'].shape
            for target_cycle_num in range(look_back_cycles, num_cycles):
                if target_cycle_num in soh_map:
                    end_idx_last_input_cycle = target_cycle_num * self.num_points_per_cycle
                    if end_idx_last_input_cycle <= timeseries_shape[0]:
                         self.samples.append((battery_id, target_cycle_num))
        if not self.samples:
            raise ValueError("No valid training samples found across all batteries.")
        print(f"--- Dataset Creation Complete: {len(self.samples)} total valid samples. ---")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        battery_id, target_cycle_num = self.samples[idx]
        data = self.all_battery_data[battery_id]
        timeseries = data['timeseries']
        soh_map = data['soh_map']
        start_input_cycle = target_cycle_num - self.look_back_cycles
        input_features_list = []
        for cycle in range(start_input_cycle, target_cycle_num):
            start_idx = cycle * self.num_points_per_cycle
            end_idx = start_idx + self.num_points_per_cycle
            if end_idx > timeseries.shape[0]: return None
            cycle_features = timeseries[start_idx:end_idx, self.feature_indices]
            if not np.all(np.isfinite(cycle_features)): return None
            features_tensor = torch.tensor(cycle_features, dtype=torch.float32).permute(1, 0)
            input_features_list.append(features_tensor)

        if not input_features_list: return None
        try:
            stacked_features_tensor = torch.stack(input_features_list, dim=0)
        except RuntimeError: return None

        target_soh = soh_map.get(target_cycle_num)
        if target_soh is None: return None
        soh_tensor = torch.tensor([target_soh], dtype=torch.float32)
        return stacked_features_tensor, soh_tensor

# --- Custom Collate Function ---
def collate_fn(batch):
    """Filters out None samples from a batch and stacks the valid ones."""
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        return None, None
    try:
        features, targets = zip(*valid_batch)
        features = torch.stack(features, dim=0)
        targets = torch.stack(targets, dim=0)
        return features, targets
    except Exception as e:
        print(f"Error during collation: {e}")
        return None, None