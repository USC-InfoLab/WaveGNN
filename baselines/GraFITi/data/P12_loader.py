import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm # For progress if needed, though current P12/P19 loading is fast
import os # For potential path joining if paths were not hardcoded

# Import the collate function from the GraFITi project structure
from grafiti.grafiti import tsdm_collate_classification

class P12ClassificationDataset(Dataset):
    def __init__(self, processed_data_list):
        self.processed_data_list = processed_data_list

    def __len__(self):
        return len(self.processed_data_list)

    def __getitem__(self, idx):
        return self.processed_data_list[idx]


def _extract_attributes_p12(data_dict, attr):
    X = [d[attr] for d in data_dict]
    X = np.stack(X, axis=0)
    X = X.astype(np.float32)
    return torch.from_numpy(X)

def _normalize_p12(data):
    # Normalize using the whole dataset - specific to P12 loader needs
    mean = data.mean(axis=(0,1) if data.ndim == 3 else 0) # Handle different dims if data changes
    std = data.std(axis=(0,1) if data.ndim == 3 else 0)
    std[std == 0] = 1.0 # Avoid division by zero if a feature has no variance
    data = (data - mean) / std
    data = torch.nan_to_num(data) # This was original, but we want NaNs for our collate fn
                                 # However, this normalize is on the *entire* dataset before splitting.
                                 # The NaNs for masking should be applied *after* this global norm and *after* padding.
    return data # Mean/std not returned to align with original usage, but could be useful.

def _load_and_process_p12_split(indices, all_timestamps, all_data, all_lengths, all_labels_y, median_len, split_name):
    processed_list = []
    num_features = None

    for i in tqdm(indices, desc=f"Processing P12 {split_name} (len={median_len})"):
        record_id = int(i) # Not used in dict, but good for debugging
        
        # Original data from tensors
        timestamp_original = all_timestamps[record_id].squeeze(-1) # Ensure it's 1D
        values_original = all_data[record_id]
        mask_original_int = torch.zeros_like(values_original, dtype=torch.int) # Create mask based on length
        seq_len = all_lengths[record_id].item()
        if values_original.ndim == 2:
            mask_original_int[:seq_len, :] = 1
        else: # Should be 1D for single feature, or error if unexpected
            mask_original_int[:seq_len] = 1
        label = all_labels_y[record_id] # Already a scalar float tensor

        if num_features is None and values_original.ndim == 2:
            num_features = values_original.shape[-1]
        elif num_features is None and values_original.ndim == 1: # Single feature case
            num_features = 1
        
        # Convert int mask to boolean mask
        mask_original_bool = mask_original_int.bool()

        # Apply truncation/padding to median_len
        current_len = timestamp_original.shape[0]
        
        # Ensure values_original is 2D if it's single feature for consistent padding
        if values_original.ndim == 1 and num_features == 1:
            values_original_2d = values_original.unsqueeze(-1)
            mask_original_bool_2d = mask_original_bool.unsqueeze(-1)
        else:
            values_original_2d = values_original
            mask_original_bool_2d = mask_original_bool

        if current_len > median_len:
            # Truncate
            timestamp_processed = timestamp_original[:median_len]
            values_processed = values_original_2d[:median_len, :]
            mask_processed_bool = mask_original_bool_2d[:median_len, :]
        elif current_len < median_len:
            # Pad
            padding_len = median_len - current_len
            timestamp_processed = F.pad(timestamp_original, (0, padding_len), 'constant', 0.0)
            values_processed = F.pad(values_original_2d, (0, 0, 0, padding_len), 'constant', 0.0) # Pad features with 0
            mask_processed_bool = F.pad(mask_original_bool_2d, (0, 0, 0, padding_len), 'constant', False) # Pad mask with False
        else: # current_len == median_len
            timestamp_processed = timestamp_original
            values_processed = values_original_2d
            mask_processed_bool = mask_original_bool_2d

        # Apply boolean mask to create NaNs for missing/padded values
        # Padded values (originally 0) will become NaN if their mask is False.
        # Original observed values remain; original unobserved (but within seq_len) also become NaN if mask was False.
        x_obs_with_nans = torch.where(mask_processed_bool, values_processed, torch.tensor(float('nan')))
        
        # If it was a single feature and we unsqueezed, squeeze it back for x_obs if needed by collate, 
        # but our collate expects (L,D) so (L,1) is fine.
        # If num_features == 1: x_obs_with_nans = x_obs_with_nans.squeeze(-1) # Not strictly necessary if collate handles (L,1)

        processed_list.append({
            't_obs': timestamp_processed.float(),
            'x_obs': x_obs_with_nans.float(), # Contains NaNs
            'class_label': label.float() # Ensure it's float
        })
    
    return processed_list, num_features


def get_p12_data(args, device): # Renamed, takes args and device
    """
    Loads and preprocesses the P12 dataset, returning DataLoaders and num_features.
    args must contain median_len and batch_size.
    """
    # Config values from the original script header
    main_path = "/storage/datasets_public/irreg_ts/datasets/" # Make sure this is accessible
    data_path = "P12data/processed_data/PTdict_list.npy"
    label_path = "P12data/processed_data/arr_outcomes.npy"
    splits_path = "P12data/splits/"
    split_num = 1  # Default split

    # Load data and labels
    raw_data_dict = np.load(os.path.join(main_path, data_path), allow_pickle=True)
    raw_labels = np.load(os.path.join(main_path, label_path), allow_pickle=True)

    # Extract arrays
    data_tensor = _extract_attributes_p12(raw_data_dict, "arr")
    timestamps_tensor = _extract_attributes_p12(raw_data_dict, "time") / 60.0  # convert to mins
    lengths_tensor = _extract_attributes_p12(raw_data_dict, "length").long()
    y_tensor = torch.from_numpy(raw_labels[:, -1]).type(torch.float32)

    # Normalize using the whole dataset - this was original P12 logic
    data_tensor_normalized = _normalize_p12(data_tensor)

    # Load split indices
    sfile = f"phy12_split{split_num}.npy"
    train_indices, val_indices, test_indices = np.load(
        os.path.join(main_path, splits_path, sfile), allow_pickle=True
    )
    
    median_len = args.median_len
    batch_size = args.batch_size

    train_processed, num_features = _load_and_process_p12_split(train_indices, timestamps_tensor, data_tensor_normalized, lengths_tensor, y_tensor, median_len, "train")
    if num_features is None and train_processed: # Should be caught by _load_and_process if all data is 1D after squeeze
         if train_processed[0]['x_obs'].ndim == 1: num_features = 1
         elif train_processed[0]['x_obs'].ndim == 2: num_features = train_processed[0]['x_obs'].shape[-1]
    
    if num_features is None: # If still None after train, try from val.
        print("Warning: num_features not found from training data for P12, attempting val.")
        val_temp, num_features_val = _load_and_process_p12_split(val_indices, timestamps_tensor, data_tensor_normalized, lengths_tensor, y_tensor, median_len, "val_temp")
        if num_features_val is not None: num_features = num_features_val
        else: raise ValueError("Could not determine num_features for P12 dataset.")

    val_processed, _ = _load_and_process_p12_split(val_indices, timestamps_tensor, data_tensor_normalized, lengths_tensor, y_tensor, median_len, "validation")
    test_processed, _ = _load_and_process_p12_split(test_indices, timestamps_tensor, data_tensor_normalized, lengths_tensor, y_tensor, median_len, "test")

    train_dataset = P12ClassificationDataset(train_processed)
    val_dataset = P12ClassificationDataset(val_processed)
    test_dataset = P12ClassificationDataset(test_processed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=tsdm_collate_classification, num_workers=min(os.cpu_count(), 4))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=tsdm_collate_classification, num_workers=min(os.cpu_count(), 4))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=tsdm_collate_classification, num_workers=min(os.cpu_count(), 4))

    return train_loader, val_loader, test_loader, num_features

# Remove the old main/example block or keep for standalone testing if desired
# if __name__ == "__main__":
#     from collections import Counter
#     # Simulate args for testing
#     class Args:
#         median_len = 50
#         batch_size = 4
#     args_test = Args()
#     train_loader, val_loader, test_loader, nf = get_data(args_test, 'cpu')
#     print(f"P12: Num features: {nf}")
#     for batch_idx, batch_data in enumerate(train_loader):
#         print(f"Batch {batch_idx}:")
#         print(f"  x_time shape: {batch_data.x_time.shape}")
#         print(f"  x_vals shape: {batch_data.x_vals.shape}")
#         print(f"  x_mask shape: {batch_data.x_mask.shape}")
#         print(f"  y_mask_for_grafiti_structure shape: {batch_data.y_mask_for_grafiti_structure.shape}")
#         print(f"  class_labels shape: {batch_data.class_labels.shape}")
#         if batch_idx == 0: # Print one batch and break
#             break