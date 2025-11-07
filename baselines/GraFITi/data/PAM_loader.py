import torch
import numpy as np
import random
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse # For testing

# Assuming tsdm_collate_classification is in GraFITi.grafiti.grafiti
# This will be resolved if the path is incorrect during integration.
from grafiti.grafiti import tsdm_collate_classification

class PAMClassificationDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def _load_raw_pam_data(args):
    """
    Loads raw PAM data files.
    args.data_path should point to the directory containing PAMdata subfolders.
    e.g. if data is in /my_disk/PAMdata/processed_data, args.data_path = /my_disk/PAMdata
    """
    data_file_path = os.path.join(args.data_path, "processed_data", "PTdict_list.npy")
    labels_file_path = os.path.join(args.data_path, "processed_data", "arr_outcomes.npy")
    splits_file_path = os.path.join(args.data_path, "splits", "PAM_split_1.npy") # Using split 1
    density_scores_path = os.path.join(args.data_path, "IG_density_scores_PAM.npy")

    try:
        # The original loader did: data_dict = np.load(data_file, allow_pickle=True); data = data_dict.astype(np.float32)
        # This implies PTdict_list.npy might be an array itself, not a dictionary of arrays.
        raw_data_array = np.load(data_file_path, allow_pickle=True).astype(np.float32)
        raw_labels_array = np.load(labels_file_path, allow_pickle=True)
        # Labels: mortality is the last one in original P12/P19. For PAM, assuming multiclass label is also the last.
        # Original PAM loader: y = labels[:, -1]. This should be integer class labels 0-7.
        raw_class_labels = raw_labels_array[:, -1].astype(np.int64) # Ensure long for CrossEntropyLoss

        train_indices, val_indices, test_indices = np.load(splits_file_path, allow_pickle=True)
        
        density_scores = None
        # Try to load density scores if the file exists. Processing function will handle if it's None.
        if os.path.exists(density_scores_path):
            try:
                density_scores = np.load(density_scores_path, allow_pickle=True)
                print(f"Info: Successfully loaded density scores from {density_scores_path}")
            except Exception as e:
                print(f"Warning: Error loading density scores from {density_scores_path}: {e}. 'fixed' irregularity might fallback or not apply as expected.")
                density_scores = None
        else:
            # This is not an error, just info. The processing function will warn if fixed type is chosen and scores are missing.
            print(f"Info: Density scores file not found at {density_scores_path}. This is okay if 'fixed' irregularity is not used or if fallback is intended.")
    
    except FileNotFoundError as e:
        print(f"Error loading PAM data files: {e}")
        print(f"Checked paths: data='{data_file_path}', labels='{labels_file_path}', splits='{splits_file_path}'")
        raise

    num_samples, seq_len, num_features = raw_data_array.shape
    
    # Timestamps are regular, scaled to minutes (original was / 60.0)
    # Original PAM loader created timestamps like: torch.arange(data.shape[1]).unsqueeze(0).repeat(data.shape[0], 1) / 60.0
    # For each sample, the timestamps will be the same.
    # We need timestamps of shape (median_len,)
    # Since median_len for PAM is 600, and original seq_len is 600.
    individual_timestamps = np.arange(seq_len, dtype=np.float32) / 60.0 

    # Lengths are all fixed at seq_len (which should be args.median_len for PAM)
    lengths = np.full(num_samples, seq_len, dtype=int)

    return (torch.from_numpy(raw_data_array),
            torch.from_numpy(individual_timestamps), # Shape (L,)
            torch.from_numpy(lengths),
            torch.from_numpy(raw_class_labels), # Shape (N,)
            train_indices, val_indices, test_indices, density_scores, num_features)

def _calculate_norm_stats(data_tensor, length_tensor, indices):
    """Calculates mean and std per feature from the specified indices of data_tensor."""
    # data_tensor: (Total_N, L, D), length_tensor: (Total_N)
    # We only use actual sequence parts for normalization. For PAM, all lengths are 600.
    
    # Slices of the data_tensor for the current split (e.g., training)
    split_data = data_tensor[indices] # (N_split, L, D)
    
    # Since all lengths are equal to L (600), we can directly calculate mean/std
    # Reshape to (N_split * L, D) to calculate mean/std over samples and time for each feature
    num_features = split_data.shape[2]
    reshaped_data = split_data.reshape(-1, num_features)
    
    data_mean = torch.mean(reshaped_data, axis=0)
    data_std = torch.std(reshaped_data, axis=0)
    data_std[data_std == 0] = 1.0 # Avoid division by zero
    
    return data_mean, data_std

def _process_pam_sample_for_classification(
    raw_data_single_sample, # (L, D) tensor
    raw_timestamps_single_sample, # (L,) tensor
    original_length, # scalar int, should be median_len (600)
    class_label_single_sample, # scalar long tensor
    
    # Irregularity parameters directly passed
    apply_irregularity_for_this_sample: bool,
    irregularity_rate: float,
    irregularity_type: str,
    median_len: int, # from args.median_len, should be 600 for PAM
    
    data_mean, data_std, # Normalization stats from training split
    sensor_density_scores_arg, # Loaded density scores (can be None)
    n_features # int
    ):
    
    # median_len = args.median_len # Should be 600 for PAM # Now passed directly

    # Ensure sequence length matches median_len (PAM sequences are fixed at 600)
    if raw_data_single_sample.shape[0] != median_len:
        # This case should ideally not happen if data prep is correct for PAM (all 600 len)
        # If it does, we'd need padding/truncation, but PAM is fixed.
        # For now, assume raw_data_single_sample.shape[0] == median_len
        print(f"Warning: PAM sample length {raw_data_single_sample.shape[0]} not matching median_len {median_len}. Check data.")
        # Apply simple truncation or padding if absolutely necessary (though not expected for PAM)
        if raw_data_single_sample.shape[0] > median_len:
            raw_data_single_sample = raw_data_single_sample[:median_len, :]
            raw_timestamps_single_sample = raw_timestamps_single_sample[:median_len]
        else: # Pad (less likely for PAM)
            pad_len = median_len - raw_data_single_sample.shape[0]
            raw_data_single_sample = torch.nn.functional.pad(raw_data_single_sample, (0,0,0,pad_len), 'constant', 0.0)
            raw_timestamps_single_sample = torch.nn.functional.pad(raw_timestamps_single_sample, (0,pad_len), 'constant', 0.0)


    # 1. Normalization
    normalized_data = (raw_data_single_sample - data_mean) / data_std
    
    # 2. Irregularity Masking
    final_mask_torch = torch.ones(median_len, n_features, dtype=torch.bool) # Start with all True

    if apply_irregularity_for_this_sample and n_features > 0 and irregularity_rate > 0:
        n_sensors_to_drop = round(irregularity_rate * n_features)

        if n_sensors_to_drop > 0:
            if irregularity_type == "fixed":
                can_use_fixed_scores = False
                if sensor_density_scores_arg is not None:
                    processed_drop_indices = np.array([], dtype=int)
                    # Try 2D interpretation (original loader likely assumed this with [:,0] for IG_density_scores_PAM.npy)
                    if sensor_density_scores_arg.ndim == 2 and sensor_density_scores_arg.shape[0] >= n_sensors_to_drop and sensor_density_scores_arg.shape[1] > 0:
                        # Assumes indices are in the first column, sorted by importance (less important first / ones to drop)
                        processed_drop_indices = sensor_density_scores_arg[:n_sensors_to_drop, 0].astype(int)
                    # Try 1D interpretation (user snippet: self.sensor_density_scores[:n_sensors_to_drop])
                    elif sensor_density_scores_arg.ndim == 1 and sensor_density_scores_arg.shape[0] >= n_sensors_to_drop:
                        processed_drop_indices = sensor_density_scores_arg[:n_sensors_to_drop].astype(int)
                    else:
                        print(f"Warning: 'fixed' irregularity chosen, but sensor_density_scores shape ({sensor_density_scores_arg.shape}) is not suitable for dropping {n_sensors_to_drop} sensors.")

                    if processed_drop_indices.size > 0:
                        # Validate indices
                        valid_indices_mask = (processed_drop_indices >= 0) & (processed_drop_indices < n_features)
                        drop_indices = processed_drop_indices[valid_indices_mask]
                        
                        if drop_indices.size != processed_drop_indices.size:
                             print(f"Warning: Some sensor indices from 'fixed' scores were out of bounds [0, {n_features-1}]. Original: {processed_drop_indices.tolist()}, Filtered: {drop_indices.tolist()}")

                        if drop_indices.size > 0:
                            final_mask_torch[:, drop_indices] = False
                            can_use_fixed_scores = True
                        # else: No valid indices after filtering, will fallback.
                
                if not can_use_fixed_scores:
                    print(f"Warning: 'fixed' irregularity type chosen for sample, but sensor_density_scores were None, unusable, or yielded no valid sensor indices (attempted to drop {n_sensors_to_drop}). Falling back to random sensor drop.")
                    if n_features > 0 : # Check if there are any features to drop from
                        actual_sensors_to_drop_random = min(n_sensors_to_drop, n_features) # Cannot drop more than available
                        if actual_sensors_to_drop_random > 0:
                           drop_indices_random = np.random.choice(n_features, actual_sensors_to_drop_random, replace=False)
                           final_mask_torch[:, drop_indices_random] = False
                    # else: n_features is 0, cannot drop anything.

            elif irregularity_type == "random":
                if n_features > 0:
                    actual_sensors_to_drop_random = min(n_sensors_to_drop, n_features)
                    if actual_sensors_to_drop_random > 0:
                        drop_indices = np.random.choice(n_features, actual_sensors_to_drop_random, replace=False)
                        final_mask_torch[:, drop_indices] = False
                # else: n_features is 0, cannot drop.
            
            elif irregularity_type == "first_n":
                if n_features > 0:
                    actual_num_to_drop_first_n = min(n_sensors_to_drop, n_features)
                    if actual_num_to_drop_first_n > 0:
                        final_mask_torch[:, :actual_num_to_drop_first_n] = False
                # else: n_features is 0, cannot drop.
            else:
                print(f"Warning: Unknown irregularity_type '{irregularity_type}'. No irregularity applied due to unknown type.")
    
    final_mask_np = final_mask_torch.numpy()
    # 3. Create x_obs with NaNs
    x_obs_np = np.where(final_mask_np, normalized_data.numpy(), np.nan)

    return {
        't_obs': raw_timestamps_single_sample.clone().detach().float(), # Should be (median_len,)
        'x_obs': torch.from_numpy(x_obs_np).float(), # (median_len, num_features) with NaNs
        'class_label': class_label_single_sample.clone().detach().long() # scalar long
    }

def get_pam_data(args, device):
    """
    Main function to load and prepare PAMAP2 DataLoaders for multiclass classification.
    args should contain: data_path, median_len (will be set to 600), batch_size, num_workers.
    Optional PAM-specific args: pam_apply_irregularity, pam_irregularity_rate, pam_irregularity_type.
    """
    if not hasattr(args, 'median_len') or args.median_len != 600:
        print(f"Warning: median_len for PAM dataset is fixed to 600. Overriding args.median_len (was {getattr(args, 'median_len', 'Not set')}).")
    args.median_len = 600 # PAM uses fixed length 600

    all_data, all_timestamps_individual, all_lengths, all_labels, \
    train_indices, val_indices, test_indices, \
    density_scores, num_features = _load_raw_pam_data(args)

    # Calculate normalization statistics from the training set
    train_data_mean, train_data_std = _calculate_norm_stats(all_data, all_lengths, train_indices)
    
    # Determine irregularity settings for val/test (train is usually not augmented with irregularity)
    # Original PAM loader applied irregularity to val/test but not train.
    # We'll allow this to be configurable via args.pam_train_apply_irregularity if needed in future.
    # For now, irregularity is only potentially applied if args.pam_apply_irregularity is True globally.
    # The density_scores loaded are global; specific handling per sample if type is 'fixed'.

    processed_train_samples = []
    apply_irreg_train = getattr(args, 'pam_train_apply_irregularity', False)
    for idx in tqdm(train_indices, desc=f"Processing PAM Train (len={args.median_len})"):
        sample = _process_pam_sample_for_classification(
            raw_data_single_sample=all_data[idx], 
            raw_timestamps_single_sample=all_timestamps_individual, 
            original_length=all_lengths[idx].item(), 
            class_label_single_sample=all_labels[idx],
            apply_irregularity_for_this_sample=apply_irreg_train,
            irregularity_rate=args.pam_irregularity_rate,
            irregularity_type=args.pam_irregularity_type,
            median_len=args.median_len, # This is fixed to 600 for PAM by args override
            data_mean=train_data_mean, 
            data_std=train_data_std, 
            sensor_density_scores_arg=density_scores, 
            n_features=num_features
        )
        processed_train_samples.append(sample)

    processed_val_samples = []
    apply_irreg_val_test = getattr(args, 'pam_apply_irregularity', False)
    for idx in tqdm(val_indices, desc=f"Processing PAM Val (len={args.median_len})"):
        sample = _process_pam_sample_for_classification(
            raw_data_single_sample=all_data[idx], 
            raw_timestamps_single_sample=all_timestamps_individual, 
            original_length=all_lengths[idx].item(), 
            class_label_single_sample=all_labels[idx],
            apply_irregularity_for_this_sample=apply_irreg_val_test,
            irregularity_rate=args.pam_irregularity_rate,
            irregularity_type=args.pam_irregularity_type,
            median_len=args.median_len,
            data_mean=train_data_mean, 
            data_std=train_data_std, 
            sensor_density_scores_arg=density_scores, 
            n_features=num_features
        )
        processed_val_samples.append(sample)

    processed_test_samples = []
    # apply_irreg_val_test is also used for test split
    for idx in tqdm(test_indices, desc=f"Processing PAM Test (len={args.median_len})"):
        sample = _process_pam_sample_for_classification(
            raw_data_single_sample=all_data[idx], 
            raw_timestamps_single_sample=all_timestamps_individual, 
            original_length=all_lengths[idx].item(), 
            class_label_single_sample=all_labels[idx],
            apply_irregularity_for_this_sample=apply_irreg_val_test,
            irregularity_rate=args.pam_irregularity_rate,
            irregularity_type=args.pam_irregularity_type,
            median_len=args.median_len,
            data_mean=train_data_mean, 
            data_std=train_data_std, 
            sensor_density_scores_arg=density_scores, 
            n_features=num_features
        )
        processed_test_samples.append(sample)

    train_dataset = PAMClassificationDataset(processed_train_samples)
    val_dataset = PAMClassificationDataset(processed_val_samples)
    test_dataset = PAMClassificationDataset(processed_test_samples)

    common_loader_params = {
        'batch_size': args.batch_size,
        'num_workers': getattr(args, 'num_workers', 0),
        'collate_fn': tsdm_collate_classification,
        'pin_memory': True
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_params)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_params)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_params)

    return train_loader, val_loader, test_loader, num_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PAM Data Loader Test")
    # Use a known path to PAMdata for testing, or make it a required argument
    parser.add_argument('--data_path', type=str, required=True, help="Path to the PAMdata directory")
    parser.add_argument('--median_len', type=int, default=600, help="Fixed sequence length for PAM (should be 600)")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of dataloader workers")
    
    # Args for irregularity, mirroring train_grafiti.py
    parser.add_argument('--pam_apply_irregularity', action='store_true', help="Apply irregularity to PAM val/test splits")
    parser.add_argument('--pam_train_apply_irregularity', action='store_true', help="Apply irregularity to PAM train split")
    parser.add_argument('--pam_irregularity_rate', type=float, default=0.1, help="PAM irregularity rate (fraction of sensors to drop, e.g., 0.1 for 10%)")
    parser.add_argument('--pam_irregularity_type', type=str, default='random', choices=['random', 'fixed', 'first_n'], help="PAM irregularity type (default random)")


    args = parser.parse_args()
    print(f"Testing PAM loader with args: {args}")

    train_loader, val_loader, test_loader, num_feats = get_pam_data(args, torch.device("cpu"))

    print(f"Number of features from PAM loader: {num_feats}")
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")

    def check_batch(loader, name):
        try:
            first_batch = next(iter(loader))
            print(f"First batch from {name} loaded successfully.")
            # Assuming the collate function returns BatchClassification which has x_vals, x_mask, class_labels
            print(f"  Batch x_time shape: {first_batch.x_time.shape}") 
            print(f"  Batch x_vals shape: {first_batch.x_vals.shape}") 
            print(f"  Batch x_vals NaNs: {torch.isnan(first_batch.x_vals).sum()} / {first_batch.x_vals.numel()}")
            print(f"  Batch x_mask shape: {first_batch.x_mask.shape}")
            print(f"  Batch y_mask_for_grafiti_structure shape: {first_batch.y_mask_for_grafiti_structure.shape}")
            print(f"  Batch class_labels shape: {first_batch.class_labels.shape}") 
            print(f"  Batch class_labels dtype: {first_batch.class_labels.dtype}")
            print(f"  Batch class_labels example: {first_batch.class_labels[:5].squeeze()}")


        except Exception as e:
            print(f"Error processing first batch from {name}: {e}")
            import traceback
            traceback.print_exc()

    if train_loader: check_batch(train_loader, "train")
    if val_loader: check_batch(val_loader, "validation")
    if test_loader: check_batch(test_loader, "test")