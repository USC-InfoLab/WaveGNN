import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os # Added for os.path.join

# Assuming tsdm_collate_classification is in GraFITi.grafiti.grafiti
# This will be resolved by the user or in a later step if the path is incorrect.
# from GraFITi.grafiti.grafiti import tsdm_collate_classification
# For now, to make it runnable standalone for testing, let's define a placeholder
# In the actual integration, train_grafiti.py will handle the import.

class P19ClassificationDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def _normalize_data(data_split):
    """Normalizes data (mean=0, std=1) column-wise."""
    mean = np.nanmean(data_split, axis=(0, 1), keepdims=True)
    std = np.nanstd(data_split, axis=(0, 1), keepdims=True)
    std[std == 0] = 1 # Avoid division by zero for constant features
    normalized_data = (data_split - mean) / std
    return normalized_data

def _load_and_process_p19_split(split_name, args):
    """
    Loads and preprocesses a specific split of the P19 dataset.
    args.data_path should point to the root directory for P19, e.g., "P19data/"
    which contains "processed_data/" and "splits/"
    """
    # Construct paths using args.data_path
    # main_path = args.data_path # e.g., /storage/datasets_public/irreg_ts/datasets/
    # data_subdir = "P19data" # This seems to be part of the original hardcoded path

    # The user's args.data_path should be the directory containing 'processed_data' and 'splits'
    # e.g. if files are in /my_data/P19/processed_data, args.data_path = /my_data/P19
    
    # Corrected path logic: args.data_path is the base P19 directory
    # like /storage/datasets_public/irreg_ts/datasets/P19data/
    data_dict_path = os.path.join(args.data_path, "processed_data", "PT_dict_list_6.npy")
    label_path = os.path.join(args.data_path, "processed_data", "arr_outcomes_6.npy")
    split_file_name = f"phy19_split1_new.npy" # Hardcoded split_num = 1
    splits_path = os.path.join(args.data_path, "splits", split_file_name)

    try:
        data_dict_raw = np.load(data_dict_path, allow_pickle=True)
        labels_raw = np.load(label_path, allow_pickle=True)
        split_indices_all = np.load(splits_path, allow_pickle=True)
    except FileNotFoundError as e:
        print(f"Error loading P19 data files: {e}")
        print(f"Looked for data_dict: {data_dict_path}")
        print(f"Looked for labels: {label_path}")
        print(f"Looked for splits: {splits_path}")
        raise

    # Extract from raw loaded data
    all_values = np.stack([d["arr"] for d in data_dict_raw], axis=0).astype(np.float32)
    all_times = np.stack([d["time"] for d in data_dict_raw], axis=0).astype(np.float32) / 60.0 # convert to mins
    all_lengths = np.stack([d["length"] for d in data_dict_raw], axis=0).astype(int)
    
    # Labels: mortality is the last outcome
    all_labels = labels_raw[:, -1].astype(np.float32)

    split_map = {'train': 0, 'val': 1, 'test': 2}
    current_split_indices = split_indices_all[split_map[split_name]]

    processed_samples = []
    num_features = all_values.shape[2] if all_values.ndim == 3 else 1 # Infer number of features

    # Normalize features per split based on training data statistics if it's 'train',
    # otherwise, it should use pre-computed stats or normalize independently.
    # For simplicity and consistency with P12, let's normalize each split independently first.
    # The original code normalized the entire dataset before splitting.
    # We'll normalize selected data for the current split.
    
    split_values = all_values[current_split_indices]
    # Normalize the 'values' part of the current split
    # The original P19 loader normalized all data together.
    # The P12 loader normalized each split independently.
    # Let's follow the P12 pattern of split-wise normalization for now.
    
    # Calculate mean and std for normalization based on the training split's values
    if split_name == 'train':
        train_values_for_norm = all_values[split_indices_all[split_map['train']]]
        # Filter out padding before calculating mean/std
        flat_train_values = []
        for i, idx in enumerate(split_indices_all[split_map['train']]):
            seq_len = all_lengths[idx].item()
            flat_train_values.append(train_values_for_norm[i, :seq_len, :])
        flat_train_values_np = np.concatenate(flat_train_values, axis=0)
        
        data_mean = np.mean(flat_train_values_np, axis=0)
        data_std = np.std(flat_train_values_np, axis=0)
        data_std[data_std == 0] = 1 # Avoid division by zero
        
        # Store these for use in val/test if needed, or pass them. For now, saved in args for simplicity.
        args.p19_data_mean = data_mean
        args.p19_data_std = data_std
    else: # val or test
        if not hasattr(args, 'p19_data_mean') or not hasattr(args, 'p19_data_std'):
            # This case should ideally not happen if train is processed first.
            # Fallback to split-specific normalization, or raise error.
            # For now, let's print a warning and normalize by current split.
            print(f"Warning: Normalizing {split_name} P19 data with its own stats because train stats not found in args.")
            current_split_values_for_norm = []
            for i, idx in enumerate(current_split_indices):
                seq_len = all_lengths[idx].item()
                current_split_values_for_norm.append(all_values[idx, :seq_len, :])
            flat_current_split_values_np = np.concatenate(current_split_values_for_norm, axis=0)

            data_mean = np.mean(flat_current_split_values_np, axis=0)
            data_std = np.std(flat_current_split_values_np, axis=0)
            data_std[data_std == 0] = 1
        else:
            data_mean = args.p19_data_mean
            data_std = args.p19_data_std

    for i, record_idx in enumerate(current_split_indices):
        original_seq_len = all_lengths[record_idx].item()
        
        # Get full sequences
        # Ensure time is 1D if it comes out as 2D from selection
        t_obs_full = all_times[record_idx].squeeze(-1) if all_times[record_idx].ndim > 1 else all_times[record_idx]
        t_obs_full = t_obs_full[:original_seq_len]

        x_obs_full = all_values[record_idx, :original_seq_len, :]
        
        # Normalize x_obs_full
        x_obs_full_normalized = (x_obs_full - data_mean) / data_std
        
        class_label = all_labels[record_idx]

        # Prepare for fixed length
        t_obs = np.zeros(args.median_len, dtype=np.float32)
        x_obs_padded = np.zeros((args.median_len, num_features), dtype=np.float32)
        mask_fixed = np.zeros(args.median_len, dtype=bool)

        # Truncate or pad
        actual_len = min(original_seq_len, args.median_len)
        
        t_obs[:actual_len] = t_obs_full[:actual_len]
        x_obs_padded[:actual_len] = x_obs_full_normalized[:actual_len]
        mask_fixed[:actual_len] = True
        
        # Introduce NaNs for missing values based on the mask
        x_obs = x_obs_padded.copy()
        x_obs[~mask_fixed] = np.nan # Apply NaN where mask is False (i.e. padded regions)

        # Sanity check for t_obs shape, should be (median_len,)
        if t_obs.ndim > 1:
             t_obs = t_obs.squeeze() # Should not happen if t_obs_full is 1D

        processed_samples.append({
            't_obs': torch.from_numpy(t_obs).float(),
            'x_obs': torch.from_numpy(x_obs).float(), # This contains NaNs
            'class_label': torch.tensor(class_label).float()
        })

    return processed_samples, num_features


def get_p19_data(args, device):
    """
    Returns P19 DataLoaders for classification.
    args should contain: data_path, median_len, batch_size, num_workers, split_num_p19
    """
    # Import here to avoid circular dependency if this file is imported elsewhere
    # And to ensure it's available when get_p19_data is called from train_grafiti
    try:
        from grafiti.grafiti import tsdm_collate_classification
    except ImportError:
        raise ImportError("Could not import tsdm_collate_classification from grafiti.grafiti. Ensure the path is correct.")

    # args.split_num_p19 is no longer needed from args, as it's hardcoded to 1.
    # if not hasattr(args, 'split_num_p19'):
    #     args.split_num_p19 = 1 # Default P19 split
    #     print(f"args.split_num_p19 not found, defaulting to {args.split_num_p19}")


    # Process training data first to get normalization stats
    train_samples, num_features = _load_and_process_p19_split('train', args)
    # Val and test splits will use normalization stats from args (set during train processing)
    val_samples, _ = _load_and_process_p19_split('val', args)
    test_samples, _ = _load_and_process_p19_split('test', args)

    train_dataset = P19ClassificationDataset(train_samples)
    val_dataset = P19ClassificationDataset(val_samples)
    test_dataset = P19ClassificationDataset(test_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=tsdm_collate_classification,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=tsdm_collate_classification,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=tsdm_collate_classification,
        pin_memory=True
    )
    
    # Clean up args to prevent them from persisting if args is mutable and shared
    if hasattr(args, 'p19_data_mean'):
        del args.p19_data_mean
    if hasattr(args, 'p19_data_std'):
        del args.p19_data_std

    return train_loader, val_loader, test_loader, num_features

# Example usage (for testing this script directly)
if __name__ == "__main__":
    import argparse
    # Define a dummy tsdm_collate_classification for standalone testing
    def tsdm_collate_classification(batch):
        # This is a simplified placeholder. The real one handles more complex collation.
        # from GraFITi.grafiti.grafiti import BatchClassification # Assuming this exists
        
        # Placeholder BatchClassification if not available
        from collections import namedtuple
        BatchClassification = namedtuple("BatchClassification", ["inputs", "targets", "class_labels", "meta_info"])
        Sample = namedtuple("Sample", ["t", "x", "t_target", "y_target", "active_entries"])


        t_obs_list = [item['t_obs'] for item in batch]
        x_obs_list = [item['x_obs'] for item in batch]
        class_labels_list = [item['class_label'] for item in batch]

        # Pad t_obs and x_obs (already done to median_len, but collate usually handles batch-wise padding)
        # For now, assume they are already padded and just stack
        t_obs_batch = torch.stack(t_obs_list)
        x_obs_batch = torch.stack(x_obs_list) # Should be (batch, median_len, num_features)
        
        # Create dummy t_target and y_target as the model expects them for its internal structure
        # These won't be used for loss in classification.
        # The values are arbitrary but shapes need to be consistent with model's expectation.
        # Let's assume target sequence length is also median_len for simplicity here.
        t_target_batch = torch.zeros_like(t_obs_batch) # Dummy
        y_target_batch = torch.zeros_like(x_obs_batch) # Dummy, shape (batch, median_len, num_features)
        
        # Active entries: a mask indicating valid data points, not padding.
        # In our case, x_obs already contains NaNs for padding.
        # The model's grafiti_layers.py uses this for creating attention masks.
        # It expects non-NaN values to be 1.0 and NaNs to be 0.0.
        # So, active_entries should be True where x_obs is not NaN (first feature).
        # Since we have fixed length, and NaN indicates padding, we can derive this.
        # However, the original collate used it for forecasting targets.
        # For classification, the specific structure of 'inputs' for grafiti_layers matters.
        # The `grafiti_` class expects inputs.x and inputs.t
        # Let's assume the model's internal padding mechanism will handle NaNs in x_obs correctly.

        # The get_contextualized_embeddings path uses `mask = ~torch.isnan(inputs.x[..., 0])`
        # So, NaNs in inputs.x are critical.
        
        inputs = Sample(t=t_obs_batch, x=x_obs_batch, t_target=t_target_batch, y_target=y_target_batch, active_entries=None)
        
        # Targets for forecasting (dummy)
        targets = y_target_batch # Or some other dummy structure based on model's forward pass

        class_labels_batch = torch.stack(class_labels_list).unsqueeze(-1) # Ensure (batch_size, 1) for BCEWithLogitsLoss

        # meta_info can be empty or contain relevant info
        meta_info = {'example_indices': [i for i in range(len(batch))]}
        
        return BatchClassification(inputs=inputs, targets=targets, class_labels=class_labels_batch, meta_info=meta_info)


    parser = argparse.ArgumentParser(description="P19 Data Loader Test")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the P19data directory (containing processed_data and splits)")
    parser.add_argument('--median_len', type=int, default=50, help="Fixed sequence length")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of dataloader workers")
    # parser.add_argument('--split_num_p19', type=int, default=1, help="P19 split number to use (1-5)") # Removed
    
    # Add dummy args that might be expected by the imported tsdm_collate_classification or model
    parser.add_argument('--ವಾscreenಹಾಕಲಾಗಿದೆ', type=str, default='absolute', help='Dummy arg for time encoding')


    args = parser.parse_args()

    print(f"Attempting to load P19 data with args: {args}")

    # In a real scenario, tsdm_collate_classification would be properly imported
    # For this test, we use the placeholder.
    # To make this testable standalone, we need to ensure the import is handled.
    # One way: if __name__ == "__main__": GraFITi.grafiti.grafiti.tsdm_collate_classification = tsdm_collate_classification
    
    # For testing, let's try to use the real collate if available, else the dummy
    try:
        from GraFITi.grafiti.grafiti import tsdm_collate_classification as real_collate
        print("Using real tsdm_collate_classification.")
    except ImportError:
        print("Real tsdm_collate_classification not found, using placeholder for testing.")
        # This assignment is tricky due to module loading.
        # Instead, get_p19_data will try to import it, and if it fails, it's an issue for integration.
        # The if __name__ == "__main__": block should use its local placeholder.

    # To ensure the test uses the local placeholder:
    global_collate_backup = None
    try:
        import GraFITi.grafiti.grafiti
        if hasattr(GraFITi.grafiti.grafiti, 'tsdm_collate_classification'):
            global_collate_backup = GraFITi.grafiti.grafiti.tsdm_collate_classification
        GraFITi.grafiti.grafiti.tsdm_collate_classification = tsdm_collate_classification
        print("Temporarily overriding tsdm_collate_classification for standalone test.")
    except ImportError:
        print("GraFITi.grafiti.grafiti module not found, test will rely on get_p19_data's import error or a miracle.")
        # If the module itself doesn't exist, this direct patching won't work.
        # The get_p19_data has its own try-except for the import.

    train_loader, val_loader, test_loader, num_feats = get_p19_data(args, torch.device("cpu"))

    print(f"Number of features: {num_feats}")
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")

    # Check one batch from train_loader
    try:
        first_batch = next(iter(train_loader))
        print("First batch loaded successfully.")
        print(f"Batch inputs t shape: {first_batch.inputs.t.shape}")
        print(f"Batch inputs x shape: {first_batch.inputs.x.shape}") # Should have NaNs
        print(f"Batch inputs x NaNs: {torch.isnan(first_batch.inputs.x).sum()} / {first_batch.inputs.x.numel()}")
        print(f"Batch class_labels shape: {first_batch.class_labels.shape}")
        print(f"Batch class_labels example: {first_batch.class_labels[:5]}")
        
        # Verify that t_target and y_target are present (even if dummy)
        if hasattr(first_batch.inputs, 't_target') and hasattr(first_batch.inputs, 'y_target'):
             print(f"Batch inputs t_target shape: {first_batch.inputs.t_target.shape}")
             print(f"Batch inputs y_target shape: {first_batch.inputs.y_target.shape}")
        else:
            print("Warning: t_target or y_target missing from batch.inputs")

    except Exception as e:
        print(f"Error processing first batch: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore backup if it was made
        if global_collate_backup is not None and 'GraFITi.grafiti.grafiti' in sys.modules:
            GraFITi.grafiti.grafiti.tsdm_collate_classification = global_collate_backup
            print("Restored original tsdm_collate_classification.")