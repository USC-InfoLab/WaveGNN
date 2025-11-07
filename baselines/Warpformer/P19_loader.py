import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from Utils import cal_tau # Assuming Utils.py is in the accessible path

def get_p19_data(args, device):
    # Hard-coded values from the original configuration for P19
    main_path = "/storage/datasets_public/irreg_ts/datasets/"
    data_path = "P19data/processed_data/PT_dict_list_6.npy"
    label_path = "P19data/processed_data/arr_outcomes_6.npy"
    splits_path = "P19data/splits/"
    split_num = 1  # Using split 1 as default, from original P19 loader

    # Read dataset
    data_file = main_path + data_path
    labels_file = main_path + label_path
    splits_dir = main_path + splits_path

    # Load the data
    data_dict = np.load(data_file, allow_pickle=True)
    labels_np = np.load(labels_file, allow_pickle=True)

    # Extract and normalize data
    data = extract_attributes(data_dict, "arr")
    timestamps = extract_attributes(data_dict, "time") / 60.0  # convert to mins
    lengths = extract_attributes(data_dict, "length").long()

    # Extract labels (mortality is the last one)
    y = labels_np[:, -1]
    y = torch.from_numpy(y).type(torch.float32) # Will be converted to long in create_dataloader

    # Normalize data (using its own mean/std)
    data, _, _ = normalize(data)

    # Split data using P19 specific split file name
    sfile = f"phy19_split{split_num}_new" # P19 specific
    train_indices, val_indices, test_indices = np.load(
        f"{splits_dir}{sfile}.npy", allow_pickle=True
    )

    # Process data for each split
    # window_size here will be args.median_len passed from Main_warp.py
    train_data = process_split_data(data, timestamps, lengths, y, train_indices, args.median_len)
    val_data = process_split_data(data, timestamps, lengths, y, val_indices, args.median_len)
    test_data = process_split_data(data, timestamps, lengths, y, test_indices, args.median_len)

    # Create dataloaders
    train_loader = create_dataloader(train_data, args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_data, args.batch_size, shuffle=False)
    test_loader = create_dataloader(test_data, args.batch_size, shuffle=False)

    # Set number of features
    args.num_types = data.shape[2]

    return train_loader, val_loader, test_loader, args.num_types

def extract_attributes(data_dict, attr):
    X = [d[attr] for d in data_dict]
    X = np.stack(X, axis=0)
    X = X.astype(np.float32)
    return torch.from_numpy(X)

def normalize(data, mean=None, std=None):
    # Calculate mean and std if not provided (e.g., per-dataset normalization)
    if mean is None:
        mean = data.mean(axis=(0, 1), keepdims=True) # Mean over samples and time
    if std is None:
        std = data.std(axis=(0, 1), keepdims=True)   # Std over samples and time
    
    # Avoid division by zero for features with no variance
    std[std == 0] = 1.0
    
    data = (data - mean) / std
    data = torch.nan_to_num(data) # Handle any NaNs that might arise
    return data, mean, std

def process_split_data(data, timestamps, lengths, labels, indices, window_size):
    processed_data = []
    # data shape: (num_total_samples, max_original_seq_len, n_sensors)
    # timestamps shape: (num_total_samples, max_original_seq_len) or (num_total_samples, max_original_seq_len, 1)
    # lengths shape: (num_total_samples,)
    n_sensors = data.shape[2]

    for idx in indices:
        original_len = lengths[idx].item()

        len_to_take = min(original_len, window_size)

        X_sample_actual = data[idx, :len_to_take, :]
        # Ensure timestamps are [len_to_take] by selecting and squeezing
        t_sample_actual = timestamps[idx, :len_to_take].squeeze() 
        if t_sample_actual.ndim == 0 and len_to_take == 1: # Handle single timestamp case
             t_sample_actual = t_sample_actual.unsqueeze(0)


        observed_data_actual_np = X_sample_actual.numpy()
        observed_tp_actual_np = t_sample_actual.numpy()
        
        observed_mask_actual_np = np.ones((len_to_take, n_sensors), dtype=np.int32)

        if len_to_take > 0:
            tau_actual_np = cal_tau(
                observed_tp_actual_np.reshape(1, len_to_take),
                observed_mask_actual_np.reshape(1, len_to_take, n_sensors)
            ).squeeze(0)
        else:
            tau_actual_np = np.zeros((0, n_sensors), dtype=np.float32)

        observed_data_final_np = np.zeros((window_size, n_sensors), dtype=np.float32)
        observed_mask_final_np = np.zeros((window_size, n_sensors), dtype=np.int32)
        observed_tp_final_np = np.zeros((window_size, 1), dtype=np.float32)
        tau_final_np = np.zeros((window_size, n_sensors), dtype=np.float32)
        sequence_end_final_np = np.zeros((window_size, 1), dtype=np.float32)

        if len_to_take > 0:
            observed_data_final_np[:len_to_take, :] = observed_data_actual_np
            observed_mask_final_np[:len_to_take, :] = observed_mask_actual_np
            observed_tp_final_np[:len_to_take, 0] = observed_tp_actual_np
            tau_final_np[:len_to_take, :] = tau_actual_np
            sequence_end_final_np[len_to_take - 1, 0] = 1.0
        
        tp_and_seq_end_final_np = np.concatenate([observed_tp_final_np, sequence_end_final_np], axis=-1)

        combined_data_sample = np.concatenate(
            (observed_data_final_np,
             observed_mask_final_np,
             tp_and_seq_end_final_np,
             tau_final_np),
            axis=-1
        )
        processed_data.append((combined_data_sample, labels[idx].item()))
    
    return processed_data

def create_dataloader(data_list, batch_size, shuffle=False):
    # data_list: list of tuples (combined_data_sample_np, label_scalar)
    # each combined_data_sample_np is (window_size, 3*K + 2)
    
    # Stack all sequence tensors: results in (B, window_size, 3*K + 2)
    data_x_tensor = torch.stack([torch.from_numpy(x[0]).float() for x in data_list])
    # Convert labels to a tensor of longs: results in (B,)
    data_y_tensor = torch.tensor([x[1] for x in data_list], dtype=torch.long)
    
    dataset = TensorDataset(data_x_tensor, data_y_tensor)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=8, # Consistent with other loaders
        drop_last=False # To ensure all samples are processed, esp. with small test sets
    )
    return dataloader

# Keep the if __name__ == "__main__": block for standalone testing if desired
# but ensure it does not interfere with the main importable functions.
# For example, by removing or commenting it out if it causes issues during import.
# if __name__ == "__main__":
#     from collections import Counter
#     # Minimal args for testing
#     class Args:
#         median_len = 60 
#         batch_size = 4
        
#     args_test = Args()
#     device_test = torch.device("cpu")
    
#     train_loader, val_loader, test_loader, num_types = get_p19_data(args_test, device_test)
    
#     print(f"Number of types (features): {num_types}")
#     print(f"Train loader batches: {len(train_loader)}")
#     for batch_idx, (data, labels) in enumerate(train_loader):
#         print(f"Train Batch {batch_idx + 1} data shape: {data.shape}, labels shape: {labels.shape}")
#         break 
#     # Further checks can be added here, e.g., t_min, t_max, label_cnt from original