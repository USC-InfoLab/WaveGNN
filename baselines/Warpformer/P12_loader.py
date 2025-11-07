import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from Utils import cal_tau

def get_p12_data(args, device):
    # Hard-coded values from the original configuration
    main_path = "/storage/datasets_public/irreg_ts/datasets/"
    data_path = "P12data/processed_data/PTdict_list.npy"
    label_path = "P12data/processed_data/arr_outcomes.npy"
    splits_path = "P12data/splits/"
    split_num = 1  # Using split 1 as default
    
    # Read dataset
    data_file = main_path + data_path
    labels_file = main_path + label_path
    splits_dir = main_path + splits_path
    
    # Load the data
    data_dict = np.load(data_file, allow_pickle=True)
    labels = np.load(labels_file, allow_pickle=True)
    
    # Extract and normalize data
    data = extract_attributes(data_dict, "arr")
    timestamps = extract_attributes(data_dict, "time") / 60.0  # convert to mins
    lengths = extract_attributes(data_dict, "length").long()
    
    # Extract labels (mortality is the last one)
    y = labels[:, -1]
    y = torch.from_numpy(y).type(torch.float32)
    
    # Normalize data
    data, _, _ = normalize(data)
    
    # Split data using split file
    sfile = f"phy12_split{split_num}"
    train_indices, val_indices, test_indices = np.load(
        f"{splits_dir}{sfile}.npy", allow_pickle=True
    )
    
    # Process data for each split
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
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data = (data - mean) / std
    data = torch.nan_to_num(data)
    return data, mean, std

def process_split_data(data, timestamps, lengths, labels, indices, window_size):
    processed_data = []
    # data shape: (num_total_samples, max_original_seq_len, n_sensors)
    # timestamps shape: (num_total_samples, max_original_seq_len, 1)
    # lengths shape: (num_total_samples,)
    n_sensors = data.shape[2]

    for idx in indices:
        original_len = lengths[idx].item()  # Actual length of the sequence for this sample

        # Determine the length to use: min of original length and window_size for truncation/padding
        len_to_take = min(original_len, window_size)

        # Slice the actual data up to len_to_take
        # X_sample_actual shape: (len_to_take, n_sensors)
        # t_sample_actual shape: (len_to_take,)
        X_sample_actual = data[idx, :len_to_take, :]
        t_sample_actual = timestamps[idx, :len_to_take].squeeze(-1)  # Ensure it's (len_to_take,)

        observed_data_actual_np = X_sample_actual.numpy()
        observed_tp_actual_np = t_sample_actual.numpy()
        
        # Create mask for the actual data part
        # observed_mask_actual_np shape: (len_to_take, n_sensors)
        observed_mask_actual_np = np.ones((len_to_take, n_sensors), dtype=np.int32)

        # Calculate tau using only the actual data part
        # tau_actual_np shape: (len_to_take, n_sensors)
        if len_to_take > 0:
            # cal_tau expects tp: (B, L), mask: (B, L, K)
            tau_actual_np = cal_tau(
                observed_tp_actual_np.reshape(1, len_to_take),
                observed_mask_actual_np.reshape(1, len_to_take, n_sensors)
            ).squeeze(0) # Remove batch dim
        else: # Handle case of empty sequence
            tau_actual_np = np.zeros((0, n_sensors), dtype=np.float32)

        # Initialize final arrays with padding (shape: window_size, ...)
        observed_data_final_np = np.zeros((window_size, n_sensors), dtype=np.float32)
        observed_mask_final_np = np.zeros((window_size, n_sensors), dtype=np.int32)
        observed_tp_final_np = np.zeros((window_size, 1), dtype=np.float32) # For tp_orig
        tau_final_np = np.zeros((window_size, n_sensors), dtype=np.float32)
        sequence_end_final_np = np.zeros((window_size, 1), dtype=np.float32)

        # Copy actual data into the final padded/truncated arrays
        if len_to_take > 0:
            observed_data_final_np[:len_to_take, :] = observed_data_actual_np
            observed_mask_final_np[:len_to_take, :] = observed_mask_actual_np
            observed_tp_final_np[:len_to_take, 0] = observed_tp_actual_np
            tau_final_np[:len_to_take, :] = tau_actual_np
            # Sequence end is at the end of the *actual* data within the window
            sequence_end_final_np[len_to_take - 1, 0] = 1.0
        
        # tp_and_seq_end_final_np shape: (window_size, 2)
        tp_and_seq_end_final_np = np.concatenate([observed_tp_final_np, sequence_end_final_np], axis=-1)

        # combined_data_sample shape: (window_size, 3*n_sensors + 2)
        combined_data_sample = np.concatenate(
            (observed_data_final_np,
             observed_mask_final_np,
             tp_and_seq_end_final_np,
             tau_final_np),
            axis=-1
        )
        processed_data.append((combined_data_sample, labels[idx].item()))
    
    return processed_data

def create_dataloader(data, batch_size, shuffle=False):
    # Convert data to tensors
    data_x_tensor = torch.stack([torch.from_numpy(x[0]).float() for x in data])  # Shape (B, L, 3K+2)
    data_y_tensor = torch.tensor([x[1] for x in data], dtype=torch.long)  # Shape (B,)
    
    # Create dataset and dataloader
    dataset = TensorDataset(data_x_tensor, data_y_tensor)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=8,
        drop_last=False
    )
    
    return dataloader

if __name__ == "__main__":
    from collections import Counter
    train_data, val_data, test_data = load_and_preprocess_p12()

    t_min, t_max = float("inf"), float("-inf")
    label_cnt = Counter()
    avr_length = 0
    for i, data in enumerate(train_data):
        if i == 0:
            for d in data:
                print((d.shape if isinstance(d, torch.Tensor) else d))
        t_min = min(t_min, data[1].min())
        t_max = max(t_max, data[1].max())
        label_cnt[data[-1].item()] += 1
        avr_length += len(data[2])
    
    avr_length /= len(train_data)
    print("Average length:", avr_length)
    print("t_min:", t_min)
    print("t_max:", t_max)
    print("label_cnt:", label_cnt)
    print("train_data:", len(train_data))