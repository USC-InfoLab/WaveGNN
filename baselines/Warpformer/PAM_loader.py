import torch
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from Utils import cal_tau

def get_pam_data(args, device):
    # Hard-coded values from the JSON configuration
    main_path = "/storage/datasets_public/irreg_ts/datasets/"
    data_path = "PAMdata/processed_data/PTdict_list.npy"
    label_path = "PAMdata/processed_data/arr_outcomes.npy"
    splits_path = "PAMdata/splits/"
    window_size = 600
    irregularity = False  # From JSON config
    irregularity_rate = 0.0  # From JSON config
    irregularity_type = "fixed"  # From JSON config
    
    # Read dataset
    data_file = main_path + data_path
    labels_file = main_path + label_path
    splits_dir = main_path + splits_path
    
    # Load the data
    data_dict = np.load(data_file, allow_pickle=True)
    labels = np.load(labels_file, allow_pickle=True)
    
    # Extract and normalize data
    data = data_dict.astype(np.float32)
    data = torch.from_numpy(data)
    data = normalize(data)
    
    # Create timestamps exactly as in original code
    timestamps = torch.arange(data.shape[1])
    timestamps = timestamps.unsqueeze(0).repeat(data.shape[0], 1) / 60.0
    
    # Create lengths (all 600)
    lengths = torch.ones(data.shape[0]) * window_size
    
    # Extract labels (mortality is the last one)
    y = labels[:, -1]
    y = torch.from_numpy(y)
    y = y.type(torch.float32)
    
    # Split data using split file number 1
    split_num = 1
    sfile = f"PAM_split_{split_num}"
    train_indices, val_indices, test_indices = np.load(
        f"{splits_dir}{sfile}.npy", allow_pickle=True
    )
    
    # Load sensor density scores if needed for fixed irregularity
    sensor_density_path = "PAMdata/IG_density_scores_PAM.npy"
    density_scores = None
    if irregularity and irregularity_type == "fixed":
        density_scores = np.load(main_path + sensor_density_path, allow_pickle=True)
    
    # Process data for each split
    train_data = process_split_data(data, timestamps, lengths, y, train_indices, window_size, 
                                  False, irregularity_rate, irregularity_type, density_scores)
    val_data = process_split_data(data, timestamps, lengths, y, val_indices, window_size,
                                irregularity, irregularity_rate, irregularity_type, density_scores)
    test_data = process_split_data(data, timestamps, lengths, y, test_indices, window_size,
                                 irregularity, irregularity_rate, irregularity_type, density_scores)
    
    # Create dataloaders
    train_loader = create_dataloader(train_data, args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_data, args.batch_size, shuffle=False)
    test_loader = create_dataloader(test_data, args.batch_size, shuffle=False)
    
    # Set number of features
    args.num_types = data.shape[2]
    
    return train_loader, val_loader, test_loader, args.num_types

def normalize(data):
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data = torch.nan_to_num(data)
    return data

def process_split_data(data, timestamps, lengths, labels, indices, window_size, 
                      irregularity, irregularity_rate, irregularity_type, density_scores):
    processed_data = []
    n_sensors = data.shape[2] # K (number of features for observed_data)
    
    for idx in indices:
        # Sample a random window
        len_sample = lengths[idx].item()
        w_idx = random.randint(0, int(len_sample) - window_size) if len_sample > window_size else 0
        
        X_sample = data[idx, w_idx:w_idx + window_size] # Shape (L, K)
        t_sample = timestamps[idx, w_idx:w_idx + window_size] # Shape (L,)
        
        # Create mask for irregularity
        if irregularity:
            n_sensors_to_drop = round(irregularity_rate * n_sensors)
            if irregularity_type == "fixed":
                if density_scores is not None:
                    density_score_indices = density_scores[:n_sensors_to_drop, 0]
                    drop_indices = np.array(density_score_indices, dtype=int)
                    base_mask_torch = torch.ones(window_size, n_sensors, dtype=torch.bool)
                    base_mask_torch[:, drop_indices] = False
                else:
                    base_mask_torch = torch.ones(window_size, n_sensors, dtype=torch.bool)
            elif irregularity_type == "random":
                drop_indices = np.random.choice(n_sensors, n_sensors_to_drop, replace=False)
                base_mask_torch = torch.ones(window_size, n_sensors, dtype=torch.bool)
                base_mask_torch[:, drop_indices] = False
            else: # 'channel'
                drop_indices = int(irregularity_rate * n_sensors)
                base_mask_torch = torch.ones(window_size, n_sensors, dtype=torch.bool)
                base_mask_torch[:, :drop_indices] = False
        else:
            base_mask_torch = torch.ones(window_size, n_sensors, dtype=torch.bool)
        
        # Apply padding mask based on length
        padding_mask_torch = torch.arange(window_size) < (len_sample - w_idx)
        final_mask_torch = base_mask_torch & padding_mask_torch.unsqueeze(1) # Shape (L, K)
        
        # Apply mask to data
        masked_sample_torch = X_sample * final_mask_torch.float() # Shape (L, K)
        
        # Convert to NumPy
        observed_data_np = masked_sample_torch.numpy()  # Shape (L, K)
        observed_mask_np = final_mask_torch.int().numpy()     # Shape (L, K)
        observed_tp_np = t_sample.numpy()         # Shape (L,)
        
        # Prepare inputs for cal_tau (expects batched input)
        observed_tp_for_tau = observed_tp_np.reshape(1, -1)  # Shape (1, L)
        observed_mask_for_tau = observed_mask_np.reshape(1, -1, n_sensors)  # Shape (1, L, K)
        
        # Calculate tau
        tau_batched = cal_tau(observed_tp_for_tau, observed_mask_for_tau)  # Shape (1, L, K)
        tau_np = tau_batched.squeeze(0) # Shape (L, K)
        
        # Prepare observed_tp and sequence_end for final concatenation
        observed_tp_reshaped_np = observed_tp_np.reshape(window_size, 1) # Shape (L, 1)
        
        sequence_end_np = np.zeros((window_size, 1), dtype=np.float32) # Shape (L, 1)
        sequence_end_np[-1, 0] = 1.0
        
        tp_and_seq_end_np = np.concatenate([observed_tp_reshaped_np, sequence_end_np], axis=-1) # Shape (L, 2)
        
        # Combine all components for a single sample
        # Expected order: data (K), mask (K), tp_orig (1), seq_end (1), tau (K)
        # Total features = K + K + 1 + 1 + K = 3K + 2
        combined_data_sample = np.concatenate(
            (observed_data_np,       # (L, K)
             observed_mask_np,       # (L, K)
             tp_and_seq_end_np,      # (L, 2)  (contains tp_orig and seq_end)
             tau_np),                # (L, K)
            axis=-1  # Concatenate along the feature dimension
        ) # Shape (L, 3K+2)
        
        processed_data.append((combined_data_sample, labels[idx].item())) # labels[idx] is a 0-dim tensor
    
    return processed_data

def create_dataloader(data, batch_size, shuffle=False):
    # data_x: list of (L, 3K+2) numpy arrays
    # data_y: list of scalar floats
    
    # Convert data to tensors
    # Each x[0] is (L, 3K+2)
    data_x_tensor = torch.stack([torch.from_numpy(x[0]).float() for x in data]) # Shape (B, L, 3K+2)
    # Each x[1] is a scalar label - convert to long for classification
    data_y_tensor = torch.tensor([x[1] for x in data], dtype=torch.long) # Shape (B,)
    
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