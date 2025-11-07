import torch
import numpy as np
import random

def load_and_preprocess_pam(apply_irregularity=False, irregularity_rate_param=0.0, irregularity_type_param="random"):
    # Hard-coded values from the JSON configuration
    main_path = "/storage/datasets_public/irreg_ts/datasets/"
    data_path = "PAMdata/processed_data/PTdict_list.npy"
    label_path = "PAMdata/processed_data/arr_outcomes.npy"
    splits_path = "PAMdata/splits/"
    window_size = 600
    
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
    if apply_irregularity and irregularity_type_param == "fixed":
        try:
            density_scores = np.load(main_path + sensor_density_path, allow_pickle=True)
        except FileNotFoundError:
            print(f"Warning: Density scores file not found at {main_path + sensor_density_path}. 'fixed' irregularity will fallback to random if used.")
            density_scores = None
    
    # Create lists to hold the processed data
    train_processed = []
    val_processed = []
    test_processed = []
    
    # Process training data - no irregularity for training as per original code
    for idx in train_indices:
        record_id, timestamp, irg_ts, irg_ts_mask, label = process_sample(
            data[idx], 
            timestamps[idx], 
            lengths[idx], 
            y[idx], 
            idx,
            window_size,
            apply_irregularity,
            irregularity_rate_param,
            irregularity_type_param,
            density_scores,
            data.shape[2]  # Number of sensors
        )
        train_processed.append((record_id, timestamp, irg_ts, irg_ts_mask, label))
    
    # Process validation data
    for idx in val_indices:
        record_id, timestamp, irg_ts, irg_ts_mask, label = process_sample(
            data[idx], 
            timestamps[idx], 
            lengths[idx], 
            y[idx], 
            idx,
            window_size,
            apply_irregularity,
            irregularity_rate_param,
            irregularity_type_param,
            density_scores,
            data.shape[2]  # Number of sensors
        )
        val_processed.append((record_id, timestamp, irg_ts, irg_ts_mask, label))
    
    # Process test data
    for idx in test_indices:
        record_id, timestamp, irg_ts, irg_ts_mask, label = process_sample(
            data[idx], 
            timestamps[idx], 
            lengths[idx], 
            y[idx], 
            idx,
            window_size,
            apply_irregularity,
            irregularity_rate_param,
            irregularity_type_param,
            density_scores,
            data.shape[2]  # Number of sensors
        )
        test_processed.append((record_id, timestamp, irg_ts, irg_ts_mask, label))
    
    return train_processed, val_processed, test_processed

def normalize(data):
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data = torch.nan_to_num(data)
    return data

def process_sample(data_sample, timestamp_sample, length, label, idx, 
                  window_size, irregularity_active, current_irregularity_rate, current_irregularity_type, 
                  density_scores, n_sensors):
    # Sample a random window as in the original code
    len_sample = length.item()
    w_idx = random.randint(0, len_sample - window_size) if len_sample > window_size else 0
    X_sample = data_sample[w_idx:w_idx + window_size]
    t_sample = timestamp_sample[w_idx:w_idx + window_size]
    
    base_mask = torch.ones(window_size, n_sensors, dtype=torch.bool)

    # Create mask for irregularity
    if irregularity_active and n_sensors > 0:
        n_sensors_to_drop = round(current_irregularity_rate * n_sensors)
        if n_sensors_to_drop > 0:
            if current_irregularity_type == "fixed":
                can_use_density_scores = False
                if density_scores is not None and density_scores.shape[0] > 0 and density_scores.ndim >= 1:
                    scores_indices_to_consider = density_scores[:, 0] if density_scores.ndim > 1 else density_scores
                    num_from_scores = min(n_sensors_to_drop, scores_indices_to_consider.shape[0])

                    selected_indices_from_scores = scores_indices_to_consider[:num_from_scores]
                    
                    drop_indices = np.array(selected_indices_from_scores, dtype=int)
                    drop_indices = drop_indices[(drop_indices >= 0) & (drop_indices < n_sensors)]
                    
                    if drop_indices.size > 0:
                        base_mask[:, drop_indices] = False
                        can_use_density_scores = True
                
                if not can_use_density_scores:
                    print(f"Warning: Patient {idx}, 'fixed' irregularity. Sensor density scores not effectively used (rate: {current_irregularity_rate}, to_drop: {n_sensors_to_drop}). Falling back to random drop.")
                    if n_sensors_to_drop <= n_sensors :
                        drop_indices_random = np.random.choice(n_sensors, n_sensors_to_drop, replace=False)
                        base_mask[:, drop_indices_random] = False
                    else:
                         print(f"Warning: Patient {idx}, cannot drop {n_sensors_to_drop} sensors from {n_sensors}. Skipping drop for this sample.")

            elif current_irregularity_type == "random":
                if n_sensors_to_drop <= n_sensors:
                    drop_indices = np.random.choice(n_sensors, n_sensors_to_drop, replace=False)
                    base_mask[:, drop_indices] = False
                else:
                    print(f"Warning: Patient {idx}, cannot drop {n_sensors_to_drop} sensors from {n_sensors}. Skipping drop for this sample.")

            else:
                print(f"Warning: Patient {idx}, unknown irregularity_type '{current_irregularity_type}' provided for PAM dataset. No specific irregularity applied for this type. Rate: {current_irregularity_rate}, to_drop: {n_sensors_to_drop}.")
    else:
        # This 'else' corresponds to 'if irregularity_active and n_sensors > 0:'
        # If not active, or no sensors, or n_sensors_to_drop is 0, base_mask remains all True.
        pass
    
    # Apply padding mask based on length, exactly as in original code
    padding_mask = torch.arange(window_size) < (len_sample - w_idx)
    mask = base_mask & padding_mask.unsqueeze(1)
    
    # Apply mask to data
    masked_sample = X_sample * mask.float()
    
    # Create the tuple elements in the requested order
    record_id = f"patient_{idx}"
    timestamp = t_sample  # Regular timestamps
    irg_ts = masked_sample  # Masked data as irregular time series
    irg_ts_mask = mask.int()  # Mask indicating which points are valid
    label_tensor = label
    
    return record_id, timestamp, irg_ts, irg_ts_mask, label_tensor