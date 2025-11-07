import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pickle
import os
from Utils import cal_tau # Ensure Utils.py is accessible

def get_mimic3_data(args, device):
    """
    Main function to load and prepare MIMIC-III IHM or PHE data.
    args.data_path should point to the specific dataset folder (ihm or pheno).
    args.task should be 'mimic3_ihm' or 'mimic3_phe'.
    """
    
    train_raw = load_mimic3_split(args.data_path, "train", args.task)
    val_raw = load_mimic3_split(args.data_path, "val", args.task)
    test_raw = load_mimic3_split(args.data_path, "test", args.task)

    if not train_raw:
        raise ValueError(f"Training data is empty for task {args.task}. Check data_path: {args.data_path}")
    
    # Determine num_types from the data (item[2] is irg_ts with shape (L, K))
    # Find the first non-empty sequence to determine num_types
    args.num_types = None
    for item in train_raw:
        if item[2].shape[0] > 0: # Check if sequence length is greater than 0
            args.num_types = item[2].shape[-1]
            break
    if args.num_types is None:
        # Fallback if all sequences in train_raw are empty or train_raw itself is empty
        # Attempt to get from val or test, or raise error
        for item_list in [val_raw, test_raw]:
            for item in item_list:
                if item[2].shape[0] > 0:
                    args.num_types = item[2].shape[-1]
                    break
            if args.num_types is not None: break
        if args.num_types is None:
            raise ValueError(f"Could not determine num_types for task {args.task}. All data splits might be empty or contain only empty sequences.")

    # Normalization: Placeholder. Ideally, compute stats on train_raw and apply to all.
    # For now, process_mimic3_list will handle data as is, assuming pre-normalization or simple per-sample normalization.

    train_processed = process_mimic3_list(train_raw, args, args.task, "train")
    val_processed = process_mimic3_list(val_raw, args, args.task, "validation")
    test_processed = process_mimic3_list(test_raw, args, args.task, "test")

    train_loader = create_mimic3_dataloader(train_processed, args.batch_size, args.task, shuffle=True)
    val_loader = create_mimic3_dataloader(val_processed, args.batch_size, args.task, shuffle=False)
    test_loader = create_mimic3_dataloader(test_processed, args.batch_size, args.task, shuffle=False)
    
    return train_loader, val_loader, test_loader, args.num_types

def load_mimic3_split(base_path, split_name_key, task_name_logging):
    """ Loads a single split (train, val, test) of MIMIC-III data. """
    file_name_map = {
        "train": "trainp2x_data.pkl",
        "val": "valp2x_data.pkl",
        "test": "testp2x_data.pkl"
    }
    pickle_file = os.path.join(base_path, file_name_map[split_name_key])
    
    if not os.path.exists(pickle_file):
        raise FileNotFoundError(f"Pickle file not found: {pickle_file} for task {task_name_logging}")
        
    with open(pickle_file, "rb") as f:
        data_from_pickle = pickle.load(f)
    
    processed_list = []
    for item_dict in tqdm(data_from_pickle, desc=f"Loading raw {split_name_key} for {task_name_logging}"):
        record_id = item_dict["name"]
        timestamp = np.array(item_dict["ts_tt"], dtype=np.float32)
        irg_ts = np.array(item_dict["irg_ts"], dtype=np.float32)
        irg_ts_mask = np.array(item_dict["irg_ts_mask"], dtype=np.int32)
        
        label = torch.tensor(item_dict["label"], dtype=torch.float32) # Keep as float initially
            
        processed_list.append((record_id, timestamp, irg_ts, irg_ts_mask, label))
    return processed_list

def process_mimic3_list(raw_data_list, args, task_name, split_desc):
    processed_data_for_model = []
    window_size = args.median_len

    for record_id, timestamp_np, irg_ts_np, irg_ts_mask_np, label_tensor in tqdm(raw_data_list, desc=f"Processing {split_desc} for {task_name}"):
        
        observed_data_np = irg_ts_np 
        original_len = observed_data_np.shape[0]
        n_sensors = args.num_types # Should be set correctly in get_mimic3_data

        len_to_take = min(original_len, window_size)

        final_observed_data = np.zeros((window_size, n_sensors), dtype=np.float32)
        final_observed_mask = np.zeros((window_size, n_sensors), dtype=np.int32)
        final_observed_tp = np.zeros((window_size, 1), dtype=np.float32)
        final_tau = np.zeros((window_size, n_sensors), dtype=np.float32)
        final_seq_end = np.zeros((window_size, 1), dtype=np.float32)

        if len_to_take > 0:
            actual_data = observed_data_np[:len_to_take, :]
            actual_mask = irg_ts_mask_np[:len_to_take, :]
            actual_tp = timestamp_np[:len_to_take]

            final_observed_data[:len_to_take, :] = actual_data
            final_observed_mask[:len_to_take, :] = actual_mask # Use the mask from data
            final_observed_tp[:len_to_take, 0] = actual_tp
            
            tau_actual_np = cal_tau(
                actual_tp.reshape(1, len_to_take),
                actual_mask.reshape(1, len_to_take, n_sensors) # Ensure correct K for mask
            ).squeeze(0)
            final_tau[:len_to_take, :] = tau_actual_np
            
            final_seq_end[len_to_take - 1, 0] = 1.0
        
        tp_and_seq_end = np.concatenate([final_observed_tp, final_seq_end], axis=-1)

        combined_sample = np.concatenate(
            (final_observed_data, final_observed_mask, tp_and_seq_end, final_tau),
            axis=-1
        )
        processed_data_for_model.append((combined_sample, label_tensor))
        
    return processed_data_for_model

def create_mimic3_dataloader(processed_data_list, batch_size, task_name, shuffle=False):
    if not processed_data_list:
        # Create empty tensors for an empty DataLoader
        # Determine feature size (3K+2) from args.num_types for combined_sample
        # This part is tricky if num_types isn't available (e.g. args is not passed here)
        # Assuming args.num_types is globally accessible or can be inferred if this function is called after its set
        # For safety, let's try to make it robust or require num_features if list is empty
        print(f"Warning: Processed data list is empty for task {task_name}. Returning empty DataLoader.")
        # Placeholder shapes for empty dataloader
        num_combined_features = 10 # A dummy value, ideally get from args.num_types*3+2
        dummy_x = torch.empty(0, args.median_len if 'args' in locals() else 50, num_combined_features) 
        dummy_y_shape = (0,)
        if task_name == 'mimic3_phe':
             dummy_y_shape = (0, 25) # For multi-label, assuming 25 labels
        dummy_y = torch.empty(dummy_y_shape)       
        dataset = TensorDataset(dummy_x, dummy_y)
        return DataLoader(dataset, batch_size=batch_size)

    data_x_tensor = torch.stack([torch.from_numpy(item[0]).float() for item in processed_data_list])
    labels_list = [item[1] for item in processed_data_list]

    if task_name == 'mimic3_ihm':
        data_y_tensor = torch.stack(labels_list).long() 
        if data_y_tensor.ndim > 1 and data_y_tensor.shape[-1] == 1:
            data_y_tensor = data_y_tensor.squeeze(-1)
    elif task_name == 'mimic3_phe':
        data_y_tensor = torch.stack(labels_list).float()
    else:
        raise ValueError(f"Unknown MIMIC-III task for dataloader: {task_name}")

    dataset = TensorDataset(data_x_tensor, data_y_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=min(os.cpu_count(), 8),
        drop_last=False 
    )
    return dataloader