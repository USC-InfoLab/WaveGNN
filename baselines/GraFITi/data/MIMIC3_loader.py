import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pickle
import os

# Import the collate function from the GraFITi project structure
from grafiti.grafiti import tsdm_collate_classification

class MIMIC3ClassificationDataset(Dataset):
    def __init__(self, processed_data_list):
        self.processed_data_list = processed_data_list

    def __len__(self):
        return len(self.processed_data_list)

    def __getitem__(self, idx):
        return self.processed_data_list[idx]

def _load_and_process_split(pickle_file_path, task_name_logging, split_name_logging, median_len):
    """
    Loads and processes a single split of MIMIC-III data.
    Ensures sequences are truncated/padded to median_len.
    Returns a list of dictionaries, each formatted for tsdm_collate_classification,
    and the number of features (num_types) if this is the training split.
    """
    if not os.path.exists(pickle_file_path):
        raise FileNotFoundError(f"Pickle file not found: {pickle_file_path} for task {task_name_logging}, split {split_name_logging}")

    with open(pickle_file_path, "rb") as f:
        data_from_pickle = pickle.load(f)

    processed_list = []
    num_types_from_split = None

    for item_dict in tqdm(data_from_pickle, desc=f"Processing {split_name_logging} for {task_name_logging} (len={median_len})"):
        # record_id = item_dict["name"] # Not directly used by the model input
        
        timestamp_np_original = np.array(item_dict["ts_tt"], dtype=np.float32)
        irg_ts_np_original = np.array(item_dict["irg_ts"], dtype=np.float32)
        irg_ts_mask_np_original = np.array(item_dict["irg_ts_mask"], dtype=np.bool_) # Ensure boolean

        label_torch = torch.tensor(item_dict["label"], dtype=torch.float32)

        # Determine num_types from the first valid sample 
        if num_types_from_split is None and irg_ts_np_original.ndim == 2 and irg_ts_np_original.shape[0] > 0:
            num_types_from_split = irg_ts_np_original.shape[-1]
        elif num_types_from_split is None and irg_ts_np_original.ndim == 1 and irg_ts_np_original.shape[0] > 0 and split_name_logging=="train":
            # This case might occur if a feature dimension was squeezed out for some reason. Add a check.
            # However, typical irg_ts is 2D. If it's 1D, it suggests an issue or a single-feature dataset.
            # For now, we assume multi-feature, so irg_ts_np_original.shape[-1] is the way.
            # If this path is hit, it might indicate data format issues.
            # num_types_from_split = 1 # Or handle as an error depending on expectation
            pass # Let it be determined by a proper 2D sample later or raise error in get_mimic3_data

        # Apply truncation/padding to median_len
        current_len = timestamp_np_original.shape[0]
        
        if current_len > median_len:
            # Truncate: take the first median_len points
            timestamp_np = timestamp_np_original[:median_len]
            irg_ts_np = irg_ts_np_original[:median_len, :]
            irg_ts_mask_np = irg_ts_mask_np_original[:median_len, :]
        elif current_len < median_len:
            # Pad
            padding_len = median_len - current_len
            timestamp_np = np.pad(timestamp_np_original, (0, padding_len), 'constant', constant_values=0.0) # Pad timestamps with 0
            if irg_ts_np_original.ndim == 2: # Should always be the case for multi-feature
                irg_ts_np = np.pad(irg_ts_np_original, ((0, padding_len), (0,0)), 'constant', constant_values=0.0) # Pad values with 0
                irg_ts_mask_np = np.pad(irg_ts_mask_np_original, ((0, padding_len), (0,0)), 'constant', constant_values=False) # Pad mask with False
            else: # Should not happen with correct data
                irg_ts_np = np.pad(irg_ts_np_original, (0, padding_len), 'constant', constant_values=0.0) 
                irg_ts_mask_np = np.pad(irg_ts_mask_np_original, (0, padding_len), 'constant', constant_values=False)
        else: # current_len == median_len
            timestamp_np = timestamp_np_original
            irg_ts_np = irg_ts_np_original
            irg_ts_mask_np = irg_ts_mask_np_original

        # Apply mask to create NaNs after padding/truncation
        # Where mask is False (missing or padded), value becomes NaN. Where True, value is kept.
        x_obs_with_nans_np = np.where(irg_ts_mask_np, irg_ts_np, np.nan)

        t_obs_tensor = torch.from_numpy(timestamp_np).float()
        x_obs_tensor = torch.from_numpy(x_obs_with_nans_np).float() # Contains NaNs
        
        # Adjust label for IHM task if it's multi-dimensional (e.g. [[0.]] or [[1.]])
        # This assumes IHM is binary, and BCEWithLogitsLoss expects a single logit.
        # If label is already scalar tensor or (1,) shape, it's fine.
        if task_name_logging == 'mimic3_ihm':
            if label_torch.ndim > 0 and label_torch.numel() == 1:
                 label_torch = label_torch.squeeze() 
            # If it's already a scalar, no change. If it's shape (1,), squeeze makes it scalar.
            # If it was, for instance, one-hot [0., 1.], this would need different handling based on expected target.
            # Assuming for IHM, label is a single float (0.0 or 1.0).

        processed_list.append({
            't_obs': t_obs_tensor,
            'x_obs': x_obs_tensor, # This has NaNs
            'class_label': label_torch
        })
    
    if not processed_list and num_types_from_split is None:
        # Handle case where a split is empty and num_types couldn't be inferred
        # This might happen if, e.g., a test split is empty.
        # The caller (get_mimic3_data) primarily relies on training data for num_types.
        pass
        
    return processed_list, num_types_from_split


def get_mimic3_data(args, device):
    """
    Main function to load and prepare MIMIC-III IHM or PHE data.
    args.data_path should point to the specific dataset folder (ihm or pheno).
    args.task_name should be 'mimic3_ihm' or 'mimic3_phe'.
    args.median_len provides the fixed sequence length.
    """
    
    # file_path is already the specific ihm/pheno path from train_grafiti.py
    base_path = args.data_path 
    task_name = args.task_name # 'mimic3_ihm' or 'mimic3_phe'
    median_len = args.median_len # Get median_len from args

    train_pickle_path = os.path.join(base_path, "trainp2x_data.pkl")
    val_pickle_path = os.path.join(base_path, "valp2x_data.pkl")
    test_pickle_path = os.path.join(base_path, "testp2x_data.pkl")

    train_processed, num_types = _load_and_process_split(train_pickle_path, task_name, "train", median_len)
    if num_types is None:
        # Try to infer from validation set if training set was empty or had no features
        val_processed_temp, num_types_val = _load_and_process_split(val_pickle_path, task_name, "val_temp_for_num_types", median_len)
        if num_types_val is not None:
            num_types = num_types_val
            print(f"Warning: num_types inferred from validation set: {num_types}")
            val_processed, _ = _load_and_process_split(val_pickle_path, task_name, "validation", median_len)
        else:
            # Try to infer from test set if training and val sets were empty or had no features
            test_processed_temp, num_types_test = _load_and_process_split(test_pickle_path, task_name, "test_temp_for_num_types", median_len)
            if num_types_test is not None:
                num_types = num_types_test
                print(f"Warning: num_types inferred from test set: {num_types}")
                val_processed, _ = _load_and_process_split(val_pickle_path, task_name, "validation", median_len)
                test_processed, _ = _load_and_process_split(test_pickle_path, task_name, "test", median_len)
            else:
                 raise ValueError(f"Could not determine num_types (number of features) for task {task_name} from any split.")
    else: # num_types was successfully inferred from training data
        val_processed, _ = _load_and_process_split(val_pickle_path, task_name, "validation", median_len)
        test_processed, _ = _load_and_process_split(test_pickle_path, task_name, "test", median_len)


    if not train_processed:
        # This case should ideally be caught by num_types check, but as a safeguard:
        print(f"Warning: Training data is empty for task {task_name} at path {base_path}.")
    
    # Create Datasets
    train_dataset = MIMIC3ClassificationDataset(train_processed)
    val_dataset = MIMIC3ClassificationDataset(val_processed)
    test_dataset = MIMIC3ClassificationDataset(test_processed)

    # Create DataLoaders
    # Batch size from args, num_workers can be set here or passed via args if needed
    # ARGS from train_grafiti.py contains batch_size
    
    # Access batch_size from the args object passed to get_mimic3_data
    current_batch_size = args.batch_size 

    train_loader = DataLoader(
        train_dataset,
        batch_size=current_batch_size,
        shuffle=True,
        num_workers=min(os.cpu_count(), 4),
        collate_fn=tsdm_collate_classification
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=current_batch_size, # Or a different val_batch_size
        shuffle=False,
        num_workers=min(os.cpu_count(), 4),
        collate_fn=tsdm_collate_classification
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=current_batch_size, # Or a different test_batch_size
        shuffle=False,
        num_workers=min(os.cpu_count(), 4),
        collate_fn=tsdm_collate_classification
    )
    
    return train_loader, val_loader, test_loader, num_types