import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pickle
import os


def load_and_preprocess_mimic3(file_path):
    """
    Load and preprocess MIMIC-III data from pickle files.
    
    Args:
        file_path (str): Path to directory containing pickle files.
        
    Returns:
        tuple: Tuple containing lists of processed data for train, val, and test sets.
               Each list contains tuples of (record_id, timestamp, irg_ts, irg_ts_mask).
    """
    # Load data from pickle files
    train_data = pickle.load(open(file_path + "/trainp2x_data.pkl", "rb"))
    val_data = pickle.load(open(file_path + "/valp2x_data.pkl", "rb"))
    test_data = pickle.load(open(file_path + "/testp2x_data.pkl", "rb"))
    
    # Process each dataset to format required by model
    train_processed = []
    for idx, item_dict in enumerate(tqdm(train_data, desc="Processing train data")):
        record_id = item_dict["name"]
        timestamp = torch.tensor(item_dict["ts_tt"], dtype=torch.float)
        irg_ts = torch.tensor(item_dict["irg_ts"], dtype=torch.float)
        irg_ts_mask = torch.tensor(item_dict["irg_ts_mask"], dtype=torch.int)
        label = torch.tensor(item_dict["label"], dtype=torch.float)
        train_processed.append((record_id, timestamp, irg_ts, irg_ts_mask, label))
    
    val_processed = []
    for idx, item_dict in enumerate(tqdm(val_data, desc="Processing validation data")):
        record_id = item_dict["name"]
        timestamp = torch.tensor(item_dict["ts_tt"], dtype=torch.float)
        irg_ts = torch.tensor(item_dict["irg_ts"], dtype=torch.float)
        irg_ts_mask = torch.tensor(item_dict["irg_ts_mask"], dtype=torch.int)
        label = torch.tensor(item_dict["label"], dtype=torch.float)
        val_processed.append((record_id, timestamp, irg_ts, irg_ts_mask, label))
    
    test_processed = []
    for idx, item_dict in enumerate(tqdm(test_data, desc="Processing test data")):
        record_id = item_dict["name"]
        timestamp = torch.tensor(item_dict["ts_tt"], dtype=torch.float)
        irg_ts = torch.tensor(item_dict["irg_ts"], dtype=torch.float)
        irg_ts_mask = torch.tensor(item_dict["irg_ts_mask"], dtype=torch.int)
        label = torch.tensor(item_dict["label"], dtype=torch.float)
        test_processed.append((record_id, timestamp, irg_ts, irg_ts_mask, label))
    
    return train_processed, val_processed, test_processed


# def extract_attributes(data_dict, attr):
#     X = [d[attr] for d in data_dict]
#     X = np.stack(X, axis=0)
#     X = X.astype(np.float32)
#     return torch.from_numpy(X)


# def load_mimic3(file_path):
#     train_data = pickle.load(open(file_path + "/trainp2x_data.pkl", "rb"))
#     val_data = pickle.load(open(file_path + "/valp2x_data.pkl", "rb"))
#     test_data = pickle.load(open(file_path + "/testp2x_data.pkl", "rb"))

#     train_dict = {
#         "reg_ts": [item_dict["reg_ts"] for item_dict in train_data],
#         "names": [item_dict["name"] for item_dict in train_data],
#         "labels": extract_attributes(train_data, "label"),
#         "ts_tt": [item_dict["ts_tt"] for item_dict in train_data],
#         "irg_ts": [item_dict["irg_ts"] for item_dict in train_data],
#         "irg_ts_mask": [item_dict["irg_ts_mask"] for item_dict in train_data],
#         "static_features": torch.randn(len(train_data), 5),
#     }

#     val_dict = {
#         "reg_ts": [item_dict["reg_ts"] for item_dict in val_data],
#         "names": [item_dict["name"] for item_dict in val_data],
#         "labels": extract_attributes(val_data, "label"),
#         "ts_tt": [item_dict["ts_tt"] for item_dict in val_data],
#         "irg_ts": [item_dict["irg_ts"] for item_dict in val_data],
#         "irg_ts_mask": [item_dict["irg_ts_mask"] for item_dict in val_data],
#         "static_features": torch.randn(len(train_data), 5),
#     }

#     test_dict = {
#         "reg_ts": [item_dict["reg_ts"] for item_dict in test_data],
#         "names": [item_dict["name"] for item_dict in test_data],
#         "labels": extract_attributes(test_data, "label"),
#         "ts_tt": [item_dict["ts_tt"] for item_dict in test_data],
#         "irg_ts": [item_dict["irg_ts"] for item_dict in test_data],
#         "irg_ts_mask": [item_dict["irg_ts_mask"] for item_dict in test_data],
#         "static_features": torch.randn(len(train_data), 5),
#     }

#     return train_dict, val_dict, test_dict


# class MIMIC3Loader(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         data_dict,
#         args,
#     ):
#         """
#         Args:
#             data_dict (dict): dictionary having 'irg_ts', 'irg_ts_mask', 'ts_tt', 'name', ,'text_data', 'text_time_to_end', and 'label' for all patients.
#             period_length (int): maximum period length for irregular time series (e.g. 48 hours for mortality prediction and 24 hours for phenotype classification).
#             window_size (int, optional): _description_. Defaults to 32.
#         """

#         self.X = data_dict["irg_ts"]  # list of irregular time series
#         self.X_mask = data_dict["irg_ts_mask"]  # list of irregular time series masks
#         # self.w_len = args.window_size
#         self.period_length = 48  # 48 hours-IHM, 24 hours-pheno
#         self.timestamps = data_dict["ts_tt"]
#         self.static_features = data_dict["static_features"]
#         self.y = data_dict["labels"]
#         self.lengths = [len(ts) for ts in self.X]
#         self.n_patients = len(self.X)
#         self.n_sensors = self.X[0].shape[-1]
#         self.features_len = (
#             self.static_features.shape[1] if self.static_features is not None else 0
#         )

#     def get_dims(self):
#         """
#         Return a dictionary of data dimensions.
#         """
#         return {
#             "n_patients": self.n_patients,
#             "n_sensors": self.n_sensors,
#             "features_len": self.features_len,
#         }

#     def __len__(self):
#         """
#         Return:
#             (Int): The number of patients in the dataset.
#         """
#         return self.n_patients

#     def __getitem__(self, idx):
#         irg_ts = torch.tensor(self.X[idx], dtype=torch.float)
#         irg_ts_mask = torch.tensor(self.X_mask[idx], dtype=torch.int)
#         timestamp = torch.tensor(self.timestamps[idx], dtype=torch.float)
#         y_sample = self.y[idx] if self.y is not None else None
#         len_sample = torch.tensor(self.lengths[idx])
#         static_features = (
#             self.static_features[idx] if self.static_features is not None else None
#         )

#         irg_ts_relative_time_stamps = get_relative_timestamps(
#             timestamp, irg_ts_mask.bool()
#         )

#         data = {
#             "irg_ts": irg_ts,
#             "irg_ts_mask": irg_ts_mask,
#             "irg_ts_timestamps": timestamp,
#             "static_features": static_features,
#             "len_sample": len_sample,
#             "y_sample": y_sample,
#             "irg_ts_relative_time_stamps": irg_ts_relative_time_stamps,
#         }
        
#         return data