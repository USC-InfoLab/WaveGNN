import torch
import numpy as np
from utils.utils import get_relative_timestamps
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pickle
import os


class MIMIC3Loader(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dict,
        args,
    ):
        """
        Args:
            data_dict (dict): dictionary having 'irg_ts', 'irg_ts_mask', 'ts_tt', 'name', ,'text_data', 'text_time_to_end', and 'label' for all patients.
            period_length (int): maximum period length for irregular time series (e.g. 48 hours for mortality prediction).
            window_size (int, optional): _description_. Defaults to 32.
        """

        self.X = data_dict["irg_ts"]  # list of irregular time series
        self.X_mask = data_dict["irg_ts_mask"]  # list of irregular time series masks
        self.w_len = args.window_size
        self.period_length = args.period_length  # 48 hours
        self.timestamps = data_dict["ts_tt"]
        self.static_features = data_dict["static_features"]
        self.y = data_dict["labels"]
        self.lengths = [len(ts) for ts in self.X]
        self.n_patients = len(self.X)
        self.n_sensors = self.X[0].shape[-1]
        self.features_len = (
            self.static_features.shape[1] if self.static_features is not None else 0
        )

    def get_dims(self):
        """
        Return a dictionary of data dimensions.
        """
        return {
            "n_patients": self.n_patients,
            "n_sensors": self.n_sensors,
            "features_len": self.features_len,
        }

    def __len__(self):
        """
        Return:
            (Int): The number of patients in the dataset.
        """
        return self.n_patients

    def __getitem__(self, idx):
        irg_ts = torch.tensor(self.X[idx], dtype=torch.float)
        irg_ts_mask = torch.tensor(self.X_mask[idx], dtype=torch.int)
        timestamp = torch.tensor(self.timestamps[idx], dtype=torch.float)
        y_sample = self.y[idx] if self.y is not None else None
        len_sample = torch.tensor(self.lengths[idx])
        static_features = (
            self.static_features[idx] if self.static_features is not None else None
        )

        irg_ts_relative_time_stamps = get_relative_timestamps(
            timestamp, irg_ts_mask.bool()
        )

        data = {
            "irg_ts": irg_ts,
            "irg_ts_mask": irg_ts_mask,
            "irg_ts_timestamps": timestamp,
            "static_features": static_features,
            "len_sample": len_sample,
            "y_sample": y_sample,
            "irg_ts_relative_time_stamps": irg_ts_relative_time_stamps,
        }

        return data


# code adapted from https://github.com/XZhang97666/MultimodalMIMIC
def mimic3_collate_fn(batch, max_window_size=500):
    irg_ts = pad_sequence(
        [item["irg_ts"] for item in batch], batch_first=True, padding_value=0
    )
    irg_ts_mask = pad_sequence(
        [item["irg_ts_mask"] for item in batch], batch_first=True, padding_value=0
    )
    irg_ts_timestamps = pad_sequence(
        [item["irg_ts_timestamps"] for item in batch], batch_first=True, padding_value=0
    )
    static_features = torch.stack([item["static_features"] for item in batch])
    len_sample = torch.stack([item["len_sample"] for item in batch])
    y_sample = torch.stack([item["y_sample"] for item in batch])

    irg_ts_relative_time_stamps = pad_sequence(
        [item["irg_ts_relative_time_stamps"] for item in batch],
        batch_first=True,
        padding_value=0,
    )

    window_length = irg_ts.shape[1]
    if window_length > max_window_size:
        return (
            irg_ts[:, window_length - max_window_size :, :],
            irg_ts_mask[:, window_length - max_window_size :, :],
            irg_ts_timestamps[:, window_length - max_window_size :],
            static_features,
            len_sample,
            y_sample,
            irg_ts_relative_time_stamps[:, window_length - max_window_size :, :],
        )

    return (
        irg_ts,
        irg_ts_mask,
        irg_ts_timestamps,
        static_features,
        len_sample,
        y_sample,
        irg_ts_relative_time_stamps,
    )
