import torch
import numpy as np
from utils.utils import get_relative_timestamps


class PAMLoader(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dict,
        window_size,
        irregularity,
        irregularity_rate,
        irregularity_type,
        args,
    ):
        """
        Args:
            data_file (str): The path to the data file.
            labels_file (str): The path to the labels file.
            window_size (int): The size of the window of data to return for each sample.
            irregularity (bool): Whether to simulate irregularity in the data.
            irregularity_rate (float): The rate of irregularity in the data.
        """
        self.X = data_dict["data"]
        self.timestamps = data_dict["timestamps"]
        self.y = data_dict["y"]
        self.static_features = data_dict["static_features"]
        self.lengths = data_dict["lengths"]
        self.n_patients = len(self.X)
        self.n_sensors = self.X[0].shape[-1]
        self.w_len = window_size
        self.features_len = (
            self.static_features.shape[1] if self.static_features is not None else 0
        )
        self.irregularity = irregularity
        self.irregularity_type = irregularity_type
        self.irregularity_rate = irregularity_rate

        self.density_scores = np.load(
            args.main_path + args.sensor_density_path, allow_pickle=True
        )

    def __len__(self):
        """
        Return:
            (Int): The number of patients in the dataset.
        """
        return self.n_patients

    def __getitem__(self, idx):
        """
        Return:
            (Tensor, Tensor, Tensor): A tuple of shape
                ([w_len, 2*n_sensors], [w_len, n_sensors], [w_len, 1])
                of the windowed data and the corresponding label (if available).
        """
        Xi = self.X[idx]
        ti = self.timestamps[idx]
        len_sample = self.lengths[idx]

        w_idx = (
            np.random.randint(0, len_sample - self.w_len)
            if len_sample > self.w_len
            else 0
        )
        X_sample = Xi[w_idx : w_idx + self.w_len]
        t_sample = ti[w_idx : w_idx + self.w_len]

        if self.irregularity:
            n_sensors_to_drop = round(self.irregularity_rate * self.n_sensors)
            if self.irregularity_type == "fixed":  # TODO: test
                density_score_indices = self.density_scores[:n_sensors_to_drop, 0]
                drop_indices = np.array(density_score_indices, dtype=int)
                base_mask = torch.ones(self.w_len, self.n_sensors, dtype=torch.bool)
                base_mask[:, drop_indices] = False
            elif self.irregularity_type == "random":
                # randomly choose which sensors to drop
                drop_indices = np.random.choice(
                    self.n_sensors,
                    n_sensors_to_drop,
                    replace=False,
                )
                base_mask = torch.ones(self.w_len, self.n_sensors, dtype=torch.bool)
                base_mask[:, drop_indices] = False
            else:
                # drop the first irregularity_rate of the sensors
                drop_indices = int(self.irregularity_rate * self.n_sensors)
                base_mask = torch.ones(self.w_len, self.n_sensors, dtype=torch.bool)
                base_mask[:, :drop_indices] = False
        else:
            base_mask = torch.ones(self.w_len, self.n_sensors, dtype=torch.bool)

        # apply padding mask based on length
        padding_mask = torch.arange(self.w_len) < (len_sample - w_idx)
        mask = base_mask & padding_mask.unsqueeze(1)

        masked_sample = X_sample * mask.float()

        relative_time_stamps = get_relative_timestamps(t_sample, mask)

        y_sample = self.y[idx] if self.y is not None else None
        f_sample = (
            self.static_features[idx] if self.static_features is not None else None
        )

        return (
            masked_sample,
            mask,
            t_sample,
            f_sample,
            len_sample,
            y_sample.long(),
            relative_time_stamps,
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
