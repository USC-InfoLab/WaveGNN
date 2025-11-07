import numpy as np
import torch

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

def load_and_preprocess_p19():
    """
    Loads and preprocesses the P19 dataset, returning train/val/test splits as lists of tuples:
    (record_id, timestamp, irregular_timeseries, irregular_timeseries_mask, label)
    """
    main_path = "/storage/datasets_public/irreg_ts/datasets/"
    data_path = "P19data/processed_data/PT_dict_list_6.npy"
    label_path = "P19data/processed_data/arr_outcomes_6.npy"
    splits_path = "P19data/splits/"
    split_num = 1  # Change if you want a different split

    data_dict = np.load(main_path + data_path, allow_pickle=True)
    labels = np.load(main_path + label_path, allow_pickle=True)

    data = extract_attributes(data_dict, "arr")
    timestamps = extract_attributes(data_dict, "time") / 60.0  # convert to mins
    lengths = extract_attributes(data_dict, "length").long()
    y = labels[:, -1]
    y = torch.from_numpy(y).type(torch.float32)

    data, _, _ = normalize(data)

    sfile = f"phy19_split{split_num}_new"
    train_indices, val_indices, test_indices = np.load(
        f"{main_path}{splits_path}{sfile}.npy", allow_pickle=True
    )

    def make_list(indices):
        result = []
        for i in indices:
            record_id = int(i)
            ts = timestamps[record_id].squeeze(-1)
            irg_ts = data[record_id]
            seq_len = lengths[record_id].item()
            mask = torch.zeros_like(irg_ts, dtype=torch.int)
            if irg_ts.ndim == 2:
                mask[:seq_len, :] = 1
            else:
                mask[:seq_len] = 1
            label = y[record_id]
            result.append((record_id, ts, irg_ts, mask, label))
        return result

    train_data = make_list(train_indices)
    val_data = make_list(val_indices)
    test_data = make_list(test_indices)

    return train_data, val_data, test_data

if __name__ == "__main__":
    from collections import Counter
    train_data, val_data, test_data = load_and_preprocess_p19()

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