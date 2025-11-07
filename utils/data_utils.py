import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
import pickle
from loaders.P12_loader import P12Loader
from loaders.P19_loader import P19Loader
from loaders.PAM_loader import PAMLoader
from loaders.MIMIC3_loader import MIMIC3Loader


def extract_attributes(data_dict, attr):
    X = [d[attr] for d in data_dict]
    X = np.stack(X, axis=0)
    X = X.astype(np.float32)
    return torch.from_numpy(X)


def train_test_split(data_file, labels_file, splits_dir, split_num=1):

    dataset_name = data_file.split("/")[-3][:3]
    print(f"Dataset: {dataset_name}")

    data, timestamps, static_features, lengths, y = read_dataset(
        data_file, labels_file, dataset_name
    )

    if dataset_name == "P12":
        sfile = "phy12_split" + str(split_num)
    elif dataset_name == "PAM":
        sfile = "PAM_split_" + str(split_num)
    elif dataset_name == "P19":
        sfile = "phy19_split" + str(split_num) + "_new"
    else:
        raise ValueError("Invalid dataset name")
    train_indices, val_indices, test_indices = np.load(
        f"{splits_dir}{sfile}.npy", allow_pickle=True
    )

    train_dict = {
        "data": data[train_indices],
        "timestamps": timestamps[train_indices],
        "static_features": static_features[train_indices],
        "lengths": lengths[train_indices],
        "y": y[train_indices],
    }

    val_dict = {
        "data": data[val_indices],
        "timestamps": timestamps[val_indices],
        "static_features": static_features[val_indices],
        "lengths": lengths[val_indices],
        "y": y[val_indices],
    }

    test_dict = {
        "data": data[test_indices],
        "timestamps": timestamps[test_indices],
        "static_features": static_features[test_indices],
        "lengths": lengths[test_indices],
        "y": y[test_indices],
    }
    return train_dict, val_dict, test_dict


def normalize(data):
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data = torch.nan_to_num(data)
    return data


def read_dataset(data_file, labels_file, dataset_name="PAM"):
    # load data and labels
    data_dict = np.load(data_file, allow_pickle=True)
    labels = np.load(labels_file, allow_pickle=True)
    # extract attributes
    if dataset_name == "P12":
        data = extract_attributes(data_dict, "arr")
        data = normalize(data)
        timestamps = extract_attributes(data_dict, "time")
        timestamps = timestamps / 60.0  # convert to mins
        static_features = extract_attributes(data_dict, "static")
        static_features = normalize(static_features)
        lengths = extract_attributes(data_dict, "length")
    elif dataset_name == "P19":
        data = extract_attributes(data_dict, "arr")
        data = normalize(data)
        timestamps = extract_attributes(data_dict, "time")
        timestamps = timestamps / 60.0  # convert to mins
        static_features = extract_attributes(data_dict, "extended_static")
        static_features = normalize(static_features)
        lengths = extract_attributes(data_dict, "length")
    else:
        data = data_dict.astype(np.float32)
        data = torch.from_numpy(data)
        data = normalize(data)
        # create timestamps
        timestamps = torch.arange(data.shape[1])
        timestamps = timestamps.unsqueeze(0).repeat(data.shape[0], 1) / 60.0
        # create static features
        static_features = torch.randn(data.shape[0], 5)
        # create lengths all lengths are 600
        lengths = torch.ones(data.shape[0]) * 600

    y = labels[:, -1]  # dataset has multiple labels, mortality is the last one
    # convert to tensor of type float32
    y = torch.from_numpy(y)
    y = y.type(torch.float32)
    return data, timestamps, static_features, lengths, y


# code from https://github.com/XZhang97666/MultimodalMIMIC
def load_clinical_notes_encoder(device):
    tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
    Bert_model = AutoModel.from_pretrained("yikuan8/Clinical-Longformer")
    Bert_model = Bert_model.to(device)

    return Bert_model, tokenizer


def load_mimic3(file_path):
    train_data = pickle.load(open(file_path + "/trainp2x_data.pkl", "rb"))
    val_data = pickle.load(open(file_path + "/valp2x_data.pkl", "rb"))
    test_data = pickle.load(open(file_path + "/testp2x_data.pkl", "rb"))

    train_dict = {
        "reg_ts": [item_dict["reg_ts"] for item_dict in train_data],
        "names": [item_dict["name"] for item_dict in train_data],
        "labels": extract_attributes(train_data, "label"),
        "ts_tt": [item_dict["ts_tt"] for item_dict in train_data],
        "irg_ts": [item_dict["irg_ts"] for item_dict in train_data],
        "irg_ts_mask": [item_dict["irg_ts_mask"] for item_dict in train_data],
        "static_features": torch.randn(len(train_data), 5),
    }

    val_dict = {
        "reg_ts": [item_dict["reg_ts"] for item_dict in val_data],
        "names": [item_dict["name"] for item_dict in val_data],
        "labels": extract_attributes(val_data, "label"),
        "ts_tt": [item_dict["ts_tt"] for item_dict in val_data],
        "irg_ts": [item_dict["irg_ts"] for item_dict in val_data],
        "irg_ts_mask": [item_dict["irg_ts_mask"] for item_dict in val_data],
        "static_features": torch.randn(len(train_data), 5),
    }

    test_dict = {
        "reg_ts": [item_dict["reg_ts"] for item_dict in test_data],
        "names": [item_dict["name"] for item_dict in test_data],
        "labels": extract_attributes(test_data, "label"),
        "ts_tt": [item_dict["ts_tt"] for item_dict in test_data],
        "irg_ts": [item_dict["irg_ts"] for item_dict in test_data],
        "irg_ts_mask": [item_dict["irg_ts_mask"] for item_dict in test_data],
        "static_features": torch.randn(len(train_data), 5),
    }

    return train_dict, val_dict, test_dict


def get_datasets(args, n_split, device="cuda"):
    ### Load datasets
    if args.dataset.startswith("MIMIC3"):
        TRAIN, VAL, TEST = load_mimic3(args.main_path + args.data_path)

        train_dataset = MIMIC3Loader(TRAIN, args=args)
        val_dataset = MIMIC3Loader(VAL, args=args)
        test_dataset = MIMIC3Loader(TEST, args=args)

    else:
        TRAIN, VAL, TEST = train_test_split(
            args.main_path + args.data_path,
            args.main_path + args.label_path,
            args.main_path + args.splits_path,
            split_num=n_split,
        )

        print("Train data shape:", TRAIN["data"].shape)
        print("Val data shape:", VAL["data"].shape)
        print("Test data shape:", TEST["data"].shape)

        if args.dataset == "P12":
            train_dataset = P12Loader(
                TRAIN,
                window_size=args.window_size,
                irregularity=args.irregularity,
                irregularity_rate=args.irregularity_rate,
            )
            val_dataset = P12Loader(
                VAL,
                window_size=args.window_size,
                irregularity=args.irregularity,
                irregularity_rate=args.irregularity_rate,
            )
            test_dataset = P12Loader(
                TEST,
                window_size=args.window_size,
                irregularity=args.irregularity,
                irregularity_rate=args.irregularity_rate,
            )
        elif args.dataset == "PAM":
            # Following the experimental setup on Raindrop irregularity is only
            # introduced on validation and test sets
            train_dataset = PAMLoader(
                TRAIN,
                window_size=args.window_size,
                irregularity=False,
                irregularity_rate=args.irregularity_rate,
                irregularity_type=args.irregularity_type,
                args=args,
            )
            val_dataset = PAMLoader(
                VAL,
                window_size=args.window_size,
                irregularity=args.irregularity,
                irregularity_rate=args.irregularity_rate,
                irregularity_type=args.irregularity_type,
                args=args,
            )

            test_dataset = PAMLoader(
                TEST,
                window_size=args.window_size,
                irregularity=args.irregularity,
                irregularity_rate=args.irregularity_rate,
                irregularity_type=args.irregularity_type,
                args=args,
            )
        else:
            train_dataset = P19Loader(
                TRAIN,
                window_size=args.window_size,
                irregularity=args.irregularity,
                irregularity_rate=args.irregularity_rate,
            )
            val_dataset = P19Loader(
                VAL,
                window_size=args.window_size,
                irregularity=args.irregularity,
                irregularity_rate=args.irregularity_rate,
            )
            test_dataset = P19Loader(
                TEST,
                window_size=args.window_size,
                irregularity=args.irregularity,
                irregularity_rate=args.irregularity_rate,
            )

    return (train_dataset, val_dataset, test_dataset)
