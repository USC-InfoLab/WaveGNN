import os, json
import random
import numpy as np
import torch
from datetime import datetime
from model.model import WaveGNN
from utils.trainer import Trainer
from utils.data_utils import get_datasets
from loaders.MIMIC3_loader import mimic3_collate_fn
from utils.utils import (
    CheckpointSaver,
    get_save_dir,
    load_model_checkpoint,
    count_parameters,
)
from utils.config import args_parser

VERBOSE = True

if __name__ == "__main__":
    args = args_parser()

    device = torch.device(
        args.device
        if torch.cuda.is_available() and args.device.startswith("cuda")
        else "cpu"
    )

    print("using device:", device)
    torch.manual_seed(args.random_seed)

    args.save_dir = get_save_dir(f"{args.save_dir}", training=True)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Save args
    args_file = os.path.join(args.save_dir, "args.json")
    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    if args.dataset == "PAM" or args.dataset.startswith("MIMIC3"):
        all_metrics = {
            "accuracy": np.zeros((args.n_splits, args.n_runs)),
            "f1": np.zeros((args.n_splits, args.n_runs)),
            "precision": np.zeros((args.n_splits, args.n_runs)),
            "recall": np.zeros((args.n_splits, args.n_runs)),
            "aucroc": np.zeros((args.n_splits, args.n_runs)),
            "auprc": np.zeros((args.n_splits, args.n_runs)),
        }
    else:
        all_metrics = {
            "aucroc": np.zeros((args.n_splits, args.n_runs)),
            "auprc": np.zeros((args.n_splits, args.n_runs)),
        }
    # Perform experiment n_runs times
    for run_id in range(args.n_runs):
        # use the different data splits as in Raindrop
        for split_id in range(args.n_splits):
            run_name = f'{args.dataset}_irreg_{args.irregularity}_rate:{args.irregularity_rate}_positional_encoding:{args.positional_encoding}_{str(datetime.now().strftime("%Y-%m-%d %H:%M"))}'

            if args.dataset == "MIMIC3-IHM":
                checkpoint_saver = CheckpointSaver(
                    args.save_dir, metric_name="auprc", maximize_metric=True
                )
            elif args.dataset == "PAM":
                checkpoint_saver = CheckpointSaver(
                    args.save_dir, metric_name="f1", maximize_metric=True
                )
            else:
                checkpoint_saver = CheckpointSaver(
                    args.save_dir, metric_name="aucroc", maximize_metric=True
                )
            train_dataset, val_dataset, test_dataset = get_datasets(
                args, n_split=split_id + 1, device=device
            )

            # create dataloaders
            if args.dataset.startswith("MIMIC3"):
                mimic_3_fn = lambda batch: mimic3_collate_fn(batch, args.window_size)
                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    collate_fn=mimic_3_fn,
                )
                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=mimic_3_fn,
                )
                test_dataloader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=mimic_3_fn,
                )
            else:
                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=args.batch_size, shuffle=True
                )
                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    drop_last=True,
                )
                test_dataloader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    drop_last=True,
                )

            stats = train_dataset.get_dims()

            # Create the model
            model = WaveGNN(
                n_sensors=stats["n_sensors"],
                static_features_len=6 if args.dataset == "P19" else 5,
                hidden_dim=128,
                args=args,
            ).to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )

            if args.scheduler == True:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=0.1,
                    patience=1,
                    threshold=0.0001,
                    threshold_mode="rel",
                    cooldown=0,
                    min_lr=1e-8,
                    eps=1e-08,
                    verbose=True,
                )
            else:
                scheduler = None

            if args.n_classes == 1 or args.n_classes == 25:
                criterion = torch.nn.BCEWithLogitsLoss()
            else:
                criterion = torch.nn.CrossEntropyLoss()

            # Train the model
            torch.autograd.set_detect_anomaly(True)
            trainer = Trainer(
                model,
                optimizer,
                criterion,
                train_dataloader,
                val_dataloader,
                device,
                checkpoint_saver=checkpoint_saver,
                args=args,
                scheduler=scheduler,
            )

            try:
                trainer.run(epochs=args.epochs, verbose=VERBOSE, patience=args.patience)
            except KeyboardInterrupt:
                print("Exiting training early...")

            # Load the best model
            best_path = os.path.join(args.save_dir, "best.pth.tar")
            best_model = load_model_checkpoint(best_path, model)

            # Test the model
            print("Testing the model...")
            test_loss, test_metrics = trainer.test(test_dataloader, best_model)
            print(f"Test Loss: {test_loss}")
            print(f"Test Metrics: {test_metrics}")

            # update metrics
            if args.dataset == "PAM" or args.dataset.startswith("MIMIC3"):
                all_metrics["accuracy"][split_id, run_id] = test_metrics["accuracy"]
                all_metrics["f1"][split_id, run_id] = test_metrics["f1"]
                all_metrics["precision"][split_id, run_id] = test_metrics["precision"]
                all_metrics["recall"][split_id, run_id] = test_metrics["recall"]
                all_metrics["aucroc"][split_id, run_id] = test_metrics["aucroc"]
                all_metrics["auprc"][split_id, run_id] = test_metrics["auprc"]
            else:
                all_metrics["aucroc"][split_id, run_id] = test_metrics["aucroc"]
                all_metrics["auprc"][split_id, run_id] = test_metrics["auprc"]

    if args.dataset == "PAM" or args.dataset.startswith("MIMIC3"):
        # get the best run for each split based on F1
        best_idx = np.argmax(all_metrics["f1"], axis=1)
        best_accuracies = [
            all_metrics["accuracy"][i, best_idx[i]] for i in range(args.n_splits)
        ]
        best_f1s = [all_metrics["f1"][i, best_idx[i]] for i in range(args.n_splits)]
        best_precisions = [
            all_metrics["precision"][i, best_idx[i]] for i in range(args.n_splits)
        ]
        best_recalls = [
            all_metrics["recall"][i, best_idx[i]] for i in range(args.n_splits)
        ]
        best_aurocs = [
            all_metrics["aucroc"][i, best_idx[i]] for i in range(args.n_splits)
        ]
        best_auprcs = [
            all_metrics["auprc"][i, best_idx[i]] for i in range(args.n_splits)
        ]
    else:
        # get the best run for each split based on AUPRC
        best_idx = np.argmax(all_metrics["auprc"], axis=1)
        best_aurocs = [
            all_metrics["aucroc"][i, best_idx[i]] for i in range(args.n_splits)
        ]
        best_auprcs = [
            all_metrics["auprc"][i, best_idx[i]] for i in range(args.n_splits)
        ]

    # display results
    print("------------------------------------------")
    print("Overall results for dataset:", args.dataset)
    if args.dataset == "PAM" or args.dataset.startswith("MIMIC3"):
        # print mean +/- std for each metric
        print(
            "Accuracy = %.1f +/- %.1f"
            % (np.mean(best_accuracies) * 100, np.std(best_accuracies) * 100)
        )
        print(
            "F1       = %.1f +/- %.1f"
            % (np.mean(best_f1s) * 100, np.std(best_f1s) * 100)
        )
        print(
            "Precision= %.1f +/- %.1f"
            % (np.mean(best_precisions) * 100, np.std(best_precisions) * 100)
        )
        print(
            "Recall   = %.1f +/- %.1f"
            % (np.mean(best_recalls) * 100, np.std(best_recalls) * 100)
        )
        print(
            "AUCROC = %.1f +/- %.1f"
            % (np.mean(best_aurocs) * 100, np.std(best_aurocs) * 100)
        )
        print(
            "AUPRC  = %.1f +/- %.1f"
            % (np.mean(best_auprcs) * 100, np.std(best_auprcs) * 100)
        )
    else:
        print(
            "AUCROC = %.1f +/- %.1f"
            % (np.mean(best_aurocs) * 100, np.std(best_aurocs) * 100)
        )
        print(
            "AUPRC  = %.1f +/- %.1f"
            % (np.mean(best_auprcs) * 100, np.std(best_auprcs) * 100)
        )
    print("------------------------------------------")
