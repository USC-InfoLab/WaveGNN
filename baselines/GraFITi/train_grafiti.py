import argparse
import sys
import time
from random import SystemRandom
from torch.optim import AdamW
import logging
import os
import random
import warnings
from datetime import datetime

import numpy as np
import torch
# import torchinfo
from IPython.core.display import HTML
from torch import Tensor, jit
import pdb
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

import wandb

# Import for classification data loading
from data.MIMIC3_loader import get_mimic3_data
from data.P12_loader import get_p12_data
from data.P19_loader import get_p19_data
from data.PAM_loader import get_pam_data

# fmt: off
parser = argparse.ArgumentParser(description="Training Script for GraFITi.")
parser.add_argument("-q",  "--quiet",        default=False,  const=True, help="kernel-inititialization", nargs="?")
parser.add_argument("-r",  "--run_id",       default=None,   type=str,   help="run_id")
parser.add_argument("-c",  "--config",       default=None,   type=str,   help="load external config", nargs=2)
parser.add_argument("-e",  "--epochs",       default=300,    type=int,   help="maximum epochs")
# parser.add_argument("-f",  "--fold",         default=2,      type=int,   help="fold number - not used by MIMIC3_loader") # Fold not used by new loader
parser.add_argument("-bs", "--batch-size",   default=32,     type=int,   help="batch-size")
parser.add_argument("-lr", "--learn-rate",   default=0.001,  type=float, help="learn-rate")
parser.add_argument("-b",  "--betas", default=(0.9, 0.999),  type=float, help="adam betas", nargs=2)
parser.add_argument("-wd", "--weight-decay", default=0.001,  type=float, help="weight-decay")
parser.add_argument("-hs", "--hidden-size",  default=32,    type=int,   help="hidden-size - Note: GraFITi uses latent-dim")
parser.add_argument("-ki", "--kernel-init",  default="skew-symmetric",   help="kernel-inititialization - Not used by GraFITi directly")
parser.add_argument("-n",  "--note",         default="",     type=str,   help="Note that can be added")
parser.add_argument("-s",  "--seed",         default=None,   type=int,   help="Set the random seed.")
parser.add_argument("-nl",  "--nlayers", default=2,   type=int,   help="Number of GraFITi layers")
parser.add_argument("-ahd",  "--attn-head", default=2,   type=int,   help="Number of attention heads")
parser.add_argument("-ldim",  "--latent-dim", default=128,   type=int,   help="Latent dimension in GraFITi")
# parser.add_argument("-dset", "--dataset", default="ushcn", type=str, help="Name of the dataset - used for saving path") # Repurposed for save path
# parser.add_argument("-ft", "--forc-time", default=0, type=int, help="forecast horizon in hours - Not used for classification")
# parser.add_argument("-ct", "--cond-time", default=36, type=int, help="conditioning range in hours - Not used by MIMIC3_loader, uses median_len")
# parser.add_argument("-nf", "--nfolds", default=5, type=int, help="#folds for crossvalidation - Not used by MIMIC3_loader")

# New arguments for classification with MIMIC3_loader
parser.add_argument("--data_path", type=str, required=True, help="Path to the specific preprocessed data folder (e.g., /path/to/data/ihm/ or /path/to/P19data/)")
parser.add_argument("--task_name", type=str, required=True, choices=['mimic3_ihm', 'mimic3_phe', 'p12', 'p19', 'pam'], help="Task name - used for data path and model output dim.")
parser.add_argument("--metric_task_type", type=str, required=True, choices=['binary', 'multilabel', 'multiclass'], help="Type of classification task for metric calculation (e.g., binary, multilabel, multiclass).")
parser.add_argument("--median_len", type=int, default=100, help="Fixed sequence length for observations after truncation/padding.")
parser.add_argument("--wandb", action='store_true', help="Enable wandb logging")
# parser.add_argument("--split_num_p19", type=int, default=1, help="Split number for P19 dataset (1-5).") # Removed

# PAM-specific irregularity arguments
parser.add_argument("--pam_apply_irregularity", action='store_true', help="Apply irregularity to PAM val/test splits (as in original PAM loader)")
parser.add_argument("--pam_train_apply_irregularity", action='store_true', help="Apply irregularity to PAM train split (original PAM loader did not)")
parser.add_argument("--pam_irregularity_rate", type=float, default=0.0, help="PAM irregularity rate (fraction of sensors to drop, default 0.0 matches original)")
parser.add_argument("--pam_irregularity_type", type=str, default='random', choices=['random', 'fixed', 'first_n'], help="PAM irregularity type (default random, original was fixed but flag was off)")

# fmt: on

ARGS = parser.parse_args()


print(" ".join(sys.argv))
# use current time as experiment id
experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
print(ARGS, experiment_id)

# --- WandB Initialization ---
wandb.init(
    mode="online" if ARGS.wandb else "offline",
    project="GraFITi-Classification", # Choose your project name
    config=vars(ARGS),
    name=f"{ARGS.task_name}-{ARGS.metric_task_type}-ld{ARGS.latent_dim}-nl{ARGS.nlayers}-ah{ARGS.attn_head}-s{ARGS.seed}-id{experiment_id}",
    resume="allow",  # Allow resuming runs
    id=f"{ARGS.run_id if ARGS.run_id else experiment_id}" # Use provided run_id or generate one
)
wandb.config.update({
    "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "script_path": os.path.abspath(__file__)
}, allow_val_change=True) # allow_val_change for items not initially in ARGS

import yaml

if ARGS.config is not None:
    cfg_file, cfg_id = ARGS.config
    with open(cfg_file, "r") as file:
        cfg_dict = yaml.safe_load(file)
        vars(ARGS).update(**cfg_dict[int(cfg_id)])

print(ARGS)


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

warnings.filterwarnings(action="ignore", category=UserWarning, module="torch")
logging.basicConfig(level=logging.WARN)
HTML("<style>.jp-OutputArea-prompt:empty {padding: 0; border: 0;}</style>")

if ARGS.seed is not None:
    torch.manual_seed(ARGS.seed)
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)

OPTIMIZER_CONFIG = {
    "lr": ARGS.learn_rate,
    "betas": tuple(ARGS.betas),
    "weight_decay": ARGS.weight_decay,
}

# Determine num_classes based on task_name
if ARGS.task_name == 'mimic3_ihm':
    num_classes = 1  # Binary classification (output one logit)
    if ARGS.metric_task_type != 'binary':
        print(f"Warning: For task {ARGS.task_name}, metric_task_type is typically 'binary'. Received {ARGS.metric_task_type}.")
elif ARGS.task_name == 'mimic3_phe':
    num_classes = 25 # Multi-label classification
    if ARGS.metric_task_type != 'multilabel':
        print(f"Warning: For task {ARGS.task_name}, metric_task_type is typically 'multilabel'. Received {ARGS.metric_task_type}.")
elif ARGS.task_name == 'p12':
    num_classes = 1 # Binary classification
    if ARGS.metric_task_type != 'binary':
        print(f"Warning: For task {ARGS.task_name}, metric_task_type is typically 'binary'. Received {ARGS.metric_task_type}.")
elif ARGS.task_name == 'p19':
    num_classes = 1 # Binary classification
    if ARGS.metric_task_type != 'binary':
        print(f"Warning: For task {ARGS.task_name}, metric_task_type is typically 'binary'. Received {ARGS.metric_task_type}.")
elif ARGS.task_name == 'pam':
    num_classes = 8 # Multiclass classification (8 classes for PAM)
    if ARGS.metric_task_type != 'multiclass':
        # Override or warn heavily if metric_task_type is not multiclass for PAM
        print(f"Warning: For task {ARGS.task_name}, metric_task_type MUST be 'multiclass'. Overriding to 'multiclass'.")
        ARGS.metric_task_type = 'multiclass'
else:
    raise ValueError(f"Unknown or unsupported task_name: {ARGS.task_name}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Loading ---
data_path_for_task = ARGS.data_path 

if ARGS.task_name.startswith('mimic3'):
    loader_args = argparse.Namespace(
        data_path=data_path_for_task, 
        task_name=ARGS.task_name,
        batch_size=ARGS.batch_size,
        median_len=ARGS.median_len,
        num_workers=getattr(ARGS, 'num_workers', 0) # Add num_workers if available in ARGS
    )
    TRAIN_LOADER, VALID_LOADER, TEST_LOADER, num_features = get_mimic3_data(loader_args, DEVICE)
elif ARGS.task_name == 'p12':
    loader_args = argparse.Namespace(
        data_path=data_path_for_task, # Should point to the dir containing P12_data_set1.npy etc.
        median_len=ARGS.median_len,   # P12 loader uses this
        batch_size=ARGS.batch_size,
        num_workers=getattr(ARGS, 'num_workers', 0),
        # P12 specific args if any, e.g. split_num if it uses one. Ensure P12_loader.py handles this if needed.
        # For P12, the original loader also had a hardcoded split_num = 1 inside its get_data.
        # If P12_loader.py needs a split number from args, it should be added here and to its parser.
    )
    TRAIN_LOADER, VALID_LOADER, TEST_LOADER, num_features = get_p12_data(loader_args, DEVICE)
elif ARGS.task_name == 'p19':
    loader_args = argparse.Namespace(
        data_path=data_path_for_task, # Should point to P19data dir
        median_len=ARGS.median_len,   # P19 loader uses this
        batch_size=ARGS.batch_size,
        num_workers=getattr(ARGS, 'num_workers', 0)
        # split_num_p19=ARGS.split_num_p19 # Removed, as it's hardcoded in P19_loader.py
        # Add dummy time_encoding if collate expects it, P19 loader doesn't use it directly
        # time_encoding=getattr(ARGS, 'time_encoding', 'absolute')
    )
    TRAIN_LOADER, VALID_LOADER, TEST_LOADER, num_features = get_p19_data(loader_args, DEVICE)
elif ARGS.task_name == 'pam':
    loader_args = argparse.Namespace(
        data_path=data_path_for_task, # Should point to PAMdata dir
        median_len=ARGS.median_len,   # PAM loader will enforce 600
        batch_size=ARGS.batch_size,
        num_workers=getattr(ARGS, 'num_workers', 0),
        # PAM-specific irregularity args
        pam_apply_irregularity=ARGS.pam_apply_irregularity,
        pam_train_apply_irregularity=ARGS.pam_train_apply_irregularity,
        pam_irregularity_rate=ARGS.pam_irregularity_rate,
        pam_irregularity_type=ARGS.pam_irregularity_type
    )
    TRAIN_LOADER, VALID_LOADER, TEST_LOADER, num_features = get_pam_data(loader_args, DEVICE)
else:
    raise ValueError(f"Data loading not implemented for task: {ARGS.task_name}")

ARGS.num_types = num_features # Store num_features determined by the loader back into global ARGS

print(f"Data loaded for task: {ARGS.task_name}. Number of features (num_types): {ARGS.num_types}")
print(f"Training batches: {len(TRAIN_LOADER)}, Validation batches: {len(VALID_LOADER)}, Test batches: {len(TEST_LOADER)}")

# Loss Function for Classification
if ARGS.metric_task_type == 'multiclass':
    LOSS = torch.nn.CrossEntropyLoss().to(DEVICE)
    print(f"Using CrossEntropyLoss for {ARGS.metric_task_type} task.")
else: # binary or multilabel
    LOSS = torch.nn.BCEWithLogitsLoss().to(DEVICE)
    print(f"Using BCEWithLogitsLoss for {ARGS.metric_task_type} task.")

from grafiti.grafiti import GraFITi

MODEL_CONFIG = {
    "input_dim": ARGS.num_types, # Determined by the data loader
    "attn_head": ARGS.attn_head,
    "latent_dim": ARGS.latent_dim,
    "n_layers": ARGS.nlayers,
    "device": DEVICE,
    "num_classes": num_classes # Added for classification head
}

MODEL = GraFITi(**MODEL_CONFIG).to(DEVICE)
# torchinfo.summary(MODEL) # This will show the new classification head too
# print model total parameters
total_params = sum(p.numel() for p in MODEL.parameters())
print(f"Total parameters: {total_params}")


def predict_fn(model, batch) -> tuple[Tensor, Tensor]: # Return (labels, logits)
    """Get targets and predictions for classification."""
    # batch is a BatchClassification object, with fields:
    # x_time: Combined observed and dummy target timestamps (B, L_total)
    # x_vals: Combined observed values (with NaNs filled by 0) and dummy target values (0) (B, L_total, D)
    # x_mask: Mask indicating true observations (True for observed, False for dummy target part) (B, L_total, D)
    # y_mask_for_grafiti_structure: Mask for the target part, all False for classification (B, L_total, D)
    # class_labels: Classification labels (B,) or (B, num_classes)

    # These correspond to the arguments expected by model.classify()
    # (x_time, x_vals, x_mask, y_mask_for_grafiti_structure)
    
    # Pass the tensors directly from the batch to the model
    t_for_model = batch.x_time.to(DEVICE)
    x_vals_for_model = batch.x_vals.to(DEVICE)
    # batch.x_mask is the observation mask, True for actual observations, False for the dummy target part.
    # This is what model.classify expects as its x_mask argument (which maps to obs_mask in get_contextualized_embeddings)
    obs_mask_for_model = batch.x_mask.to(DEVICE)
    
    # batch.y_mask_for_grafiti_structure is the dummy target mask, all False for classification.
    # This is what model.classify expects as its y_mask_for_grafiti_structure argument.
    target_mask_for_model = batch.y_mask_for_grafiti_structure.to(DEVICE)

    labels = batch.class_labels.to(DEVICE)

    # Call the model's classify method
    logits = model.classify(t_for_model, x_vals_for_model, obs_mask_for_model, target_mask_for_model)
    
    # Adjust label shape and type based on task type and num_classes
    processed_labels = labels # Start with original labels
    if ARGS.metric_task_type == 'multiclass':
        # For CrossEntropyLoss, labels should be (B,) and long type
        processed_labels = processed_labels.squeeze(-1).long() 
    elif num_classes == 1: # Binary classification
        # For BCEWithLogitsLoss with num_classes=1, labels should be (B, 1) and float
        if processed_labels.ndim == 1:
            processed_labels = processed_labels.unsqueeze(1)
        processed_labels = processed_labels.float()
    else: # Multilabel classification
        # For BCEWithLogitsLoss with num_classes > 1, labels should be (B, num_classes) and float
        processed_labels = processed_labels.float()
    
    return processed_labels, logits


# Reset
MODEL.zero_grad(set_to_none=True)

# Helper function for metrics
def compute_classification_metrics(y_true_np, y_pred_binary_np, y_pred_probs_np, metric_task_type, num_classes_for_multiclass=None):
    """
    Computes classification metrics based on task type.
    metric_task_type: 'binary', 'multilabel', or 'multiclass'
    y_true_np: Ground truth labels (Numpy array)
    y_pred_binary_np: Binary predictions (0/1) after thresholding (Numpy array)
    y_pred_probs_np: Raw probabilities (sigmoid output for binary/multilabel, softmax for multiclass) (Numpy array)
    num_classes_for_multiclass: Required if metric_task_type is 'multiclass' for one-hot encoding.
    """
    results = {}

    # Accuracy
    try:
        results["accuracy"] = accuracy_score(y_true_np, y_pred_binary_np)
    except ValueError as e:
        print(f"Metric Error (Accuracy): {e}")
        results["accuracy"] = 0.0

    # F1 Score, Precision, Recall based on t-PatchGNN logic
    f1_avg = 'macro'  # Default for multilabel
    if metric_task_type == 'binary':
        f1_avg = 'binary'
    elif metric_task_type == 'multiclass':
        f1_avg = 'weighted'
    
    precision_avg = 'weighted' # Default for multiclass or multilabel (as per t-PatchGNN: 'binary' if binary else 'weighted')
    if metric_task_type == 'binary':
        precision_avg = 'binary'
    elif metric_task_type == 'multilabel': # t-PatchGNN uses weighted for precision/recall in non-binary cases.
        precision_avg = 'macro' # Common alternative for multilabel if weighted isn't desired, or stick to weighted.
                                # The provided snippet implies weighted for non-binary precision/recall.
                                # Let's stick to the snippet's logic: binary if binary, else weighted.
        precision_avg = 'weighted' 

    recall_avg = precision_avg # Same as precision averaging in the snippet

    try:
        results["f1"] = f1_score(y_true_np, y_pred_binary_np, average=f1_avg, zero_division=0)
        results["precision"] = precision_score(y_true_np, y_pred_binary_np, average=precision_avg, zero_division=0)
        results["recall"] = recall_score(y_true_np, y_pred_binary_np, average=recall_avg, zero_division=0)
    except ValueError as e:
        print(f"Metric Error (F1/Precision/Recall): {e}")
        results["f1"] = 0.0
        results["precision"] = 0.0
        results["recall"] = 0.0

    # AUROC and AUPRC
    try:
        if metric_task_type == "binary":
            # y_pred_probs_np is (N,) or (N,1) for the positive class after sigmoid.
            # y_true_np is (N,) or (N,1).
            # The t-PatchGNN example uses all_probs[:, 1] for binary. This implies all_probs might be (N,2) from a softmax.
            # Our current binary setup (num_classes=1 with BCEWithLogitsLoss) gives probs as (N,1). So squeeze is fine.
            scores_for_roc_pr = y_pred_probs_np.squeeze()
            labels_for_roc_pr = y_true_np.squeeze()
            if len(np.unique(labels_for_roc_pr)) > 1: 
                results["auroc"] = roc_auc_score(labels_for_roc_pr, scores_for_roc_pr)
                results["auprc"] = average_precision_score(labels_for_roc_pr, scores_for_roc_pr)
            else:
                print(f"Skipping AUROC/AUPRC for binary task ({metric_task_type}): only one class present in y_true.")
                results["auroc"] = 0.0
                results["auprc"] = 0.0

        elif metric_task_type == "multilabel":
            # y_true_np is (N, num_labels), y_pred_probs_np is (N, num_labels)
            # t-PatchGNN example uses these directly (average='macro' or 'samples' for sklearn is typical)
            results["auroc"] = roc_auc_score(y_true_np, y_pred_probs_np, average="macro") 
            results["auprc"] = average_precision_score(y_true_np, y_pred_probs_np, average="macro")
        
        elif metric_task_type == "multiclass":
            if num_classes_for_multiclass is None:
                raise ValueError("num_classes_for_multiclass must be provided for multiclass AUROC/AUPRC.")
            from sklearn.preprocessing import label_binarize
            y_true_one_hot = label_binarize(y_true_np, classes=range(num_classes_for_multiclass))
            results["auroc"] = roc_auc_score(y_true_one_hot, y_pred_probs_np, multi_class='ovr', average='macro') # or 'ovo'
            results["auprc"] = average_precision_score(y_true_one_hot, y_pred_probs_np, average='macro')

    except ValueError as e:
        print(f"Metric Error (AUROC/AUPRC): {e}")
        pdb.set_trace()
        results["auroc"] = 0.0
        results["auprc"] = 0.0
        
    return results


# ## Initialize Optimizer
OPTIMIZER = AdamW(MODEL.parameters(), **OPTIMIZER_CONFIG)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    OPTIMIZER, "min", patience=5, factor=0.5, min_lr=0.00001
)

es = False
best_val_loss = 10e8
total_num_batches = 0
early_stop_counter = 0
for epoch in range(1, ARGS.epochs + 1):
    MODEL.train() # Set model to training mode
    epoch_loss_list = []
    start_time = time.time()
    
    # Wrap TRAIN_LOADER with tqdm for a progress bar
    train_pbar = tqdm(TRAIN_LOADER, desc=f"Epoch {epoch}/{ARGS.epochs} [TRAIN]", unit="batch")
    for batch in train_pbar:
        OPTIMIZER.zero_grad()
        LABELS, LOGITS = predict_fn(MODEL, batch)
        R = LOSS(LOGITS, LABELS)
        assert torch.isfinite(R).item(), "Model Collapsed during training!"
        epoch_loss_list.append(R.item())
        # Backward
        R.backward()
        OPTIMIZER.step()
        train_pbar.set_postfix(loss=R.item())
        
    epoch_time = time.time()
    train_loss = np.mean(epoch_loss_list)
    
    MODEL.eval() # Set model to evaluation mode
    val_epoch_loss_list = []
    all_val_labels = []
    all_val_preds_binary = []
    all_val_probs = []
    # Wrap VALID_LOADER with tqdm
    val_pbar = tqdm(VALID_LOADER, desc=f"Epoch {epoch}/{ARGS.epochs} [VALID]", unit="batch")
    with torch.no_grad():
        for batch in val_pbar:
            LABELS, LOGITS = predict_fn(MODEL, batch)
            R_val = LOSS(LOGITS, LABELS)
            if R_val.isnan():
                print("NaN loss in validation!")
                pdb.set_trace()
            val_epoch_loss_list.append(R_val.item())
            val_pbar.set_postfix(loss=R_val.item())

            # Store labels and predictions for metrics
            if ARGS.metric_task_type == 'multiclass':
                probs = torch.softmax(LOGITS, dim=-1)
                preds_binary = torch.argmax(probs, dim=-1) 
            else: # binary or multilabel
                probs = torch.sigmoid(LOGITS)
                preds_binary = (probs >= 0.5).float()

            all_val_labels.append(LABELS.cpu())
            all_val_preds_binary.append(preds_binary.cpu())
            all_val_probs.append(probs.cpu())
            
    val_loss = np.mean(val_epoch_loss_list)

    # Concatenate all validation outputs and compute metrics
    if all_val_labels: # Ensure there was validation data
        y_true_val = torch.cat(all_val_labels, dim=0).numpy()
        y_pred_binary_val = torch.cat(all_val_preds_binary, dim=0).numpy()
        y_pred_probs_val = torch.cat(all_val_probs, dim=0).numpy()
        
        val_metrics = compute_classification_metrics(
            y_true_val, y_pred_binary_val, y_pred_probs_val, 
            ARGS.metric_task_type, 
            num_classes_for_multiclass=MODEL_CONFIG['num_classes'] if ARGS.metric_task_type == 'multiclass' else None
        )
        print(f"Validation Metrics: {val_metrics}")
    else:
        val_metrics = {}

    print(
        f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
        f"Epoch Time: {int(epoch_time - start_time)}s, LR: {OPTIMIZER.param_groups[0]['lr']:.6f}"
    )

    log_dict = {
        "train/loss": train_loss,
        "val/epoch": epoch,
        "val/loss": val_loss,
        "val/learning_rate": OPTIMIZER.param_groups[0]['lr']
    }
    if val_metrics: # Log validation metrics if they exist
        for k, v in val_metrics.items():
            log_dict[f"val/{k}"] = v
    wandb.log(log_dict, step=epoch)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wandb.summary["best_val_loss"] = best_val_loss
        wandb.summary["best_val_epoch"] = epoch
        if val_metrics: # Log best metrics too
            for k, v in val_metrics.items():
                wandb.summary[f"best_val/{k}"] = v

        # Save checkpoint using task_name for clarity
        save_dir = "saved_models"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{ARGS.task_name}_{experiment_id}.h5")
        torch.save(
            {
                "epoch": epoch,
                "state_dict": MODEL.state_dict(),
                "optimizer_state_dict": OPTIMIZER.state_dict(),
                "loss": train_loss,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
            },
            save_path,
        )
        print(f"Model saved to {save_path}")
        early_stop_counter = 0 # Reset counter
    else:
        early_stop_counter += 1
    
    if early_stop_counter >= 30:
        print("Early stopping due to no improvement in validation loss for 30 epochs.")
        es = True
    
    scheduler.step(val_loss)

    if (epoch == ARGS.epochs) or (es == True):
        print("Loading best model for testing...")
        # Load best checkpoint
        load_path = os.path.join(save_dir, f"{ARGS.task_name}_{experiment_id}.h5")
        if os.path.exists(load_path):
            chp = torch.load(load_path, weights_only=False)
            MODEL.load_state_dict(chp["state_dict"])
            print(f"Loaded checkpoint from epoch {chp['epoch']} with best val loss {chp.get('val_loss', best_val_loss):.4f}")
        else:
            print("No checkpoint found to load for testing. Using current model state.")

        MODEL.eval()
        test_loss_list = []
        all_test_labels = []
        all_test_preds_binary = []
        all_test_probs = []
        # Wrap TEST_LOADER with tqdm
        test_pbar = tqdm(TEST_LOADER, desc="Testing", unit="batch")
        with torch.no_grad():
            for batch in test_pbar:
                LABELS, LOGITS = predict_fn(MODEL, batch)
                R_test = LOSS(LOGITS, LABELS)
                assert torch.isfinite(R_test).item(), "Model Collapsed during testing!"
                test_loss_list.append(R_test.item())
                test_pbar.set_postfix(loss=R_test.item())

                # Store labels and predictions for metrics
                if ARGS.metric_task_type == 'multiclass':
                    probs = torch.softmax(LOGITS, dim=-1)
                    preds_binary = torch.argmax(probs, dim=-1)
                else: # binary or multilabel
                    probs = torch.sigmoid(LOGITS)
                    preds_binary = (probs >= 0.5).float()

                all_test_labels.append(LABELS.cpu())
                all_test_preds_binary.append(preds_binary.cpu())
                all_test_probs.append(probs.cpu())
        
        test_loss = np.mean(test_loss_list)

        # Concatenate all test outputs and compute metrics
        if all_test_labels: # Ensure there was test data
            y_true_test = torch.cat(all_test_labels, dim=0).numpy()
            y_pred_binary_test = torch.cat(all_test_preds_binary, dim=0).numpy()
            y_pred_probs_test = torch.cat(all_test_probs, dim=0).numpy()
            
            test_metrics = compute_classification_metrics(
                y_true_test, y_pred_binary_test, y_pred_probs_test, 
                ARGS.metric_task_type,
                num_classes_for_multiclass=MODEL_CONFIG['num_classes'] if ARGS.metric_task_type == 'multiclass' else None
            )
            print(f"Test Metrics: {test_metrics}")
        else:
            test_metrics = {}

        test_log_dict = {
            "test/loss": test_loss
        }
        if test_metrics:
            for k, v in test_metrics.items():
                test_log_dict[f"test/{k}"] = v
        wandb.log(test_log_dict, step=epoch) # Log at the final epoch step

        # Update summary for key final values if desired, but primary logging via wandb.log()
        wandb.summary["best_val_loss"] = best_val_loss # Already being updated when best is found
        wandb.summary["final_test_loss_summary"] = test_loss # Can keep a summary version
        if test_metrics.get('accuracy'): # Example: add final test accuracy to summary
            wandb.summary["final_test_accuracy_summary"] = test_metrics['accuracy']

        print(
            f"End of Training. Best Val Loss: {best_val_loss:.4f}, Test Loss: {test_loss:.4f}"
        )
        # Optionally, save test_metrics to a file or the checkpoint dictionary if needed for later summary
        # For instance, could update the loaded checkpoint object if we want to resave it with test results, though usually separate logs are better.
        if os.path.exists(load_path):
            chp["test_loss"] = test_loss
            chp["test_metrics"] = test_metrics
            # torch.save(chp, load_path) # Example: resaving checkpoint with test results - use with caution

        break

wandb.finish()
print("Training finished.")
