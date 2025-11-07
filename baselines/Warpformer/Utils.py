import numpy as np
import pandas as pd
import os
import math
import sys
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, average_precision_score, roc_auc_score
import time
import random
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb


def cal_tau(observed_tp, observed_mask):
    # input [B,L,K], [B,L]
    # return [B,L,K]
    # observed_mask, observed_tp = x[:, :, input_dim:2 * input_dim], x[:, :, -1]
    if observed_tp.ndim == 2:
        tmp_time = observed_mask * np.expand_dims(observed_tp,axis=-1) # [B,L,K]
    else:
        tmp_time = observed_tp.copy()
        
    b,l,k = tmp_time.shape
    
    new_mask = observed_mask.copy()
    new_mask[:,0,:] = 1
    tmp_time[new_mask == 0] = np.nan
    tmp_time = tmp_time.transpose((1,0,2)) # [L,B,K]
    tmp_time = np.reshape(tmp_time, (l,b*k)) # [L, B*K]

    # padding the missing value with the next value
    df1 = pd.DataFrame(tmp_time)
    df1 = df1.fillna(method='ffill')
    tmp_time = np.array(df1)

    tmp_time = np.reshape(tmp_time, (l,b,k))
    tmp_time = tmp_time.transpose((1,0,2)) # [B,L,K]
    
    tmp_time[:,1:] -= tmp_time[:,:-1]
    del new_mask
    return tmp_time * observed_mask


def pred_loss(prediction, truth, loss_func):
    """ supervised prediction loss, cross entropy or label smoothing. 
    prediction: [B, 2]
    label: [B]
    """
    loss = loss_func(prediction, truth)
    loss = torch.sum(loss)
    return loss



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, save_path=None, dp_flag=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path
        self.dp_flag = dp_flag
        self.best_epoch = -1

    def __call__(self, val_loss, model, classifier=None, time_predictor=None, decoder=None,epoch=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, classifier, time_predictor, decoder, dp_flag=self.dp_flag)
            if epoch is not None:
                self.best_epoch = epoch
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience} (Best: {-self.best_score:.6f}, Current: {-score:.6f})')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, classifier, time_predictor, decoder, dp_flag=self.dp_flag)
            if epoch is not None:
                self.best_epoch = epoch
            self.counter = 0

    def save_checkpoint(self, val_loss, model, classifier=None, time_predictor=None, decoder=None, dp_flag=False):
        '''
        Saves model when validation loss decrease.
        '''
        if self.verbose:
            print(f'Validation AUROC improved (Val_Loss was {-self.val_loss_min:.6f} --> {-val_loss:.6f}). Saving model ...')

        classifier_state_dict = None

        if dp_flag:
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        if classifier is not None:
            classifier_state_dict = classifier.state_dict()
            
        if self.save_path is not None:
            torch.save({
                'model_state_dict':model_state_dict,
                'classifier_state_dict': classifier_state_dict,
            }, self.save_path)
        else:
            print("no path assigned")  

        self.val_loss_min = val_loss


def log_info(opt, phase, epoch, acc, precision, recall, f1, rmse=0.0, start=0.0, value_rmse=0.0, auroc=0.0, auprc=0.0, loss=0.0, save=False):
    log_message = (
        f'  -({phase}) epoch: {epoch}, acc: {acc:8.5f}, precision: {precision:8.5f}, recall: {recall:8.5f}, f1: {f1:8.5f}, '
        f'AUROC: {auroc:8.5f}, AUPRC: {auprc:8.5f}, RMSE: {rmse:8.5f}, Value_RMSE: {value_rmse:8.5f}, loss: {loss:8.5f}, '
        f'elapse: {(time.time() - start) / 60:3.3f} min'
    )
    print(log_message)

    # Log to wandb if enabled
    if hasattr(opt, 'wandb') and opt.wandb:
        wandb.log({
            f'{phase}/accuracy': acc,
            f'{phase}/precision': precision,
            f'{phase}/recall': recall,
            f'{phase}/f1': f1,
            f'{phase}/auroc': auroc,
            f'{phase}/auprc': auprc,
            f'{phase}/rmse': rmse,
            f'{phase}/value_rmse': value_rmse,
            f'{phase}/loss': loss,
            'epoch': epoch
        })

    if save and opt.log is not None:
        with open(opt.log, 'a') as f:
            f_message = (
                f'{phase}:\t{epoch}, ACC: {acc:8.5f}, Precision: {precision:8.5f}, Recall: {recall:8.5f}, F1: {f1:8.5f}, '
                f'AUROC: {auroc:8.5f}, AUPRC: {auprc:8.5f}, TimeRMSE: {rmse:8.5f}, ValueRMSE: {value_rmse:8.5f}, Loss: {loss:8.5f}\n'
            )
            f.write(f_message)
                

def load_checkpoints(save_path, model, classifier=None, time_predictor=None, decoder=None, dp_flag=False, use_cpu=False):
    if not os.path.exists(save_path) or not os.path.getsize(save_path) > 0: 
        print(save_path, " is None or empty file")
        return model, classifier, time_predictor, decoder

    if use_cpu:
        checkpoint = torch.load(save_path,map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(save_path)
    
    if dp_flag:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])


    if classifier is not None and checkpoint['classifier_state_dict'] is not None:
        classifier.load_state_dict(checkpoint['classifier_state_dict'])

    if time_predictor is not None and 'time_predictor_state_dict' in checkpoint and checkpoint['time_predictor_state_dict'] is not None :
        time_predictor.load_state_dict(checkpoint['time_predictor_state_dict'])

    if decoder is not None and 'decoder_state_dict' in checkpoint and checkpoint['decoder_state_dict'] is not None:
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

    return model, classifier, time_predictor, decoder


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)  # gpu
    

def evaluate_mc(label, pred, n_class):
    """
    Evaluate metrics for multiclass or binary classification.
    Args:
        label (np.array): True labels, shape [n_samples].
        pred (np.array): Predicted probabilities, shape [n_samples, num_classes].
        n_class (int): Number of classes.
    Returns:
        tuple: acc, precision, recall, f1, auroc, auprc
    """
    y_true = label
    y_prob = pred
    y_pred_labels = np.argmax(y_prob, axis=-1)

    # Accuracy
    acc = accuracy_score(y_true, y_pred_labels)

    # Initialize metrics for safety in case of errors
    precision, recall, f1, auroc, auprc = 0.0, 0.0, 0.0, 0.0, 0.0

    try:
        if n_class > 2:  # Multiclass (e.g., PAM with n_class=8)
            f1 = f1_score(y_true, y_pred_labels, average='weighted', zero_division=0)
            precision = precision_score(y_true, y_pred_labels, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred_labels, average='weighted', zero_division=0)
            
            y_true_one_hot = label_binarize(y_true, classes=range(n_class))
            # Ensure y_true_one_hot has n_class columns even if some classes are not in this batch
            if y_true_one_hot.shape[1] < n_class:
                 y_true_one_hot_full = np.zeros((y_true_one_hot.shape[0], n_class))
                 y_true_one_hot_full[:, :y_true_one_hot.shape[1]] = y_true_one_hot
                 y_true_one_hot = y_true_one_hot_full

            auroc = roc_auc_score(y_true_one_hot, y_prob, multi_class='ovr', average='weighted')
            auprc = average_precision_score(y_true_one_hot, y_prob, average='weighted')

        elif n_class == 2:  # Binary classification
            # For binary, argmax gives 0 or 1, which is suitable for f1, precision, recall.
            f1 = f1_score(y_true, y_pred_labels, average='binary', zero_division=0)
            precision = precision_score(y_true, y_pred_labels, average='binary', zero_division=0)
            recall = recall_score(y_true, y_pred_labels, average='binary', zero_division=0)

            # AUROC/AUPRC use probability of the positive class (class 1)
            y_prob_positive_class = y_prob[:, 1]
            auroc = roc_auc_score(y_true, y_prob_positive_class)
            auprc = average_precision_score(y_true, y_prob_positive_class)
        else: # n_class < 2, not typical for classification, return zeros
            pass

    except ValueError as e:
        print(f"ValueError during metric calculation: {e}. Returning 0 for affected metrics.")
        # AUROC/AUPRC are often sensitive to single-class presence in batches
        # Other metrics might also fail if y_true or y_pred_labels are problematic

    return acc, precision, recall, f1, auroc, auprc

def evaluate_ml(true, pred):
    """
    Evaluate metrics for multilabel classification.
    Args:
        true (np.array): True binary labels, shape [n_samples, n_classes].
        pred (np.array): Predicted probabilities, shape [n_samples, n_classes].
    Returns:
        tuple: acc, precision, recall, f1, auroc, auprc
    """
    y_true_ml = true
    y_prob_ml = pred
    # For F1, precision, recall, accuracy, convert probabilities to binary predictions
    y_pred_labels_ml = np.array(y_prob_ml > 0.5, dtype=int)

    # Accuracy for multilabel can be exact match ratio or hamming score based.
    # sklearn.metrics.accuracy_score is subset accuracy (exact match)
    acc = accuracy_score(y_true_ml, y_pred_labels_ml)
    
    precision, recall, f1, auroc, auprc = 0.0, 0.0, 0.0, 0.0, 0.0

    try:
        # F1 Score: "macro" for multilabel
        f1 = f1_score(y_true_ml, y_pred_labels_ml, average='macro', zero_division=0)
        
        # Precision & Recall: "weighted" for multilabel
        precision = precision_score(y_true_ml, y_pred_labels_ml, average='weighted', zero_division=0)
        recall = recall_score(y_true_ml, y_pred_labels_ml, average='weighted', zero_division=0)
        
        # AUROC & AUPRC for multilabel: use raw probabilities
        # Rule: average_precision_score(all_labels, all_probs)
        # Rule: roc_auc_score(all_labels, all_probs)
        # Default for roc_auc_score for multilabel is 'macro' if not specified
        # Default for average_precision_score for multilabel is 'macro' if not specified
        auroc = roc_auc_score(y_true_ml, y_prob_ml, average='macro') # Using 'macro' as it's common and was in original
        auprc = average_precision_score(y_true_ml, y_prob_ml, average='macro') # Using 'macro'

    except ValueError as e:
        print(f"ValueError during multilabel metric calculation: {e}. Returning 0 for affected metrics.")

    return acc, precision, recall, f1, auroc, auprc
