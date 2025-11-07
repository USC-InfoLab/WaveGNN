import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import NamedTuple, List
from grafiti import grafiti_layers
from torch.nn.utils.rnn import pad_sequence
import pdb


class Batch(NamedTuple):
    r"""A single sample of the data."""

    x_time: Tensor  # B×(N+K):   the input timestamps.
    x_vals: Tensor  # B×(N+K)×D: the input values.
    x_mask: Tensor  # B×(N+K)×D: the input mask.

    y_vals: Tensor  # B×(N+K)×D: the target values.
    y_mask: Tensor  # B×(N+K)×D: teh target mask.


class Inputs(NamedTuple):
    r"""A single sample of the data."""

    t: Tensor
    x: Tensor
    t_target: Tensor


class Sample(NamedTuple):
    r"""A single sample of the data."""

    key: int
    inputs: Inputs
    targets: Tensor


def tsdm_collate(batch: list[Sample]) -> Batch:
    r"""Collate tensors into batch.

    Transform the data slightly: t, x, t_target → T, X where X[t_target:] = NAN
    """
    x_vals: list[Tensor] = []
    y_vals: list[Tensor] = []
    x_time: list[Tensor] = []
    x_mask: list[Tensor] = []
    y_mask: list[Tensor] = []

    context_x: list[Tensor] = []
    context_vals: list[Tensor] = []
    context_mask: list[Tensor] = []
    target_vals: list[Tensor] = []
    target_mask: list[Tensor] = []

    for sample in batch:
        t, x, t_target = sample.inputs
        y = sample.targets

        # get whole time interval
        sorted_idx = torch.argsort(t)

        # create a mask for looking up the target values
        mask_y = y.isfinite()
        mask_x = x.isfinite()

        # nan to zeros
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)

        x_vals.append(x[sorted_idx])
        x_time.append(t[sorted_idx])
        x_mask.append(mask_x[sorted_idx])

        y_vals.append(y)
        y_mask.append(mask_y)
        context_x.append(torch.cat([t, t_target], dim=0))

        x_vals_temp = torch.zeros_like(x)
        y_vals_temp = torch.zeros_like(y)
        context_vals.append(torch.cat([x, y_vals_temp], dim=0))
        context_mask.append(torch.cat([mask_x, y_vals_temp], dim=0))
        # context_y = torch.cat([context_vals, context_mask], dim=2)

        target_vals.append(torch.cat([x_vals_temp, y], dim=0))
        target_mask.append(torch.cat([x_vals_temp, mask_y], dim=0))
        # target_y = torch.cat([target_vals, target_mask], dim=2)

    return Batch(
        x_time=pad_sequence(context_x, batch_first=True),
        x_vals=pad_sequence(context_vals, batch_first=True, padding_value=0),
        x_mask=pad_sequence(context_mask, batch_first=True),
        y_vals=pad_sequence(target_vals, batch_first=True, padding_value=0),
        y_mask=pad_sequence(target_mask, batch_first=True),
    )


# New NamedTuple for classification batches
class BatchClassification(NamedTuple):
    x_time: Tensor       # BxL_obs: observed timestamps
    x_vals: Tensor       # BxL_obsxD: observed values
    x_mask: Tensor       # BxL_obsxD: observed mask
    # For structural compatibility with grafiti_layers.py during embedding extraction
    # these will be mostly empty/False for the "target" part of the sequence
    # but are needed for the combined sequence length L_total = L_obs + L_dummy_target
    y_mask_for_grafiti_structure: Tensor # BxL_totalxD: mask that includes a dummy target part
    class_labels: Tensor # B: class labels or BxC for multi-label


# New collate function for classification
def tsdm_collate_classification(batch: list[dict]) -> BatchClassification:
    r"""Collate tensors into a batch for classification.

    Each item in the batch list is expected to be a dictionary with keys:
    - 't_obs': Tensor of observed timestamps (L_obs,)
    - 'x_obs': Tensor of observed values (L_obs, D) - SHOULD CONTAIN NaNs for MISSING VALUES
    - 'class_label': Tensor for the class label (scalar or vector)
    """
    x_time_list: list[Tensor] = []
    x_vals_list: list[Tensor] = []
    x_mask_list: list[Tensor] = []
    y_mask_for_grafiti_list: list[Tensor] = []
    class_labels_list: list[Tensor] = []

    # Determine a minimal dummy target length, e.g., 1, if not all sequences are empty
    # This is to ensure that the grafiti_layers.py can process sequences even if they are only obs.
    # The actual target part of the masks/values will be False/0 for classification.
    dummy_target_len = 1

    for sample_dict in batch:
        t_obs = sample_dict['t_obs']
        x_obs = sample_dict['x_obs'] # Expected to have NaNs
        class_label = sample_dict['class_label']

        L_obs = t_obs.shape[0]
        D = x_obs.shape[-1]

        # Create observed mask from NaNs in x_obs
        current_x_mask_obs = ~torch.isnan(x_obs)
        current_x_vals_obs = torch.nan_to_num(x_obs, nan=0.0) # Fill NaNs for processing

        # Create dummy target tensors (empty time, zero values, false mask)
        # The length of these dummies ensures that the full sequence (obs + dummy_target) is processed by grafiti_layers
        t_dummy_target = torch.zeros(dummy_target_len, dtype=t_obs.dtype, device=t_obs.device)
        # x_dummy_target_vals = torch.zeros(dummy_target_len, D, dtype=x_obs.dtype, device=x_obs.device)
        x_dummy_target_mask = torch.zeros(dummy_target_len, D, dtype=torch.bool, device=x_obs.device)

        # Concatenate observed and dummy target parts for grafiti_layers processing
        # x_time will be the combined t_obs and t_dummy_target (effectively just t_obs if dummy times are not used in model)
        # For classification, we only care about obs_mask part of the combined mask later on.
        # The grafiti_layers expects values and masks for the *full* length (obs + target_placeholder)
        
        combined_t = torch.cat([t_obs, t_dummy_target], dim=0)
        # For x_vals passed to grafiti_layers.py, the target part should be masked out by obs_mask
        # So, we pass observed values and zeros for the dummy target part.
        combined_x_vals = torch.cat([current_x_vals_obs, torch.zeros(dummy_target_len, D, dtype=x_obs.dtype, device=x_obs.device)], dim=0)
        
        # x_mask for grafiti_layers is the obs_mask part
        # This will be used as `obs_mask` argument in grafiti_layers.get_contextualized_embeddings
        # This is the actual mask of observed values. The target part is False.
        current_x_mask_for_input = torch.cat([current_x_mask_obs, torch.zeros(dummy_target_len, D, dtype=torch.bool, device=x_obs.device)], dim=0)

        # y_mask_for_grafiti_structure is the target_mask part for grafiti_layers
        # This will be used as `target_mask` argument in grafiti_layers.get_contextualized_embeddings
        # For classification this is essentially a dummy, all False, but needs to exist for structure.
        current_y_mask_for_grafiti_structure = torch.cat([torch.zeros_like(current_x_mask_obs), x_dummy_target_mask], dim=0)

        x_time_list.append(combined_t)
        x_vals_list.append(combined_x_vals) # Has observed values + zero padding for dummy target part
        x_mask_list.append(current_x_mask_for_input) # Mask for observed part, False for dummy target part
        y_mask_for_grafiti_list.append(current_y_mask_for_grafiti_structure) # False for obs part, False for dummy target part
        class_labels_list.append(class_label)

    return BatchClassification(
        x_time=pad_sequence(x_time_list, batch_first=True, padding_value=0.0),
        x_vals=pad_sequence(x_vals_list, batch_first=True, padding_value=0.0), # Pad with 0, NaNs already handled
        x_mask=pad_sequence(x_mask_list, batch_first=True, padding_value=False), # Pad with False
        y_mask_for_grafiti_structure=pad_sequence(y_mask_for_grafiti_list, batch_first=True, padding_value=False),
        class_labels=torch.stack(class_labels_list)
    )


class GraFITi(nn.Module):

    def __init__(
        self, input_dim=41, attn_head=4, latent_dim=128, n_layers=2, device="cuda",
        num_classes=None # Add num_classes for classification
    ):
        super().__init__()
        self.dim = input_dim  # input dimensions
        self.attn_head = attn_head  # no. of attention heads
        self.latent_dim = latent_dim  # latend dimension
        self.n_layers = n_layers  # number of grafiti layers
        self.device = device  # cpu or gpu
        self.grafiti_ = grafiti_layers.grafiti_(
            self.dim, self.latent_dim, self.n_layers, self.attn_head, device=device
        )  # applying grafiti

        self.num_classes = num_classes
        if self.num_classes is not None:
            # The input to classification_head will be 3 * latent_dim based on concatenated embeddings
            self.classification_head = nn.Linear(3 * self.latent_dim, self.num_classes)

    def forward(self, x_time, x_vals, x_mask, y_mask):
        yhat = self.grafiti_(x_time, x_vals, x_mask, y_mask)
        return yhat

    def classify(self, x_time, x_vals, x_mask, y_mask_for_grafiti_structure):
        """
        Performs classification based on the contextualized embeddings from GraFITi layers.

        Args:
            x_time: Combined time points (observed + dummy target); Tensor (B, L_total)
            x_vals: Observed values (target part is dummy); Tensor (B, L_total, D)
            x_mask: Mask for actual observed values (target part is False); Tensor (B, L_total, D)
                    This corresponds to the `obs_mask` for get_contextualized_embeddings.
            y_mask_for_grafiti_structure: Dummy target mask for structural compatibility; Tensor (B, L_total, D)
                                          Corresponds to the `target_mask` for get_contextualized_embeddings.

        Returns:
            logits: Classification logits (B, num_classes)
        """
        if self.num_classes is None:
            raise RuntimeError("GraFITi model was not initialized for classification (num_classes is None).")

        # Get contextualized embeddings from the grafiti_layers
        # x_mask here is the obs_mask for the grafiti_layers.get_contextualized_embeddings
        # y_mask_for_grafiti_structure is the target_mask for get_contextualized_embeddings
        edge_emb, t_gathered, c_gathered, valid_flat_mask = self.grafiti_.get_contextualized_embeddings(
            time_points=x_time,
            values=x_vals, # Contains actual observed values and zeros for dummy target part
            obs_mask=x_mask, # Masks only the observed part of x_vals
            target_mask=y_mask_for_grafiti_structure # Masks only the dummy target part (all False)
        )
        # edge_emb, t_gathered, c_gathered are (B, K', M)
        # valid_flat_mask is (B, K') - True for valid flattened points (observed points)

        # Concatenate the embeddings
        # We are interested in the embeddings corresponding to the *observed* points.
        # The valid_flat_mask tells us which of the K' flattened points are actual observations.
        concatenated_embeddings = torch.cat([edge_emb, t_gathered, c_gathered], dim=-1)  # (B, K', 3*M)

        # Masked Average Pooling
        # We want to pool only over the valid, observed embeddings.
        # Add a dimension to valid_flat_mask for broadcasting and expand
        mask_for_pooling = valid_flat_mask.unsqueeze(-1).expand_as(concatenated_embeddings) # (B, K', 3*M)
        
        # Apply mask: zero out embeddings for non-valid points
        masked_embeddings = concatenated_embeddings * mask_for_pooling.float()
        
        # Sum valid embeddings and divide by the number of valid embeddings per batch instance
        summed_embeddings = masked_embeddings.sum(dim=1)  # (B, 3*M)
        num_valid_points = valid_flat_mask.sum(dim=1, keepdim=True).float() # (B, 1)
        # Avoid division by zero if a sample has no valid points (though unlikely with proper data)
        num_valid_points = torch.clamp(num_valid_points, min=1.0)
        
        pooled_embeddings = summed_embeddings / num_valid_points # (B, 3*M)

        logits = self.classification_head(pooled_embeddings)
        return logits
