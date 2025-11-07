import gc
import numpy as np
import sklearn as sk
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu


import lib.utils as utils
from lib.utils import get_device

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Independent


def one_hot(labels, num_classes):
	"""
	Convert labels to one-hot encoding
	Args:
		labels: Tensor or numpy array of shape (n_samples,)
		num_classes: Number of classes
	Returns:
		Tensor or numpy array of shape (n_samples, num_classes)
	"""
	if isinstance(labels, torch.Tensor):
		labels = labels.long()
		return torch.eye(num_classes, device=labels.device)[labels]
	else:
		labels = labels.astype(np.int64)
		return np.eye(num_classes)[labels]

def gaussian_log_likelihood(mu_2d, data_2d, obsrv_std, indices = None):
	n_data_points = mu_2d.size()[-1]

	if n_data_points > 0:
		gaussian = Independent(Normal(loc = mu_2d, scale = obsrv_std.repeat(n_data_points)), 1)
		log_prob = gaussian.log_prob(data_2d) 
		log_prob = log_prob / n_data_points 
	else:
		log_prob = torch.zeros([1]).to(get_device(data_2d)).squeeze()
	return log_prob


def poisson_log_likelihood(masked_log_lambdas, masked_data, indices, int_lambdas):
	# masked_log_lambdas and masked_data 
	n_data_points = masked_data.size()[-1]

	if n_data_points > 0:
		log_prob = torch.sum(masked_log_lambdas) - int_lambdas[indices]
		#log_prob = log_prob / n_data_points
	else:
		log_prob = torch.zeros([1]).to(get_device(masked_data)).squeeze()
	return log_prob



def compute_binary_CE_loss(label_predictions, mortality_label):
	mortality_label = mortality_label.reshape(-1)

	if len(label_predictions.size()) == 1:
		label_predictions = label_predictions.unsqueeze(0)
 
	n_traj_samples = label_predictions.size(0)
	label_predictions = label_predictions.reshape(n_traj_samples, -1)
	
	idx_not_nan = ~torch.isnan(mortality_label)
	if len(idx_not_nan) == 0.:
		print("All are labels are NaNs!")
		ce_loss = torch.Tensor(0.).to(get_device(mortality_label))

	label_predictions = label_predictions[:,idx_not_nan]
	mortality_label = mortality_label[idx_not_nan]

	if torch.sum(mortality_label == 0.) == 0 or torch.sum(mortality_label == 1.) == 0:
		print("Warning: all examples in a batch belong to the same class -- please increase the batch size.")

	assert(not torch.isnan(label_predictions).any())
	assert(not torch.isnan(mortality_label).any())

	# For each trajectory, we get n_traj_samples samples from z0 -- compute loss on all of them
	mortality_label = mortality_label.repeat(n_traj_samples, 1)
	ce_loss = nn.BCEWithLogitsLoss()(label_predictions, mortality_label)

	# divide by number of patients in a batch
	ce_loss = ce_loss / n_traj_samples
	return ce_loss


def compute_multiclass_CE_loss(label_predictions, true_label, mask):
	if (len(label_predictions.size()) == 3):
		label_predictions = label_predictions.unsqueeze(0)

	n_traj_samples, n_traj, n_tp, n_dims = label_predictions.size()

	# assert(not torch.isnan(label_predictions).any())
	# assert(not torch.isnan(true_label).any())

	# For each trajectory, we get n_traj_samples samples from z0 -- compute loss on all of them
	true_label = true_label.repeat(n_traj_samples, 1, 1)

	label_predictions = label_predictions.reshape(n_traj_samples * n_traj * n_tp, n_dims)
	true_label = true_label.reshape(n_traj_samples * n_traj * n_tp, n_dims)

	# choose time points with at least one measurement
	mask = torch.sum(mask, -1) > 0

	# repeat the mask for each label to mark that the label for this time point is present
	pred_mask = mask.repeat(n_dims, 1,1).permute(1,2,0)

	label_mask = mask
	pred_mask = pred_mask.repeat(n_traj_samples,1,1,1)
	label_mask = label_mask.repeat(n_traj_samples,1,1,1)

	pred_mask = pred_mask.reshape(n_traj_samples * n_traj * n_tp,  n_dims)
	label_mask = label_mask.reshape(n_traj_samples * n_traj * n_tp, 1)

	if (label_predictions.size(-1) > 1) and (true_label.size(-1) > 1):
		assert(label_predictions.size(-1) == true_label.size(-1))
		# targets are in one-hot encoding -- convert to indices
		_, true_label = true_label.max(-1)

	res = []
	for i in range(true_label.size(0)):
		pred_masked = torch.masked_select(label_predictions[i], pred_mask[i].bool())
		labels = torch.masked_select(true_label[i], label_mask[i].bool())
	
		pred_masked = pred_masked.reshape(-1, n_dims)

		if (len(labels) == 0):
			continue

		ce_loss = nn.CrossEntropyLoss()(pred_masked, labels.long())
		res.append(ce_loss)

	ce_loss = torch.stack(res, 0).to(get_device(label_predictions))
	ce_loss = torch.mean(ce_loss)
	# # divide by number of patients in a batch
	# ce_loss = ce_loss / n_traj_samples
	return ce_loss


def compute_masked_likelihood(mu, data, mask, likelihood_func):
	# Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
	n_traj_samples, n_traj, n_timepoints, n_dims = data.size()

	res = []
	for i in range(n_traj_samples):
		for k in range(n_traj):
			for j in range(n_dims):
				data_masked = torch.masked_select(data[i,k,:,j], mask[i,k,:,j].bool())
				
				#assert(torch.sum(data_masked == 0.) < 10)

				mu_masked = torch.masked_select(mu[i,k,:,j], mask[i,k,:,j].bool())
				# Take mean over the length of sequence
				log_prob = likelihood_func(mu_masked, data_masked, indices = (i,k,j))
				# print(mu_masked.shape, log_prob)
				res.append(log_prob)
	# shape: [n_traj*n_traj_samples, 1]

	res = torch.stack(res, 0).to(get_device(data))
	res = res.reshape((n_traj_samples, n_traj, n_dims))
	# Take mean over the number of dimensions
	res = torch.mean(res, -1) # !!!!!!!!!!! changed from sum to mean
	res = res.transpose(0,1) # (B, 1)
	return res


def masked_gaussian_log_density(mu, data, obsrv_std, mask = None):
	# these cases are for plotting through plot_estim_density
	if (len(mu.size()) == 3):
		# add additional dimension for gp samples
		mu = mu.unsqueeze(0)

	if (len(data.size()) == 2):
		# add additional dimension for gp samples and time step
		data = data.unsqueeze(0).unsqueeze(2)
	elif (len(data.size()) == 3):
		# add additional dimension for gp samples
		data = data.unsqueeze(0)

	n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()

	assert(data.size()[-1] == n_dims)

	# Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
	if mask is None:
		mu_flat = mu.reshape(n_traj_samples*n_traj, n_timepoints * n_dims)
		n_traj_samples, n_traj, n_timepoints, n_dims = data.size()
		data_flat = data.reshape(n_traj_samples*n_traj, n_timepoints * n_dims)
	
		res = gaussian_log_likelihood(mu_flat, data_flat, obsrv_std)
		res = res.reshape(n_traj_samples, n_traj).transpose(0,1)
	else:
		# Compute the likelihood per patient so that we don't priorize patients with more measurements
		func = lambda mu, data, indices: gaussian_log_likelihood(mu, data, obsrv_std = obsrv_std, indices = indices)
		res = compute_masked_likelihood(mu, data, mask, func)
	return res



def mse(mu, data, indices = None):
	n_data_points = mu.size()[-1]

	if n_data_points > 0:
		mse = nn.MSELoss()(mu, data)
	else:
		mse = torch.zeros([1]).to(get_device(data)).squeeze()
	return mse


def compute_mse(mu, data, mask = None):
	"""
	these cases are for plotting through plot_estim_density
	mu = pred
	data = groud_truth
	"""
	if (len(mu.size()) == 3):
		# add additional dimension for gp samples
		mu = mu.unsqueeze(0)

	if (len(data.size()) == 2):
		# add additional dimension for gp samples and time step
		data = data.unsqueeze(0).unsqueeze(2)
	elif (len(data.size()) == 3):
		# add additional dimension for gp samples
		data = data.unsqueeze(0)

	n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()
	assert(data.size()[-1] == n_dims)

	# Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
	if mask is None:
		mu_flat = mu.reshape(n_traj_samples*n_traj, n_timepoints * n_dims)
		n_traj_samples, n_traj, n_timepoints, n_dims = data.size()
		data_flat = data.reshape(n_traj_samples*n_traj, n_timepoints * n_dims)
		res = mse(mu_flat, data_flat)
	else:
		# Compute the likelihood per patient so that we don't priorize patients with more measurements
		# (n_traj_samples, n_traj, n_timepoints, n_dims) -> (n_traj (bs), 1)
		res = compute_masked_likelihood(mu, data, mask, mse) 
		# print(mu.shape, res.shape)
	return res


def compute_poisson_proc_likelihood(truth, pred_y, info, mask = None):
	# Compute Poisson likelihood
	# https://math.stackexchange.com/questions/344487/log-likelihood-of-a-realization-of-a-poisson-process
	# Sum log lambdas across all time points
	if mask is None:
		poisson_log_l = torch.sum(info["log_lambda_y"], 2) - info["int_lambda"]
		# Sum over data dims
		poisson_log_l = torch.mean(poisson_log_l, -1)
	else:
		# Compute likelihood of the data under the predictions
		truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
		mask_repeated = mask.repeat(pred_y.size(0), 1, 1, 1)

		# Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
		int_lambda = info["int_lambda"]
		f = lambda log_lam, data, indices: poisson_log_likelihood(log_lam, data, indices, int_lambda)
		poisson_log_l = compute_masked_likelihood(info["log_lambda_y"], truth_repeated, mask_repeated, f)
		poisson_log_l = poisson_log_l.permute(1,0)
		# Take mean over n_traj
		#poisson_log_l = torch.mean(poisson_log_l, 1)
		
	# poisson_log_l shape: [n_traj_samples, n_traj]
	return poisson_log_l
	

def compute_error(truth, pred_y, mask, func, reduce, norm_dict=None):
	# pred_y shape [n_traj_samples, n_batch, n_tp, n_dim]
	# truth shape  [n_bacth, n_tp, n_dim] or [B, L, n_dim]

	if len(pred_y.shape) == 3: 
		pred_y = pred_y.unsqueeze(dim=0)
	n_traj_samples, n_batch, n_tp, n_dim = pred_y.size()
	truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
	mask = mask.repeat(pred_y.size(0), 1, 1, 1)

	if(func == "MSE"):
		error = ((truth_repeated - pred_y)**2) * mask # (n_traj_samples, n_batch, n_tp, n_dim)
	elif(func == "MAE"):
		error = torch.abs(truth_repeated - pred_y) * mask # (n_traj_samples, n_batch, n_tp, n_dim)
	elif(func == "MAPE"):
		if(norm_dict == None):
			mask = (truth_repeated != 0) * mask
			truth_div = truth_repeated + (truth_repeated == 0) * 1e-8
			error = torch.abs(truth_repeated - pred_y) / truth_div * mask
		else:
			data_max = norm_dict["data_max"]
			data_min = norm_dict["data_min"]
			truth_rescale = truth_repeated * (data_max - data_min) + data_min
			pred_y_rescale = pred_y * (data_max - data_min) + data_min
			mask = (truth_rescale != 0) * mask
			truth_rescale_div = truth_rescale + (truth_rescale == 0) * 1e-8
			error = torch.abs(truth_rescale - pred_y_rescale) / truth_rescale_div * mask
	else:
		raise Exception("Error function not specified")

	error_var_sum = error.reshape(-1, n_dim).sum(dim=0) # (n_dim, )
	mask_count = mask.reshape(-1, n_dim).sum(dim=0) # (n_dim, )

	if(reduce == "mean"):
		### 1. Compute avg error of each variable first 
		### 2. Compute avg error along the variables 
		error_var_avg = error_var_sum / (mask_count + 1e-8) # (n_dim, ) 
		# print("error_var_avg", error_var_avg.max().item(), error_var_avg.min().item(), (1.0*error_var_avg).mean().item())
		n_avai_var = torch.count_nonzero(mask_count)
		error_avg = error_var_avg.sum() / n_avai_var # (1, )
		
		return error_avg # a scalar (1, ) 
	
	elif(reduce == "sum"):
		# (n_dim, ) , (n_dim, ) 
		return error_var_sum, mask_count  

	else:
		raise Exception("Reduce argument not specified!")


def compute_all_losses(model, batch_dict, task_type=None):
	"""
	Compute losses based on model predictions
	
	Args:
		model: The model to make predictions
		batch_dict: Dictionary containing batch data
		task_type: str, 'forecasting' or 'classification'. If None, determine automatically.
	
	Returns:
		Dictionary with computed losses and metrics
	"""
	# Determine the task type if not specified
	if task_type is None:
		task_type = 'classification' if 'label' in batch_dict else 'forecasting'
	
	results = {}
	
	if task_type == 'forecasting':
		# Forecasting task
		pred_y = model.forecasting(
			batch_dict["tp_to_predict"],
			batch_dict["observed_data"], 
			batch_dict["observed_tp"], 
			batch_dict["observed_mask"]
		)
		
		# Compute regression metrics
		mse = compute_error(batch_dict["data_to_predict"], pred_y, 
							mask=batch_dict["mask_predicted_data"], 
							func="MSE", reduce="mean")
		rmse = torch.sqrt(mse)
		mae = compute_error(batch_dict["data_to_predict"], pred_y, 
							mask=batch_dict["mask_predicted_data"], 
							func="MAE", reduce="mean")
		
		# Use MSE as the primary loss
		loss = mse
		
		results["loss"] = loss
		results["mse"] = mse.item()
		results["rmse"] = rmse.item()
		results["mae"] = mae.item()
		
	else:  # classification task, task_type in ["binary", "multiclass", "multilabel"]
		# Classification task
		# print(batch_dict["observed_data"].shape, batch_dict["observed_tp"].shape, batch_dict["observed_mask"].shape)
		class_logits = model.classify(
			batch_dict["observed_data"],
			batch_dict["observed_tp"],
			batch_dict["observed_mask"]
		)
		
		# Get labels
		labels = batch_dict["label"]
		
		if task_type in ["binary", "multiclass"]:
			loss = nn.CrossEntropyLoss()(class_logits, labels.long())
		elif task_type == "multilabel":
			loss = nn.BCEWithLogitsLoss()(class_logits, labels.float())
		
		# Compute metrics
		with torch.no_grad():
			if task_type in ["binary", "multiclass"]:
				# For binary and multiclass classification
				probs = torch.softmax(class_logits, dim=1)
				_, predictions = torch.max(class_logits, dim=1)
				correct = (predictions == labels.long()).float().sum()
				accuracy = correct / labels.size(0)
				# print("probs", probs)
				# print("predictions", predictions, "labels", labels)
			elif task_type == "multilabel":
				probs = torch.sigmoid(class_logits)
				predictions = (probs > 0.5).float()
				correct = (predictions == labels.float()).float().sum()
				accuracy = correct / (labels.size(0) * labels.size(1))
			else:
				raise ValueError("Invalid task type for classification.")
		
		results["loss"] = loss
		results["accuracy"] = accuracy.item()
	
	return results

def evaluation(model, dataloader, n_batches, task_type=None):
	"""
	Evaluate model on dataloader
	
	Args:
		model: The model to evaluate
		dataloader: DataLoader containing evaluation data
		n_batches: Number of batches to evaluate
		task_type: str, 'forecasting' or 'classification'. If None, determine automatically.
	
	Returns:
		Dictionary with averaged metrics
	"""
	# Check the first batch to determine task_type if not specified
	if task_type is None:
		batch_dict = utils.get_next_batch(dataloader)
		task_type = 'classification' if 'label' in batch_dict else 'forecasting'
	
	if task_type == 'forecasting':
		return evaluation_forecasting(model, dataloader, n_batches)
	else:
		return evaluation_classification(model, dataloader, n_batches, task_type)

def evaluation_classification(model, dataloader, n_batches, task_type):
	"""
	Evaluate classification model on dataloader
	"""
	model.eval()  # Set model to evaluation mode
	
	total_samples = 0
	total_results = {
		"loss": 0,
		"accuracy": 0,
		"precision": 0,
		"recall": 0,
		"f1": 0,
		"auprc": 0
	}
	
	# For more robust metrics calculation across the entire dataset
	all_labels = []
	all_predictions = []
	all_probs = []
	
	with torch.no_grad():  # Disable gradient computation
		for _ in range(n_batches):
			batch_dict = utils.get_next_batch(dataloader)
			batch_size = batch_dict["observed_data"].size(0)
			
			# Get batch results
			results = compute_all_losses(model, batch_dict, task_type=task_type)
			
			# Accumulate weighted metrics for simple averaging
			for key in total_results:
				if key not in results:
					continue
				total_results[key] += results[key] * batch_size
			
			total_samples += batch_size
			
			# Store predictions, probabilities, and labels for dataset-level metrics
			class_logits = model.classify(
				batch_dict["observed_data"],
				batch_dict["observed_tp"],
				batch_dict["observed_mask"]
			)
			if task_type in ["binary", "multiclass"]:
				probs = torch.softmax(class_logits, dim=1)
				_, predictions = torch.max(class_logits, dim=1)
			elif task_type == "multilabel":
				probs = torch.sigmoid(class_logits)
				predictions = (probs > 0.5).float()
			
			if batch_dict["label"].device.type == 'cuda':
				all_labels.extend(batch_dict["label"].cpu().numpy().tolist())
				all_predictions.extend(predictions.cpu().numpy().tolist())
				all_probs.append(probs.cpu().numpy())
			else:
				all_labels.extend(batch_dict["label"].numpy().tolist())
				all_predictions.extend(predictions.numpy().tolist())
				all_probs.append(probs.numpy())
	
	# Convert lists to numpy arrays for dataset-level metrics

	# from collections import Counter
	# print(all_labels, file=open("all_labels.txt", "w"))
	# print(all_predictions, file=open("all_predictions.txt", "w"))
	# print("all preds: ", Counter(all_predictions))
	# print("all labels: ", Counter(all_labels))

	all_labels = np.array(all_labels)
	all_predictions = np.array(all_predictions)
	all_probs = np.vstack(all_probs) if all_probs else np.array([])
	
	# Average the results
	for key in total_results:
		total_results[key] /= total_samples
	
	# Compute dataset-level metrics (more accurate than averaging batch metrics)
	total_results["f1"] = sk.metrics.f1_score(all_labels, all_predictions, average=(
		"weighted" if task_type == "multiclass" else ("binary" if task_type == "binary" else "macro")
	), zero_division=0)
	total_results["precision"] = sk.metrics.precision_score(all_labels, all_predictions, average=("binary" if task_type == "binary" else "weighted"), zero_division=0)
	total_results["recall"] = sk.metrics.recall_score(all_labels, all_predictions, average=("binary" if task_type == "binary" else "weighted"), zero_division=0)
	
	if task_type == "multiclass":
		total_results["auprc"] = sk.metrics.average_precision_score(one_hot(all_labels, num_classes=all_probs.shape[-1]), all_probs)
		total_results["auroc"] = sk.metrics.roc_auc_score(one_hot(all_labels, num_classes=all_probs.shape[-1]), all_probs)
	elif task_type == "binary":
		total_results["auprc"] = sk.metrics.average_precision_score(all_labels, all_probs[:, 1])
		total_results["auroc"] = sk.metrics.roc_auc_score(all_labels, all_probs[:, 1])
	elif task_type == "multilabel":
		total_results["auprc"] = sk.metrics.average_precision_score(all_labels, all_probs)
		total_results["auroc"] = sk.metrics.roc_auc_score(all_labels, all_probs)
	else: # raise error
		raise ValueError("Invalid task type for dataset-level metrics.")
	
	return total_results

def evaluation_forecasting(model, dataloader, n_batches):
	# This is the original evaluation function for forecasting
	n_eval_samples = 0
	n_eval_samples_mape = 0
	total_results = {}
	total_results["loss"] = 0
	total_results["mse"] = 0
	total_results["mae"] = 0
	total_results["rmse"] = 0
	total_results["mape"] = 0

	for _ in range(n_batches):
		batch_dict = utils.get_next_batch(dataloader)

		pred_y = model.forecasting(batch_dict["tp_to_predict"], 
			batch_dict["observed_data"], batch_dict["observed_tp"], 
			batch_dict["observed_mask"]) 
		
		# (n_dim, ) , (n_dim, ) 
		se_var_sum, mask_count = compute_error(batch_dict["data_to_predict"], pred_y, mask=batch_dict["mask_predicted_data"], func="MSE", reduce="sum") # a vector

		ae_var_sum, _ = compute_error(batch_dict["data_to_predict"], pred_y, mask = batch_dict["mask_predicted_data"], func="MAE", reduce="sum") # a vector

		ape_var_sum, mask_count_mape = compute_error(batch_dict["data_to_predict"], pred_y, mask = batch_dict["mask_predicted_data"], func="MAPE", reduce="sum") # a vector

		# add a tensor (n_dim, )
		total_results["loss"] += se_var_sum
		total_results["mse"] += se_var_sum
		total_results["mae"] += ae_var_sum
		total_results["mape"] += ape_var_sum
		n_eval_samples += mask_count
		n_eval_samples_mape += mask_count_mape

	# ... existing averaging code ...
	n_avai_var = torch.count_nonzero(n_eval_samples)
	n_avai_var_mape = torch.count_nonzero(n_eval_samples_mape)
	
	### 1. Compute avg error of each variable first
	### 2. Compute avg error along the variables 
	total_results["loss"] = (total_results["loss"] / (n_eval_samples + 1e-8)).sum() / n_avai_var
	total_results["mse"] = (total_results["mse"] / (n_eval_samples + 1e-8)).sum() / n_avai_var
	total_results["mae"] = (total_results["mae"] / (n_eval_samples + 1e-8)).sum() / n_avai_var
	total_results["rmse"] = torch.sqrt(total_results["mse"])
	total_results["mape"] = (total_results["mape"] / (n_eval_samples_mape + 1e-8)).sum() / n_avai_var_mape

	for key, var in total_results.items(): 
		if isinstance(var, torch.Tensor):
			var = var.item()
		total_results[key] = var

	return total_results

