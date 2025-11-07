import os
import sys
sys.path.append("..")

import time
import datetime
import argparse
import numpy as np
import pandas as pd
import random
from random import SystemRandom
from sklearn import model_selection

import torch
import torch.nn as nn
import torch.optim as optim

import lib.utils as utils
from lib.parse_datasets import parse_datasets
from model.tPatchGNN import *
import wandb

parser = argparse.ArgumentParser('IMTS Forecasting')

parser.add_argument('--state', type=str, default='def')
parser.add_argument('-n',  type=int, default=int(1e8), help="Size of the dataset")
parser.add_argument('--hop', type=int, default=1, help="hops in GNN")
parser.add_argument('--nhead', type=int, default=1, help="heads in Transformer")
parser.add_argument('--tf_layer', type=int, default=1, help="# of layer in Transformer")
parser.add_argument('--nlayer', type=int, default=1, help="# of layer in TSmodel")
parser.add_argument('--epoch', type=int, default=1000, help="training epoches")
parser.add_argument('--patience', type=int, default=20, help="patience for early stop")
parser.add_argument('--history', type=int, default=24, help="number of hours (months for ushcn and ms for activity) as historical window")
parser.add_argument('-ps', '--patch_size', type=float, default=24, help="window size for a patch")
parser.add_argument('--stride', type=float, default=24, help="period stride for patch sliding")
parser.add_argument('--logmode', type=str, default="a", help='File mode of logging.')

parser.add_argument('--lr',  type=float, default=1e-3, help="Starting learning rate.")
parser.add_argument('--w_decay', type=float, default=0.0, help="weight decay.")
parser.add_argument('-b', '--batch_size', type=int, default=32)

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('--seed', type=int, default=1, help="Random seed")
parser.add_argument('--dataset', type=str, default='physionet', help="Dataset to load. Available: physionet, mimic, ushcn")
parser.add_argument("--num_classes", type=int, default=2, help="Number of classes for classification task. Default: 2 (binary classification)")
parser.add_argument("--task_type", type=str, help="Type of task: [binary, multiclass, multilabel]")
parser.add_argument("--early_stop_metric", type=str, default="f1", help="Metric for early stopping. Default: auroc")
parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")

# Irregularity parameters for PAM dataset
parser.add_argument('--irregularity', action='store_true', help="Enable irregularity for PAM dataset")
parser.add_argument('--irregularity_rate', type=float, default=0.0, help="Rate of irregularity for PAM dataset (fraction of sensors to drop)")
parser.add_argument('--irregularity_type', type=str, default='random', choices=['random', 'fixed'], help="Type of irregularity for PAM dataset ('random' or 'fixed')")

# value 0 means using original time granularity, Value 1 means quantization by 1 hour, 
# value 0.1 means quantization by 0.1 hour = 6 min, value 0.016 means quantization by 0.016 hour = 1 min
parser.add_argument('--quantization', type=float, default=0.0, help="Quantization on the physionet dataset.")
parser.add_argument('--model', type=str, default='tPatchGNN', help="Model name")
parser.add_argument('--outlayer', type=str, default='Linear', help="Model name")
parser.add_argument('-hd', '--hid_dim', type=int, default=64, help="Number of units per hidden layer")
parser.add_argument('-td', '--te_dim', type=int, default=10, help="Number of units for time encoding")
parser.add_argument('-nd', '--node_dim', type=int, default=10, help="Number of units for node vectors")
parser.add_argument('--gpu', type=str, default='0', help='which gpu to use.')

args = parser.parse_args()
args.npatch = int(np.ceil((args.history - args.patch_size) / args.stride)) + 1 # (window size for a patch)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
file_name = os.path.basename(__file__)[:-3]
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.PID = os.getpid()
print("PID, device:", args.PID, args.device)

#####################################################################################################

if __name__ == '__main__':
	utils.setup_seed(args.seed)

	experimentID = args.load
	if experimentID is None:
		# Make a new experiment ID based on current system time in year-month-day-hour-minute format
		experimentID = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
	ckpt_path = os.path.join(args.save, "experiment_" + str(args.dataset) + '-' + experimentID)
	
	input_command = sys.argv
	ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
	if len(ind) == 1:
		ind = ind[0]
		input_command = input_command[:ind] + input_command[(ind+2):]
	input_command = " ".join(input_command)

	# utils.makedirs("results/")

	##################################################################
	data_obj = parse_datasets(args, patch_ts=True)
	input_dim = data_obj["input_dim"]
	
	### Model setting ###
	args.ndim = input_dim
	model = tPatchGNN(args).to(args.device)

	##################################################################
	
	# # Load checkpoint and evaluate the model
	# if args.load is not None:
	# 	utils.get_ckpt_model(ckpt_path, model, args.device)
	# 	exit()

	##################################################################

	wandb.init(
		mode="online" if args.wandb else "offline",  # Toggle this based on your needs
		project="t-PatchGNN",  # Use a consistent project name
		config=vars(args),
		name=f"{args.dataset}-{args.irregularity_type}-{100*args.irregularity_rate:.0f}%-dim{args.hid_dim}-p{args.patch_size}-s{args.stride}-seed{args.seed}",
	)
	wandb.config.update({"input_command": input_command})
	wandb.config.update({"start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
	wandb.config.update({"script_path": os.path.abspath(__file__)})

	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	num_batches = data_obj["n_train_batches"] # n_sample / batch_size
	print("n_train_batches:", num_batches)

	early_stop_best_metric = float("-inf")
	test_res = None
	for itr in range(args.epoch):
		st = time.time()

		### Training ###
		train_loss = 0
		model.train()
		for _ in range(num_batches):
			optimizer.zero_grad()
			batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
			train_res = compute_all_losses(model, batch_dict, task_type=args.task_type)
			train_res["loss"].backward()
			optimizer.step()
			train_loss += train_res["loss"].item()
			# print(train_res["loss"].item())
		wandb.log({
			"epoch": itr,
			"train_loss": train_loss / num_batches,
		}, step=itr)

		### Validation ###
		model.eval()
		with torch.no_grad():
			val_res = evaluation(model, data_obj["val_dataloader"], data_obj["n_val_batches"], task_type=args.task_type)
			
			# Log training and validation metrics
			wandb.log({
				"epoch": itr,
				"val_loss": val_res["loss"],
				"val_accuracy": val_res["accuracy"],
				"val_precision": val_res["precision"],
				"val_recall": val_res["recall"],
				"val_f1": val_res["f1"],
				"val_auprc": val_res["auprc"],
				"val_auroc": val_res["auroc"],
				"time_spent": time.time()-st,
			}, step=itr)
			print({
				"epoch": itr,
				"val_loss": val_res["loss"],
				"val_accuracy": val_res["accuracy"],
				"val_precision": val_res["precision"],
				"val_recall": val_res["recall"],
				"val_f1": val_res["f1"],
				"val_auprc": val_res["auprc"],
				"val_auroc": val_res["auroc"],
				"time_spent": time.time()-st,
			})

			if(val_res[args.early_stop_metric] > early_stop_best_metric):
				early_stop_best_metric = val_res[args.early_stop_metric]
				best_iter = itr
				# run test dataset on the current best model
				test_res = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"], task_type=args.task_type)
				# log the test results
				wandb.log({
					"epoch": itr,
					"test_loss": test_res["loss"],
					"test_accuracy": test_res["accuracy"],
					"test_precision": test_res["precision"],
					"test_recall": test_res["recall"],
					"test_f1": test_res["f1"],
					"test_auprc": test_res["auprc"],
					"test_auroc": test_res["auroc"],
				}, step=itr)
				print({
					"epoch": itr,
					"test_loss": test_res["loss"],
					"test_accuracy": test_res["accuracy"],
					"test_precision": test_res["precision"],
					"test_recall": test_res["recall"],
					"test_f1": test_res["f1"],
					"test_auprc": test_res["auprc"],
					"test_auroc": test_res["auroc"],
				})
				# save the model
				utils.save_checkpoint(state=model.state_dict(), save=ckpt_path, epoch=itr)
    
		if(itr - best_iter >= args.patience):
			print("Exp has been early stopped!")
			if test_res is not None:
				wandb.run.summary["best_test_epoch"] = best_iter
				wandb.run.summary["best_test_loss"] = test_res["loss"]
				wandb.run.summary["best_test_accuracy"] = test_res["accuracy"]
				wandb.run.summary["best_test_precision"] = test_res["precision"]
				wandb.run.summary["best_test_recall"] = test_res["recall"]
				wandb.run.summary["best_test_f1"] = test_res["f1"]
				wandb.run.summary["best_test_auprc"] = test_res["auprc"]
			wandb.run.summary["early_stopped"] = True
			break

	with open("results.log", "a") as f:
		f.write(f"[{args.dataset}-{args.irregularity_type}-{100*args.irregularity_rate:.0f}%]: {test_res["accuracy"] * 100:.3f} {test_res["precision"] * 100:.3f} {test_res["recall"] * 100:.3f} {test_res["f1"] * 100:.3f}\n")
	wandb.finish()
	sys.exit(0)


