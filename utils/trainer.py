import torch
import numpy as np
from utils.utils import CheckpointSaver, one_hot
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
)
from utils.utils import thresh_max_f1
from torch.nn import functional as F


class Trainer:
    """
    Trainer class for training the model
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        device,
        args,
        checkpoint_saver=None,
        scheduler=None,
    ):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_saver = checkpoint_saver
        self.best_threshold = 0.5
        self.args = args
        self.scheduler = scheduler

        if args.dataset == "PAM":
            self.task = "multiclass"
        elif args.dataset == "MIMIC3-PHE":
            self.task = "multilabel"
        else:
            self.task = "binary"

    def run(self, epochs=100, verbose=True, patience=10):
        e = 0
        train_loss = []
        eval_loss = []
        patience_count = 0
        prev_val_loss = 1e8

        while True:
            # train
            tloss = self.train_grad_acc()
            # eval
            eloss, metrics = self.eval()
            # verbose
            if verbose and e % 10 == 0:
                print(
                    {
                        "epoch": e,
                        "train_loss": tloss,
                        "eval_loss": eloss,
                        "accuracy": metrics["accuracy"],
                        "f1_score": metrics["f1"],
                        "aucroc": metrics["aucroc"],
                        "auprc": metrics["auprc"],
                        "best_threshold": metrics["best_threshold"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                    }
                )

            train_loss.append(tloss)
            eval_loss.append(eloss)

            # save model
            if self.args.dataset == "MIMIC3-IHM":
                best_model = self.checkpoint_saver.save(
                    e, self.model, self.optimizer, metrics["auprc"]
                )
            elif self.args.dataset == "PAM":
                best_model = self.checkpoint_saver.save(
                    e, self.model, self.optimizer, metrics["f1"]
                )
            else:
                best_model = self.checkpoint_saver.save(
                    e, self.model, self.optimizer, metrics["aucroc"]
                )

            if eloss < prev_val_loss:
                patience_count = 0
                prev_val_loss = eloss
            else:
                patience_count += 1

            e += 1
            if e >= epochs or patience_count >= patience:
                break

            # step the scheduler
            if self.scheduler is not None:
                if self.args.dataset.startswith("MIMIC3") or self.args.dataset == "PAM":
                    self.scheduler.step(metrics["f1"])
                else:
                    self.scheduler.step(metrics["auprc"])
        return (train_loss, eval_loss)

    def train(self):
        self.model.train()
        train_loss = []
        for i, batch in tqdm(
            enumerate(self.train_loader), total=len(self.train_loader)
        ):
            # get batch
            x = batch[0].unsqueeze(-1).to(self.device)

            mask = batch[1].to(self.device)
            timestamps = batch[2].to(self.device)
            static_features = batch[3].to(self.device)
            length = batch[4].to(self.device)
            y = batch[5].to(self.device)
            relative_timestamps = batch[6].unsqueeze(-1).to(self.device)
            # Zero the gradients
            self.optimizer.zero_grad()
            # Forward pass
            logits = (
                self.model(x, mask, timestamps, static_features, relative_timestamps)
                .clone()
                .squeeze()
            )
            # if batch size is 1, unsqueeze the logits
            if y.shape[0] == 1:
                logits = logits.unsqueeze(0)

            if self.task == "binary" or self.task == "multilabel":
                output_probs = torch.sigmoid(logits)
            else:
                output_probs = torch.softmax(logits, dim=-1)
            loss = self.criterion(logits, y)
            # Save the loss
            train_loss.append(loss.item())
            # Backward pass
            loss.backward()
            self.optimizer.step()

        # return the average loss
        return np.mean(train_loss)

    def train_grad_acc(self):
        self.model.train()
        train_loss = []
        self.optimizer.zero_grad()

        for step, batch in tqdm(
            enumerate(self.train_loader), total=len(self.train_loader)
        ):
            # get batch
            x = batch[0].unsqueeze(-1).to(self.device)
            mask = batch[1].to(self.device)
            timestamps = batch[2].to(self.device)
            static_features = batch[3].to(self.device)
            length = batch[4].to(self.device)
            y = batch[5].to(self.device)
            relative_timestamps = batch[6].unsqueeze(-1).to(self.device)
            logits = (
                self.model(x, mask, timestamps, static_features, relative_timestamps)
                .clone()
                .squeeze()
            )
            # if batch size is 1, unsqueeze the logits
            if y.shape[0] == 1:
                logits = logits.unsqueeze(0)

            if self.task == "binary" or self.task == "multilabel":
                output_probs = torch.sigmoid(logits)
            else:
                output_probs = F.softmax(logits, dim=-1)

            loss = self.criterion(logits, y)

            # Save the loss
            train_loss.append(loss.item())

            # Backward pass
            loss.backward()
            if (step + 1) % self.args.gradient_accumulation_step == 0 or step == len(
                self.train_loader
            ) - 1:
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

        # return the average loss
        return np.mean(train_loss)

    def eval(self):
        self.model.eval()
        val_loss = []
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for batch in self.val_loader:
                # get batch
                x = batch[0].unsqueeze(-1).to(self.device)
                mask = batch[1].to(self.device)
                timestamps = batch[2].to(self.device)
                static_features = batch[3].to(self.device)
                length = batch[4].to(self.device)
                y = batch[5].to(self.device)
                relative_timestamps = batch[6].unsqueeze(-1).to(self.device)
                # compute logits
                logits = (
                    self.model(
                        x, mask, timestamps, static_features, relative_timestamps
                    )
                    .clone()
                    .squeeze()
                )
                if self.task == "binary" or self.task == "multilabel":
                    output_probs = torch.sigmoid(logits).cpu().numpy()
                    y_pred.extend((output_probs > self.best_threshold).astype(int))
                else:
                    output_probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    y_pred.extend(np.argmax(output_probs, axis=1))
                loss = self.criterion(logits, y)
                val_loss.append(loss.item())
                # compute metrics
                y_true.extend(y.cpu().numpy())
                y_prob.extend(output_probs)
        # compute metrics
        self.best_threshold = (
            thresh_max_f1(
                y_true=np.array(y_true),
                y_prob=np.array(y_prob),
                n_classes=self.args.n_classes,
            )
            if self.task == "binary" or self.task == "multilabel"
            else 0.5
        )
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(
            y_true,
            y_pred,
            average=(
                "weighted"
                if self.task == "multiclass"
                else ("binary" if self.task == "binary" else "macro")
            ),
        )
        aucroc = (
            roc_auc_score(y_true=y_true, y_score=y_prob)
            if self.task == "binary" or self.task == "multilabel"
            else roc_auc_score(y_true=one_hot(y_true), y_score=y_prob)
        )
        auprc = (
            average_precision_score(y_true=y_true, y_score=y_prob)
            if self.task == "binary" or self.task == "multilabel"
            else average_precision_score(y_true=one_hot(y_true), y_score=y_prob)
        )
        precision = precision_score(
            y_true,
            y_pred,
            average=(
                "weighted"
                if self.task == "multiclass" or self.task == "multilabel"
                else "binary"
            ),
        )
        recall = recall_score(
            y_true,
            y_pred,
            average=(
                "weighted"
                if self.task == "multiclass" or self.task == "multilabel"
                else "binary"
            ),
        )

        metrics = {
            "accuracy": accuracy,
            "f1": f1,
            "aucroc": aucroc,
            "best_threshold": self.best_threshold,
            "auprc": auprc,
            "precision": precision,
            "recall": recall,
        }
        # return the average loss
        return np.mean(val_loss), metrics

    def test(self, test_loader, best_model):
        self.model = best_model
        val_loss = []
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                # get batch
                x = batch[0].unsqueeze(-1).to(self.device)
                mask = batch[1].to(self.device)
                timestamps = batch[2].to(self.device)
                static_features = batch[3].to(self.device)
                length = batch[4].to(self.device)
                y = batch[5].to(self.device)
                relative_timestamps = batch[6].unsqueeze(-1).to(self.device)
                logits = (
                    self.model(
                        x, mask, timestamps, static_features, relative_timestamps
                    )
                    .clone()
                    .squeeze()
                )
                if self.task == "binary" or self.task == "multilabel":
                    output_probs = torch.sigmoid(logits).cpu().numpy()
                    y_pred.extend((output_probs > self.best_threshold).astype(int))
                else:
                    output_probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    y_pred.extend(np.argmax(output_probs, axis=1))
                loss = self.criterion(logits, y)
                val_loss.append(loss.item())
                # compute metrics
                y_true.extend(y.cpu().numpy())
                y_prob.extend(output_probs)
        # compute metrics
        self.best_threshold = (
            thresh_max_f1(
                y_true=np.array(y_true),
                y_prob=np.array(y_prob),
                n_classes=self.args.n_classes,
            )
            if self.task == "binary" or self.task == "multilabel"
            else 0.5
        )
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(
            y_true,
            y_pred,
            average=(
                "weighted"
                if self.task == "multiclass"
                else ("binary" if self.task == "binary" else "macro")
            ),
        )
        aucroc = (
            roc_auc_score(y_true=y_true, y_score=y_prob)
            if self.task == "binary" or self.task == "multilabel"
            else roc_auc_score(y_true=one_hot(y_true), y_score=y_prob)
        )
        auprc = (
            average_precision_score(y_true=y_true, y_score=y_prob)
            if self.task == "binary" or self.task == "multilabel"
            else average_precision_score(y_true=one_hot(y_true), y_score=y_prob)
        )
        precision = precision_score(
            y_true,
            y_pred,
            average=(
                "weighted"
                if self.task == "multiclass" or self.task == "multilabel"
                else "binary"
            ),
        )
        recall = recall_score(
            y_true,
            y_pred,
            average=(
                "weighted"
                if self.task == "multiclass" or self.task == "multilabel"
                else "binary"
            ),
        )

        metrics = {
            "accuracy": accuracy,
            "f1": f1,
            "aucroc": aucroc,
            "best_threshold": self.best_threshold,
            "auprc": auprc,
            "precision": precision,
            "recall": recall,
        }
        # return the average loss
        return np.mean(val_loss), metrics
