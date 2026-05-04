# ================================================================
# it_binding_model.py 
# ================================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
import wandb

from torchmetrics.classification import (
        BinaryAUROC,
        BinaryAveragePrecision,
        BinaryAccuracy,
        BinaryPrecision,
        BinaryRecall,
        BinaryF1Score,
    )

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
)

from src.architectures.binding_predictor import DNABindingPredictor

# ================================================================
# STEP-BASED WARMUP + COSINE LR
# ================================================================

class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        # warmup
        if step <= self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]

        # cosine decay
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cos_scale = 0.5 * (1 + np.cos(np.pi * progress))
        return [base_lr * cos_scale for base_lr in self.base_lrs]



# ================================================================
#  LIT MODEL — BALANCED TRAINING, STEP LR, THRESHOLD SWEEP
# ================================================================

class LitDNABindingModel(pl.LightningModule):

    def __init__(
        self,
        protein_in_dim=512,
        d_model=128,
        dropout=0.3,
        lr=1e-4,
        weight_decay=1e-5,
        warmup_steps=1000,     # STEP-BASED WARMUP
        total_steps=None,      # COMPUTED AUTOMATICALLY IF None
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = DNABindingPredictor(
            protein_in_dim=protein_in_dim,
            d_model=d_model,
            dropout=dropout,
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps  # If None → infer automatically

        # Balanced sampling 
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        self.val_auroc = BinaryAUROC()
        self.val_auprc = BinaryAveragePrecision()

        self.val_logits = []
        self.val_targets = []

        self.test_probs = []
        self.test_targets = []

        self.best_threshold = None

    # ----------------------------------------------------------


    # ----------------------------------------------------------
    # OPTIMIZER + STEP LR SCHEDULER
    # ----------------------------------------------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        sched = WarmupCosineLR(
            opt,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step"
            }
    }



    # ----------------------------------------------------------
    def training_step(self, batch, batch_idx):
        dna, prot, mask, labels, tf_idx = batch

        prot = prot.to(dna.device, non_blocking=True)
        mask = mask.to(dna.device, non_blocking=True)
        labels = labels.to(dna.device, non_blocking=True)

        logits = self.model(dna, prot, protein_mask=mask).squeeze(-1)
        loss = self.loss_fn(logits, labels.float())

        self.log(
            "train/loss",
            loss,
            on_epoch=True,
            batch_size=dna.size(0),
            sync_dist=False,
        )

        return loss


    # ----------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        dna, prot, mask, labels, tf_idx = batch

        prot = prot.to(dna.device, non_blocking=True)
        mask = mask.to(dna.device, non_blocking=True)
        labels = labels.to(dna.device, non_blocking=True)

        logits = self.model(dna, prot, protein_mask=mask).squeeze(-1)

        loss = self.loss_fn(logits, labels.float())
        self.log(
            "val/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=dna.size(0),
            sync_dist=False,
        )

        probs = torch.sigmoid(logits)

        self.val_auroc.update(probs.detach(), labels.int())
        self.val_auprc.update(probs.detach(), labels.int())

        return loss
    
    def on_train_batch_start(self, batch, batch_idx):
        self._batch_start_time = time.time()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if hasattr(self, "_batch_start_time"):
            step_time = time.time() - self._batch_start_time

            self.log(
                "train/step_time_sec",
                step_time,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
            )

    # ----------------------------------------------------------
    # VALIDATION — threshold sweep + PR/ROC logging
    # ----------------------------------------------------------
    def on_validation_epoch_end(self):
        if not self.val_logits:
            return

        probs = torch.cat(self.val_logits, dim=0).view(-1)
        targets = torch.cat(self.val_targets, dim=0).view(-1).int()

        self.val_logits.clear()
        self.val_targets.clear()

        # AUROC / AUPRC
        auroc = self.val_auroc.compute()
        auprc = self.val_auprc.compute()

        self.log("val/roc_auc", auroc, prog_bar=True)
        self.log("val/pr_auc", auprc, prog_bar=True)

        self.val_auroc.reset()
        self.val_auprc.reset()

        # Threshold sweep
        thresholds = torch.linspace(0.0, 0.3, steps=301, device=probs.device)

        preds = probs[:, None] >= thresholds[None, :]
        targets_bool = targets.bool()[:, None]

        tp = (preds & targets_bool).sum(dim=0).float()
        fp = (preds & ~targets_bool).sum(dim=0).float()
        fn = (~preds & targets_bool).sum(dim=0).float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        best_idx = torch.argmax(f1)

        best_f1 = f1[best_idx]
        best_t = thresholds[best_idx]
        best_p = precision[best_idx]
        best_r = recall[best_idx]

        self.best_threshold = best_t
        self.log("val/best_f1", best_f1, prog_bar=True, sync_dist=True)
        self.log("val/best_precision", best_p, sync_dist=True)
        self.log("val/best_recall", best_r, sync_dist=True)
        self.log("val/best_threshold", best_t, sync_dist=True)

        # PR / ROC curves to W&B
        try:
            p_np = probs.numpy()
            y_np = targets.numpy()

            pre, rec, _ = precision_recall_curve(y_np, p_np)
            fpr, tpr, _ = roc_curve(y_np, p_np)

            self.logger.experiment.log({
                "val/pr_curve": wandb.plot.line_series([rec], [pre], keys=["precision"], xname="Recall"),
                "val/roc_curve": wandb.plot.line_series([fpr], [tpr], keys=["TPR"], xname="FPR"),
            })
        except Exception as e:
            print("[WARN] PR/ROC curve error:", e)



# TEST
# ----------------------------------------------------------
    def test_step(self, batch, batch_idx):
        dna, prot, mask, labels, tf_idx = batch

        prot = prot.to(dna.device, non_blocking=True)
        mask = mask.to(dna.device, non_blocking=True)
        labels = labels.to(dna.device, non_blocking=True)

        logits = self.model(dna, prot, protein_mask=mask).squeeze(-1)
        probs = torch.sigmoid(logits)

        self.test_probs.append(probs.detach().cpu())
        self.test_targets.append(labels.detach().cpu())

        loss = self.loss_fn(logits, labels.float())

        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )

        return loss

    
    
    #macro metrices
    def on_test_epoch_end(self):
        

        parent_dir = "eval_out"
        data_dir = os.path.join(parent_dir, "metrics_data")
        out_dir  = os.path.join(parent_dir, "eval_results")

        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        # --------------------------------------------------
        # Collect predictions
        # --------------------------------------------------
        probs = torch.cat(self.test_probs, dim=0).view(-1)
        targets = torch.cat(self.test_targets, dim=0).view(-1).int()

        # to numpy (REQUIRED for sklearn)
        probs_np = probs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        # --------------------------------------------------
        # Save raw data
        # --------------------------------------------------
        np.save(f"{data_dir}/probs.npy", probs_np)
        np.save(f"{data_dir}/targets.npy", targets_np)

        # --------------------------------------------------
        # Thresholded predictions
        # --------------------------------------------------
        thr = 0.3
        preds_np = (probs_np >= thr).astype(int)

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------
        auroc = roc_auc_score(targets_np, probs_np)
        aupr  = average_precision_score(targets_np, probs_np)

        acc  = accuracy_score(targets_np, preds_np)
        prec = precision_score(targets_np, preds_np, zero_division=0)
        rec  = recall_score(targets_np, preds_np, zero_division=0)
        f1   = f1_score(targets_np, preds_np, zero_division=0)

        print("========== Test Metrics ==========")
        print(f"AUROC : {auroc:.4f}")
        print(f"AUPR  : {aupr:.4f}")
        print(f"ACC   : {acc:.4f}")
        print(f"Prec  : {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1    : {f1:.4f}")

        with open(f"{out_dir}/metrics.txt", "w") as f:
            f.write(f"AUROC  {auroc:.6f}\n")
            f.write(f"AUPR   {aupr:.6f}\n")
            f.write(f"ACC    {acc:.6f}\n")
            f.write(f"PREC   {prec:.6f}\n")
            f.write(f"RECALL {rec:.6f}\n")
            f.write(f"F1     {f1:.6f}\n")

        # --------------------------------------------------
        # ROC Curve
        # --------------------------------------------------
        fpr, tpr, _ = roc_curve(targets_np, probs_np)
        np.savez(f"{out_dir}/roc_data.npz", fpr=fpr, tpr=tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC (AUC={auroc:.3f})")
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/roc_curve.png", dpi=300)
        plt.close()

        # --------------------------------------------------
        # Precision–Recall Curve
        # --------------------------------------------------
        precision, recall, _ = precision_recall_curve(targets_np, probs_np)
        np.savez(f"{out_dir}/pr_data.npz", precision=precision, recall=recall)

        plt.figure()
        plt.plot(recall, precision, label=f"PR (AUPRC={aupr:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/pr_curve.png", dpi=300)
        plt.close()

        print(f"\nAll results saved to: {out_dir}")




            