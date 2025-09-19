
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Optional
import wandb
import numpy as np
import pandas as pd
from pathlib import Path
from utils import calc_metrics


def run_epoch(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device,
              optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, torch.Tensor]:
    """
    Run one epoch (train or eval depending on whether optimizer is given).
    Returns average loss, logits and targets.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_logits, all_targets = [], []

    context = torch.enable_grad() if is_train else torch.inference_mode()
    with context:
        pbar = tqdm(dataloader, desc="Train" if is_train else "Eval", leave=False)
        for X, y in pbar:
            X, y = X.to(device), y.to(device).float()

            logits = model(X)
            loss = loss_fn(logits, y)
            total_loss += loss.item()

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            all_logits.append(logits.detach())
            all_targets.append(y.detach())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    return {"loss": avg_loss, "logits": logits, "targets": targets}

def evaluate_metrics(logits: torch.Tensor,
                     targets: torch.Tensor,
                     class_names: List[str],
                     optimal_thresholds: Optional[Dict[str, float]] = None,
                     threshold: Optional[float] = None) -> Dict[str, Dict]:
    """Wrapper to compute all metrics in a structured format."""
    if optimal_thresholds:
        acc, prec, rec, f1, ap, mAP = calc_metrics(
            targets=targets, logits=logits,
            optimal_thresholds=optimal_thresholds,
            class_names=class_names
        )
    elif threshold:
        acc, prec, rec, f1, ap, mAP = calc_metrics(
            targets=targets, logits=logits,
            threshold=threshold,
            class_names=class_names
        )
    else:
        raise ValueError("Either optimal_thresholds or threshold must be provided.")

    per_class = {
        name: {
            "accuracy": acc[i],
            "precision": prec[i],
            "recall": rec[i],
            "f1": f1[i],
            "ap": ap[i]
        }
        for i, name in enumerate(class_names)
    }

    return {
        "per_class": per_class,
        "macro": {"mAP": mAP}
    }

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: Optional[torch.utils.data.DataLoader],
          test_dataloader: Optional[torch.utils.data.DataLoader],
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          class_names: List[str],
          device: torch.device,
          threshold: float = 0.5,
          scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> None:
    """
    Train + optional validate + test a PyTorch model with W&B logging.
    """

    # Tracking
    epoch_idx = []
    loss_train, loss_val, loss_test = [], [], []
    precision_data = {name: [] for name in class_names}
    recall_data = {name: [] for name in class_names}
    optimal_thresholds = {name: threshold for name in class_names}

    for epoch in tqdm(range(epochs)):
        # --- Train ---
        train_out = run_epoch(model, train_dataloader, loss_fn, device, optimizer)
        train_metrics = evaluate_metrics(train_out["logits"], train_out["targets"],
                                         class_names, threshold=threshold)

        # --- Validation (optional) ---
        val_out, val_metrics = None, None
        if val_dataloader is not None and len(val_dataloader) > 0:
            val_out = run_epoch(model, val_dataloader, loss_fn, device)
            val_metrics = evaluate_metrics(val_out["logits"], val_out["targets"],
                                           class_names, threshold=threshold)

        # --- Test (optional) ---
        test_out, test_metrics = None, None
        if test_dataloader is not None and len(test_dataloader) > 0:
            test_out = run_epoch(model, test_dataloader, loss_fn, device)
            test_metrics = evaluate_metrics(test_out["logits"], test_out["targets"],
                                            class_names, optimal_thresholds=optimal_thresholds)

        # Scheduler step
        if scheduler:
            scheduler.step()

        # --- Store losses ---
        epoch_idx.append(epoch + 1)
        loss_train.append(train_out["loss"])
        if val_out is not None:
            loss_val.append(val_out["loss"])
        if test_out is not None:
            loss_test.append(test_out["loss"])

        # --- Print summary ---
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train - Loss: {train_out['loss']:.4f}")
        if val_out is not None:
            print(f"  Val   - Loss: {val_out['loss']:.4f}")
        if test_out is not None:
            print(f"  Test  - Loss: {test_out['loss']:.4f}")
            print(f"  Test  - mAP:  {test_metrics['macro']['mAP']:.4f}")

            for cname in class_names:
                m = test_metrics["per_class"][cname]
                print(f"    {cname}: "
                      f"Acc {m['accuracy']:.4f} | Prec {m['precision']:.4f} | "
                      f"Rec {m['recall']:.4f} | F1 {m['f1']:.4f} | AP {m['ap']:.4f}")

        # --- Prepare W&B log dict ---
        log_dict = {
            "epoch": epoch + 1,
            "loss_train": train_out["loss"],
        }
        if val_out is not None:
            log_dict["loss_val"] = val_out["loss"]
        if test_out is not None:
            log_dict["loss_test"] = test_out["loss"]
            log_dict["mAP"] = test_metrics["macro"]["mAP"]

        # Add per-class metrics (from test set)
        if test_metrics is not None:
            for cname in class_names:
                for metric, val in test_metrics["per_class"][cname].items():
                    log_dict[f"{metric}_{cname}"] = val

        # --- Loss plot ---
        ys, keys = [loss_train], ["train"]
        if loss_val:
            ys.append(loss_val); keys.append("val")
        if loss_test:
            ys.append(loss_test); keys.append("test")
        log_dict["loss_plot"] = wandb.plot.line_series(
            xs=epoch_idx, ys=ys, keys=keys,
            title="Loss per Epoch", xname="Epoch"
        )

        # --- Precision/Recall curves ---
        if test_metrics is not None:
            for cname in class_names:
                precision_data[cname].append(test_metrics["per_class"][cname]["precision"])
                recall_data[cname].append(test_metrics["per_class"][cname]["recall"])

                log_dict[f"precision_recall_{cname}"] = wandb.plot.line_series(
                    xs=epoch_idx,
                    ys=[precision_data[cname], recall_data[cname]],
                    keys=["precision", "recall"],
                    title=f"Precision & Recall - {cname}",
                    xname="Epoch"
                )

        # --- Per-class confusion matrices (multi-label safe) ---
        if test_out is not None:
            preds_bin = (test_out["logits"].sigmoid() > threshold).cpu().numpy()
            y_true_bin = test_out["targets"].cpu().numpy()

            for i, cname in enumerate(class_names):
                cm_preds = preds_bin[:, i]
                cm_true = y_true_bin[:, i]

                log_dict[f"conf_matrix_{cname}"] = wandb.plot.confusion_matrix(
                    preds=cm_preds,
                    y_true=cm_true,
                    class_names=[f"not_{cname}", cname]
                )

        # --- Push to W&B ---
        wandb.log(log_dict, step=epoch + 1)
