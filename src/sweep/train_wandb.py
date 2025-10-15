import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb_variant.engine_update as engine_update
import wandb
# custom classes
from backbone.effnet_b0 import EfficientNetB0
from backbone.effnet_b2 import EfficientNetB2
from backbone.vit_b16 import ViTB16
from backbone.convnext_tiny import ConvNeXtTiny
from data_setup import create_dataloaders
from utils import (analyze_class_distribution_from_path, get_class_names,
                   save_model)

IMAGES_DIR = Path("data/images")
LABELS_CSV = Path("data/labels/labels.csv")
NUM_WORKERS = 2

EPISODE_SPLITS = [
    {"train": [["02-04-04","03-04-17","03-04-03","cook-1","miss-piggy-1", "fozzie-bear"]], "test": [["02-01-01"]], "val":[[]]},
    {"train": [["02-01-01","03-04-17","03-04-03","cook-1","miss-piggy-1", "fozzie-bear"]], "test": [["02-04-04"]], "val":[[]]},
    {"train": [["02-01-01","02-04-04","03-04-03","cook-1","miss-piggy-1", "fozzie-bear"]], "test": [["03-04-17"]], "val":[[]]},
    {"train": [["02-01-01","02-04-04","03-04-17","cook-1","miss-piggy-1", "fozzie-bear"]], "test": [["03-04-03"]], "val":[[]]},
]

def create_model(model_name: str, out_features: int, device: torch.device):
    if model_name in ["effnetb0", "effnetb2", "convnext_tiny"]:
        if model_name == "effnetb0":
            wrapper = EfficientNetB0(device=device, unfreeze_last_n=0, out_features=out_features)
        elif model_name == "effnetb2":
            wrapper = EfficientNetB2(device=device, unfreeze_last_n=0, out_features=out_features)
        elif model_name == "convnext_tiny":
            wrapper = ConvNeXtTiny(device=device, unfreeze_last_n=0, out_features=out_features)
        return wrapper.model, wrapper.train_transform, wrapper.test_transform
    elif model_name == "vitb16":
        wrapper = ViTB16(device=device, unfreeze_last_n=0, out_features=out_features)
        return wrapper.model, wrapper.train_transform, wrapper.test_transform
    else:
        raise ValueError(f"Unknown model: {model_name}")


def freeze_backbone(model: nn.Module, model_name: str) -> None:
    if model_name in ["effnetb2", "effnetb0", "convnext_tiny"]:
        for p in model.features.parameters():
            p.requires_grad = False
    elif model_name == "vitb16":
        for p in model.encoder.parameters():
            p.requires_grad = False


def unfreeze_last_n_blocks(model: nn.Module, model_name: str, n: int) -> None:
    if n is None or n <= 0:
        return
    if model_name in ["effnetb2", "effnetb0", "convnext_tiny"]:
        for layer in model.features[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
    elif model_name == "vitb16":
        for layer in model.encoder.layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True


def head_parameters(model: nn.Module, model_name: str):
    if model_name in ["effnetb2", "effnetb0", "convnext_tiny"]:
        return (p for p in model.classifier.parameters() if p.requires_grad)
    elif model_name == "vitb16":
        return (p for p in model.heads.parameters() if p.requires_grad)


def backbone_blocks_last_to_first(model: nn.Module, model_name: str):
    if model_name in ["effnetb2", "effnetb0", "convnext_tiny"]:
        return list(model.features)[::-1]
    elif model_name == "vitb16":
        return list(model.encoder.layers)[::-1]

def requires_grad_params(module: nn.Module) -> List[nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]

def build_param_groups_llrd(model: nn.Module,
                            model_name: str,
                            lr_head: float,
                            lr_backbone: float,
                            llrd_gamma: float) -> List[Dict]:
    groups: List[Dict] = []
    # head
    groups.append({"params": list(head_parameters(model, model_name)), "lr": lr_head})
    # backbone blocks from last to first, LR decays as we go deeper
    blocks = backbone_blocks_last_to_first(model, model_name)
    lr = lr_backbone
    for blk in blocks:
        params = requires_grad_params(blk)
        if params:
            groups.append({"params": params, "lr": lr})
            lr *= llrd_gamma
    return groups


def run_stage(model: nn.Module,
              model_name: str,
              stage_name: str,
              epochs: int,
              lr_head: float,
              lr_backbone: float,
              llrd_gamma: float,
              weight_decay: float,
              train_loader,
              val_loader,
              test_loader,
              class_names: List[str],
              device: torch.device,
              threshold: float,
              do_eval: bool = True) -> None:
    """Build param groups for this stage and train using engine.train."""
    
    if lr_backbone == 0.0:
        # head-only training
        params = list(head_parameters(model, model_name))
        optimizer = torch.optim.AdamW([{"params": params, "lr": lr_head}], weight_decay=weight_decay)
    else:
        param_groups = build_param_groups_llrd(model, model_name, lr_head, lr_backbone, llrd_gamma)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    wandb.log({f"stage_{stage_name}_start": 1}, commit=False)
    engine_update.train(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader if do_eval else None,
        test_dataloader=test_loader if do_eval else None,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=epochs,
        threshold=threshold,
        class_names=class_names,
        device=device,
        scheduler = scheduler
    )
    wandb.log({f"stage_{stage_name}_end": 1}, commit=True)


def main():
    run = wandb.init(project=wandb.config.get("project", "muppet-show-classifier"),
                     group=wandb.config.get("group", None))
    cfg = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(cfg.seed)

    # 2) Data + class names
    # analyze_class_distribution_from_path(LABELS_CSV)
    out_features = len(get_class_names(csv_path=LABELS_CSV))

    # 3) Model
    model, train_transform, test_transform = create_model(cfg.model_name, out_features, device)
    model_name_for_log = cfg.model_name

    # 4) Dataloaders (episode vs fraction)
    split = EPISODE_SPLITS[cfg.episode_index]
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        random_seed=42,
        images_dir=IMAGES_DIR,
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=cfg.batch_size,
        num_workers=NUM_WORKERS,
        device=device,
        episode_splits=split
    )
    split_tag = f"episodes#{cfg.episode_index}"

    freeze_backbone(model, model_name_for_log)

    run_stage(
        model=model,
        model_name=model_name_for_log,
        stage_name="warmup_head",
        epochs=int(cfg.warmup_epochs),
        lr_head=float(cfg.lr_head_warmup),
        lr_backbone=0.0,  # head-only
        llrd_gamma=float(cfg.llrd_gamma),
        weight_decay=float(cfg.weight_decay),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        threshold=float(cfg.threshold),
        do_eval = False
    )


    # Stage 2: finetune (unfreeze top-n, small LR with LLRD)
    # ensure clean state
    freeze_backbone(model, model_name_for_log)
    unfreeze_last_n_blocks(model, model_name_for_log, cfg.unfreeze_last_n)

    run_stage(
        model=model,
        model_name=model_name_for_log,
        stage_name=f"finetune_top{cfg.unfreeze_last_n}",
        epochs=int(cfg.finetune_epochs),
        lr_head=float(cfg.lr_head_finetune),
        lr_backbone=float(cfg.lr_backbone),
        llrd_gamma=float(cfg.llrd_gamma),
        weight_decay=float(cfg.weight_decay),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        threshold=float(cfg.threshold),
        do_eval = True
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_name = f"{model_name_for_log}_{split_tag}_wu{cfg.warmup_epochs}_uf{cfg.unfreeze_last_n}_{timestamp}.pth"
    save_model(model, target_dir="models", model_name=ckpt_name)
    wandb.save(os.path.join("models", ckpt_name))
    wandb.finish()

DEFAULT_CONFIG = dict(
    project="muppet-show-classifier",
    group=None,
    seed=42,
    model_name="effnetb2",
    batch_size=32,
    warmup_epochs=2,
    finetune_epochs=5,
    lr_head_warmup=1e-3,
    lr_head_finetune=1e-4,
    lr_backbone=1e-5,
    llrd_gamma=0.9,
    weight_decay=1e-4,
    threshold=0.5,
    unfreeze_last_n=2,
    episode_index=0
)

if __name__ == "__main__":
    run = wandb.init(project=DEFAULT_CONFIG["project"], config=DEFAULT_CONFIG)
    cfg = wandb.config
    main()