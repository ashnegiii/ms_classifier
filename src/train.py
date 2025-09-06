import itertools
import random
import string
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer

import torch

import engine
import utils
import wandb
from backbone.effnet_b0 import EfficientNetB0
from backbone.effnet_b2 import EfficientNetB2
from backbone.vit_b16 import ViTB16
from data_setup import create_dataloaders
from experiment_config import ExperimentConfig
from utils import (analyze_class_distribution_from_path, calc_pos_weight,
                   create_weighted_sampler_from_csv, get_class_names)

# Config
NUM_WORKERS = 2

images_dir = Path("data/images")
labels_dir = Path("data/labels/labels.csv")


def generate_experiment_group_id():
    """Generate a random group identifier for this experiment batch"""
    random_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"exp-{random_id}"


def create_model(model_name, out_features, unfreeze_encoder_layers, device):
    if model_name == "effnetb0":
        return EfficientNetB0(
            out_features=out_features,
            device=device,
            unfreeze_last_n=unfreeze_encoder_layers,
        )
    elif model_name == "vitb16":
        return ViTB16(
            out_features=out_features,
            unfreeze_last_n=unfreeze_encoder_layers,
            device=device,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def run_single_experiment(
    config_dict, experiment_id, total_experiments, experiment_group
):
    """Run a single experiment with the given configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(ExperimentConfig.RANDOM_SEED)
    (
        torch.cuda.manual_seed(ExperimentConfig.RANDOM_SEED)
        if device.type == "cuda"
        else None
    )

    print(f"\n{'='*60}")
    print(f"EXPERIMENT {experiment_id}/{total_experiments}")
    print(f"Config: {config_dict}")
    print(f"{'='*60}")

    out_features = len(get_class_names(csv_path=labels_dir))

    model = create_model(
        config_dict["model_name"],
        out_features,
        config_dict["unfreeze_encoder_layers"],
        device,
    )

    if config_dict["episodes"] is None:
        # FRACTION MODE
        train_dataloader, val_dataloader, test_dataloader, class_names = (
            create_dataloaders(
                images_dir=images_dir,
                train_transform=model.train_transform,
                test_transform=model.test_transform,
                batch_size=config_dict["batch_size"],
                num_workers=NUM_WORKERS,
                device=device,
                train_split=config_dict["train_split"],
                test_split=config_dict["test_split"],
                val_split=config_dict["val_split"],
            )
        )
        split_tag = f"dl{int(config_dict['train_split']*100)}"
    else:
        # EPISODE MODE
        train_dataloader, val_dataloader, test_dataloader, class_names = (
            create_dataloaders(
                images_dir=images_dir,
                train_transform=model.train_transform,
                test_transform=model.test_transform,
                batch_size=config_dict["batch_size"],
                num_workers=NUM_WORKERS,
                device=device,
                episode_splits=config_dict["episodes"],
            )
        )
        split_tag = "episodes"

    # --------------------------
    # Experiment naming
    # --------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = (
        f"{model.model_name}_unfreeze{config_dict['unfreeze_encoder_layers']}_"
        f"{split_tag}_"
        f"e{config_dict['num_epochs']}_bs{config_dict['batch_size']}_"
        f"lr{config_dict['learning_rate']}_wd{config_dict['weight_decay']}_"
        f"th{config_dict['output_threshold']}_mw{config_dict['max_weight']}_{timestamp}"
    )

    model_name = f"{experiment_name}.pth"

    # --------------------------
    # Training setup
    # --------------------------
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.model.parameters(),
        lr=config_dict["learning_rate"],
        weight_decay=config_dict["weight_decay"],
    )

    # --------------------------
    # W&B init
    # --------------------------
    wandb.init(
        project="muppet-show-classifier",
        group=experiment_group,
        job_type="hyperparameter_sweep",
        name=experiment_name,
        config={
            "model": model.model_name,
            "unfreeze_encoder_layers": config_dict["unfreeze_encoder_layers"],
            "train_split": (
                config_dict["train_split"]
                if ExperimentConfig.SPLIT_MODE == "fraction"
                else None
            ),
            "test_split": (
                config_dict["test_split"]
                if ExperimentConfig.SPLIT_MODE == "fraction"
                else None
            ),
            "val_split": (
                config_dict["val_split"]
                if ExperimentConfig.SPLIT_MODE == "fraction"
                else None
            ),
            "episodes": (
                config_dict["episodes"]
                if ExperimentConfig.SPLIT_MODE == "episode"
                else None
            ),
            "epochs": config_dict["num_epochs"],
            "batch_size": config_dict["batch_size"],
            "loss_fn": loss_fn.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
            "learning_rate": config_dict["learning_rate"],
            "weight_decay": config_dict["weight_decay"],
            "output_threshold": config_dict["output_threshold"],
            "max_weight": config_dict["max_weight"],
            "device": str(device),
            "num_classes": out_features,
            "class_names": class_names,
        },
    )

    # --------------------------
    # Training
    # --------------------------
    print(f"[INFO] Starting experiment: {experiment_name}")
    start = timer()
    try:
        engine.train(
            model=model.model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=config_dict["num_epochs"],
            threshold=config_dict["output_threshold"],
            class_names=class_names,
            device=device,
        )

        training_time = timer() - start
        print(f"[INFO] Training completed in {training_time:.3f}s")

        # Save model
        utils.save_model(model.model, target_dir="models", model_name=model_name)

    except Exception as e:
        print(f"[ERROR] Experiment {experiment_name} failed: {str(e)}")
    finally:
        wandb.finish()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Device: {device}")

    # Generate random group ID for this experiment batch
    experiment_group = generate_experiment_group_id()
    print(f"[INFO] Experiment group: {experiment_group}")
    print(f"[INFO] Data analysis:")
    analyze_class_distribution_from_path(labels_dir)

    if ExperimentConfig.SPLIT_MODE == "fraction":
        experiment_combinations = list(
            itertools.product(
                ExperimentConfig.model_name,
                ExperimentConfig.UNFREEZE_ENCODER_LAYERS,
                ExperimentConfig.NUM_EPOCHS,
                ExperimentConfig.BATCH_SIZE,
                ExperimentConfig.LEARNING_RATE,
                ExperimentConfig.WEIGHT_DECAY,
                ExperimentConfig.OUTPUT_THRESHOLD,
                ExperimentConfig.MAX_WEIGHT,
                ExperimentConfig.TRAIN_SPLIT,
                ExperimentConfig.TEST_SPLIT,
                ExperimentConfig.VAL_SPLIT,
            )
        )
    elif ExperimentConfig.SPLIT_MODE == "episode":
        experiment_combinations = list(
            itertools.product(
                ExperimentConfig.model_name,
                ExperimentConfig.UNFREEZE_ENCODER_LAYERS,
                ExperimentConfig.NUM_EPOCHS,
                ExperimentConfig.BATCH_SIZE,
                ExperimentConfig.LEARNING_RATE,
                ExperimentConfig.WEIGHT_DECAY,
                ExperimentConfig.OUTPUT_THRESHOLD,
                ExperimentConfig.MAX_WEIGHT,
                ExperimentConfig.EPISODE_SPLITS,
            )
        )
    else:
        raise RuntimeError("The split does not exist. Try 'fraction' or 'episode'.")

    total_experiments = len(experiment_combinations)
    print(f"[INFO] Total experiments to run: {total_experiments}")

    overall_start = timer()

    for i, combo in enumerate(experiment_combinations, start=1):
        if ExperimentConfig.SPLIT_MODE == "fraction":
            (
                model_name,
                unfreeze_layers,
                num_epochs,
                batch_size,
                learning_rate,
                weight_decay,
                output_threshold,
                max_weight,
                train_split,
                test_split,
                val_split,
            ) = combo

            config_dict = {
                "model_name": model_name,
                "unfreeze_encoder_layers": unfreeze_layers,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "output_threshold": output_threshold,
                "max_weight": max_weight,
                "train_split": train_split,
                "test_split": test_split,
                "val_split": val_split,
                "episodes": None,
            }

        else:  # episode mode
            (
                model_name,
                unfreeze_layers,
                num_epochs,
                batch_size,
                learning_rate,
                weight_decay,
                output_threshold,
                max_weight,
                episodes,
            ) = combo

            config_dict = {
                "model_name": model_name,
                "unfreeze_encoder_layers": unfreeze_layers,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "output_threshold": output_threshold,
                "max_weight": max_weight,
                "train_split": None,
                "test_split": None,
                "val_split": None,
                "episodes": episodes,
            }

        run_single_experiment(
            config_dict=config_dict,
            experiment_id=i,
            total_experiments=total_experiments,
            experiment_group=experiment_group,
        )

    # Print final summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments: {total_experiments}")

    total_time = timer() - overall_start
    print(f"[INFO] Total training time: {total_time:.2f}s ({total_time/3600:.2f}h)")


if __name__ == "__main__":
    main()
