import itertools
import random
import string
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer

import torch

from backbone.clip_vit_b16 import CLIPViTB16
import engine
import utils
import wandb
from backbone.convnext_tiny import ConvNeXtTiny
from backbone.effnet_b0 import EfficientNetB0
from backbone.effnet_b2 import EfficientNetB2
from backbone.vit_b16 import ViTB16
from data_setup import create_dataloaders
from exp_config import ExperimentConfig
from utils import get_class_names

# Config
NUM_WORKERS = 2
images_dir = Path("data/images")
labels_dir = Path("data/labels/labels.csv")


def generate_experiment_group_id():
    random_id = "".join(random.choices(
        string.ascii_lowercase + string.digits, k=4))
    return f"exp-{random_id}"


def create_model(model_name, out_features, unfreeze_encoder_layers, device):

    if model_name == "effnetb2":
        return EfficientNetB2(out_features=out_features, unfreeze_last_n=unfreeze_encoder_layers, device=device)
    elif model_name == "convnext_tiny":
        return ConvNeXtTiny(out_features=out_features, unfreeze_last_n=unfreeze_encoder_layers, device=device)
    elif model_name == "vitb16":
        return ViTB16(out_features=out_features, unfreeze_last_n=unfreeze_encoder_layers, device=device)
    elif model_name == "clip_vitb16":
        return CLIPViTB16(device=device, unfreeze_last_n=unfreeze_encoder_layers, out_features=out_features)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def run_single_experiment(config_dict, experiment_id, total_experiments, experiment_group):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config_dict["random_seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed(config_dict["random_seed"])

    print(f"\n{'='*60}")
    print(f"EXPERIMENT {experiment_id}/{total_experiments}")
    print(f"Config: {config_dict}")
    print(f"{'='*60}")

    out_features = len(get_class_names(csv_path=labels_dir))
    model = create_model(config_dict["model_name"], out_features,
                         config_dict["unfreeze_encoder_layers"], device)

    # Always episode mode
    train_dl, val_dl, test_dl, class_names = create_dataloaders(
        random_seed=config_dict["random_seed"],
        images_dir=images_dir,
        train_transform=model.train_transform,
        test_transform=model.test_transform,
        batch_size=config_dict["batch_size"],
        num_workers=NUM_WORKERS,
        device=device,
        episode_splits=config_dict["episodes"],
    )

    # Naming
    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    test_eps = "-".join(config_dict["episodes"]["test"]
                        ) if config_dict["episodes"]["test"] else "noTest"

    experiment_name = (
        f"{model.model_name}"
        f"_tes{test_eps}"
        f"_uf{config_dict['unfreeze_encoder_layers']}"
        f"_e{config_dict['num_epochs']}"
        f"_bs{config_dict['batch_size']}"
        f"_lr{config_dict['learning_rate']}"
        f"_{timestamp}"
    )

    model_name = f"{experiment_name}.pth"

    # Loss & optimizer
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.model.parameters(),
        lr=config_dict["learning_rate"],
        weight_decay=config_dict["weight_decay"],
    )

    # Scheduler
    scheduler = None
    if config_dict["scheduler"] == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config_dict["step_size"], gamma=config_dict["gamma"]
        )
    elif config_dict["scheduler"] == "None":
        scheduler = None

    # W&B logging
    wandb.init(
        project="muppet-show-classifier",
        group=experiment_group,
        job_type="hyperparameter_sweep",
        name=experiment_name,
        config={
            "tag": config_dict["tag"],
            "group": experiment_group,
            "model": model.model_name,
            "class_names": class_names,
            "episodes_train": config_dict["episodes"]["train"],
            "episodes_test": config_dict["episodes"]["test"],
            "episodes_val": config_dict["episodes"]["val"],
            "unfreeze_encoder_layers": config_dict["unfreeze_encoder_layers"],
            "epochs": config_dict["num_epochs"],
            "batch_size": config_dict["batch_size"],
            "loss_fn": loss_fn.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
            "learning_rate": config_dict["learning_rate"],
            "weight_decay": config_dict["weight_decay"],
            "output_threshold": config_dict["output_threshold"],
            "scheduler": config_dict["scheduler"],
            "step_size": config_dict["step_size"],
            "gamma": config_dict["gamma"],
            "device": str(device),
            "num_classes": out_features,
        },
    )

    print(f"[INFO] Starting experiment: {experiment_name}")
    start = timer()
    try:
        engine.train(
            model=model.model,
            train_dataloader=train_dl,
            val_dataloader=val_dl,
            test_dataloader=test_dl,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=config_dict["num_epochs"],
            class_names=class_names,
            device=device,
            threshold=config_dict["output_threshold"],
            early_stopping_patience=3,
            scheduler=scheduler
        )
        print(f"[INFO] Training completed in {timer() - start:.3f}s")
        model_path = utils.save_model(
            model.model, target_dir="models", model_name=model_name)

        experiment_name = (
            f"{experiment_group}"
            f"_{model.model_name}")

        artifact = wandb.Artifact(
            name=experiment_name,
            type="model",
            description=f"Model trained with config {config_dict}",
            metadata=config_dict
        )
        artifact.add_file(local_path=model_path, name="model")
        wandb.log_artifact(artifact)

    except Exception as e:
        print(f"[ERROR] Experiment {experiment_name} failed: {str(e)}")
    finally:
        wandb.finish()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    experiment_group = generate_experiment_group_id()
    print(f"[INFO] Experiment group: {experiment_group}")

    experiment_combinations = list(
        itertools.product(
            ExperimentConfig.RANDOM_SEED,
            ExperimentConfig.TAG,
            ExperimentConfig.MODEL_NAME,
            ExperimentConfig.UNFREEZE_ENCODER_LAYERS,
            ExperimentConfig.NUM_EPOCHS,
            ExperimentConfig.BATCH_SIZE,
            ExperimentConfig.LEARNING_RATE,
            ExperimentConfig.WEIGHT_DECAY,
            ExperimentConfig.OUTPUT_THRESHOLD,
            ExperimentConfig.MAX_WEIGHT,
            ExperimentConfig.EPISODE_SPLITS,
            ExperimentConfig.SCHEDULER,
            ExperimentConfig.STEP_SIZE,
            ExperimentConfig.GAMMA,
        )
    )
    total_experiments = len(experiment_combinations)
    print(f"[INFO] Total experiments to run: {total_experiments}")

    overall_start = timer()
    for i, combo in enumerate(experiment_combinations, start=1):
        (
            random_seed,
            tag,
            model_name,
            unfreeze_layers,
            num_epochs,
            batch_size,
            learning_rate,
            weight_decay,
            output_threshold,
            max_weight,
            episodes,
            scheduler_name,
            step_size,
            gamma,
        ) = combo

        config_dict = {
            "random_seed": random_seed,
            "tag": tag,
            "model_name": model_name,
            "unfreeze_encoder_layers": unfreeze_layers,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "output_threshold": output_threshold,
            "max_weight": max_weight,
            "episodes": episodes,
            "scheduler": scheduler_name,
            "step_size": step_size,
            "gamma": gamma,
        }

        run_single_experiment(
            config_dict, i, total_experiments, experiment_group)

    print(f"\n{'='*60}\nEXPERIMENT SUMMARY\n{'='*60}")
    print(f"Total experiments: {total_experiments}")
    print(f"[INFO] Total training time: {timer() - overall_start:.2f}s")


if __name__ == "__main__":
    main()
