import itertools
import string
import torch
from pathlib import Path

from backbone.effnet_b0 import EfficientNetB0
from backbone.effnet_b2 import EfficientNetB2
from backbone.vit_b16 import ViTB16
from data_setup import create_dataloaders
from timeit import default_timer as timer
from datetime import datetime
from utils import calc_pos_weight, analyze_class_distribution_from_csv, create_weighted_sampler_from_csv
from utils import get_class_names
from experiment_config import ExperimentConfig

import engine, utils
import wandb
import random

# Config
RANDOM_SEED = 42
NUM_WORKERS = 2

train_dir = Path("data/train")
test_dir  = Path("data/test")
val_dir = Path("data/val")
train_labels_dir = Path("data/train_labels/labels.csv")

def generate_experiment_group_id():
    """Generate a random group identifier for this experiment batch"""
    random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"exp-{random_id}"

def create_model(model_name, out_features, unfreeze_encoder_layers, device):
    if model_name == "effnetb0":
        return EfficientNetB0(out_features=out_features, device=device, unfreeze_last_n=unfreeze_encoder_layers)
    elif model_name == "vitb16":
        return ViTB16(out_features=out_features, unfreeze_last_n=unfreeze_encoder_layers, device=device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    

def run_single_experiment(config_dict, experiment_id, total_experiments, experiment_group):
    """Run a single experiment with the given configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED) if device.type == "cuda" else None
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT {experiment_id}/{total_experiments}")
    print(f"Config: {config_dict}")
    print(f"{'='*60}")
    
    out_features = len(get_class_names(csv_path=train_labels_dir))
    
    model = create_model(config_dict['model_name'], out_features, config_dict['unfreeze_encoder_layers'], device)
    
    # sampler = create_weighted_sampler_from_csv(train_labels_dir)
    
    train_dataloader, val_dataloader, test_dataloader, class_names = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        train_data_fraction=config_dict['train_data_fraction'],
        test_data_fraction=config_dict['test_data_fraction'],
        val_data_fraction=config_dict['val_data_fraction'],
        train_transform=model.train_transform,
        test_transform=model.test_transform,
        sampler=None,
        batch_size=config_dict['batch_size'],
        num_workers=NUM_WORKERS,
        device=device
    )
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = (f"{model.model_name}_unfreeze{config_dict['unfreeze_encoder_layers']}_"
                      f"dl{int(config_dict['train_data_fraction']*100)}_"
                      f"e{config_dict['num_epochs']}_bs{config_dict['batch_size']}_"
                      f"lr{config_dict['learning_rate']}_wd{config_dict['weight_decay']}_"
                      f"th{config_dict['output_threshold']}_mw{config_dict['max_weight']}_{timestamp}")

    model_name = f"{experiment_name}.pth"
    
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.model.parameters(), 
        lr=config_dict['learning_rate'], 
        weight_decay=config_dict['weight_decay']
    )
    
    wandb.init(
        project="muppet-show-classifier", 
        group=experiment_group,
        job_type="hyperparameter_sweep",
        name=experiment_name,
        config={
            "model": model.model_name,
            "unfreeze_encoder_layers": config_dict['unfreeze_encoder_layers'],
            "train_data_fraction": config_dict['train_data_fraction'],
            "test_data_fraction": config_dict['test_data_fraction'],
            "val_data_fraction": config_dict['val_data_fraction'],
            "epochs": config_dict['num_epochs'],
            "batch_size": config_dict['batch_size'],
            "loss_fn": loss_fn.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
            "learning_rate": config_dict['learning_rate'],
            "weight_decay": config_dict['weight_decay'],
            "output_threshold": config_dict['output_threshold'],
            "max_weight": config_dict['max_weight'],
            "device": str(device),
            "num_classes": out_features,
            "class_names": class_names,
        }
    )

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
            epochs=config_dict['num_epochs'],
            threshold=config_dict['output_threshold'],
            class_names=class_names,
            device=device
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
    
    analyze_class_distribution_from_csv(train_labels_dir)

    experiment_combinations = list(itertools.product(
        ExperimentConfig.model_name,
        ExperimentConfig.UNFREEZE_ENCODER_LAYERS,
        ExperimentConfig.NUM_EPOCHS,
        ExperimentConfig.BATCH_SIZE,
        ExperimentConfig.LEARNING_RATE,
        ExperimentConfig.WEIGHT_DECAY,
        ExperimentConfig.OUTPUT_THRESHOLD,
        ExperimentConfig.MAX_WEIGHT,
        ExperimentConfig.TRAIN_DATA_FRACTION,
        ExperimentConfig.TEST_DATA_FRACTION,
        ExperimentConfig.VAL_DATA_FRACTION
    ))
    
    total_experiments = len(experiment_combinations)
    print(f"[INFO] Total experiments to run: {total_experiments}")
    
    overall_start = timer()
    
    for i, combo in enumerate(experiment_combinations, start=1):
        (model_name, unfreeze_encoder_layers, num_epochs, batch_size, learning_rate, 
         weight_decay, output_threshold, max_weight, train_fraction, 
         test_fraction, val_fraction) = combo
        
        config_dict = {
            'model_name': model_name,
            'unfreeze_encoder_layers': unfreeze_encoder_layers,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'output_threshold': output_threshold,
            'max_weight': max_weight,
            'train_data_fraction': train_fraction,
            'test_data_fraction': test_fraction,
            'val_data_fraction': val_fraction
        }

        run_single_experiment(
            config_dict=config_dict,
            experiment_id=i,
            total_experiments=total_experiments,
            experiment_group=experiment_group
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
