import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import tqdm
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score)
from torch import nn
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms


def find_optimal_thresholds(
    model: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    class_names: List[str],
    device: torch.device,
    threshold_range: np.ndarray = np.arange(0.1, 0.95, 0.05),
    metric: str = 'f1'
) -> Dict[str, float]:
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.inference_mode():
        pbar = tqdm(val_dataloader, desc="Validation", leave=False)
        for X, y in pbar:
            X = X.to(device)
            y = y.to(device)
            
            logits = model(X)
            probs = torch.sigmoid(logits)
            
            all_predictions.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    optimal_thresholds = {}
    
    for class_idx, class_name in enumerate(class_names):
        y_true = all_targets[:, class_idx]
        y_scores = all_predictions[:, class_idx]
        
        best_threshold = 0.5
        best_metric_value = 0.0
        
        for threshold in threshold_range:
            y_pred = (y_scores >= threshold).astype(int)
            
            # skip if all predictions are same
            if len(np.unique(y_pred)) == 1: 
                continue
            
            if metric == 'f1':
                metric_value = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                metric_value = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                metric_value = recall_score(y_true, y_pred, zero_division=0)

            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_threshold = threshold

        optimal_thresholds[class_name] = best_threshold    
    
    return optimal_thresholds


def calc_metrics(targets: torch.Tensor, 
                logits: torch.Tensor, 
                optimal_thresholds: Dict[str, float] = None, 
                class_names: List[str] = None,
                threshold: float = 0.5):
    """Calculate metrics with optional per-class optimal thresholds"""
    
    # Convert logits -> probabilities
    probs = torch.sigmoid(logits).cpu().numpy()
    y_true = targets.cpu().numpy().astype(int)
    
    C = y_true.shape[1]
    
    # Apply thresholds
    if optimal_thresholds is not None:
        y_pred_np = np.zeros_like(probs, dtype=int)
        for c, name in enumerate(class_names):
            t = optimal_thresholds.get(name)
            y_pred_np[:, c] = (probs[:, c] >= t).astype(int)
    else:
        y_pred_np = (probs >= threshold).astype(int)
    accuracy_per_class = []
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    ap_per_class = []
    
    
    for c in range(C):
        yt = y_true[:, c]
        yp = probs[:, c]
        yhat = y_pred_np[:, c]        
        
        accuracy_per_class.append(accuracy_score(yt, yhat))
        precision_per_class.append(precision_score(yt, yhat, zero_division=0))
        recall_per_class.append(recall_score(yt, yhat, zero_division=0))
        f1_per_class.append(f1_score(yt, yhat, zero_division=0))
        ap_per_class.append(average_precision_score(yt, yp))

    mAP = float(np.mean(ap_per_class))

    return accuracy_per_class, precision_per_class, recall_per_class, f1_per_class, ap_per_class, mAP

def analyze_class_distribution_from_path(csv_path):
    df = pd.read_csv(csv_path)
    print(f"CSV Path: {csv_path}")
    analyze_class_distribution_from_df(df)


def analyze_class_distribution_from_df(df, label="CSV"):
    label_cols = [c for c in df.columns if c != 'filename']
    
    total_samples = len(df)
    
    print(f"\n=== CLASS DISTRIBUTION FROM {label}===")
    print(f"Total samples: {total_samples}")
    print("-" * 50)
    
    class_stats = []
    for col in label_cols:
        positive_count = df[col].sum()
        negative_count = total_samples - positive_count
        positive_percent = (positive_count / total_samples) * 100
        negative_percent = (negative_count / total_samples) * 100
        
        # Calculate imbalance ratio
        if positive_count > 0:
            imbalance_ratio = negative_count / positive_count
        else:
            imbalance_ratio = float('inf')
        
        class_stats.append({
            'class': col,
            'positive': positive_count,
            'negative': negative_count,
            'pos_percent': positive_percent,
            'imbalance_ratio': imbalance_ratio
        })
        
        print(f"{col:15} | Pos: {positive_count:4d} ({positive_percent:5.1f}%) | "
              f"Neg: {negative_count:4d} ({negative_percent:5.1f}%) | "
              f"Ratio: 1:{imbalance_ratio:.1f}")
    
    print("-" * 50)
    
    # Sort by imbalance ratio to show most problematic classes
    #class_stats.sort(key=lambda x: x['imbalance_ratio'], reverse=True)
    #print("\nMOST IMBALANCED CLASSES (worst first):")
    #for stat in class_stats:
    #    if stat['imbalance_ratio'] != float('inf'):
    #        print(f"{stat['class']:15} - 1:{stat['imbalance_ratio']:6.1f} "
    #              f"({stat['pos_percent']:4.1f}% positive)")
    
    #print("=" * 50)
    
    return class_stats


def create_weighted_sampler_from_csv(csv_path, oversample_factor=10):
    """
    Create WeightedRandomSampler based on class frequencies in CSV.
    Heavily oversamples minority classes like cook.
    
    Args:
        csv_path: Path to labels CSV file
        oversample_factor: How much to boost minority classes (default 10x)
    
    Returns:
        WeightedRandomSampler
    """
    df = pd.read_csv(csv_path)
    label_cols = [c for c in df.columns if c != 'filename']
    
    # Calculate class frequencies
    class_frequencies = {}
    for column in label_cols:
        positive_examples = df[column].sum()
        class_frequencies[column] = positive_examples / len(df)
    
    # Find rarest class frequency for normalization
    min_frequency = min(class_frequencies.values())
    
    # Calculate sample weights
    sample_weights = []
    for idx, row in df.iterrows():
        # Start with base weight
        sample_weight = 1.0
        
        # For each active class in this sample
        for col in label_cols:
            if row[col] == 1:
                # Weight inversely proportional to class frequency
                class_weight = min_frequency / class_frequencies[col]
                # Apply oversample factor for very rare classes
                if class_frequencies[col] < 0.05:  # Less than 5%
                    class_weight *= oversample_factor
                
                # Use the maximum weight (rarest class dominates)
                sample_weight = max(sample_weight, class_weight)
        
        sample_weights.append(sample_weight)
    
    print(f"[INFO] Sample weights range: {min(sample_weights):.3f} to {max(sample_weights):.3f}")
    print(f"[INFO] Cook samples will be seen ~{oversample_factor}x more often")
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )



def calc_pos_weight(csv_path, device, max_weight: float = None):
    df = pd.read_csv(csv_path)
    
    label_cols = [c for c in df.columns if c != 'filename']
    
    weights = []
    for column in label_cols:
        positive_examples = df[column].sum()
        negative_examples = df.shape[0] - positive_examples

        if positive_examples > 0 and max_weight is None:
            weight = negative_examples / positive_examples if positive_examples > 0 else 1.0
        elif positive_examples > 0:
            weight = min(negative_examples / positive_examples, max_weight)
        else:
            weight = 1.0
        
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32).to(device)

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)
  
  return model_save_path


def load_backbone(model_class, model_path: str, num_classes: int, device: torch.device):
    """
    Einheitlicher Loader für alle Backbone-Klassen (ViT, EfficientNet, ...)

    Args:
        model_class: Klassenobjekt (z.B. ViTB16 oder EfficientNetB2)
        model_path: Pfad zur gespeicherten .pth Datei
        num_classes: Anzahl der Klassen für den Klassifikator
        device: torch.device ("cpu" oder "cuda")

    Returns:
        Instanziertes Backbone mit geladenen Gewichten (im eval-Modus)
    """
    # Backbone instanziieren
    backbone = model_class(device=device, out_features=num_classes, pretrained=False,  augmentation=False, unfreeze_encoder_layers=2)

    # State dict laden
    state_dict = torch.load(model_path, map_location=device)
    backbone.model.load_state_dict(state_dict)

    return backbone.model.to(device).eval()

def load_model(model_path: str, device: torch.device):
    """
    Lädt ein gespeichertes PyTorch Model von einer .pth Datei
    
    Args:
        model_path: Pfad zur .pth Datei
        device: torch.device (cpu oder cuda)
    
    Returns:
        Geladenes Model im eval() Modus
    """
    model = torch.load(model_path, map_location=device)
    return model.to(device).eval()


def load_vit_model(model_path: str, num_classes: int, device: torch.device):
    """
    Lädt ein ViT-B16 Model mit angepasstem Klassifikator
    
    Args:
        model_path: Pfad zur .pth Datei
        num_classes: Anzahl der Klassen für den Klassifikator
        device: torch.device (cpu oder cuda)
    
    Returns:
        Geladenes Model im eval() Modus
    """
    # Model-Architektur erstellen (ohne pretrained weights)
    model = models.vit_b_16(weights=None)
    
    # Klassifikator anpassen (genau wie beim Training)
    original_head_in_features = model.heads[0].in_features 
    model.heads = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(in_features=original_head_in_features, out_features=num_classes, bias=True)
    )
    
    # State dict laden und ins Model
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    return model.to(device).eval()



def get_class_names(csv_path: str):
    df = pd.read_csv(csv_path)
    return [c for c in df.columns if c != 'filename']