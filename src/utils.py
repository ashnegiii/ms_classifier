import shutil
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, accuracy_score
import torch
from pathlib import Path
import cv2
import pandas as pd
from typing import Dict, Tuple, List, Optional, Union
from PIL import Image
import torchvision.models as models
from torch import nn
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import WeightedRandomSampler
import tqdm

SUPPORTED_VIDEO_EXTS = [".mp4", ".avi"]

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



def analyze_class_distribution_from_csv(csv_path):
    """Analyze class distribution from labels CSV file"""
    df = pd.read_csv(csv_path)
    label_cols = [c for c in df.columns if c != 'filename']
    
    total_samples = len(df)
    
    print("\n=== CLASS DISTRIBUTION FROM CSV ===")
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
    class_stats.sort(key=lambda x: x['imbalance_ratio'], reverse=True)
    print("\nMOST IMBALANCED CLASSES (worst first):")
    for stat in class_stats:
        if stat['imbalance_ratio'] != float('inf'):
            print(f"{stat['class']:15} - 1:{stat['imbalance_ratio']:6.1f} "
                  f"({stat['pos_percent']:4.1f}% positive)")
    
    print("=" * 50)
    
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

def _find_video_path(video_dir: Path, prefix: str) -> Path:
    for ext in [".mp4", ".avi"]:
        p = (video_dir / f"{prefix}{ext}")
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No video found for prefix '{prefix}'. Looked for: "
        + ", ".join(str(video_dir / f"{prefix}{e}") for e in SUPPORTED_VIDEO_EXTS)
    )
    
def _extract_and_merge_for_prefix(
    prefix: str,
    video_dir: Path,
    csv_dir: Path,
    images_out_dir: Path,
    images_out_dir_val: Path = None,
    val_frac: float = 0.0
) -> pd.DataFrame:
    """Extract frames for a single prefix and return a labels DataFrame (with filename column, no '.jpg')."""
    video_path = _find_video_path(video_dir, prefix)
    csv_path   = csv_dir / f"{prefix}.csv"

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    frame_col = "frame"

    # Subsample rows
    df_kept = df.iloc[::1].reset_index(drop=True)
    kept_frames = df_kept[frame_col].astype(int).tolist()
    kept_set = set(kept_frames)

    # Prepare video read
    images_out_dir.mkdir(parents=True, exist_ok=True)
    
    if val_frac > 0 and images_out_dir_val is not None:
        images_out_dir_val.mkdir(parents=True, exist_ok=True)
        
        # Split dataframe into train/val
        df_val = df_kept.sample(frac=val_frac, random_state=42)
        df_train = df_kept.drop(df_val.index).reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)
        
        # Get frame sets for each split
        val_frames = set(df_val[frame_col].astype(int))
        train_frames = set(df_train[frame_col].astype(int))
        
        # Read video once and save frames to appropriate folders
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        saved_train = 0
        saved_val = 0
        frame_idx = 0
        
        while True:
            ok, frame = cap.read()
            if not ok:
                break
                
            if frame_idx in kept_set:
                if frame_idx in val_frames:
                    out_path = images_out_dir_val / f"{prefix}_{frame_idx}.jpg"
                    cv2.imwrite(str(out_path), frame)
                    saved_val += 1
                elif frame_idx in train_frames:
                    out_path = images_out_dir / f"{prefix}_{frame_idx}.jpg"
                    cv2.imwrite(str(out_path), frame)
                    saved_train += 1
            
            frame_idx += 1
        
        cap.release()
        
        print(f"[{prefix}] saved {saved_train} frames to {images_out_dir}")
        print(f"[{prefix}] saved {saved_val} frames to {images_out_dir_val}")
        
        # Create label dataframes
        label_cols = [c for c in df_train.columns if c != frame_col]
        
        labels_out_train = df_train[label_cols].copy()
        labels_out_train.insert(0, "filename", [f"{prefix}_{f}.jpg" for f in df_train[frame_col]])
        
        labels_out_val = df_val[label_cols].copy()
        labels_out_val.insert(0, "filename", [f"{prefix}_{f}.jpg" for f in df_val[frame_col]])
        
        return labels_out_train, labels_out_val
    
    else:
        # No validation split
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        saved = 0
        frame_idx = 0
        
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx in kept_set:
                out_path = images_out_dir / f"{prefix}_{frame_idx}.jpg"
                cv2.imwrite(str(out_path), frame)
                saved += 1
            frame_idx += 1
        
        cap.release()
        
        print(f"[{prefix}] saved {saved} frames to {images_out_dir}")
        
        label_cols = [c for c in df_kept.columns if c != frame_col]
        labels_out = df_kept[label_cols].copy()
        labels_out.insert(0, "filename", [f"{prefix}_{f}.jpg" for f in kept_frames])
        return labels_out


def build_split(
    split: str,
    prefixes: List[str],            
    video_dir: str | Path,          
    csv_dir: str | Path,            
    data_root: str | Path = "data",
    val_frac: float = 0.0
) -> Tuple[Path, Path]:
    """
    One-call pipeline per split: extracts frames from videos AND writes merged labels.csv.
    Returns: (images_dir, merged_labels_csv_path) or ((train_images_dir, val_images_dir), (train_labels_path, val_labels_path)) if val_frac > 0
    """
    split = split.lower()
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'.")
    
    video_dir = Path(video_dir)
    csv_dir = Path(csv_dir)
    data_root = Path(data_root)

    if val_frac == 0.0:
        # no validation split
        images_out_dir = data_root / split
        labels_out_dir = data_root / f"{split}_labels"
        
        # Remove existing directory and its content
        if images_out_dir.exists():
            shutil.rmtree(images_out_dir)
            
        labels_out_dir.mkdir(parents=True, exist_ok=True)

        merged_parts = []
        for p in prefixes:
            part = _extract_and_merge_for_prefix(
                prefix=p,
                video_dir=video_dir,
                csv_dir=csv_dir,
                images_out_dir=images_out_dir,
            )
            merged_parts.append(part)

        merged_df = pd.concat(merged_parts, ignore_index=True)

        # Ensure class columns are exactly those in CSVs beyond filename
        cols = ["filename"] + [c for c in merged_df.columns if c != "filename"]
        merged_df = merged_df[cols]

        merged_csv_path = labels_out_dir / "labels.csv"
        merged_df.to_csv(merged_csv_path, index=False)
        print(f"[{split}] merged labels saved to: {merged_csv_path} (rows={len(merged_df)})")
        
        return images_out_dir, merged_csv_path
    
    else:
        # Validation split logic
        images_out_dir_train = data_root / "train"
        images_out_dir_val = data_root / "val"
        labels_out_dir_train = data_root / "train_labels"
        labels_out_dir_val = data_root / "val_labels"
        
        # Clean existing directories
        if images_out_dir_train.exists():
            shutil.rmtree(images_out_dir_train)
        if images_out_dir_val.exists():
            shutil.rmtree(images_out_dir_val)
            
        labels_out_dir_train.mkdir(parents=True, exist_ok=True)
        labels_out_dir_val.mkdir(parents=True, exist_ok=True)
        
        merged_parts_train = []
        merged_parts_val = []
        
        for p in prefixes:
            train_part, val_part = _extract_and_merge_for_prefix(
                prefix=p,
                video_dir=video_dir,
                csv_dir=csv_dir,
                images_out_dir=images_out_dir_train,
                images_out_dir_val=images_out_dir_val,
                val_frac=val_frac
            )
            merged_parts_train.append(train_part)
            merged_parts_val.append(val_part)
        
        # Merge train parts
        merged_df_train = pd.concat(merged_parts_train, ignore_index=True)
        cols = ["filename"] + [c for c in merged_df_train.columns if c != "filename"]
        merged_df_train = merged_df_train[cols]
        
        # Merge val parts
        merged_df_val = pd.concat(merged_parts_val, ignore_index=True)
        merged_df_val = merged_df_val[cols]
        
        # Save merged labels
        merged_csv_path_train = labels_out_dir_train / "labels.csv"
        merged_csv_path_val = labels_out_dir_val / "labels.csv"
        
        merged_df_train.to_csv(merged_csv_path_train, index=False)
        merged_df_val.to_csv(merged_csv_path_val, index=False)
        
        print(f"[train] merged labels saved to: {merged_csv_path_train} (rows={len(merged_df_train)})")
        print(f"[val] merged labels saved to: {merged_csv_path_val} (rows={len(merged_df_val)})")
        
        return (images_out_dir_train, images_out_dir_val), (merged_csv_path_train, merged_csv_path_val)


def build_train(prefixes: List[str], video_dir: str | Path, csv_dir: str | Path,
                data_root: str | Path = "data", val_frac: float = 0.0):
    return build_split("train", prefixes, video_dir, csv_dir, data_root, val_frac)


def build_test(prefixes: List[str], video_dir: str | Path, csv_dir: str | Path,
               data_root: str | Path = "data"):
    return build_split("test", prefixes, video_dir, csv_dir, data_root)

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

def pred_and_plot_image_multilabel(
    model: torch.nn.Module,
    image_paths: List[Union[str, Path]],
    class_names: List[str],
    grid: Tuple[int, int] = (2, 3),
    image_size: Tuple[int, int] = (224, 224),
    transform: Optional[transforms.Compose] = None,
    device: torch.device = torch.device("cpu"),
    thresh: float = 0.5,
    suptitle: Optional[str] = "Model Predictions"
):
    model = model.to(device).eval()
    image_paths = [Path(p) for p in image_paths]
    rows, cols = grid
    max_slots = rows * cols
    paths = image_paths[:max_slots]

    PRINT_EPS = 5e-4
    # ✅ Use caller-provided transform, else safe default with ImageNet normalization
    tfm = transform
    imgs_pil = [Image.open(p).convert("RGB") for p in paths]
    tensors = [tfm(im) for im in imgs_pil]
    batch = torch.stack(tensors, dim=0).to(device)

    with torch.inference_mode():
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu()

    pred_idx_lists = [(p >= thresh).nonzero(as_tuple=True)[0].tolist() for p in probs]
    pred_name_lists = [[class_names[i] for i in idxs] for idxs in pred_idx_lists]

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax_i in range(max_slots):
        ax = axes[ax_i]
        if ax_i < len(paths):
            im = imgs_pil[ax_i]
            names = pred_name_lists[ax_i]
            pr = probs[ax_i]
            shown = [
                f"{cls} ({pr[class_names.index(cls)]:.2f})"
                for cls in names
                if round(pr[class_names.index(cls)].item(), 2) != 0.0
            ] if names else []
            title = ", ".join(shown) if shown else f"none ≥ {thresh:.2f}"

            ax.imshow(im)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        else:
            ax.axis("off")

    if suptitle:
        plt.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    plt.show()

    return [(p, pred_name_lists[i], probs[i]) for i, p in enumerate(paths)]



def get_class_names(csv_path: str):
    df = pd.read_csv(csv_path)
    return [c for c in df.columns if c != 'filename']