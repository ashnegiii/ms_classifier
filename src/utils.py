import shutil
import torch
from pathlib import Path
import cv2
import pandas as pd
from typing import Tuple, List, Optional, Union
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt



def calc_pos_weight(csv_path, device):
    df = pd.read_csv(csv_path)
    
    label_cols = [c for c in df.columns if c != 'filename']
    
    weights = []
    for column in label_cols:
        positive_examples = df[column].sum()
        negative_examples = df.shape[0] - positive_examples
        
        if positive_examples > 0:
            weight = negative_examples / positive_examples
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


def _extract_and_merge_for_prefix(
    prefix: str,
    video_dir: Path,
    csv_dir: Path,
    images_out_dir: Path,
) -> pd.DataFrame:
    """Extract frames for a single prefix and return a labels DataFrame (with filename column, no '.jpg')."""
    video_path = video_dir / f"{prefix}.avi"
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
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    saved = 0
    frame_idx = 0
    target_remaining = set(kept_frames)

    while True and target_remaining:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in kept_set:
            out_path = images_out_dir / f"{prefix}_{frame_idx}.jpg"
            cv2.imwrite(str(out_path), frame)
            target_remaining.discard(frame_idx)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"[{prefix}] saved {saved} frames to {images_out_dir}")

    label_cols = [c for c in df.columns if c != frame_col]
    labels_out = df_kept[label_cols].copy()
    labels_out.insert(0, "filename", [f"{prefix}_{f}.jpg" for f in kept_frames])
    return labels_out

def build_split(
    split: str,
    prefixes: List[str],            
    video_dir: str | Path,          
    csv_dir: str | Path,            
    data_root: str | Path = "data",
) -> Tuple[Path, Path]:
    """
    One-call pipeline per split: extracts frames from videos AND writes merged labels.csv.
    Returns: (images_dir, merged_labels_csv_path)
    """
    split = split.lower()
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'.")

    video_dir = Path(video_dir)
    csv_dir = Path(csv_dir)
    data_root = Path(data_root)

    images_out_dir = data_root / split        # data/train or data/test
    labels_out_dir = data_root / f"{split}_labels"  # data/train_labels or data/test_labels
    # remove directory and its content
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

    # OPTIONAL: ensure class columns are exactly those in CSVs beyond filename
    # and sort columns: filename first, then classes
    cols = ["filename"] + [c for c in merged_df.columns if c != "filename"]
    merged_df = merged_df[cols]

    merged_csv_path = labels_out_dir / "labels.csv"
    merged_df.to_csv(merged_csv_path, index=False)
    print(f"[{split}] merged labels saved to: {merged_csv_path} (rows={len(merged_df)})")

    return images_out_dir, merged_csv_path

def build_train(prefixes: List[str], video_dir: str | Path, csv_dir: str | Path,
                data_root: str | Path = "data"):
    return build_split("train", prefixes, video_dir, csv_dir, data_root)

def build_test(prefixes: List[str], video_dir: str | Path, csv_dir: str | Path,
               data_root: str | Path = "data"):
    return build_split("test", prefixes, video_dir, csv_dir, data_root)

def load_model(model_path: str, model_class, model_args: dict, device: torch.device, num_classes: int):
    model = model_class(**model_args)

    # Change classifier to match number of classes from training
    if hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Sequential):
        in_features = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
        )

    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)  # no mismatch now

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


def analyze_class_distribution(dataloader):
    all_labels = []
    for _, labels in dataloader:
        all_labels.append(labels)
