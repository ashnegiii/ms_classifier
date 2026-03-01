
"""
This script runs a trained model on a video and outputs per-frame, per-character predictions to a CSV.

Usage (from repository root):
    PYTHONPATH=. python video_prediction/predict_video.py --video path/to/video.mp4 --model path/to/model.pth [--model-type clip_vitb16]

Output: CSV with same name as video, in same folder (e.g. video.mp4 -> video.csv)
Format: frame,kermit,miss_piggy,cook,statler_waldorf,rowlf_the_dog,fozzie_bear
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import torch
from PIL import Image
from torchvision import transforms

from src.backbone.clip_vit_b16 import CLIPViTB16
from src.backbone.convnext_tiny import ConvNeXtTiny
from src.backbone.effnet_b2 import EfficientNetB2
from src.backbone.resnet import ResNet50
from src.utils import load_backbone, load_vit_model

CLASS_NAMES = [
    "kermit",
    "miss_piggy",
    "cook",
    "statler_waldorf",
    "rowlf_the_dog",
    "fozzie_bear",
]
NUM_CLASSES = len(CLASS_NAMES)

# ImageNet transform for CNN backbones
TRANSFORM_IMAGENET = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
# CLIP-specific transform
TRANSFORM_CLIP = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    ),
])

MODEL_TYPES = {"effnetb2", "resnet50", "convnext_tiny", "clip_vitb16"}



def _infer_model_type(model_path: str) -> str:
    """Infer model type from filename (e.g. clip_vitb16.pth -> clip_vitb16)."""
    name = Path(model_path).stem.lower()
    if "clip" in name or "clip_vitb16" in name:
        return "clip_vitb16"
    if "effnet" in name or "effnet_b2" in name:
        return "effnetb2"
    if "resnet" in name or "resnet_50" in name:
        return "resnet50"
    if "convnext" in name or "conv-next" in name:
        return "convnext_tiny"
    raise ValueError(
        f"Cannot infer model type from '{model_path}'. "
        f"Use --model-type. Allowed: {', '.join(sorted(MODEL_TYPES))}"
    )


def load_model_and_transform(model_path: str, model_type: str, device: torch.device):
    """Load model and return (model, transform) for inference."""
    if model_type == "effnetb2":
        model = load_backbone(
            EfficientNetB2,
            model_path=model_path,
            num_classes=NUM_CLASSES,
            device=device,
        )
        return model, TRANSFORM_IMAGENET
    if model_type == "resnet50":
        model = load_backbone(
            ResNet50,
            model_path=model_path,
            num_classes=NUM_CLASSES,
            device=device,
        )
        return model, TRANSFORM_IMAGENET
    if model_type == "convnext_tiny":
        model = load_backbone(
            ConvNeXtTiny,
            model_path=model_path,
            num_classes=NUM_CLASSES,
            device=device,
        )
        return model, TRANSFORM_IMAGENET
    if model_type == "clip_vitb16":
        backbone = CLIPViTB16(
            device=device,
            pretrained=False,
            augmentation=False,
            unfreeze_last_n=0,
            out_features=NUM_CLASSES,
        )
        state = torch.load(model_path, map_location=device)
        backbone.load_state_dict(state, strict=True)
        backbone.eval()
        return backbone, TRANSFORM_CLIP
    raise ValueError(f"Unknown model_type: {model_type!r}")


def predict_video(video_path: Path, model_path: Path, model_type: str, output_csv: Path):
    """Run model on each frame and write predictions to CSV."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = load_model_and_transform(str(model_path), model_type, device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    rows = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR -> RGB, then to PIL for transform
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        row = [frame_idx] + [float(probs[i]) for i in range(NUM_CLASSES)]
        rows.append(row)
        frame_idx += 1

    cap.release()

    # Write CSV
    header = "frame," + ",".join(CLASS_NAMES)
    with open(output_csv, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")

    print(f"Wrote {len(rows)} frames to {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Run model on video and output per-frame predictions to CSV."
    )
    parser.add_argument("--video", required=True, type=Path, help="Path to video file")
    parser.add_argument("--model", required=True, type=Path, help="Path to .pth model file")
    parser.add_argument(
        "--model-type",
        choices=sorted(MODEL_TYPES),
        default=None,
        help="Model architecture (inferred from filename if not given)",
    )
    args = parser.parse_args()

    if not args.video.exists():
        sys.exit(f"Video not found: {args.video}")
    if not args.model.exists():
        sys.exit(f"Model not found: {args.model}")

    model_type = args.model_type or _infer_model_type(str(args.model))
    output_csv = args.video.with_suffix(".csv")

    predict_video(args.video, args.model, model_type, output_csv)


if __name__ == "__main__":
    main()
