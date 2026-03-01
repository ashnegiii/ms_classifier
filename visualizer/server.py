import base64
import io
import os
import sys
from pathlib import Path
from typing import Any

# Project root so "src" is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms

from src.backbone.clip_vit_b16 import CLIPViTB16
from src.backbone.convnext_tiny import ConvNeXtTiny
from src.backbone.effnet_b2 import EfficientNetB2
from src.backbone.resnet import ResNet50
from src.backbone.vit_b16 import ViTB16
from src.eval.gradcam import GradCAM
from src.utils import load_backbone, load_vit_model

# config
CLASS_NAMES = [
    "kermit",
    "miss_piggy",
    "cook",
    "statler_waldorf",
    "rowlf_the_dog",
    "fozzie_bear",
]

# Map frontend character IDs to backend class name
CHARACTER_ID_TO_CLASS: dict[str, str] = {
    "kermit": "kermit",
    "miss_piggy": "miss_piggy",
    "cook": "cook",
    "the_cook": "cook",
    "statler_waldorf": "statler_waldorf",
    "rowlf_the_dog": "rowlf_the_dog",
    "fozzie_bear": "fozzie_bear",
}
DISPLAY_NAMES: dict[str, str] = {
    "kermit": "Kermit",
    "miss_piggy": "Miss Piggy",
    "cook": "The Swedish Chef",
    "statler_waldorf": "Statler & Waldorf",
    "rowlf_the_dog": "Rowlf The Dog",
    "fozzie_bear": "Fozzie Bear",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = len(CLASS_NAMES)

# Standard ImageNet transform for CNN backbones
test_transform_imagenet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Allowed model types
MODEL_TYPES = {"effnetb2", "resnet50", "convnext_tiny", "clip_vitb16"}

# Folder for .pth checkpoints
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "visualizer/models")).resolve()

# Filenames per architecture
MODEL_TYPE_TO_FILENAME = {
    "resnet50": "resnet_50.pth",
    "effnetb2": "effnet_b2.pth",
    "convnext_tiny": "convnext_tiny.pth",
    "clip_vitb16": "clip_vitb16.pth",
}


def _find_last_conv2d(module: nn.Module) -> nn.Module | None:
    """Find the last Conv2d in a module tree (depth-first order)."""
    last_conv: nn.Module | None = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv


def _get_target_layer_and_transform(
    model: nn.Module,
    model_type: str,
) -> tuple[nn.Module, Any]:
    """Return (target_layer for Grad-CAM, transform) for the given loaded model and type."""
    if model_type == "effnetb2":
        target = None
        for m in reversed(list(model.features[-1].modules())):
            if isinstance(m, nn.Conv2d):
                target = m
                break
        if target is None:
            raise ValueError("Could not find Conv2d in EfficientNet-B2 features")
        return target, test_transform_imagenet

    if model_type == "resnet50":
        # ResNet50: last conv in layer4
        target = None
        for m in reversed(list(model.layer4.modules())):
            if isinstance(m, nn.Conv2d):
                target = m
                break
        if target is None:
            raise ValueError("Could not find Conv2d in ResNet50 layer4")
        return target, test_transform_imagenet

    if model_type == "convnext_tiny":
        target = _find_last_conv2d(model.features)
        if target is None:
            raise ValueError("Could not find Conv2d in ConvNeXt-Tiny features")
        return target, test_transform_imagenet

    if model_type == "clip_vitb16":
        # CLIP ViT: visual encoder; use first/last Conv2d (patch embed)
        target = _find_last_conv2d(model.visual)
        if target is None:
            raise ValueError("Could not find Conv2d in CLIP visual encoder")
        # CLIP uses its own normalizer in the backbone
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
        return target, transform

    raise ValueError(f"Unknown model_type: {model_type}")


def _load_model_by_type(model_type: str, model_path: str) -> tuple[nn.Module, nn.Module, Any]:
    """Load model from path for the given type. Returns (model, target_layer, transform)."""
    if model_type == "effnetb2":
        model = load_backbone(
            EfficientNetB2,
            model_path=model_path,
            num_classes=NUM_CLASSES,
            device=device,
        )
    elif model_type == "resnet50":
        model = load_backbone(
            ResNet50,
            model_path=model_path,
            num_classes=NUM_CLASSES,
            device=device,
        )
    elif model_type == "convnext_tiny":
        model = load_backbone(
            ConvNeXtTiny,
            model_path=model_path,
            num_classes=NUM_CLASSES,
            device=device,
        )
    elif model_type == "clip_vitb16":
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
        model = backbone
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model_type: {model_type!r}")

    target_layer, transform = _get_target_layer_and_transform(model, model_type)
    return model, target_layer, transform


app = FastAPI(title="Muppet Show Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# App state: model, gradcam, transform (set by POST /api/model)
app.state.model = None
app.state.gradcam = None
app.state.test_transform = test_transform_imagenet


class LoadModelRequest(BaseModel):
    modelType: str


class PredictRequest(BaseModel):
    frameBase64: str
    characterIds: list[str]
    frameNumber: int | None = None


class PredictionItem(BaseModel):
    characterId: str
    characterName: str
    confidence: float
    gradCamImageBase64: str


class PredictResponse(BaseModel):
    predictions: list[PredictionItem]
    frameNumber: int | None = None


def _get_class_index(character_id: str) -> int | None:
    """Map frontend character ID to backend class index."""
    class_name = CHARACTER_ID_TO_CLASS.get(character_id)
    if class_name is None:
        return None
    try:
        return CLASS_NAMES.index(class_name)
    except ValueError:
        return None


def _build_gradcam_overlay(
    img_np: np.ndarray,
    heatmap: np.ndarray,
) -> np.ndarray:
    """Build overlay image: heatmap (JET) resized to image size, blended with original (60% original, 40% heatmap)."""
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    heatmap_colored = cv2.resize(
        heatmap_colored, (img_np.shape[1], img_np.shape[0])
    )
    overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
    return overlay


@app.post("/api/model")
async def load_model(body: LoadModelRequest):
    """Load a model from the backend visualizer/models/ folder. Architecture selects the file automatically."""
    model_type = body.modelType.strip().lower()
    if model_type not in MODEL_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"modelType must be one of: {', '.join(sorted(MODEL_TYPES))}",
        )
    filename = MODEL_TYPE_TO_FILENAME[model_type]
    model_path = MODEL_DIR / filename
    if not model_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Model file not found: {filename}. Put {filename} in the '{MODEL_DIR}' folder.",
        )

    try:
        model, target_layer, transform = _load_model_by_type(model_type, str(model_path))
        model.eval()

        gradcam = GradCAM(model, target_layer)
        app.state.model = model
        app.state.gradcam = gradcam
        app.state.test_transform = transform
        return {"status": "ok", "message": "Model loaded", "modelType": model_type, "filename": filename}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load model: {str(e)}",
        )


def _decode_frame(frame_base64: str) -> tuple[np.ndarray, Image.Image]:
    """Decode base64 or data URL to numpy RGB and PIL Image."""
    raw = frame_base64
    if "," in raw and raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    img_bytes = base64.b64decode(raw)
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img_pil)
    return img_np, img_pil


@app.post("/api/predict", response_model=PredictResponse)
async def predict(body: PredictRequest):
    """Run prediction and Grad-CAM for the given frame and selected character IDs."""
    model = app.state.model
    gradcam = app.state.gradcam
    if model is None or gradcam is None:
        raise HTTPException(
            status_code=503,
            detail="Load model first (POST /api/model with .pth file)",
        )
    if not body.characterIds:
        raise HTTPException(
            status_code=400,
            detail="characterIds must not be empty",
        )

    try:
        img_np, img_pil = _decode_frame(body.frameBase64)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image data: {str(e)}",
        )

    transform = app.state.test_transform
    transformed = transform(img_pil)
    img_tensor = transformed.unsqueeze(0).to(device)
    img_tensor.requires_grad_(True)

    with torch.enable_grad():
        outputs = model(img_tensor)

    with torch.no_grad():
        probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

    predictions: list[PredictionItem] = []
    for character_id in body.characterIds:
        class_idx = _get_class_index(character_id)
        if class_idx is None:
            continue
        confidence = float(probs[class_idx])
        class_name = CLASS_NAMES[class_idx]
        display_name = DISPLAY_NAMES.get(class_name, class_name)

        heatmap = gradcam.generate(class_idx, outputs, img_tensor)
        overlay = _build_gradcam_overlay(img_np, heatmap)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".png", overlay_rgb)
        gradcam_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        predictions.append(
            PredictionItem(
                characterId=character_id,
                characterName=display_name,
                confidence=confidence,
                gradCamImageBase64=gradcam_b64,
            )
        )

    return PredictResponse(
        predictions=predictions,
        frameNumber=body.frameNumber,
    )
