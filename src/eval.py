from pathlib import Path
import torch
import torchvision.models as models
from data_setup import create_dataloaders
from train import BATCH_SIZE, NUM_WORKERS
from utils import load_model, pred_and_plot_image_multilabel
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get class names first
_, test_dataloader, class_names = create_dataloaders(
    train_dir=Path("data/train"),
    test_dir=Path("data/test"),
    train_transform=None,
    test_transform=None,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    device=device
)

# Load model with the correct architecture
model = load_model(
    model_path="models/effnet_b0_dataloader_10_3_epochs_2025-08-12_05-51-37.pth",
    model_class=models.efficientnet_b0,
    model_args={"weights": None},
    device=device,
    num_classes=len(class_names)
)

# Pick test images
test_dir = Path("data/test")
imgs = random.sample(list(Path("data/test").rglob("*.jpg")), 6)

_ = pred_and_plot_image_multilabel(
    model=model,
    image_paths=imgs,
    class_names=class_names,
    grid=(2, 2),
    image_size=(224, 224),
    device=device,
    thresh=0
)