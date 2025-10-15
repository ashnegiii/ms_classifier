import random
from engine import test_step_video

from pathlib import Path
import torch
from data_setup import create_dataloaders
from train import NUM_WORKERS
from utils import load_vit_model, pred_and_plot_image_multilabel
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Get class names first
    predict_dir = Path("data/predict/03-04-17")

    # Create dataset from all images (ignore labels)
    dataset = datasets.ImageFolder(
        root=predict_dir.parent,
        transform=test_transform
    )
    # All images are in one folder:
    dataset.samples = [(str(p), 0) for p in predict_dir.glob("*.jpg")]
    dataset.targets = [0] * len(dataset.samples)

    # Dataloader
    test_dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    # Define class names manually
    class_names = ["kermit", "miss_piggy", "cook", "statler_waldorf",
                   "rowlf_the_dog", "fozzie_bear"]
    
    # Load model with the correct architecture
    model = load_vit_model(
        model_path="models/vitb16_unfreeze3_dl80_e3_bs32_lr0.001_wd0.001_th0.4_mw3_2025-08-23_22-28-25.pth",
        num_classes=len(class_names),
        device=device
    )
    
    optimal_thresholds = {
            'kermit': 0.8,
            'miss_piggy': 0.8,
            'cook': 0.8,
            'statler_waldorf': 0.8,
            'rowlf_the_dog': 0.8,
            'fozzie_bear': 0.8
    }
    
    test_step_video(class_names=class_names,
                    dataloader=test_dataloader,
                    optimal_thresholds=optimal_thresholds,
                    model=model,
                    device=device,
                    output_csv_path="data/predict/03-04-17.xlsx")

    imgs = random.sample(list(Path("data/predict/03-04-17").rglob("*.jpg")), 12)

    _ = pred_and_plot_image_multilabel(
        model=model,
        image_paths=imgs,
        class_names=class_names,
        grid=(3, 3),
        image_size=(224, 224),
        transform=test_transform,
        device=device,
        thresh=0.0,
    )

if __name__ == "__main__":
    main()