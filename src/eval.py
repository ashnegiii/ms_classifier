from pathlib import Path
from pyexpat import model
import PIL
import torch
import torchvision.models as models
from data_setup import create_dataloaders
from engine import test_step
from train import BATCH_SIZE, NUM_WORKERS
from utils import load_model, load_vit_model, pred_and_plot_image_multilabel
import random
from torchvision import transforms

from utils import calc_pos_weight

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Get class names first
    _, test_dataloader, class_names = create_dataloaders(
        train_dir=Path("data/train"),
        test_dir=Path("data/test"),
        train_data_fraction=0.1,
        test_data_fraction=0.1,
        train_transform=None,
        test_transform=test_transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=device
    )

    # Load model with the correct architecture
    model = load_vit_model(
        model_path="models/vit_b_16_dl100_e3_bs32_lr0.0001_2025-08-17_21-23-09.pth",
        num_classes=len(class_names),
        device=device
    )
    pos_weight = calc_pos_weight("data/train_labels/labels.csv", device)
    print(f"Following weights were calculated: {pos_weight}")
    
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    #test_metrics = test_step(model, test_dataloader, loss_fn, device)

    #print(f"  Test - Loss: {test_metrics['loss']:.4f}")
    #print(f"  Test - Accuracy per class: {[f'{a:.4f}' for a in test_metrics['accuracy_per_class']]}")
    #print(f"  Test - Precision per class: {[f'{a:.4f}' for a in test_metrics['precision_per_class']]}")
    #print(f"  Test - Recall per class: {[f'{a:.4f}' for a in test_metrics['recall_per_class']]}")
    #print(f"  Test - Average Precision per class: {[f'{a:.4f}' for a in test_metrics['ap_per_class']]}")
    #print(f"  Test - mAP: {test_metrics['mAP_macro']:.4f}")


    # Pick test images
    imgs = random.sample(list(Path("data/test").rglob("*.jpg")), 12)

    _ = pred_and_plot_image_multilabel(
        model=model,
        image_paths=imgs,
        class_names=class_names,
        grid=(3, 3),
        image_size=(224, 224),
        transform=test_transform,
        device=device,
        thresh=0.0
    )

if __name__ == "__main__":
    main()