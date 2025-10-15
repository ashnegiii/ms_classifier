from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from backbone.effnet_b2 import EfficientNetB2
from exp_config import ExperimentConfig
from gradcam import GradCAM
from utils import load_backbone

# --- CONFIG ---
MODEL_PATH = "models/effnetb2_unfreeze2_episodes_e2_bs32_lr0.0001_wd0.001_th0.4_mw3_2025-09-17_22-36-00.pth"
IMAGE_DIR = Path("data/custom_images")  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.3),

    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomAutocontrast(p=0.2),
    transforms.RandomAdjustSharpness(2, p=0.3),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
])


# Define same transform used at test time
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def show_original_and_transformed(img_path, transform):
    """Load an image, apply transform, and show both original and transformed."""
    img = Image.open(img_path).convert("RGB")

    # Apply transform
    transformed = transform(img)

    # Convert back to show (denormalize)
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        return tensor * std + mean

    img_transformed = denormalize(transformed).clamp(0,1).permute(1,2,0)

    # Plot side by side
    fig, ax = plt.subplots(1,2, figsize=(8,4))
    ax[0].imshow(img)
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(img_transformed)
    ax[1].set_title("Transformed")
    ax[1].axis("off")

    plt.show()
    return transformed.unsqueeze(0)  # add batch dim

def main():
    class_names = ['kermit', 'miss_piggy', 'cook',
                   'statler_waldorf', 'rowlf_the_dog', 'fozzie_bear']

    model = load_backbone(EfficientNetB2,
                          model_path=MODEL_PATH,
                          num_classes=len(class_names),
                          device=device)
    model.eval()
    
    # Get the last conv layer in EfficientNet features
    target_layer = None
    for module in reversed(list(model.features[-1].modules())):
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
            break
    
    print(f"Using target layer: {target_layer}")
    gradcam = GradCAM(model, target_layer)

    for img_path in IMAGE_DIR.glob("*.*"):
        print(f"\nProcessing: {img_path}")
        
        #img_tensor = show_original_and_transformed(img_path, test_transform).to(device)
        img = Image.open(img_path).convert("RGB")
        transformed = test_transform(img)
        img_tensor = transformed.unsqueeze(0).to(device)
        img = Image.open(img_path).convert("RGB")
        
        
        img_np = np.array(img)

        # Forward pass with gradients
        img_tensor.requires_grad_(True)
        outputs = model(img_tensor)
        
        # Get probabilities
        with torch.no_grad():
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

        print("Predictions:", [cls for cls, p in zip(class_names, probs) if p >= 0.5])

        # Generate GradCAM for predicted classes
        for class_idx, p in enumerate(probs):
            if p >= 0.10:  # Show for any class with >10% confidence
                print(f"Generating GradCAM for {class_names[class_idx]} ({p:.3f})")
                
                heatmap = gradcam.generate(class_idx, outputs, img_tensor)
                
                # Create overlay
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                heatmap_colored = cv2.resize(heatmap_colored, (img_np.shape[1], img_np.shape[0]))
                overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)

                # Display
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                ax[0].imshow(img_np)
                ax[0].set_title("Original")
                ax[0].axis("off")
                ax[1].imshow(overlay)
                ax[1].set_title(f"GradCAM: {class_names[class_idx]} ({p:.3f})")
                ax[1].axis("off")
                plt.tight_layout()
                plt.show()


if __name__ == "__main__":
    main()
