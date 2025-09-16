import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class EfficientNetB2():
    def __init__(self, device: torch.device, unfreeze_last_n: int = 0, out_features: int = 0):
        super().__init__()
        self.out_features = out_features
        self.weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
        self.model = torchvision.models.efficientnet_b2(weights = self.weights).to(device)

        for param in self.model.features.parameters():
            param.requires_grad = False
        
        if unfreeze_last_n > 0:
            for layer in self.model.features[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        self.model_name ="effnetb2"
        
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1408, out_features=self.out_features,
            bias=True).to(device)
        )
        
        
        self.train_transform = transforms.Compose([
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

        self.test_transform = transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"[INFO] Created new {self.model_name} model.")
        print(f"[INFO] Total parameters: {total_params:,}")
        print(f"[INFO] Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
