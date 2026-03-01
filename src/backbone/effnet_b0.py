import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class EfficientNetB0():
    def __init__(self, device: torch.device, pretrained: bool, augmentation: bool, unfreeze_last_n: int = 0, out_features: int = 0):
        super().__init__()
        self.out_features = out_features
        self.weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        self.model = torchvision.models.efficientnet_b0(
            weights=self.weights if pretrained else None)

        for param in self.model.features.parameters():
            param.requires_grad = False

        if unfreeze_last_n > 0:
            for layer in self.model.features[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True

        self.model_name = "effnetb0"

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=self.out_features,
                      bias=True)
        )

        self.model = self.model.to(device)

        if augmentation:
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225]),
            ])
        else:
            self.train_transform = transforms.Compose([
                transforms.Resize(
                    (224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225]),
            ])

        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)

        print(f"[INFO] Created new {self.model_name} model.")
        print(f"[INFO] Total parameters: {total_params:,}")
        print(
            f"[INFO] Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
