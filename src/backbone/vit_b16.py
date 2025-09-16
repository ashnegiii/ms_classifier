import torch
from torch import nn
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class ViTB16():
    def __init__(self, device: torch.device, unfreeze_last_n: int = 0, out_features: int = 0):
        super().__init__()
        self.out_features = out_features
        self.weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        self.model = models.vit_b_16(weights=self.weights)
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        if unfreeze_last_n > 0:
            for layer in self.model.encoder.layers[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        original_head_in_features = self.model.heads[0].in_features 
        
        self.model.heads = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=original_head_in_features, out_features=self.out_features, bias=True)
        )
        
        self.model_name = "vitb16"
        self.model = self.model.to(device)
        
        
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        #self.train_transform = transforms.Compose([
        #    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        #    transforms.RandomHorizontalFlip(),
        #    transforms.RandomRotation(15),
        #    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
        #    transforms.RandomPerspective(distortion_scale=0.3, p=0.3),

        #   transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        #    transforms.RandomGrayscale(p=0.1),
        #    transforms.RandomAutocontrast(p=0.2),
        #    transforms.RandomAdjustSharpness(2, p=0.3),
        #    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),

        #    transforms.ToTensor(),
        #    transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                        std=[0.229, 0.224, 0.225]),
        #    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        #])

        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"[INFO] Created new {self.model_name} model.")
        print(f"[INFO] Total parameters: {total_params:,}")
        print(f"[INFO] Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        
        
        