import torch
from torch import nn
import torchvision
from torchvision import transforms


class EfficientNetB0():
    def __init__(self, out_features, device: torch.device):
        super().__init__()
        self.out_features = out_features
        self.weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        self.model = torchvision.models.efficientnet_b0(weights = self.weights)

        for param in self.model.features.parameters():
            param.requires_grad = False
        
        self.model_name ="effnetb0"

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=self.out_features,
            bias=True)
        )

        self.model = self.model.to(device)
        
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print(f"[INFO] Created new {self.model_name} model.")
