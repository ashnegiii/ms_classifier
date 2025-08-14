import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class EfficientNetB2():
    def __init__(self, out_features, device: torch.device):
        super().__init__()
        self.out_features = out_features
        self.weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
        self.model = torchvision.models.efficientnet_b2(weights = self.weights).to(device)

        for param in self.model.features.parameters():
            param.requires_grad = False
        
        self.model_name ="effnetb2"
        
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1408, out_features=self.out_features,
            bias=True).to(device)
        )
        
        self.train_transform = transforms.Compose([
            transforms.Resize(288, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(288),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(288, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        print(f"INFO] Created new {self.model_name} model.")




