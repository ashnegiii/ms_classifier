import torch
from torch import nn
import open_clip
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class CLIPViTB16:
    def __init__(self, device: torch.device, unfreeze_last_n: int = 0, out_features: int = 0):
        super().__init__()
        self.model_name = "clip_vitb16"
        self.out_features = out_features

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai"
        )
        self.model = model.visual.to(device)

        for p in self.model.parameters():
            p.requires_grad = False
            
        if unfreeze_last_n > 0:
            for layer in list(self.model.transformer.resblocks)[-unfreeze_last_n:]:
                for p in layer.parameters():
                    p.requires_grad = True

        self.model.proj = nn.Linear(768, out_features).to(device)

        self.train_transform = preprocess
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])
