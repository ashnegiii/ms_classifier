import torch
from torch import nn
import open_clip
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class CLIPViTB16(nn.Module):  # Inherit from nn.Module
    def __init__(self, device: torch.device, pretrained: bool, augmentation: bool, unfreeze_last_n: int = 0, out_features: int = 0):
        super().__init__()
        self.model_name = "clip_vitb16"
        self.out_features = out_features

        # load pretrained CLIP (only visual backbone)
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai" if pretrained else None
        )
        self.visual = model.visual.to(device)

        # freeze everything
        for p in self.visual.parameters():
            p.requires_grad = False

        # unfreeze last n transformer blocks
        if unfreeze_last_n > 0:
            for layer in list(self.visual.transformer.resblocks)[-unfreeze_last_n:]:
                for p in layer.parameters():
                    p.requires_grad = True

        # Get the output dimension from the visual model
        # ViT-B-16 outputs 512-dim, ViT-B-32 outputs 768-dim
        visual_output_dim = self.visual.output_dim

        # add custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(visual_output_dim, out_features, bias=True)
        ).to(device)

        # use CLIP preprocessing for train/test

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
            transforms.Resize(
                (224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        # stats
        total_params = sum(p.numel() for p in self.visual.parameters()) \
            + sum(p.numel() for p in self.classifier.parameters())
        trainable_params = sum(p.numel() for p in self.visual.parameters() if p.requires_grad) \
            + sum(p.numel() for p in self.classifier.parameters())

        print(f"[INFO] Created new {self.model_name} model.")
        print(f"[INFO] Total parameters: {total_params:,}")
        print(f"[INFO] Trainable parameters: {trainable_params:,} "
              f"({trainable_params/total_params*100:.2f}%)")

    def forward(self, x):
        feats = self.visual(x)          # extract CLIP visual features
        out = self.classifier(feats)    # pass through classifier head
        return out
