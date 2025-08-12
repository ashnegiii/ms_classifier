# 0. Write a custom dataset class
from torch.utils.data import Dataset
import torch
from PIL import Image
from typing import Tuple, Dict, List
import pandas as pd
from torchvision import transforms
from pathlib import Path

class MultiLabelImageDataset(Dataset):
  def __init__(self,
               root: str,
               label_csv_path: str,
               transform=None):
    self.image_dir = Path(root)
    self.labels_df = pd.read_csv(label_csv_path)
    self.transform = transform
    self.classes, self.class_idx = self.find_classes(pd.read_csv(label_csv_path).columns)

  def __len__(self):
    " Returns the total number of samples (already removes the header)."
    return len(self.labels_df)

  def __getitem__(self, index: int):
    row = self.labels_df.iloc[index]
    filename = row["filename"]

    # 1. Load image
    img = Image.open(self.image_dir / filename).convert("RGB")

    # 2. Apply transform or default ToTensor
    if self.transform:
        img = self.transform(img)
    else:
        img = transforms.ToTensor()(img)

    # 3. Build multi-hot label vector
    label_vector = torch.zeros(len(self.classes), dtype=torch.float32)
    active_classes = row[row == 1].index.tolist()  # columns with value 1
    active_indices = [self.class_idx[name] for name in active_classes]
    label_vector[active_indices] = 1.0

    # 4. Return tensors
    return img, label_vector


  def find_classes(self, csv_columns: pd.Index) -> Tuple[List[str], Dict[str, int]]:
    """Finds the classes and indexes in a csv file, excluding the first column."""
    # Skip the first column
    class_names = list(csv_columns[1:])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
    return class_names, class_to_idx
