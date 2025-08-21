"""
Contains functionality for creating PyTorch DataLoader's for image classification data.
"""

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import MultiLabelImageDataset
from experiment_config import ExperimentConfig
from utils import analyze_class_distribution_from_df


def create_dataloaders(
    images_dir: str,
    train_transform: transforms.Compose,
    test_transform: transforms.Compose,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    train_split: float = 0.8,
    test_split: float = 0.2,
    val_split: float = 0.0,
    episode_splits: dict = None,
):
    """
    Creates training, validation, and testing DataLoaders.
    Supports either fraction-based splits or explicit episode-based splits.
    """

    # get data
    csv_path = images_dir.parent / "labels" / "labels.csv"
    df = pd.read_csv(csv_path)

    # -------------------------
    # FRACTION MODE
    # -------------------------
    if episode_splits is None:
        train_df, temp_df = train_test_split(
            df,
            train_size=train_split,
            random_state=ExperimentConfig.RANDOM_SEED,
            shuffle=True,
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_split,
            random_state=ExperimentConfig.RANDOM_SEED,
            shuffle=True,
        )
    else:

        def filter_by_episode(df, episodes):
            if not episodes:
                return pd.DataFrame(columns=df.columns)
            mask = df["filename"].str.split("_").str[0].isin(episodes)
            return df[mask]

        train_df = filter_by_episodes(df, episode_splits["train"])
        val_df = filter_by_episodes(df, episode_splits["val"])
        test_df = filter_by_episodes(df, episode_splits["test"])

    analyze_class_distribution_from_df(df=train_df, label="TRAINING")
    if val_split > 0.0 or (episode_splits and episode_splits["val"]):
        analyze_class_distribution_from_df(df=train_df, label="VALIDATION")
    analyze_class_distribution_from_df(df=test_df, label="TESTING")
    # -------------------------
    # Create datasets
    # -------------------------
    train_data = MultiLabelImageDataset(
        root=images_dir,
        df=train_df,
        transform=train_transform,
    )
    val_data = MultiLabelImageDataset(
        root=images_dir,
        df=val_df,
        transform=test_transform,
    )
    test_data = MultiLabelImageDataset(
        root=images_dir,
        df=test_df,
        transform=test_transform,
    )

    class_names = train_data.classes

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    return train_dataloader, val_dataloader, test_dataloader, class_names
