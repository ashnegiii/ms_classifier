import torch
from pathlib import Path

from torchvision import transforms
from backbone.effnet_b0 import EfficientNetB0
from backbone.effnet_b2 import EfficientNetB2
from data_setup import create_dataloaders
import engine, utils
from timeit import default_timer as timer
from datetime import datetime
import pandas as pd
from utils import calc_pos_weight

import os
# Config
RANDOM_SEED = 42
NUM_WORKERS = 2
print(os.cpu_count())

# Hyperparameter
TRAIN_DATA_FRACTION = 0.02
TEST_DATA_FRACTION = 0.02
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
OUTPUT_THRESHOLD = 0.5

train_dir = Path("data/train")
test_dir  = Path("data/test")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Following device active:", device)
   
    

    effnet_b0 = EfficientNetB2(out_features=6, device=device)
    
    train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=train_dir,
                                                                        test_dir=test_dir,
                                                                  train_data_fraction=TRAIN_DATA_FRACTION,
                                                                  test_data_fraction=TEST_DATA_FRACTION,
                                                                  train_transform = effnet_b0.train_transform,
                                                                  test_transform = effnet_b0.test_transform,
                                                                  batch_size = BATCH_SIZE,
                                                                  num_workers=NUM_WORKERS,
                                                                  device=device)
    
    pos_weight = calc_pos_weight("data/train_labels/labels.csv", device)
    print(f"Following weights were calculated: {pos_weight}")
    


    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(effnet_b0.model.parameters(), lr=LEARNING_RATE)

    start = timer()
    engine.train(
        model=effnet_b0.model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=NUM_EPOCHS,
        device=device,
    )
    print(f"[INFO] Total training time: {timer() - start:.3f}s")

    dataloader_name = effnet_b0.model_name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Add timestamp to model filename
    save_filepath = f"{dataloader_name}_{NUM_EPOCHS}_epochs_{timestamp}.pth"

    utils.save_model(effnet_b0.model, target_dir="models",
                     model_name=save_filepath)

if __name__ == "__main__":
    main()
