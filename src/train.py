import torch
import torchvision
from torch import nn
from pathlib import Path

from torchvision import transforms
from data_setup import create_dataloaders
import engine, utils
from timeit import default_timer as timer
import os
from datetime import datetime


RANDOM_SEED = 42
NUM_EPOCHS = 3
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001
OUTPUT_THRESHOLD = 0.5
NUM_WORKERS = os.cpu_count()

train_dir = Path("data/train")
test_dir  = Path("data/test")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Following device active:", device)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], # values per colour channel [red, green, blue]
                                 std=[0.229, 0.224, 0.225])

    simple_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
    ])

    train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=train_dir,
                                                                  test_dir = test_dir,
                                                                  train_transform = simple_transform,
                                                                  test_transform = simple_transform,
                                                                  batch_size = BATCH_SIZE,
                                                                  num_workers=NUM_WORKERS,
                                                                  device=device)

    model.classifier = nn.Sequential(
      nn.Dropout(p=0.2, inplace=True),
      nn.Linear(in_features=1280, out_features=len(class_names), bias=True)
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start = timer()
    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=NUM_EPOCHS,
        threshold=OUTPUT_THRESHOLD,
        device=device,
    )
    print(f"[INFO] Total training time: {timer() - start:.3f}s")

    dataloader_name = "dataloader_10"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Add timestamp to model filename
    save_filepath = f"effnet_b0_{dataloader_name}_{NUM_EPOCHS}_epochs_{timestamp}.pth"

    utils.save_model(model, target_dir="models",
                     model_name=save_filepath)

if __name__ == "__main__":
    main()
