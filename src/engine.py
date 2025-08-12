
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torchmetrics.classification import MultilabelF1Score 


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               f1_metric,
               device: torch.device) -> Tuple[float, float]:
  """Trains a PyTorch model for a single epoch.

  Turns a target PyTorch model to training mode and then
  runs through all of the required training steps (forward
  pass, loss calculation, optimizer step).

  Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
  """
  # Put model in train mode
  model.train()

  # Setup train loss and train accuracy values
  running_loss = 0

  # Loop through data loader data batches
  pbar = tqdm(dataloader, desc="Train", leave=False)

  for X, y in pbar:
      # Send data to target device
      X, y = X.to(device), y.to(device).float()

      # 1. fwd + loss
      logits = model(X)
      loss = loss_fn(logits, y)
      running_loss += loss.item()

      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # 4. Loss backward
      loss.backward()

      # 5. Optimizer step
      optimizer.step()


      # Store the predictions and targets for later
      with torch.inference_mode():
        f1_metric.update(torch.sigmoid(logits), y.int())

      pbar.set_postfix(
        loss=f"{loss.item():.4f}",
        f1_score=f"{f1_metric.compute():.3f}"
    )

  # Adjust metrics to get average loss and accuracy per batch
  avg_loss = running_loss / len(dataloader)
  avg_f1 = f1_metric.compute().item()
  f1_metric.reset()
  return avg_loss, avg_f1

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              f1_metric,
              device: torch.device) -> Tuple[float, float]:
  """Tests a PyTorch model for a single epoch.

  Turns a target PyTorch model to "eval" mode and then performs
  a forward pass on a testing dataset.

  Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
  """
  # Put model in eval mode
  model.eval()
  running_loss = 0.0

  with torch.inference_mode():
      pbar = tqdm(dataloader, desc="Eval", leave=False)
      for X, y in pbar:
          X, y = X.to(device), y.to(device).float()

          logits = model(X)
          loss = loss_fn(logits, y)
          running_loss += loss.item()

          with torch.inference_mode():
            f1_metric.update(torch.sigmoid(logits), y.int())

          pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            f1_score=f"{f1_metric.compute():.3f}"
            )


  running_loss = running_loss / len(dataloader)
  avg_f1 = f1_metric.compute().item()
  f1_metric.reset()
  return running_loss, avg_f1

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          threshold: int,
          device: torch.device) -> Dict[str, List[float]]:
  """Trains and tests a PyTorch model.

  Passes a target PyTorch models through train_step() and test_step()
  functions for a number of epochs, training and testing the model
  in the same epoch loop.

  Calculates, prints and stores evaluation metrics throughout.

  Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]}
    For example if training for epochs=2:
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]}
  """
  # Create empty results dictionary
  
  num_classes = model.classifier[-1].out_features
  f1_metric = MultilabelF1Score(num_labels=num_classes, threshold=threshold, average="macro")

  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
      train_loss, train_f1 = train_step(model, train_dataloader, loss_fn, optimizer, f1_metric, device)
      test_loss, test_f1 = test_step(model, test_dataloader, loss_fn, f1_metric, device)

      # Print out what's happening
      print(f"Epoch {epoch+1} | "
          f"train_loss: {train_loss:.4f} | train_f1: {train_f1:.4f} | "
          f"test_loss: {test_loss:.4f} | test_f1: {test_f1:.4f}")



def multilabel_preds(logits, thresh=0.5):
  return (torch.sigmoid(logits) >= thresh).float()
