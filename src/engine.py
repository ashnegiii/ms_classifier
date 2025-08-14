
import torch
from tqdm.auto import tqdm
from typing import Dict, List
from sklearn.metrics import average_precision_score, recall_score, accuracy_score, precision_score
import numpy as np
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Dict[str, float]:
    """Enhanced train step with metrics tracking"""
    model.train()
    running_loss = 0
    
    pbar = tqdm(dataloader, desc="Train", leave=False)
    
    for (X, y) in pbar:
        X, y = X.to(device), y.to(device).float()
        
        # Forward pass
        logits = model(X)
        loss = loss_fn(logits, y)
        running_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()                
        optimizer.step()

        
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    avg_loss = running_loss / len(dataloader)

    return {"loss": avg_loss}

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device,
              threshold = 0.5) -> Dict[str, float]:
    """Enhanced test step with metrics tracking"""
    model.eval()
    running_loss = 0.0
    all_logits = []
    all_targets = []
    
    with torch.inference_mode():
        pbar = tqdm(dataloader, desc="Eval", leave=False)
        for X, y in pbar:
            X, y = X.to(device), y.to(device).float()
            
            logits = model(X)
            loss = loss_fn(logits, y)
            running_loss += loss.item()
            
            with torch.inference_mode():
                all_logits.append(logits)
                all_targets.append(y)
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    avg_loss = running_loss / len(dataloader)
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    accuracy_per_class, precision_per_class, recall_per_class, ap_per_class, mAP = calc_metrics(targets=targets, logits=logits)
    
    return {
            "loss": avg_loss,
            "recall_per_class": recall_per_class,
            "accuracy_per_class": accuracy_per_class,
            "precision_per_class": precision_per_class,
            "ap_per_class": ap_per_class,
            "mAP_macro": mAP}
    

def calc_metrics(targets: torch.Tensor, logits: torch.Tensor, threshold = 0.5): 
    # Convert logits -> probabilities
    
    probs = torch.sigmoid(logits).cpu().numpy()
    y_true = targets.cpu().numpy().astype(int)
    
    y_pred_np = (probs >= threshold).astype(int)
    
    C = y_true.shape[1]
    
    accuracy_per_class = []
    precision_per_class = []
    ap_per_class = []
    recall_per_class = []
    
    for c in range(C):
        yt = y_true[:, c]
        yp = probs[:, c]
        yhat = y_pred_np[:, c]        
        
        accuracy_per_class.append(accuracy_score(yt, yhat))
        precision_per_class.append(precision_score(yt, yhat, average='binary'))
        recall_per_class.append(recall_score(yt, yhat, average='binary'))
        ap_per_class.append(average_precision_score(yt, yp))

    mAP = float(np.mean(ap_per_class))
    
    return accuracy_per_class, precision_per_class, recall_per_class, ap_per_class, mAP


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List[float]]:
  """Trains and tests a PyTorch model.

  Passes a target PyTorch models through train_step() and test_step()
  functions for a number of epochs, training and testing the model
  in the same epoch loop.
  """
  
  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
      # Train and test with metrics
        train_metrics = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_metrics = test_step(model, test_dataloader, loss_fn, device)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}")
        print(f"  Test - Loss: {test_metrics['loss']:.4f}")
        print(f"  Test - Accuracy per class: {[f'{a:.4f}' for a in test_metrics['accuracy_per_class']]}")
        print(f"  Test - Precision per class: {[f'{a:.4f}' for a in test_metrics['precision_per_class']]}")
        print(f"  Test - Recall per class: {[f'{a:.4f}' for a in test_metrics['recall_per_class']]}")
        print(f"  Test - Average Precision per class: {[f'{a:.4f}' for a in test_metrics['ap_per_class']]}")
        print(f"  Test - mAP: {test_metrics['mAP_macro']:.4f}")

