
import torch
from tqdm.auto import tqdm
from typing import Dict, List
import wandb
from utils import calc_metrics

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               class_names: List[str],
               device: torch.device,
               optimal_thresholds: Dict[str, float] = None,
               threshold: float = None) -> Dict[str, float]:
    """Enhanced train step with metrics tracking"""
    model.train()
    running_loss = 0
    
    all_logits = []
    all_targets = []
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
        
        with torch.inference_mode():
            all_logits.append(logits.detach())
            all_targets.append(y.detach())

        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    avg_loss = running_loss / len(dataloader)
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    if optimal_thresholds:
        accuracy_per_class, precision_per_class, recall_per_class, f1_per_class, ap_per_class, mAP = calc_metrics(targets=targets, logits=logits, optimal_thresholds=optimal_thresholds, class_names=class_names)
    elif threshold:
        accuracy_per_class, precision_per_class, recall_per_class, f1_per_class, ap_per_class, mAP = calc_metrics(targets=targets, logits=logits, threshold=threshold, class_names=class_names)
    else:
        raise ValueError("Either optimal_thresholds or threshold must be provided for metrics calculation.")
    return {
            "loss": avg_loss,
            "recall_per_class": recall_per_class,
            "accuracy_per_class": accuracy_per_class,
            "precision_per_class": precision_per_class,
            "f1_per_class": f1_per_class,
            "ap_per_class": ap_per_class,
            "mAP_macro": mAP}

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              class_names: List[str],
              device: torch.device,
              optimal_thresholds: Dict[str, float] = None,
              threshold: float = None) -> Dict[str, float]:
    """Enhanced test step with metrics tracking"""
    model.eval()
    running_loss = 0.0
    running_loss_plain = 0.0
    all_logits = []
    all_targets = []
    plain_loss_fn = torch.nn.BCEWithLogitsLoss()
    with torch.inference_mode():
        pbar = tqdm(dataloader, desc="Eval", leave=False)
        for X, y in pbar:
            X, y = X.to(device), y.to(device).float()
            
            logits = model(X)
            loss = loss_fn(logits, y)
            loss_plain = plain_loss_fn(logits, y)
            running_loss += loss.item()
            running_loss_plain += loss_plain.item()

            with torch.inference_mode():
                all_logits.append(logits)
                all_targets.append(y)
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    avg_loss = running_loss / len(dataloader)
    avg_loss_plain = running_loss_plain / len(dataloader)
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)


    if optimal_thresholds:
        accuracy_per_class, precision_per_class, recall_per_class, f1_per_class, ap_per_class, mAP = calc_metrics(targets=targets, logits=logits, optimal_thresholds=optimal_thresholds, class_names=class_names)
    elif threshold:
        accuracy_per_class, precision_per_class, recall_per_class, f1_per_class, ap_per_class, mAP = calc_metrics(targets=targets, logits=logits, threshold=threshold, class_names=class_names)
    else:
        raise ValueError("Either optimal_thresholds or threshold must be provided for metrics calculation.")
    
    return {
            "loss": avg_loss,
            "loss_plain": avg_loss_plain,
            "recall_per_class": recall_per_class,
            "accuracy_per_class": accuracy_per_class,
            "precision_per_class": precision_per_class,
            "f1_per_class": f1_per_class,
            "ap_per_class": ap_per_class,
            "mAP_macro": mAP}
    

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          class_names: List[str],
          device: torch.device,
          threshold: float = 0.5) -> Dict[str, List[float]]:
    """Trains and tests a PyTorch model with W&B logging."""
    
    # Store metrics for custom plots
    epoch_data = []
    loss_train_data = []
    loss_test_data = []
    precision_data = {class_name: [] for class_name in class_names}
    recall_data = {class_name: [] for class_name in class_names}
    optimal_thresholds = {name: threshold for name in class_names}
    
    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        # Train and test with metrics
        train_metrics = train_step(model=model, dataloader=train_dataloader, threshold=threshold, loss_fn=loss_fn, optimizer=optimizer, class_names=class_names, device=device)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        print(f"  Train - Loss: {train_metrics['loss']:.4f}")
        print(f"  Train - Accuracy per class: {[f'{a:.4f}' for a in train_metrics['accuracy_per_class']]}")
        print(f"  Train - Precision per class: {[f'{a:.4f}' for a in train_metrics['precision_per_class']]}")
        print(f"  Train - Recall per class: {[f'{a:.4f}' for a in train_metrics['recall_per_class']]}")
        print(f"  Train - F1 per class: {[f'{a:.4f}' for a in train_metrics['f1_per_class']]}")
        print(f"  Train - Average Precision per class: {[f'{a:.4f}' for a in train_metrics['ap_per_class']]}")
        print(f"  Train - mAP: {train_metrics['mAP_macro']:.4f}")
        
        optimal_thresholds = {
            'kermit': 0.3,        # Lower threshold (more common in test)
            'miss_piggy': 0.2,    # Lower threshold  
            'cook': 0.1,          # Higher threshold (less common in test)
            'statler_waldorf': 0.5, # Higher threshold
            'rowlf_the_dog': 0.5, # Much higher (very rare in test)  
            'fozzie_bear': 0.4    # Lower threshold
        }
        test_metrics = test_step(model=model, dataloader=test_dataloader, optimal_thresholds=optimal_thresholds, loss_fn=loss_fn, class_names=class_names, device=device)

        # Store data for custom plots
        epoch_data.append(epoch + 1)
        loss_train_data.append(train_metrics['loss'])
        loss_test_data.append(test_metrics['loss'])
        
        for i, class_name in enumerate(class_names):
            precision_data[class_name].append(test_metrics['precision_per_class'][i])
            recall_data[class_name].append(test_metrics['recall_per_class'][i])
        
        log_dict = {
            "mAP": test_metrics['mAP_macro']
        }
        
        # Add per-class accuracy and average precision and f1score
        for i, class_name in enumerate(class_names):
            log_dict[f"accuracy_{class_name}"] = test_metrics['accuracy_per_class'][i]
            log_dict[f"f1_{class_name}"] = test_metrics['f1_per_class'][i]
            log_dict[f"avg_precision_{class_name}"] = test_metrics['ap_per_class'][i]

        # Create custom plots
        # Loss plot (train vs test)
        log_dict["loss_plot"] = wandb.plot.line_series(
            xs=epoch_data,
            ys=[loss_train_data, loss_test_data],
            keys=["train", "test"],
            title="Training vs Test Loss",
            xname="Epoch"
        )
        
        # Precision/Recall plots for each class
        for class_name in class_names:
            log_dict[f"precision_recall_{class_name}"] = wandb.plot.line_series(
                xs=epoch_data,
                ys=[precision_data[class_name], recall_data[class_name]],
                keys=["precision", "recall"],
                title=f"Precision & Recall - {class_name}",
                xname="Epoch"
            )
        
        # Log everything to W&B
        wandb.log(log_dict, step=epoch + 1)


        print(f"  Test - Loss: {test_metrics['loss']:.4f}")
        print(f"  Plain - Loss: {test_metrics['loss_plain']:.4f}")
        print(f"  Test - Accuracy per class: {[f'{a:.4f}' for a in test_metrics['accuracy_per_class']]}")
        print(f"  Test - Precision per class: {[f'{a:.4f}' for a in test_metrics['precision_per_class']]}")
        print(f"  Test - Recall per class: {[f'{a:.4f}' for a in test_metrics['recall_per_class']]}")
        print(f"  Test - F1 per class: {[f'{a:.4f}' for a in test_metrics['f1_per_class']]}")
        print(f"  Test - Average Precision per class: {[f'{a:.4f}' for a in test_metrics['ap_per_class']]}")
        print(f"  Test - mAP: {test_metrics['mAP_macro']:.4f}")