from typing import Dict

"""EarlyStopping class to monitor validation metrics and stop training after {patience} epochs early if no improvement is seen for the specified metrics."""
class EarlyStopping:
    def __init__(self, patience: int = 2, metric: str = "mAP"):
        self.patience = patience
        self.metric = metric
        self.best_score = None
        self.epochs_no_improve = 0
        self.best_epoch = 0
        self.should_stop = False

    def __call__(self, epoch: float, val_metrics: Dict):
        """Returns true if training should stop early."""
        if val_metrics is None or self.metric not in val_metrics:
            return False

        score = val_metrics[self.metric]

        # Update best score and check for improvement
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

        # Check if we have reached the patience limit and stop if so
        if self.epochs_no_improve >= self.patience:
            self.should_stop = True
            print(
                f"[EARLY STOP] No improvement for {self.patience} epochs. Best: {self.best_score:.4f} at epoch {self.best_epoch}")

        return self.should_stop
