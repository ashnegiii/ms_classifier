import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in multi-label classification.
    Better than weighted BCE for extreme imbalances.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(inputs)
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate p_t
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Calculate focal weight
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss