"""
CoralVisionNet IO - Loss Functions
====================================
Implementation of Focal Loss to handle severe class imbalance,
such as the ~0.8% PartiallyBleached class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Balances the loss by dynamically scaling the cross entropy based on prediction confidence.
    """
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        """
        Args:
            weight (Tensor, optional): A manual rescaling weight given to each class.
            gamma (float): Focusing parameter. Default is 2.0.
            reduction (str): Specifies the reduction to apply to the output.
        """
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C) unnormalized logits
            targets: (B,) class indices
        """
        # compute standard unweighted cross entropy to get true probabilities
        ce_loss_unweighted = F.cross_entropy(inputs, targets, reduction='none')
        
        # pt is the probability of the true class
        pt = torch.exp(-ce_loss_unweighted)
        
        # focal loss formulation
        # If weight is provided, we compute the weighted ce_loss for the loss scaling
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
