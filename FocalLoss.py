import torch
import torch.nn as nn

class FocalLoss1(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss




