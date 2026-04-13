import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss


#amd [3,  1, 4, 2.5]
#gamma[1, 2.5, 3]
class FocalLoss1(nn.Module):
    def __init__(self, alpha=torch.tensor([1, 2.5, 3]), gamma=2, reduction='mean'):
        super(FocalLoss1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        self.alpha = self.alpha.to(targets.device)

        if self.alpha is not None:
            if self.alpha.size(0) == inputs.size(1):  # Ensure alpha size matches number of classes
                alpha_factor = torch.gather(self.alpha, 0, targets)
                focal_loss = alpha_factor * focal_loss
            else:
                raise ValueError("Alpha size does not match number of classes")

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss




