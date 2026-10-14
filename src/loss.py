import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    """
    Soft Multi-Class Dice Loss with support for ignore_index.
    Mitigates severe class imbalance in marine habitat segmentation.
    """
    def __init__(self, num_classes: int, ignore_index: int = 255, smooth: float = 1e-5):
        super(SoftDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        valid_mask = (targets != self.ignore_index)
        targets_clean = torch.where(valid_mask, targets, torch.zeros_like(targets))
        
        # Shape: (B, C, H, W)
        one_hot = F.one_hot(targets_clean, self.num_classes).permute(0, 3, 1, 2).float()
        valid_mask = valid_mask.unsqueeze(1).float()
        
        probs = probs * valid_mask
        one_hot = one_hot * valid_mask
        
        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dims)
        cardinality = torch.sum(probs + one_hot, dims)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - torch.mean(dice_score)


class CombinedLoss(nn.Module):
    """
    Composite Loss Function combining Class-Weighted Cross-Entropy and Soft Dice Loss.
    Loss = ce_weight * CrossEntropy + dice_weight * SoftDice
    """
    def __init__(
        self,
        weight: torch.Tensor = None,
        num_classes: int = 6,
        ignore_index: int = 255,
        ce_weight: float = 0.6,
        dice_weight: float = 0.4
    ):
        super(CombinedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.dice = SoftDiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return (self.ce_weight * ce_loss) + (self.dice_weight * dice_loss)


def compute_elastic_regularization(
    model: nn.Module,
    l1_lambda: float = 1e-5,
    l2_lambda: float = 1e-4
) -> torch.Tensor:
    """
    Computes Elastic Net (L1 + L2) Regularization on trainable parameters.
    """
    l1_norm = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
    l2_norm = sum((p ** 2).sum() for p in model.parameters() if p.requires_grad)
    return (l1_lambda * l1_norm) + (l2_lambda * l2_norm)
