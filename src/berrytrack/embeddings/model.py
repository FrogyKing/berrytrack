"""Small trainable fruit encoder and supervised contrastive loss."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CompactEncoder(nn.Module):
    def __init__(self, dimension: int = 64) -> None:
        super().__init__()
        if dimension not in (32, 64, 128):
            raise ValueError("Embedding dimension must be 32, 64, or 128")
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.projection = nn.Linear(64, dimension)

    def forward(self, crops: Tensor) -> Tensor:
        return F.normalize(self.projection(self.features(crops)), dim=-1)


def supervised_contrastive_loss(
    features: Tensor, labels: Tensor, temperature: float = 0.1
) -> Tensor:
    """Exclude self-pairs; anchors without positives contribute no loss."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    features = F.normalize(features, dim=-1)
    logits = features @ features.T / temperature
    eye = torch.eye(len(features), device=features.device, dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    logits = logits.masked_fill(eye, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    count = positive.sum(1)
    loss = -(log_prob * positive).sum(1) / count.clamp_min(1)
    valid = count > 0
    return loss[valid].mean() if valid.any() else features.sum() * 0
