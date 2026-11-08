"""Optional sparse graph attention for research; no untrained weights used by default."""

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor, nn


class SparseContextAttention(nn.Module):
    """Aggregate k local appearance/relative-position messages in O(NkD) space.

    Neighbor selection is geometric and non-differentiable; message and attention
    projections are differentiable. This research alternative needs ID-supervised
    training and a controlled ablation before replacing the deterministic context.
    """

    def __init__(self, dimension: int = 64, neighbors: int = 4) -> None:
        super().__init__()
        self.neighbors = neighbors
        self.message = nn.Linear(dimension + 2, dimension)
        self.score = nn.Sequential(nn.Linear(dimension + 2, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, positions: Tensor, appearance: Tensor, scales: Tensor) -> Tensor:
        if len(positions) < 2:
            return appearance
        k = min(self.neighbors + 1, len(positions))
        _, indices = cKDTree(positions.detach().cpu().numpy()).query(
            positions.detach().cpu().numpy(), k=k
        )
        indices = torch.as_tensor(np.asarray(indices)[:, 1:], device=appearance.device)
        relative = (positions[indices] - positions[:, None]) / scales[:, None, None].clamp_min(1)
        inputs = torch.cat((appearance[indices], relative), dim=-1)
        weights = self.score(inputs).softmax(dim=1)
        messages = (weights * self.message(inputs)).sum(dim=1)
        return torch.nn.functional.normalize(appearance + messages, dim=1)
