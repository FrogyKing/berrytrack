import json

import pytest
import torch

from berrytrack.embeddings.model import CompactEncoder, supervised_contrastive_loss
from berrytrack.infrastructure.budget import estimate_cost, reserve


def test_encoder_and_contrastive_gradient():
    model = CompactEncoder(32)
    encoded = model(torch.rand(4, 3, 32, 32))
    assert encoded.shape == (4, 32)
    assert torch.allclose(encoded.norm(dim=1), torch.ones(4), atol=1e-5)
    loss = supervised_contrastive_loss(encoded, torch.tensor([1, 1, 2, 2]))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.projection.weight.grad is not None


def test_budget_refuses_exhaustion_even_if_cap_tampered(tmp_path):
    path = tmp_path / "budget.json"
    record = reserve(path, "T4", 120, "smoke")
    assert record["reserved_usd"] == estimate_cost("T4", 120)
    path.write_text(json.dumps({"budget_usd": 1e6, "runs": [{"reserved_usd": 29.99}]}))
    with pytest.raises(RuntimeError, match="Budget exhausted"):
        reserve(path, "T4", 120, "smoke")
    with pytest.raises(ValueError):
        estimate_cost("H100", 36000)


def test_sparse_context_is_permutation_equivariant():
    from berrytrack.embeddings.context import SparseContextAttention

    torch.manual_seed(42)
    layer = SparseContextAttention(32, neighbors=2)
    xy = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [7.0, 1.0]])
    features = torch.rand(4, 32)
    scale = torch.ones(4)
    original = layer(xy, features, scale)
    permutation = torch.tensor([2, 0, 3, 1])
    changed = layer(xy[permutation], features[permutation], scale[permutation])
    assert torch.allclose(original[permutation], changed, atol=1e-6)
    original.sum().backward()
    assert layer.message.weight.grad is not None
