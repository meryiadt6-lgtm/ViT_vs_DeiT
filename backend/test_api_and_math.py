import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient

from backend.main import app, IMAGENET_MEAN, IMAGENET_STD


client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "vit" in data["models"]
    assert "deit" in data["models"]


def test_model_info():
    res = client.get("/api/model-info")
    assert res.status_code == 200
    data = res.json()
    # ViT-Base 16 specs
    assert data["vit"]["num_patches"] == 196
    assert data["vit"]["hidden_size"] == 768
    assert data["vit"]["tokens"] == 197
    assert data["vit"]["layers"] == 12
    # DeiT-Base specs
    assert data["deit"]["tokens"] == 198
    assert data["deit"]["distillation"] is True


def test_mathematical_consistency_patch_slicing():
    """Vérifie mathématiquement que N = (H*W)/P^2 et D = P*P*C"""
    H, W = 224, 224
    for P in [8, 16, 32]:
        N = (H * W) // (P * P)
        D = P * P * 3
        if P == 16:
            assert N == 196
            assert D == 768
        elif P == 8:
            assert N == 784
            assert D == 192
        elif P == 32:
            assert N == 49
            assert D == 3072


def test_mathematical_consistency_layer_norm():
    """Vérifie que le calcul manuel de LayerNorm correspond exactement à torch.nn.functional.layer_norm"""
    x_np = np.array([1.0, 3.0, 3.0, 9.0], dtype=np.float32)
    # Calcul manuel
    mu = float(np.mean(x_np))
    assert mu == 4.0
    var = float(np.mean((x_np - mu) ** 2))
    assert var == 9.0
    std = math.sqrt(var)
    assert std == 3.0

    manual_norm = (x_np - mu) / std
    expected = np.array([-1.0, -0.33333334, -0.33333334, 1.6666666], dtype=np.float32)
    np.testing.assert_allclose(manual_norm, expected, atol=1e-5)

    # PyTorch LayerNorm
    x_torch = torch.tensor([1.0, 3.0, 3.0, 9.0], dtype=torch.float32)
    torch_norm = F.layer_norm(x_torch, (4,), eps=0.0).numpy()
    np.testing.assert_allclose(manual_norm, torch_norm, atol=1e-5)


def test_mathematical_consistency_attention():
    """Vérifie le produit scalaire mis à l'échelle QK^T / sqrt(d_k) et le Softmax"""
    d_k = 4
    scale = math.sqrt(d_k)  # 2.0
    Q = torch.tensor([[2.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    K = torch.tensor([[1.0, 1.0], [0.0, 2.0]], dtype=torch.float32)

    # K^T
    KT = K.T
    expected_KT = torch.tensor([[1.0, 0.0], [1.0, 2.0]])
    torch.testing.assert_close(KT, expected_KT)

    # Q K^T
    scores = torch.matmul(Q, KT)
    expected_scores = torch.tensor([[2.0, 0.0], [2.0, 2.0]])
    torch.testing.assert_close(scores, expected_scores)

    # Scale by sqrt(4) = 2.0
    scaled_scores = scores / scale
    expected_scaled = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    torch.testing.assert_close(scaled_scores, expected_scaled)

    # Softmax row 1
    # exp(1) / (exp(1) + exp(0)) = 2.71828 / 3.71828 = 0.73105
    # exp(0) / (exp(1) + exp(0)) = 1.0 / 3.71828 = 0.26894
    attn_weights = F.softmax(scaled_scores, dim=-1)
    assert abs(attn_weights[0, 0].item() - 0.73105) < 1e-4
    assert abs(attn_weights[0, 1].item() - 0.26894) < 1e-4


def test_mathematical_consistency_gelu():
    """Vérifie la formule exacte de GELU vs torch.nn.functional.gelu"""
    vals = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    for v in vals:
        t_val = torch.tensor([v], dtype=torch.float32)
        # Formule exacte: 0.5 * x * (1 + erf(x / sqrt(2)))
        exact_formula = 0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0)))
        torch_gelu = F.gelu(t_val, approximate="none").item()
        assert abs(exact_formula - torch_gelu) < 1e-4

