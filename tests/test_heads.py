import torch

from config import ConfigManager
from training.model_factory import build_head


def test_every_head_runs_and_backprops():
    K, d, B = 10, 16, 20
    g = torch.Generator().manual_seed(0)
    f = torch.nn.functional.normalize(torch.randn(B, d, generator=g), dim=1)
    P = torch.nn.functional.normalize(torch.randn(K, d, generator=g), dim=1).requires_grad_(True)
    y = torch.arange(K).repeat_interleave(2)
    cm = ConfigManager()
    for head in ("cosine", "arcface", "cosface", "circleloss", "vpl"):
        cfg = cm.load_config(cli_args=["--loss.enable_head", "true", "--loss.head", head])
        loss_fn = build_head(cfg, K, d)
        loss = loss_fn(f, P, y, None)
        assert torch.isfinite(loss), head
        loss.backward()
        assert P.grad is not None and torch.isfinite(P.grad).all(), head
        P.grad = None
