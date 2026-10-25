import torch

from losses import (
    AdaptiveMarginSeparation,
    CombinedLoss,
    FeatureMixUp,
    FeaturePrototypeTripletLoss,
    NormalizedLoss,
    PrototypeCenterLoss,
    batch_hard_triplet,
)
from models.prototype_model import PrototypeReID


class _Backbone(torch.nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.embed_dim = d
        self.lin = torch.nn.Linear(12, d)

    def forward(self, x):
        return self.lin(x.flatten(1))


def _batch(K=8, n=4, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    f = torch.nn.functional.normalize(torch.randn(K * n, d, generator=g), dim=1)
    y = torch.arange(K).repeat_interleave(n)
    P = torch.nn.functional.normalize(torch.randn(K, d, generator=g), dim=1)
    S = P @ P.t()
    S.fill_diagonal_(-1)
    return f, y, P, S


def test_fpt_zero_when_separated():
    # features equal to their prototype and prototypes orthogonal -> margin satisfied
    P = torch.eye(4)
    f = P.repeat_interleave(2, dim=0)
    y = torch.arange(4).repeat_interleave(2)
    loss = FeaturePrototypeTripletLoss(margin=0.3, use_similarity_weighting=False)(f, P, y)
    assert loss.item() == 0.0


def test_fpt_positive_and_weighted():
    f, y, P, S = _batch()
    plain = FeaturePrototypeTripletLoss(margin=0.3, use_similarity_weighting=False)(f, P, y)
    weighted = FeaturePrototypeTripletLoss(margin=0.3, use_similarity_weighting=True, similarity_weight_strength=2.0)(f, P, y)
    assert plain.item() > 0 and weighted.item() > 0


def test_center_loss_frechet_vs_projected_and_stopgrad():
    f, y, P, S = _batch()
    P = P.clone().requires_grad_(True)
    for mean_type in ("frechet", "euclidean_projected"):
        loss = PrototypeCenterLoss(mean_type=mean_type)(f, P, y, S)
        assert torch.isfinite(loss) and loss.item() > 0
    # stop-gradient: the loss no longer depends on P (features are constants here)
    loss = PrototypeCenterLoss(stop_gradient_on_prototype=True)(f, P, y, S)
    assert not loss.requires_grad
    # ...but it still does during warmup (the detach is gated off there)
    loss_w = PrototypeCenterLoss(stop_gradient_on_prototype=True)(f, P, y, S, is_warmup_phase=True)
    assert loss_w.requires_grad


def test_bht():
    f, y, P, S = _batch()
    assert batch_hard_triplet(f, y, 0.3).item() > 0
    one_id = torch.zeros(8, dtype=torch.long)
    assert batch_hard_triplet(f[:8], one_id, 0.3).item() == 0.0  # no negatives


def test_norm_loss_zero_for_unit_vectors():
    f, *_ = _batch()
    assert NormalizedLoss()(f).item() < 1e-6
    assert NormalizedLoss()(3 * f).item() > 1.0


def test_brm_pairs_and_fallback():
    f, y, P, S = _batch()
    mix = FeatureMixUp(alpha=0.4, mode="hard_target")
    pairs = mix.mix_features_batch_with_boundary(f, y, P, S, tau=-2.0)  # every cross-id pair confusable
    K, n = 8, 4
    assert isinstance(pairs, dict) and pairs["idx_i"].numel() == (K * n) ** 2 // 2 - K * n * n // 2
    loss = FeatureMixUp.compute_brm_loss(f, P, pairs, margin=0.5, beta=0.7)
    assert torch.isfinite(loss) and loss.item() > 0
    fallback = mix.mix_features_batch_with_boundary(f, y, P, S, tau=2.0)
    assert isinstance(fallback, tuple) and len(fallback) == 4


def test_brm_repulsion_inactive_far_from_boundary():
    P = torch.eye(4)
    f = P.repeat_interleave(2, dim=0)
    y = torch.arange(4).repeat_interleave(2)
    S = torch.zeros(4, 4)
    S.fill_diagonal_(-1)
    pairs = FeatureMixUp(alpha=0).mix_features_batch_with_boundary(f, y, P, S, tau=-1.0)
    # lambda = 1 -> mixed == f_i, boundary == p_i: attraction 0, |f_i - b| = 0 -> penalty m, |f_j - b| = sqrt 2 > m
    loss = FeatureMixUp.compute_brm_loss(f, P, pairs, margin=0.5, beta=1.0)
    assert abs(loss.item() - 0.5) < 1e-5


def test_neighbor_mixup_mixes_with_similar_prototype():
    f, y, P, S = _batch()
    mix = FeatureMixUp(alpha=0.4, mode="neighbor_hard_target", neighbor_k=1)
    mixed, a, b, lam = mix.mix_features_with_neighbors(f, y, P, S)
    assert mixed.shape == f.shape and (a != b).all()
    assert (b == S[y].argmax(dim=1)).all()  # k=1 -> most similar identity


def test_adaptive_separation_scopes():
    f, y, P, S = _batch()
    for scope in ("global", "batch"):
        sep = AdaptiveMarginSeparation(num_classes=8, gamma=1.5, tau=-2.0, pair_scope=scope)
        sep.update_spread(f, y)
        loss, n = sep(P, S, targets=y)
        assert n == 28 and torch.isfinite(loss)
    sep = AdaptiveMarginSeparation(num_classes=8, tau=2.0)
    assert sep(P, S, targets=y)[1] == 0


def test_combined_and_model_roundtrip():
    f, y, P, S = _batch()
    model = PrototypeReID(_Backbone(), num_classes=8, ema_decay=0.999)
    x = torch.randn(32, 3, 2, 2)
    feats, raw = model(x, return_unnormalized=True)
    crit = CombinedLoss(
        fpt_loss=FeaturePrototypeTripletLoss(),
        center_loss=PrototypeCenterLoss(),
        norm_loss=NormalizedLoss(),
        brm_lambda=0.4,
    )
    model.compute_prototypes_similarity()
    loss, d = crit(
        feats, model.prototypes, y, unnorm_features=raw, unnorm_prototypes=model.unnormalized_prototypes,
        prototypes_similarity=model.prototypes_similarity, brm_loss=torch.tensor(1.0),
    )
    assert set(d) == {"fpt_loss", "center_loss", "norm_loss", "brm_loss"}
    loss.backward()
    model.update_prototypes()  # warmup: renormalize
    model.use_ema_prototypes = True
    model.update_prototypes()  # fine-tune: EMA
    assert torch.allclose(model._prototypes.norm(dim=1), torch.ones(8), atol=1e-5)
    sd = model.state_dict()
    assert {"_prototypes", "ema_prototypes", "_prototypes_similarity"} <= set(sd)
