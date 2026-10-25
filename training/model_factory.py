"""Model, criterion and optimizer construction."""

import torch

from config import UnifiedConfig
from losses import (
    AdaptiveMarginSeparation,
    ArcFaceLoss,
    CircleLoss,
    CombinedLoss,
    CosFaceLoss,
    CosineSimilarityLoss,
    FeaturePrototypeTripletLoss,
    NormalizedLoss,
    PrototypeCenterLoss,
    VPLArcFaceLoss,
)
from models.prototype_model import PrototypeReID


def build_head(config: UnifiedConfig, num_classes: int, embed_dim: int):
    """Optional classification head over the prototypes (Table 4 baselines)."""
    L = config.loss
    if L.head == "cosine":
        return CosineSimilarityLoss()
    if L.head == "arcface":
        return ArcFaceLoss(
            scale=L.arcface_scale,
            margin=L.arcface_margin,
            easy_margin=L.arcface_easy_margin,
            label_smoothing=L.arcface_label_smoothing,
        )
    if L.head == "cosface":
        return CosFaceLoss(scale=L.cosface_scale, margin=L.cosface_margin, label_smoothing=L.cosface_label_smoothing)
    if L.head == "circleloss":
        return CircleLoss(scale=L.circle_scale, margin=L.circle_margin)
    if L.head == "vpl":
        return VPLArcFaceLoss(
            num_classes=num_classes,
            feature_dim=embed_dim,
            scale=L.vpl_scale,
            margin=L.vpl_margin,
            injection_weight=L.vpl_injection_weight,
            prototype_weight=L.vpl_prototype_weight,
            life_span=L.vpl_life_span,
            start_step=L.vpl_start_step,
            easy_margin=L.vpl_easy_margin,
            label_smoothing=L.vpl_label_smoothing,
        )
    raise ValueError(f"Unknown loss.head {L.head!r}: expected cosine | arcface | cosface | circleloss | vpl")


def create_model_and_losses(config: UnifiedConfig, num_classes: int, backbone):
    """Returns (model, criterion, device). The criterion carries the adaptive
    separation module as ``criterion.adaptive_sep`` (None when disabled)."""
    device = torch.device("cuda" if config.system.device == "cuda" and torch.cuda.is_available() else "cpu")
    L, T = config.loss, config.training

    model = PrototypeReID(
        backbone,
        num_classes=num_classes,
        ema_decay=config.model.ema_decay,
        init_gain=config.model.prototype_init_gain,
    ).to(device)

    fpt = (
        FeaturePrototypeTripletLoss(
            margin=L.fpt_margin,
            use_similarity_weighting=L.fpt_use_similarity_weighting,
            similarity_weight_strength=L.fpt_similarity_weight_strength,
        )
        if L.enable_fpt
        else None
    )
    center = (
        PrototypeCenterLoss(
            mean_type=L.center_mean_type,
            frechet_iterations=L.center_frechet_iterations,
            use_similarity_weighting=L.center_use_similarity_weighting,
            similarity_weight_strength=L.center_similarity_weight_strength,
            use_variance_weighting=L.center_use_variance_weighting,
            variance_weight_strength=L.center_variance_weight_strength,
            stop_gradient_on_prototype=L.center_stop_gradient_on_prototype,
        )
        if L.enable_center
        else None
    )
    norm = NormalizedLoss(reduction="mean") if L.enable_norm else None
    head = build_head(config, num_classes, model.embed_dim) if L.enable_head else None

    criterion = CombinedLoss(
        fpt_loss=fpt,
        fpt_weight=L.fpt_weight,
        center_loss=center,
        center_weight=L.center_weight,
        norm_loss=norm,
        norm_weight=L.norm_weight,
        head_loss=head,
        head_weight=L.head_weight,
        brm_lambda=T.brm_lambda,
    ).to(device)

    criterion.adaptive_sep = None
    if T.enable_brm and T.enable_adaptive_separation and T.brm_lambda > 0:
        criterion.adaptive_sep = AdaptiveMarginSeparation(
            num_classes=num_classes,
            gamma=T.sep_gamma,
            tau=T.brm_tau,
            ema_decay=T.sep_ema_decay,
            pair_scope=T.sep_pair_scope,
            mean_type=L.center_mean_type,
            frechet_iters=L.center_frechet_iterations,
        ).to(device)

    return model, criterion, device


def _adamw(params, lr, weight_decay):
    return torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay), eps=1e-8)


def _scheduler(optimizer, kind, total_epochs, min_lr, step_size=30, gamma=0.1):
    total_epochs = max(1, int(total_epochs))
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=float(min_lr))
    if kind == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(step_size), gamma=float(gamma))
    if kind == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    raise ValueError(f"Unknown scheduler {kind!r}: expected cosine | step | none")


def create_warmup_optimizer_and_scheduler(config: UnifiedConfig, model: PrototypeReID):
    """Stage 1: prototypes only, AdamW, cosine warmup_lr -> warmup_min_lr."""
    T = config.training
    optimizer = _adamw([model._prototypes], T.warmup_lr, T.weight_decay)
    scheduler = _scheduler(optimizer, "cosine", T.prototype_warmup_epochs, T.warmup_min_lr)
    return optimizer, scheduler


def create_dual_finetune_optimizers_and_schedulers(config: UnifiedConfig, model: PrototypeReID):
    """Stage 2 (paper): encoder (cosine) and prototype (step) optimizers."""
    T = config.training
    remaining = T.num_epochs - T.prototype_warmup_epochs
    enc_opt = _adamw(model.model.parameters(), T.encoder_lr, T.weight_decay)
    enc_sched = _scheduler(enc_opt, T.encoder_scheduler, remaining, T.encoder_min_lr, T.encoder_step_size, T.encoder_gamma)
    pro_opt = _adamw([model._prototypes], T.prototype_lr, T.weight_decay)
    pro_sched = _scheduler(
        pro_opt, T.prototype_scheduler, remaining, T.prototype_min_lr, T.prototype_step_size, T.prototype_gamma
    )
    return enc_opt, enc_sched, pro_opt, pro_sched


def create_single_finetune_optimizer_and_scheduler(config: UnifiedConfig, model: PrototypeReID):
    """Stage 2 ablation (Supp. Table 2): one shared learning rate for encoder + prototypes."""
    T = config.training
    remaining = T.num_epochs - T.prototype_warmup_epochs
    optimizer = _adamw(model.parameters(), T.single_lr, T.weight_decay)
    scheduler = _scheduler(optimizer, "cosine", remaining, T.single_min_lr)
    return optimizer, scheduler


def create_mixed_precision_scaler(config: UnifiedConfig, device: torch.device):
    if config.system.mixed_precision and device.type == "cuda":
        return torch.amp.GradScaler("cuda")
    return None
