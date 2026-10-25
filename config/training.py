"""Training schedule, optimizers, MixUp / BRM configuration."""

from dataclasses import dataclass

from .base import BaseConfig


@dataclass
class TrainingConfig(BaseConfig):
    """Two-stage training protocol (paper Sec. 3.2.3) and BRM (Sec. 3.2.2).

    Stage 1 (prototype warmup): backbone frozen, only the prototypes are
    optimized with ``warmup_lr`` (AdamW, cosine decay) for
    ``prototype_warmup_epochs`` epochs. L_bht and L_BRM are disabled.

    Stage 2 (joint fine-tuning): backbone unfrozen, two optimizers (encoder:
    cosine ``encoder_lr`` -> ``encoder_min_lr``; prototypes: step
    ``prototype_lr`` / ``prototype_step_size``), prototypes updated by EMA,
    L_bht and L_BRM active.
    """

    # --- schedule -----------------------------------------------------------
    num_epochs: int = 200
    prototype_warmup_epochs: int = 10
    eval_period: int = 10  # evaluate every N epochs (fine-tuning only) + last epoch
    log_interval: int = 100  # print a loss line every N batches

    # --- optimizers (AdamW everywhere, paper: weight decay 0.01) -------------
    weight_decay: float = 0.01

    # Stage 1 — prototypes only, direct (non-EMA) updates, cosine warmup_lr -> warmup_min_lr.
    warmup_lr: float = 1e-3
    warmup_min_lr: float = 1e-4

    # Stage 2 — dual learning-rate schedule (paper default). Set
    # use_dual_optimizers=false for the Supp. Table 2 "single shared rate" ablation.
    use_dual_optimizers: bool = True

    encoder_lr: float = 1e-5
    encoder_min_lr: float = 1e-7
    encoder_scheduler: str = "cosine"  # cosine | step | none
    encoder_step_size: int = 30
    encoder_gamma: float = 0.1

    prototype_lr: float = 1e-1
    prototype_scheduler: str = "step"  # step | cosine | none
    prototype_step_size: int = 15
    prototype_gamma: float = 0.5
    prototype_min_lr: float = 1e-5  # cosine only

    # Single shared optimizer over backbone + prototypes (use_dual_optimizers=false).
    single_lr: float = 1e-5
    single_min_lr: float = 1e-7

    # --- feature-level MixUp (joint fine-tuning only) -----------------------
    enable_mixup: bool = True
    mixup_alpha: float = 0.4  # lambda ~ Beta(alpha, alpha)
    mixup_prob: float = 1.0  # probability of applying MixUp/BRM on a batch
    # hard_target: random pairing, loss = lambda L(y_i) + (1-lambda) L(y_j)
    #              (Table 5 "MixUp"). neighbor_hard_target: neighbour-guided
    #              MixUp with the most similar prototypes (Table 5 "NG").
    mixup_mode: str = "hard_target"  # hard_target | neighbor_hard_target
    mixup_neighbor_k: int = 5
    mixup_neighbor_temperature: float = 1.0

    # --- Boundary Repulsion MixUp (paper Eq. 7-13) -------------------------
    # Requires enable_mixup. Confusable pairs: S_ab > brm_tau. Falls back to
    # standard random MixUp when no confusable pair is present in the batch.
    enable_brm: bool = True
    brm_tau: float = 0.7  # confusability threshold tau
    brm_margin: float = 0.5  # repulsion margin m_BRM (Euclidean, Eq. 11)
    brm_beta: float = 0.7  # repulsion strength beta (Eq. 13)
    brm_lambda: float = 0.4  # outer weight lambda_BRM (Eq. 1); 0 disables BRM

    # Adaptive-margin prototype separation (second term of Eq. 13):
    # [gamma_sep (sigma_a + sigma_b) - (1 - cos(p_a, p_b))]_+ over confusable
    # class pairs, sigma_k = EMA of the intra-class angular spread.
    enable_adaptive_separation: bool = True
    sep_gamma: float = 1.5
    sep_ema_decay: float = 0.9
    sep_weight: float = 1.0  # internal weight inside L_BRM (paper: 1)
    # Which confusable class pairs the separation term ranges over:
    #   "global": every pair of identities with S_ab > tau (every step)
    #   "batch":  only pairs among the identities present in the batch (P_S of Eq. 13)
    sep_pair_scope: str = "global"  # global | batch
