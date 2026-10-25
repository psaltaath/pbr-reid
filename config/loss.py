"""Loss configuration (paper Sec. 3.2.1 and Table 4 / Table 5 ablations)."""

from dataclasses import dataclass

from .base import BaseConfig


@dataclass
class LossConfig(BaseConfig):
    """Terms of the training objective (paper Eq. 1 / Eq. 13).

    L = lambda_fpt L_fpt + lambda_bht L_bht + lambda_cen L_cen + lambda_norm L_norm
        + lambda_BRM L_BRM                      (BRM lives in TrainingConfig)
    """

    # --- L_fpt: feature-prototype triplet (Eq. 2-4) --------------------------
    # Prototype-anchored batch-hard triplet with the adaptive weight
    # w_k = exp(gamma_fpt * p_k^T f_neg).
    enable_fpt: bool = True
    fpt_weight: float = 1.0
    fpt_margin: float = 0.3
    fpt_use_similarity_weighting: bool = True
    fpt_similarity_weight_strength: float = 2.0  # gamma_fpt

    # --- L_bht: batch-hard feature triplet (Eq. 5-6), fine-tuning only -----
    enable_bht: bool = True
    bht_weight: float = 1.0
    bht_margin: float = 0.3

    # --- L_cen: prototype center loss (Eq. 7-11) ----------------------------
    enable_center: bool = True
    center_weight: float = 0.5
    # Class centroid on the sphere: "frechet" (paper, intrinsic mean via a few
    # Riemannian steps) or "euclidean_projected" (normalized Euclidean mean).
    center_mean_type: str = "frechet"
    center_frechet_iterations: int = 2
    center_use_similarity_weighting: bool = True
    center_similarity_weight_strength: float = 1.0  # gamma_center^sim
    center_use_variance_weighting: bool = True
    center_variance_weight_strength: float = 2.5  # gamma_center^var
    # Ablation (Sec. 4.3): detach the prototype so only features are pulled.
    center_stop_gradient_on_prototype: bool = False

    # --- L_norm: soft unit-norm regularizer on features and prototypes (Eq. 12)
    enable_norm: bool = True
    norm_weight: float = 0.005

    # --- Classification head slot (off in PBR-ReID; Table 4 baselines) ------
    # When enabled, a per-identity margin head over the prototypes is added:
    #   cosine     1 - cos(f, p_y)
    #   arcface    additive angular margin softmax (Deng et al. 2019)
    #   cosface    additive cosine margin softmax (Wang et al. 2018)
    #   circleloss class-level Circle Loss (Sun et al. 2020)
    #   vpl        Variational Prototype Learning + ArcFace margin (Deng et al. 2021)
    enable_head: bool = False
    head: str = "cosine"  # cosine | arcface | cosface | circleloss | vpl
    head_weight: float = 1.0

    arcface_scale: float = 64.0
    arcface_margin: float = 0.5  # radians
    arcface_easy_margin: bool = False
    arcface_label_smoothing: float = 0.0

    cosface_scale: float = 64.0
    cosface_margin: float = 0.35
    cosface_label_smoothing: float = 0.0

    circle_scale: float = 128.0
    circle_margin: float = 0.25

    vpl_scale: float = 64.0
    vpl_margin: float = 0.5
    vpl_injection_weight: float = 0.15  # lambda_2
    vpl_prototype_weight: float = 1.0  # lambda_1
    vpl_life_span: int = 100  # delta t (steps)
    vpl_start_step: int = 0
    vpl_easy_margin: bool = False
    vpl_label_smoothing: float = 0.0
