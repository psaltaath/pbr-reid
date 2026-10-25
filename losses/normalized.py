import torch
from torch import nn
from torch.nn import functional as F


class NormalizedLoss(nn.Module):
    """
    Regularization loss that penalizes the difference between unnormalized and normalized vectors.

    This loss encourages the model to produce features/prototypes that are already close to
    being normalized, potentially improving training stability and consistency.

    L = ||unnorm_vectors - normalize(unnorm_vectors)||²

    Args:
        norm_dim (int): Dimension along which to normalize (default: 1 for feature dimension)
        reduction (str): 'mean', 'sum', or 'none'
        eps (float): Small epsilon for numerical stability in normalization
    """

    def __init__(self, norm_dim=1, reduction="mean", eps=1e-8):
        super().__init__()
        self.norm_dim = norm_dim
        self.reduction = reduction
        self.eps = eps

    def forward(self, unnorm_vectors):
        """
        Args:
            unnorm_vectors: Tensor of shape (batch_size, feature_dim) or (num_classes, feature_dim)
                          representing unnormalized features or prototypes

        Returns:
            loss: Scalar tensor representing the normalization regularization loss
        """
        # Compute normalized version
        normalized_vectors = F.normalize(
            unnorm_vectors, p=2, dim=self.norm_dim, eps=self.eps
        )

        # Compute L2 difference between unnormalized and normalized
        diff = unnorm_vectors - normalized_vectors
        loss = torch.norm(diff, p=2, dim=self.norm_dim)

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class FeatureNormalizedLoss(NormalizedLoss):
    """
    Normalized loss specifically for feature vectors.
    Inherits from NormalizedLoss but provides semantic clarity.
    """

    def __init__(self, reduction="mean", eps=1e-8):
        super().__init__(norm_dim=1, reduction=reduction, eps=eps)


class PrototypeNormalizedLoss(NormalizedLoss):
    """
    Normalized loss specifically for prototype vectors.
    Inherits from NormalizedLoss but provides semantic clarity.
    """

    def __init__(self, reduction="mean", eps=1e-8):
        super().__init__(norm_dim=1, reduction=reduction, eps=eps)
