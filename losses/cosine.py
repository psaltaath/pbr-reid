"""Cosine head: L(f, p_y) = 1 - cos(f, p_y) averaged over the batch."""

from torch import nn
from torch.nn import functional as F


class CosineSimilarityLoss(nn.Module):
    def forward(self, features, prototypes, targets, prototypes_similarity=None):
        cos_sim = F.cosine_similarity(features, prototypes[targets], dim=-1)
        return (1.0 - cos_sim).mean()
