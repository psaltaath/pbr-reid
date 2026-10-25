"""L_bht — batch-hard feature triplet (paper Eq. 5-6; Hermans et al. 2017)."""

import torch
import torch.nn.functional as F


def batch_hard_triplet(features, targets, margin=0.3):
    """Instance-anchored batch-hard triplet on L2-normalized features (cosine distance).

    For every anchor i with at least one positive and one negative in the batch:
        d_ap = max_{j: y_j = y_i, j != i} (1 - cos(f_i, f_j))
        d_an = min_{k: y_k != y_i}        (1 - cos(f_i, f_k))
        L_i  = [d_ap - d_an + margin]_+
    Returns the mean over valid anchors (0 if none).
    """
    f = F.normalize(features, dim=1)
    dist = 1.0 - f @ f.t()
    B = f.shape[0]
    same = targets.unsqueeze(0) == targets.unsqueeze(1)
    eye = torch.eye(B, dtype=torch.bool, device=f.device)
    pos_mask = same & ~eye
    neg_mask = ~same

    valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    if not valid.any():
        return features.new_zeros(())

    neg_inf = torch.finfo(dist.dtype).min
    pos_inf = torch.finfo(dist.dtype).max
    d_ap = torch.where(pos_mask, dist, dist.new_full((), neg_inf)).max(dim=1).values
    d_an = torch.where(neg_mask, dist, dist.new_full((), pos_inf)).min(dim=1).values
    return F.relu(d_ap[valid] - d_an[valid] + margin).mean()
