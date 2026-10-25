import numpy as np
import torch
from torch.nn import functional as F

from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

console = Console()


def extract_features(model, data_loader, device, desc="Extracting features"):
    """Extract features from the model for ReID evaluation with progress bar"""
    model.eval()
    features = []
    pids = []
    camids = []

    with torch.no_grad():
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            try:
                total = len(data_loader)
            except TypeError:
                console.print("[red]Error: data_loader must have known length.[/red]")
                raise ValueError("data_loader must support len()")
            task = progress.add_task(f"[cyan]{desc}", total=total)

            for batch_data in data_loader:
                # val_collate_fn returns: imgs, pids, camids, camids_batch, viewids, img_paths
                imgs = batch_data[0].to(device)
                batch_pids = batch_data[1]
                batch_camids = batch_data[2]

                # Ensure pids/camids are converted to list
                if isinstance(batch_pids, torch.Tensor):
                    batch_pids = batch_pids.tolist()
                if isinstance(batch_camids, torch.Tensor):
                    batch_camids = batch_camids.tolist()

                # Use autocast only if CUDA available
                if device.type == "cuda":
                    with torch.autocast("cuda"):
                        feats = model(
                            imgs
                        )  # Use the model directly unless model.model is needed
                else:
                    feats = model(imgs)

                features.append(feats.cpu())
                pids.extend(batch_pids)
                camids.extend(batch_camids)

                progress.update(task, advance=1)

    features = torch.cat(features, dim=0)
    return features, pids, camids


def compute_distance_matrix(query_features, gallery_features):
    """Compute cosine distance matrix between query and gallery features"""
    # Normalize features
    query_features = F.normalize(query_features, p=2, dim=1)
    gallery_features = F.normalize(gallery_features, p=2, dim=1)

    # Compute cosine similarity and convert to distance
    similarity = torch.mm(query_features, gallery_features.t())
    distance = 1 - similarity

    return distance.cpu().numpy()  # Ensure on CPU before numpy


def evaluate_reid(
    distance_matrix, query_pids, gallery_pids, query_camids, gallery_camids, max_rank=50
):
    """Evaluate ReID performance using mAP and CMC

    Handles two evaluation protocols:
    1. Multi-camera ReID (Market1501, DukeMTMC, etc.):
       - Matches require same PID but different camera
       - Removes gallery images from same camera as query

    2. Single-camera ReID (VehicleID):
       - Matches require only same PID (no camera constraint)
       - Removes query image itself from gallery by index
    """
    num_q, num_g = distance_matrix.shape

    if num_g < max_rank:
        max_rank = num_g
        console.print(
            f"[yellow]Note: max_rank is set to {max_rank} since gallery size is {num_g}[/yellow]"
        )

    indices = torch.argsort(torch.from_numpy(distance_matrix), dim=1)
    matches = torch.zeros((num_q, num_g), dtype=torch.bool)

    # Detect evaluation protocol: check if all cameras are the same
    # VehicleID and similar datasets have no real camera info (all camid=0)
    unique_query_cams = set(query_camids)
    unique_gallery_cams = set(gallery_camids)
    single_camera_protocol = (
        len(unique_query_cams) == 1
        and len(unique_gallery_cams) == 1
        and unique_query_cams == unique_gallery_cams
    )

    if single_camera_protocol:
        console.print(
            "[cyan]Using single-camera evaluation protocol (VehicleID-style)[/cyan]"
        )
        # VehicleID protocol: Match by PID only, no camera constraint
        for q_idx in range(num_q):
            q_pid = query_pids[q_idx]
            for g_idx in range(num_g):
                # Match if same vehicle ID
                if gallery_pids[g_idx] == q_pid:
                    matches[q_idx, g_idx] = True
    else:
        # Multi-camera protocol: Match requires same PID AND different camera
        for q_idx in range(num_q):
            q_pid = query_pids[q_idx]
            q_camid = query_camids[q_idx]
            for g_idx in range(num_g):
                if gallery_pids[g_idx] == q_pid and gallery_camids[g_idx] != q_camid:
                    matches[q_idx, g_idx] = True

    all_cmc = []
    all_AP = []

    for q_idx in range(num_q):
        order = indices[q_idx]
        q_pid = query_pids[q_idx]
        q_camid = query_camids[q_idx]

        if single_camera_protocol:
            # VehicleID protocol: Remove query image itself (same index position)
            # Query and gallery are the same dataset, so q_idx maps to g_idx
            remove = torch.tensor([i == q_idx for i in range(num_g)])
        else:
            # Multi-camera protocol: Remove images from same PID and same camera
            remove = torch.tensor(
                [
                    gallery_pids[i] == q_pid and gallery_camids[i] == q_camid
                    for i in range(num_g)
                ]
            )

        keep = ~remove
        orig_cmc = matches[q_idx][order]

        # If after removing, no valid gallery items, skip.
        if not keep[order].any():
            continue

        cmc = orig_cmc[keep[order]]
        cmc = cmc.cumsum(dim=0)
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])

        # Compute Average Precision
        num_rel = orig_cmc[keep[order]].sum()
        if num_rel > 0:
            tmp_cmc = orig_cmc[keep[order]]
            tmp_cmc = tmp_cmc.cumsum(dim=0)
            tmp_cmc = tmp_cmc / (torch.arange(len(tmp_cmc)) + 1)
            tmp_cmc = tmp_cmc * orig_cmc[keep[order]]
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP.item())

    if len(all_cmc) == 0:
        console.print("[red]Error: No valid queries found![/red]")
        return 0.0, [0.0] * max_rank

    all_cmc = torch.stack(all_cmc).float()
    cmc = all_cmc.mean(dim=0)
    mAP = sum(all_AP) / len(all_AP) if all_AP else 0.0

    return mAP, cmc.tolist()


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    """k-reciprocal encoding re-ranking (Zhong et al., CVPR 2017).

    Refines a query→gallery distance matrix using the k-reciprocal nearest
    neighbours of every probe/gallery item, blending the resulting Jaccard
    distance with the original distance. Operates on the test set only — it is a
    pure inference-time post-processing step that requires no retraining and
    chiefly recovers the cross-camera recall tail (the residual Market gap).

    Args:
        q_g_dist: (num_q, num_g) query→gallery distances.
        q_q_dist: (num_q, num_q) query→query distances.
        g_g_dist: (num_g, num_g) gallery→gallery distances.
        k1: size of the k-reciprocal neighbour set.
        k2: size of the local query-expansion neighbourhood (k2=1 disables it).
        lambda_value: blend weight on the original distance (0 → pure Jaccard).

    Returns: (num_q, num_g) re-ranked distance matrix.
    """
    original_dist = np.concatenate(
        [
            np.concatenate([q_q_dist, q_g_dist], axis=1),
            np.concatenate([q_g_dist.T, g_g_dist], axis=1),
        ],
        axis=0,
    ).astype(np.float32)
    original_dist = np.power(original_dist, 2)
    original_dist = np.transpose(1.0 * original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist, dtype=np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    all_num = original_dist.shape[0]

    for i in range(all_num):
        forward_k_neigh_index = initial_rank[i, : k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, : k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            half = int(np.around(k1 / 2.0)) + 1
            candidate_forward = initial_rank[candidate, :half]
            candidate_backward = initial_rank[candidate_forward, :half]
            fi_c = np.where(candidate_backward == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward[fi_c]
            if len(
                np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)
            ) > 2.0 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index
                )
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = 1.0 * weight / np.sum(weight)

    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank

    inv_index = [np.where(V[:, i] != 0)[0] for i in range(all_num)]
    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)
    for i in range(query_num):
        temp_min = np.zeros((1, all_num), dtype=np.float32)
        ind_non_zero = np.where(V[i, :] != 0)[0]
        ind_images = [inv_index[ind] for ind in ind_non_zero]
        for j in range(len(ind_non_zero)):
            temp_min[0, ind_images[j]] += np.minimum(
                V[i, ind_non_zero[j]], V[ind_images[j], ind_non_zero[j]]
            )
        jaccard_dist[i] = 1 - temp_min / (2.0 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist, V, jaccard_dist
    return final_dist[:query_num, query_num:]


def _metrics_dict(mAP, cmc):
    """Pack (mAP, cmc) into the standard metrics dict (percentages)."""
    return {
        "mAP": mAP * 100,
        "R-1": cmc[0] * 100 if len(cmc) > 0 else 0.0,
        "R-5": cmc[4] * 100 if len(cmc) > 4 else 0.0,
        "R-10": cmc[9] * 100 if len(cmc) > 9 else 0.0,
        "R-20": cmc[19] * 100 if len(cmc) > 19 else 0.0,
    }


def _eval_from_features(
    qf, qp, qc, gf, gp, gc, max_rank=50,
    use_reranking=False, rerank_k1=20, rerank_k2=6, rerank_lambda=0.3,
):
    """Evaluate from already-extracted features (single or re-ranked)."""
    qf_n = F.normalize(qf, p=2, dim=1)
    gf_n = F.normalize(gf, p=2, dim=1)
    if use_reranking:
        console.print(
            f"[yellow]Re-ranking (k-reciprocal, k1={rerank_k1}, k2={rerank_k2}, "
            f"λ={rerank_lambda})...[/yellow]"
        )
        q_g = (1 - qf_n @ gf_n.t()).cpu().numpy()
        q_q = (1 - qf_n @ qf_n.t()).cpu().numpy()
        g_g = (1 - gf_n @ gf_n.t()).cpu().numpy()
        distance_matrix = re_ranking(q_g, q_q, g_g, rerank_k1, rerank_k2, rerank_lambda)
    else:
        distance_matrix = (1 - qf_n @ gf_n.t()).cpu().numpy()
    mAP, cmc = evaluate_reid(distance_matrix, qp, gp, qc, gc, max_rank)
    return _metrics_dict(mAP, cmc)


def validate_reid_model(
    model, query_loader, gallery_loader, device, max_rank=50,
    use_reranking=False, rerank_k1=20, rerank_k2=6, rerank_lambda=0.3,
):
    """Validate the ReID model and return mAP / Rank-k metrics.

    When use_reranking=True, applies k-reciprocal re-ranking to the
    query→gallery distances before scoring (test-time only).
    """
    query_features, query_pids, query_camids = extract_features(
        model, query_loader, device, "Extracting query features"
    )
    gallery_features, gallery_pids, gallery_camids = extract_features(
        model, gallery_loader, device, "Extracting gallery features"
    )
    console.print("[yellow]Evaluating ReID metrics...[/yellow]")
    return _eval_from_features(
        query_features, query_pids, query_camids,
        gallery_features, gallery_pids, gallery_camids,
        max_rank, use_reranking, rerank_k1, rerank_k2, rerank_lambda,
    )


def validate_reid_both(
    model, query_loader, gallery_loader, device, max_rank=50,
    rerank_k1=20, rerank_k2=6, rerank_lambda=0.3,
):
    """Evaluate WITH and WITHOUT re-ranking from a single feature extraction.

    Returns: (plain_metrics, rerank_metrics).
    """
    query_features, query_pids, query_camids = extract_features(
        model, query_loader, device, "Extracting query features"
    )
    gallery_features, gallery_pids, gallery_camids = extract_features(
        model, gallery_loader, device, "Extracting gallery features"
    )
    args = (
        query_features, query_pids, query_camids,
        gallery_features, gallery_pids, gallery_camids,
        max_rank,
    )
    plain = _eval_from_features(*args, use_reranking=False)
    rerank = _eval_from_features(
        *args, use_reranking=True,
        rerank_k1=rerank_k1, rerank_k2=rerank_k2, rerank_lambda=rerank_lambda,
    )
    return plain, rerank
