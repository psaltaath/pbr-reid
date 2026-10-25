"""Evaluation configuration."""

from dataclasses import dataclass

from .base import BaseConfig


@dataclass
class EvaluationConfig(BaseConfig):
    """Retrieval evaluation (CMC / mAP, cosine nearest neighbour)."""

    max_rank: int = 50

    # k-reciprocal re-ranking (Zhong et al. 2017) is a test-time post-process
    # NOT used for any number in the paper. When true, eval.py and the
    # end-of-training summary additionally report re-ranked metrics.
    rerank: bool = False
    rerank_k1: int = 20
    rerank_k2: int = 6
    rerank_lambda: float = 0.3
