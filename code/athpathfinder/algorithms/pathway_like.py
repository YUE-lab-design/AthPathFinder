"""PATHWAY-LIKE stage: ŷ₁(G) ∈ {0,1} (F2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.base import clone

from ..distances import OptimalTransportDistance, jaccard_similarity, nearest_euclidean
from ..problem import OT_SINKHORN_EPS, OT_SINKHORN_ITERS


@dataclass
class SupervisedPathwayLike:
    """Binary classifier on φ(G). Default hypothesis class is LightGBM."""

    estimator: object | None = None
    fitted_: object | None = None

    def fit(self, X, y) -> "SupervisedPathwayLike":
        from .hypothesis_classes import deployed_stage1
        model = self.estimator if self.estimator is not None else deployed_stage1()
        self.fitted_ = clone(model).fit(X, y)
        return self

    def score(self, X) -> np.ndarray:
        return self.fitted_.predict_proba(X)[:, 1]


@dataclass
class JaccardPathwayLike:
    """F4: ŷ₁ = 1 iff min gene-set Jaccard distance ≤ τ, equivalently max similarity ≥ 1-τ.

    The returned score is maximum Jaccard similarity to a training pathway so
    that the same balanced-accuracy gate rule as Algorithm 1 can choose τ.
    """

    references: list[list[str]] = field(default_factory=list)

    def fit(self, gene_sets: Sequence[Sequence[str]], y: Sequence[int]) -> "JaccardPathwayLike":
        self.references = [list(g) for g, yi in zip(gene_sets, y) if int(yi) == 1]
        return self

    def score(self, gene_sets: Sequence[Sequence[str]]) -> np.ndarray:
        return np.array([
            max((jaccard_similarity(g, ref) for ref in self.references), default=0.0)
            for g in gene_sets
        ], dtype=float)


@dataclass
class PhiSpacePathwayLike:
    """F4: ŷ₁ uses minimum Euclidean distance in φ-space to a training pathway.

    Score is -min distance so that larger values are more pathway-like.
    """

    references: np.ndarray | None = None

    def fit(self, X, y) -> "PhiSpacePathwayLike":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.references = X[y == 1]
        return self

    def score(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        scores = []
        for row in X:
            _, dist = nearest_euclidean(row, self.references)
            scores.append(-dist)
        return np.asarray(scores, dtype=float)


@dataclass
class GOOTPathwayLike:
    """PATHWAY-LIKE score = −min Sinkhorn-Wasserstein to a training pathway.

    Histograms live on the label-free GO term universe plus one unannotated bin.
    Ground cost is Jaccard distance between term gene-support sets.
    """

    terms: list[str] = field(default_factory=list)
    gene_to_go: dict | None = None
    ot: OptimalTransportDistance | None = None
    references: np.ndarray | None = None

    def fit(
        self,
        histograms: np.ndarray,
        y: Sequence[int],
        *,
        terms: list[str],
        cost: np.ndarray,
        gene_to_go: dict,
        eps: float = OT_SINKHORN_EPS,
        n_iter: int = OT_SINKHORN_ITERS,
    ) -> "GOOTPathwayLike":
        y = np.asarray(y)
        self.terms = list(terms)
        self.gene_to_go = gene_to_go
        self.ot = OptimalTransportDistance(cost, eps, n_iter)
        self.references = np.asarray(histograms, dtype=float)[y == 1]
        return self

    def score(self, histograms: np.ndarray) -> np.ndarray:
        dist = self.ot.min_distance(np.asarray(histograms, dtype=float), self.references)
        return -dist
