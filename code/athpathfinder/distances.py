"""Set distances for metric annotation (F4).

Optimal transport is a Stage-1 candidate distance on GO histograms, not a
third AthPathFinder. This module ships Jaccard (on gene sets), Euclidean
distance in φ-space, and entropy-regularized Sinkhorn on GO-term measures.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class SetDistance(Protocol):
    """Distance between two gene sets, optionally using embeddings."""

    name: str

    def pairwise(self, a: Sequence[str], b: Sequence[str]) -> float:
        ...


def jaccard_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    aa, bb = set(a), set(b)
    union = aa | bb
    if not union:
        return 0.0
    return len(aa & bb) / len(union)


def jaccard_distance(a: Sequence[str], b: Sequence[str]) -> float:
    return 1.0 - jaccard_similarity(a, b)


class JaccardDistance:
    name = "jaccard"

    def pairwise(self, a: Sequence[str], b: Sequence[str]) -> float:
        return jaccard_distance(a, b)


class PhiSpaceDistance:
    """Euclidean distance between precomputed φ vectors (aligned columns)."""

    name = "phi_euclidean"

    def pairwise_vectors(self, u: np.ndarray, v: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(u, dtype=float) - np.asarray(v, dtype=float)))

    def pairwise(self, a: Sequence[str], b: Sequence[str]) -> float:
        raise TypeError(
            "PhiSpaceDistance operates on φ vectors via pairwise_vectors; "
            "embed both sets before comparing."
        )


def support_jaccard_cost(gene_to_go: dict[str, set[str]], terms: list[str]) -> np.ndarray:
    """Pairwise Jaccard distance between GO-term gene supports, plus an unannotated bin."""
    term_index = {term: i for i, term in enumerate(terms)}
    supports = [set() for _ in terms]
    for gene, gos in gene_to_go.items():
        for term in gos:
            j = term_index.get(term)
            if j is not None:
                supports[j].add(gene)
    n = len(terms) + 1
    cost = np.ones((n, n), dtype=float)
    np.fill_diagonal(cost, 0.0)
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            union = supports[i] | supports[j]
            value = 1.0 if not union else 1.0 - len(supports[i] & supports[j]) / len(union)
            cost[i, j] = cost[j, i] = value
    cost[-1, :] = 1.0
    cost[:, -1] = 1.0
    cost[-1, -1] = 0.0
    return cost


def sinkhorn_kernel(cost: np.ndarray, eps: float) -> np.ndarray:
    return np.exp(-np.asarray(cost, dtype=float) / float(eps))


def min_sinkhorn_to_references(
    queries: np.ndarray,
    references: np.ndarray,
    kernel: np.ndarray,
    cost: np.ndarray,
    n_iter: int,
) -> np.ndarray:
    """Entropy-regularized 1-Wasserstein from each query to the nearest reference.

    Vectorized over references for one query at a time. Score for PATHWAY-LIKE
    is the negative of this distance.
    """
    queries = np.asarray(queries, dtype=float)
    references = np.asarray(references, dtype=float)
    if len(references) == 0:
        return np.full(len(queries), np.inf)
    out = np.empty(len(queries), dtype=float)
    kt = kernel.T
    eps = 1e-12
    for i, a in enumerate(queries):
        u = np.ones_like(references)
        v = np.ones_like(references)
        for _ in range(n_iter):
            kv = v @ kt
            u = a[None, :] / np.maximum(kv, eps)
            ktu = u @ kernel
            v = references / np.maximum(ktu, eps)
        coupling_cost = ((u @ (kernel * cost)) * v).sum(axis=1)
        out[i] = float(coupling_cost.min())
    return out


class OptimalTransportDistance:
    """Entropy-regularized OT on GO-frequency histograms (Sinkhorn)."""

    name = "go_histogram_sinkhorn"

    def __init__(self, cost: np.ndarray, eps: float, n_iter: int):
        self.cost = np.asarray(cost, dtype=float)
        self.eps = float(eps)
        self.n_iter = int(n_iter)
        self.kernel = sinkhorn_kernel(self.cost, self.eps)

    def min_distance(self, queries: np.ndarray, references: np.ndarray) -> np.ndarray:
        return min_sinkhorn_to_references(
            queries, references, self.kernel, self.cost, self.n_iter
        )

    def pairwise(self, a: Sequence[str], b: Sequence[str]) -> float:
        raise TypeError(
            "OptimalTransportDistance compares GO histograms via min_distance."
        )


def max_jaccard(candidate: set[str], reference_sets: list[set[str]]) -> float:
    best = 0.0
    for ref in reference_sets:
        union = candidate | ref
        best = max(best, len(candidate & ref) / len(union) if union else 0.0)
    return best


def nearest_jaccard(
    query: Sequence[str],
    references: Sequence[Sequence[str]],
) -> tuple[int, float]:
    """Return (index, similarity) of the nearest reference set."""
    if not references:
        return -1, 0.0
    scores = [jaccard_similarity(query, ref) for ref in references]
    idx = int(np.argmax(scores))
    return idx, float(scores[idx])


def nearest_euclidean(query: np.ndarray, references: np.ndarray) -> tuple[int, float]:
    if len(references) == 0:
        return -1, float("inf")
    delta = references - query
    distances = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    idx = int(np.argmin(distances))
    return idx, float(distances[idx])
