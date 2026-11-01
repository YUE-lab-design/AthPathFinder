"""FAMILY stage: ŷ₂(G) ∈ F given pathway-like (F2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.base import clone

from ..distances import nearest_euclidean, nearest_jaccard


@dataclass
class SupervisedFamily:
    """Multiclass classifier on φ(G). Default hypothesis class is Extra Trees."""

    families: list[str] = field(default_factory=list)
    estimator: object | None = None
    fitted_: object | None = None

    def fit(self, X, y_family: Sequence[str]) -> "SupervisedFamily":
        from .hypothesis_classes import deployed_stage2
        self.families = sorted(set(y_family))
        family_to_int = {family: i for i, family in enumerate(self.families)}
        codes = np.array([family_to_int[v] for v in y_family])
        model = self.estimator if self.estimator is not None else deployed_stage2()
        self.fitted_ = clone(model).fit(X, codes)
        return self

    def predict(self, X) -> np.ndarray:
        codes = self.fitted_.predict(X).astype(int)
        return np.array([self.families[i] for i in codes])


@dataclass
class JaccardFamily:
    """F4: family of the nearest training pathway under gene-set Jaccard."""

    families: list[str] = field(default_factory=list)
    references: list[list[str]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    def fit(
        self,
        gene_sets: Sequence[Sequence[str]],
        y_family: Sequence[str],
    ) -> "JaccardFamily":
        self.references = [list(g) for g in gene_sets]
        self.labels = list(y_family)
        self.families = sorted(set(y_family))
        return self

    def predict_with_support(self, gene_sets):
        """Keep first-max labels; attach zero-overlap support without abstaining."""
        from ..distances import nearest_jaccard
        if not self.references:
            raise ValueError("A nonempty family reference catalog is required")
        labels, similarities = [], []
        for genes in gene_sets:
            index, similarity = nearest_jaccard(genes, self.references)
            labels.append(self.labels[index]); similarities.append(float(similarity))
        return np.asarray(labels), np.asarray(similarities)

    def predict(self, gene_sets: Sequence[Sequence[str]]) -> np.ndarray:
        return self.predict_with_support(gene_sets)[0]


@dataclass
class PhiSpaceFamily:
    """F4: family of the nearest training pathway in Euclidean φ-space."""

    families: list[str] = field(default_factory=list)
    references: np.ndarray | None = None
    labels: list[str] = field(default_factory=list)

    def fit(self, X, y_family: Sequence[str]) -> "PhiSpaceFamily":
        self.references = np.asarray(X, dtype=float)
        self.labels = list(y_family)
        self.families = sorted(set(y_family))
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = []
        for row in X:
            idx, _ = nearest_euclidean(row, self.references)
            out.append(self.labels[idx] if idx >= 0 else self.families[0])
        return np.array(out)


@dataclass
class PhiPrototypeFamily:
    """F4: nearest family mean in φ-space."""

    families: list[str] = field(default_factory=list)
    prototypes: np.ndarray | None = None

    def fit(self, X, y_family: Sequence[str]) -> "PhiPrototypeFamily":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y_family)
        self.families = sorted(set(y_family))
        self.prototypes = np.vstack([
            X[y == family].mean(axis=0) for family in self.families
        ])
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = []
        for row in X:
            idx, _ = nearest_euclidean(row, self.prototypes)
            out.append(self.families[idx] if idx >= 0 else self.families[0])
        return np.array(out)
