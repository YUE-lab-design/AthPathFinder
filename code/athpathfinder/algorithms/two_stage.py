"""Two-stage annotator: ŷ(G) = ⊥ if ŷ₁=0 else ŷ₂(G) (F2 / Algorithm 1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from ..problem import NON_METABOLIC


def threshold_by_balanced_accuracy(y, score) -> float:
    """Frozen publication gate rule: maximize balanced accuracy on development OOF."""
    values = np.unique(np.r_[0.0, score, 1.0])
    candidates = []
    for value in values:
        pred = (score >= value).astype(int)
        candidates.append((balanced_accuracy_score(y, pred), -abs(value - .5), value))
    return float(max(candidates)[2])


def compose_end_to_end(stage1_pred: np.ndarray, family_pred: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(stage1_pred) == 0, NON_METABOLIC, family_pred)


@dataclass
class TwoStageAnnotator:
    """Algorithm 1 wrapper around a PATHWAY-LIKE scorer, gate τ, and FAMILY head.

    Stage 1 and Stage 2 may use different representations: LightGBM on φ and
    Jaccard 1-NN on gene sets is the deployed FAMILY head after hybrid selection.
    """

    stage1: object
    stage2: object
    gate: float
    families: list[str]
    stage1_uses_gene_sets: bool = False
    stage2_uses_gene_sets: bool = False
    uses_gene_sets: bool = False
    uses_phi: bool = True

    def pathway_like_score(self, X=None, gene_sets=None) -> np.ndarray:
        if self.uses_gene_sets or self.stage1_uses_gene_sets:
            return self.stage1.score(gene_sets)
        return self.stage1.score(X)

    def predict_family(self, X=None, gene_sets=None) -> np.ndarray:
        if self.uses_gene_sets or self.stage2_uses_gene_sets:
            return self.stage2.predict(gene_sets)
        return self.stage2.predict(X)

    def predict(self, X=None, gene_sets=None) -> dict[str, np.ndarray]:
        score = self.pathway_like_score(X=X, gene_sets=gene_sets)
        stage1_pred = (score >= self.gate).astype(int)
        family_pred = self.predict_family(X=X, gene_sets=gene_sets)
        return {
            "stage1_score": score,
            "pathway_like": stage1_pred,
            "family": family_pred,
            "label": compose_end_to_end(stage1_pred, family_pred),
        }
