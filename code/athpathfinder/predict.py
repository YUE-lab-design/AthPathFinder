"""Public inference API: predict(gene_list) outside the training loop."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.base import clone

from . import paths
from .embedding import build_features, select_go_terms, normalize_gene_set, FEATURE_VERSION
from .io_data import parse_genes, read_current_go
from .problem import (
    DEPLOYED_FAMILY_HEAD,
    FAMILIES,
    GATE_SELECTION,
    NON_METABOLIC,
    SEED,
    STAGE1_HYPOTHESIS,
    STAGE2_HYPOTHESIS,
)


def validate_query(genes, gene_to_go):
    raw = list(genes) if not isinstance(genes, str) else genes.replace(',', ' ').replace(';', ' ').split()
    tokens = sorted({str(g).strip().upper() for g in raw if str(g).strip()})
    parsed = parse_genes(tokens)
    retained = normalize_gene_set(parsed, gene_to_go)
    discarded = sorted(set(tokens) - set(retained))
    if len(retained) < 5:
        raise ValueError(f"At least 5 distinct retained GO-universe genes are required; retained {len(retained)}, discarded {len(discarded)}")
    return retained, discarded

@dataclass
class Prediction:
    stage1_score: float
    pathway_like: bool
    family: str | None
    label: str
    genes: list[str] = field(default_factory=list)
    discarded_genes: list[str] = field(default_factory=list)
    family_jaccard: float | None = None
    no_gene_overlap_support: bool | None = None

    def as_dict(self) -> dict:
        return {
            "stage1_score": self.stage1_score,
            "pathway_like": self.pathway_like,
            "family": self.family,
            "label": self.label,
            "genes": self.genes,
            "discarded_genes": self.discarded_genes,
            "family_jaccard": self.family_jaccard,
            "no_gene_overlap_support": self.no_gene_overlap_support,
            "feature_version": FEATURE_VERSION,
        }


@dataclass
class AthPathFinder:
    """Frozen Algorithm 1 fit: φ, PATHWAY-LIKE, gate τ, FAMILY."""

    selected_go: list[str]
    gene_to_go: dict[str, set[str]]
    stage1: object
    stage2: object
    gate: float
    families: list[str]
    stage1_name: str = STAGE1_HYPOTHESIS
    stage2_name: str = "Jaccard 1-NN"
    family_kind: str = "jaccard"
    feature_names: list[str] | None = None

    def embed(self, genes: Sequence[str], sample_id: str = "query") -> pd.DataFrame:
        retained, _ = validate_query(genes, self.gene_to_go)
        frame = pd.DataFrame({"sample_id": [sample_id], "genes_list": [retained]})
        x, names = build_features(frame, self.selected_go, self.gene_to_go)
        self.feature_names = names
        return x

    def predict_one(self, genes: Sequence[str], sample_id: str = "query") -> Prediction:
        cleaned, discarded = validate_query(genes, self.gene_to_go)
        x = self.embed(cleaned, sample_id=sample_id)
        score = float(self.stage1.predict_proba(x)[:, 1][0])
        pathway_like = bool(score >= self.gate)
        if self.family_kind == "jaccard":
            labels, similarities = self.stage2.predict_with_support([cleaned])
            family = str(labels[0]); family_jaccard = float(similarities[0])
        else:
            family = str(self.families[int(self.stage2.predict(x)[0])])
        if self.family_kind != "jaccard":
            family_jaccard = None
        label = NON_METABOLIC if not pathway_like else family
        return Prediction(
            stage1_score=score,
            pathway_like=pathway_like,
            family=family if pathway_like else None,
            label=label,
            genes=cleaned,
            discarded_genes=discarded,
            family_jaccard=family_jaccard if pathway_like else None,
            no_gene_overlap_support=(family_jaccard == 0.0) if pathway_like and family_jaccard is not None else None,
        )

    def save(self, path: Path | None = None) -> Path:
        path = Path(path) if path is not None else paths.FITS / "publication_athpathfinder_geneset_v2.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_version": FEATURE_VERSION,
            "selected_go": self.selected_go,
            "gate": self.gate,
            "families": self.families,
            "stage1_name": self.stage1_name,
            "stage2_name": self.stage2_name,
            "family_kind": self.family_kind,
            "stage1": self.stage1,
            "stage2": self.stage2,
            "seed": SEED,
            "gate_selection": GATE_SELECTION,
        }
        path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        meta = path.with_suffix(".json")
        meta.write_text(json.dumps({
            "feature_version": FEATURE_VERSION,
            "selected_go": self.selected_go,
            "gate": self.gate,
            "families": self.families,
            "stage1_name": self.stage1_name,
            "stage2_name": self.stage2_name,
            "family_kind": self.family_kind,
            "n_go": len(self.selected_go),
            "pickle": str(path),
        }, indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: Path | None = None, gene_to_go: dict[str, set[str]] | None = None) -> "AthPathFinder":
        path = Path(path) if path is not None else paths.FITS / "publication_athpathfinder_geneset_v2.pkl"
        payload = pickle.loads(path.read_bytes())
        if payload.get("feature_version") != FEATURE_VERSION:
            raise ValueError("Saved fit uses an incompatible feature version; refit the gene-set-seeded implementation")
        return cls(
            selected_go=payload["selected_go"],
            gene_to_go=gene_to_go if gene_to_go is not None else read_current_go(),
            stage1=payload["stage1"],
            stage2=payload["stage2"],
            gate=float(payload["gate"]),
            families=list(payload["families"]),
            stage1_name=payload.get("stage1_name", STAGE1_HYPOTHESIS),
            stage2_name=payload.get("stage2_name", STAGE2_HYPOTHESIS),
            family_kind=payload.get("family_kind", "supervised"),
        )


def fit_publication_model(*, save: bool = True) -> AthPathFinder:
    """Fit Algorithm 1 on the frozen development partition.

    Uses the fixed LightGBM/Jaccard architecture and archived current
    development folds. The gene-set-seeded feature rule is used throughout;
    the gate is selected only from development OOF predictions.
    """
    from .training import fit_current_model
    model, _ = fit_current_model()
    if save:
        model.save()
    return model


def load_publication_model(*, fit_if_missing: bool = True) -> AthPathFinder:
    pickle_path = paths.FITS / "publication_athpathfinder_geneset_v2.pkl"
    if pickle_path.exists():
        model = AthPathFinder.load(pickle_path)
        if model.family_kind != "jaccard":
            raise ValueError("The repaired implementation requires the fixed Jaccard family head; run --fit")
        return model
    elif not fit_if_missing:
        raise FileNotFoundError(
            f"No frozen fit at {pickle_path}. Run python -m athpathfinder --fit"
        )
    return fit_publication_model(save=True)


def predict(genes: Sequence[str], model: AthPathFinder | None = None) -> dict:
    """Annotate one gene set. Returns stage1_score, pathway_like, family, label."""
    if model is None:
        validate_query(genes, read_current_go())
    fitted = model if model is not None else load_publication_model()
    return fitted.predict_one(genes).as_dict()
