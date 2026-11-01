"""Set embedding φ(G): mean-pooled GO indicators plus optional descriptors.

Genes are instances and G is a bag (F3). GO frequencies are mean pooling of
gene-level GO indicators. The optional pairwise block summarizes gene-level
GO-set Jaccard similarities. The optional size block contains log(1+|G|),
the unnormalized positive-frequency self-information
-∑_{f_t(G)>0} f_t(G) log2 f_t(G) (column name go_entropy; not Shannon
entropy of a probability vector), and the standard deviation of per-gene
GO counts.

Supervised mutual-information ranking of GO terms is fold-wise.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_selection import mutual_info_classif

from .io_data import stable_rng
from .problem import (
    K_GO,
    MAX_GO_PERCENT,
    MIN_GO_GENES,
    OT_N_TERMS,
    PAIRWISE_CAP,
    SEED,
    VARIANCE_QUANTILE,
)

FEATURE_VERSION = "gene_set_seed_v2"

def normalize_gene_set(genes, gene_to_go):
    """Canonical retained universe members; independent of names and input order."""
    return sorted({str(g).strip().upper() for g in genes if str(g).strip().upper() in gene_to_go})

PAIRWISE_NAMES = (
    "jaccard_mean",
    "jaccard_min",
    "jaccard_max",
    "jaccard_std",
)
SIZE_NAMES = (
    "log_gene_count",
    "go_entropy",
    "per_gene_go_count_std",
)


def go_feature_names(selected_go: list[str]) -> list[str]:
    return [f"gofreq::{term}" for term in selected_go]


def feature_names(selected_go: list[str], *, pairwise: bool = True, size: bool = True) -> list[str]:
    names = go_feature_names(selected_go)
    if pairwise:
        names.extend(PAIRWISE_NAMES)
    if size:
        names.extend(SIZE_NAMES)
    return names


def feature_groups(names: list[str]) -> dict[str, list[str]]:
    go = [x for x in names if x.startswith("gofreq::")]
    jaccard = [x for x in names if x.startswith("jaccard_")]
    size = [x for x in names if x in SIZE_NAMES]
    return {
        "Full model": go + jaccard + size,
        "Without GO frequency": jaccard + size,
        "Without Jaccard": go + size,
        "Without size/entropy": go + jaccard,
        "GO frequency only": go,
        "Jaccard only": jaccard,
        "Size/entropy only": size,
    }


def label_free_go_terms(
    gene_to_go: dict[str, set[str]],
    n_terms: int = OT_N_TERMS,
) -> tuple[list[str], dict]:
    """Frequency/variance-filtered GO terms. No outcome labels."""
    genes = sorted(gene_to_go)
    gene_index = {g: i for i, g in enumerate(genes)}
    terms = sorted({t for values in gene_to_go.values() for t in values})
    term_index = {t: i for i, t in enumerate(terms)}
    row, col = [], []
    for gene, go_ids in gene_to_go.items():
        for go_id in go_ids:
            row.append(gene_index[gene])
            col.append(term_index[go_id])
    matrix = csr_matrix(
        (np.ones(len(row), dtype=np.uint8), (row, col)),
        shape=(len(genes), len(terms)),
    )
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    keep_freq = (counts >= MIN_GO_GENES) & (counts <= int(len(genes) * MAX_GO_PERCENT))
    freq_idx = np.flatnonzero(keep_freq)
    p = counts[freq_idx] / len(genes)
    variance = p * (1 - p)
    cutoff = np.quantile(variance, VARIANCE_QUANTILE)
    var_idx = freq_idx[variance > cutoff]
    var_vals = (counts[var_idx] / len(genes)) * (1 - counts[var_idx] / len(genes))
    order = np.argsort(-var_vals, kind="stable")[: min(n_terms, len(var_idx))]
    selected = [terms[var_idx[i]] for i in order]
    return selected, {
        "n_genes": len(genes),
        "n_all": len(terms),
        "n_frequency": int(len(freq_idx)),
        "n_variance": int(len(var_idx)),
        "n_ot_terms": len(selected),
        "selected_go": selected,
        "selection": "label_free_frequency_variance_top_variance",
    }


def go_histograms(
    gene_sets: list[list[str]],
    terms: list[str],
    gene_to_go: dict[str, set[str]],
) -> np.ndarray:
    """Mean-pooled GO frequencies on ``terms``, row-normalized to a discrete measure."""
    hist = np.zeros((len(gene_sets), len(terms) + 1), dtype=float)
    term_index = {term: j for j, term in enumerate(terms)}
    for i, genes in enumerate(gene_sets):
        n = max(len(genes), 1)
        for gene in genes:
            for term in gene_to_go.get(gene, ()):
                j = term_index.get(term)
                if j is not None:
                    hist[i, j] += 1.0
        hist[i, :-1] /= n
        if hist[i, :-1].sum() <= 0:
            hist[i, -1] = 1.0
        else:
            hist[i] /= hist[i].sum()
    return hist


def select_go_terms(gene_to_go: dict[str, set[str]], positive_gene_universe: set[str]):
    genes = sorted(gene_to_go)
    gene_index = {g: i for i, g in enumerate(genes)}
    terms = sorted({t for values in gene_to_go.values() for t in values})
    term_index = {t: i for i, t in enumerate(terms)}
    row, col = [], []
    for gene, go_ids in gene_to_go.items():
        for go_id in go_ids:
            row.append(gene_index[gene])
            col.append(term_index[go_id])
    matrix = csr_matrix(
        (np.ones(len(row), dtype=np.uint8), (row, col)),
        shape=(len(genes), len(terms)),
    )
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    keep_freq = (counts >= MIN_GO_GENES) & (counts <= int(len(genes) * MAX_GO_PERCENT))
    freq_idx = np.flatnonzero(keep_freq)
    p = counts[freq_idx] / len(genes)
    variance = p * (1 - p)
    cutoff = np.quantile(variance, VARIANCE_QUANTILE)
    var_idx = freq_idx[variance > cutoff]
    y = np.array([g in positive_gene_universe for g in genes], dtype=np.uint8)
    mi = mutual_info_classif(matrix[:, var_idx], y, discrete_features=True, random_state=SEED)
    order = np.argsort(-mi, kind="stable")[: min(K_GO, len(mi))]
    selected = [terms[var_idx[i]] for i in order]
    return selected, {
        "n_genes": len(genes),
        "n_all": len(terms),
        "n_frequency": len(freq_idx),
        "n_variance": len(var_idx),
        "n_selected": len(selected),
        "selected_go": selected,
        "selected_mi": [float(mi[i]) for i in order],
        "min_go_genes": MIN_GO_GENES,
        "max_go_percent": MAX_GO_PERCENT,
        "variance_quantile": VARIANCE_QUANTILE,
        "k_go": K_GO,
    }


def _go_frequencies(genes: list[str], selected_go: list[str], gene_to_go: dict[str, set[str]]) -> list[float]:
    n = len(genes)
    return [sum(go_id in gene_to_go.get(g, set()) for g in genes) / n for go_id in selected_go]


def _pairwise_go_jaccard(
    genes: list[str],
    gene_to_go: dict[str, set[str]],
    sample_id: str,
) -> list[float]:
    genes = normalize_gene_set(genes, gene_to_go)
    sampled = genes
    if len(genes) > PAIRWISE_CAP:
        sampled = stable_rng("pairs|gene-set-v2|" + "|".join(genes)).sample(genes, PAIRWISE_CAP)
    sims = []
    for a, b in combinations(sampled, 2):
        aa, bb = gene_to_go.get(a, set()), gene_to_go.get(b, set())
        sims.append(len(aa & bb) / len(aa | bb) if aa | bb else 0.0)
    return [
        float(np.mean(sims)) if sims else 0.0,
        float(np.min(sims)) if sims else 0.0,
        float(np.max(sims)) if sims else 0.0,
        float(np.std(sims)) if sims else 0.0,
    ]


def _positive_frequency_self_information(freq: list[float]) -> float:
    """-∑_{f>0} f log2 f. Not Shannon entropy: frequencies are not renormalized."""
    positive = np.array([x for x in freq if x > 0], dtype=float)
    if len(positive) == 0:
        return 0.0
    return float(-np.sum(positive * np.log2(positive)))


def _size_descriptors(genes: list[str], freq: list[float], gene_to_go: dict[str, set[str]]) -> list[float]:
    counts = [len(gene_to_go.get(g, set())) for g in genes]
    return [
        math.log1p(len(genes)),
        _positive_frequency_self_information(freq),
        float(np.std(counts)) if counts else 0.0,
    ]


def phi(
    genes: list[str],
    selected_go: list[str],
    gene_to_go: dict[str, set[str]],
    sample_id: str = "",
    *,
    pairwise: bool = True,
    size: bool = True,
) -> list[float]:
    """Permutation-invariant set embedding φ(G)."""
    genes = normalize_gene_set(genes, gene_to_go)
    if not genes:
        n_go = len(selected_go)
        n_extra = (len(PAIRWISE_NAMES) if pairwise else 0) + (len(SIZE_NAMES) if size else 0)
        return [0.0] * (n_go + n_extra)
    freq = _go_frequencies(genes, selected_go, gene_to_go)
    values = list(freq)
    if pairwise:
        values.extend(_pairwise_go_jaccard(genes, gene_to_go, sample_id))
    if size:
        values.extend(_size_descriptors(genes, freq, gene_to_go))
    return values


def featurize(
    genes: list[str],
    selected_go: list[str],
    gene_to_go: dict[str, set[str]],
    sample_id: str,
) -> list[float]:
    """Complete 67-d publication recipe: 60 GO frequencies + 4 pairwise + 3 size."""
    return phi(genes, selected_go, gene_to_go, sample_id, pairwise=True, size=True)


def build_features(
    df: pd.DataFrame,
    selected_go: list[str],
    gene_to_go: dict[str, set[str]],
):
    names = feature_names(selected_go)
    values = [
        featurize(r.genes_list, selected_go, gene_to_go, str(r.sample_id))
        for r in df.itertuples()
    ]
    return pd.DataFrame(values, columns=names, index=df.index), names


def select_and_feature(train, test, gene_to_go):
    positive_genes = set().union(*(set(x) for x in train.loc[train.label == 1, "genes_list"]))
    selected, audit = select_go_terms(gene_to_go, positive_genes)
    x_train, names = build_features(train, selected, gene_to_go)
    x_test, _ = build_features(test, selected, gene_to_go)
    return x_train, x_test, names, audit
