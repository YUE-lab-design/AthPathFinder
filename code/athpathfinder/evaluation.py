"""Current cohort reconstruction and archived grouped split helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from . import paths
from .grouping import assign_similarity_groups
from .io_data import generate_constructed_controls,load_canonical
from .problem import FAMILY_MIN,N_CONSTRUCTED_PRIMARY,NON_METABOLIC,SEED

def development_splits(data: pd.DataFrame, seed: int = SEED):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return list(cv.split(data, data.split_label, data.similarity_group))

def choose_outer_split(data: pd.DataFrame):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    candidates = []
    overall = data.split_label.value_counts(normalize=True)
    eligible = set(
        data.loc[data.label.eq(1) & data.family_eligible, "family"]
        .value_counts()
        .loc[lambda s: s.ge(FAMILY_MIN)]
        .index
    )
    for fold, (dev, test) in enumerate(splitter.split(data, data.split_label, data.similarity_group), 1):
        test_labels = set(data.iloc[test].loc[data.iloc[test].label.eq(1), "family"])
        missing = len(eligible - test_labels)
        observed = data.iloc[test].split_label.value_counts(normalize=True)
        distance = float(sum(abs(observed.get(k, 0) - overall.get(k, 0)) for k in overall.index))
        candidates.append((missing, distance, fold, dev, test))
    _, _, fold, dev, test = sorted(candidates, key=lambda x: (x[0], x[1], x[2]))[0]
    return data.iloc[dev].copy().reset_index(drop=True), data.iloc[test].copy().reset_index(drop=True), fold

def primary_cohort(positives, curated, gene_to_go, n_constructed=N_CONSTRUCTED_PRIMARY):
    constructed = generate_constructed_controls(
        positives, gene_to_go, n_constructed, f"grouped_{n_constructed}"
    )
    constructed["family_eligible"] = False
    constructed["n_collapsed_records"] = 1
    cohort = pd.concat([positives, curated, constructed], ignore_index=True, sort=False)
    cohort, near = assign_similarity_groups(cohort)
    common_family = cohort.loc[
        cohort.label.eq(1) & ~cohort.family.str.startswith("Ambiguous"), "family"
    ].value_counts().index[0]
    cohort["split_label"] = np.where(
        cohort.label.eq(0), NON_METABOLIC,
        np.where(cohort.family.str.startswith("Ambiguous"), common_family, cohort.family),
    )
    return cohort, near

def reconstruct_frozen_split() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
    """Rebuild the publication cohort and attach the frozen independent partition."""
    positives, curated, gene_to_go = load_canonical()
    cohort, _ = primary_cohort(positives, curated, gene_to_go, N_CONSTRUCTED_PRIMARY)
    part = pd.read_csv(paths.GROUPED_OUTPUT / "independent_partition.csv")
    mapped = cohort.merge(part[["sample_id", "partition"]], on="sample_id", how="inner")
    missing = set(part.sample_id) - set(mapped.sample_id)
    if missing:
        raise RuntimeError(
            f"Reconstructed cohort is missing {len(missing)} frozen-partition ids; "
            "constructed-control RNG or canonical table does not match."
        )
    extra = set(mapped.sample_id) - set(part.sample_id)
    if extra:
        raise RuntimeError(f"Reconstructed cohort has {len(extra)} unexpected sample ids.")
    dev = mapped[mapped.partition.eq("development")].copy().reset_index(drop=True)
    test = mapped[mapped.partition.eq("independent_test")].copy().reset_index(drop=True)
    return dev, test, gene_to_go
