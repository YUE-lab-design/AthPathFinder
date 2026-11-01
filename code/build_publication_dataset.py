#!/usr/bin/env python3
"""Rebuild and audit the current pathway cohort before model evaluation.

The supplied curated table is treated only as the reviewed source of pathway
labels and the 67 curated non-metabolic controls. Gene memberships are rebuilt
from the distributed KEGG/AraCyc snapshots, restricted to loci present in the
distributed GO snapshot. Exact gene-set duplicates are collapsed. Exact sets
with conflicting family labels are marked ambiguous and excluded from Stage 2.
Connected components at Jaccard >= 0.80 define indivisible split groups.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from athpathfinder import paths
from athpathfinder.grouping import assign_similarity_groups, collapse_exact
from athpathfinder.io_data import read_current_go
from athpathfinder.problem import MIN_SET_SIZE, SIMILARITY_THRESHOLD

paths.refresh()
OUT = paths.PROCESSED


def raw_memberships(go_genes: set[str]) -> dict[str, set[str]]:
    kegg = pd.read_csv(
        paths.DATA / "kegg_ath_pathway_gene.txt", sep="\t", header=None,
        names=["pathway", "gene"],
    )
    kegg["gene"] = kegg.gene.str.replace("ath:", "", regex=False).str.upper()
    result = {
        f"KEGG::{pid}": set(group.gene) & go_genes
        for pid, group in kegg.groupby("pathway")
    }
    ara = pd.read_csv(paths.DATA / "aracyc_pathways.20251021.txt", sep="\t")
    ara["gene"] = ara["Gene-id"].astype(str).str.upper()
    result.update({
        f"AraCyc::{pid}": set(group.gene) & go_genes
        for pid, group in ara.groupby("Pathway-id")
    })
    return result


def reviewed_labels() -> pd.DataFrame:
    supplied = pd.read_csv(paths.CURATED)
    keep = supplied[
        supplied.label.eq(1) | (supplied.label.eq(0) & supplied.strategy.eq("positive"))
    ][["sample_id", "label", "source", "family"]].drop_duplicates("sample_id")
    keep["role"] = keep.label.map({1: "metabolic_pathway", 0: "curated_nonmetabolic"})
    return keep.sort_values(["label", "source", "sample_id"], ascending=[False, True, True])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gene_to_go = read_current_go()
    raw = raw_memberships(set(gene_to_go))
    labels = reviewed_labels()
    labels.to_csv(OUT / "reviewed_pathway_labels.csv", index=False)

    rows, membership_audit = [], []
    for row in labels.itertuples():
        genes = sorted(raw.get(row.sample_id, set()))
        membership_audit.append({
            "sample_id": row.sample_id,
            "raw_go_covered_gene_count": len(genes),
            "retained": len(genes) >= MIN_SET_SIZE,
        })
        if len(genes) >= MIN_SET_SIZE:
            rows.append({
                "sample_id": row.sample_id, "label": int(row.label),
                "source": row.source, "family": row.family,
                "genes_list": genes,
            })
    rebuilt = pd.DataFrame(rows)
    canonical, exact_audit = collapse_exact(rebuilt)
    canonical, near_pairs = assign_similarity_groups(canonical, SIMILARITY_THRESHOLD)

    canonical.to_csv(OUT / "current_canonical_pathways_and_controls.csv", index=False)
    pd.DataFrame(membership_audit).to_csv(OUT / "raw_membership_audit.csv", index=False)
    exact_audit.to_csv(OUT / "exact_duplicate_audit.csv", index=False)
    near_pairs.to_csv(OUT / "near_duplicate_pairs_jaccard_ge_0.80.csv", index=False)

    positives = canonical[canonical.label.eq(1)]
    controls = canonical[canonical.label.eq(0)]
    summary = {
        "source_label_records": int(len(labels)),
        "retained_after_raw_membership_and_five_gene_filter": int(len(rebuilt)),
        "canonical_metabolic_pathways": int(len(positives)),
        "canonical_curated_nonmetabolic_controls": int(len(controls)),
        "stage2_family_eligible_pathways": int(positives.family_eligible.sum()),
        "exact_duplicate_records_collapsed": int(len(rebuilt) - len(canonical)),
        "ambiguous_exact_set_family_groups": int((positives.family_eligible == False).sum()),
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "similarity_groups": int(canonical.similarity_group.nunique()),
        "near_duplicate_pairs": int(len(near_pairs)),
    }
    (OUT / "dataset_construction_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
