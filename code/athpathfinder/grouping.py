"""Exact-set collapse and Jaccard-0.80 connected components."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import pandas as pd

from .problem import SIMILARITY_THRESHOLD


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def collapse_exact(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    buckets: dict[tuple[str, ...], list] = defaultdict(list)
    for row in rows.itertuples():
        buckets[tuple(row.genes_list)].append(row)
    collapsed, audit = [], []
    for genes, members in buckets.items():
        labels = sorted({int(x.label) for x in members})
        if len(labels) != 1:
            decision = "excluded_label_conflict"
            for x in members:
                audit.append({
                    "canonical_id": "",
                    "sample_id": x.sample_id,
                    "decision": decision,
                    "family": x.family,
                })
            continue
        families = sorted({str(x.family) for x in members})
        family_conflict = labels[0] == 1 and len(families) > 1
        digest = hashlib.sha256("|".join(genes).encode()).hexdigest()[:12]
        canonical_id = f"canonical::{labels[0]}::{digest}"
        family = "Ambiguous exact-set family conflict" if family_conflict else families[0]
        collapsed.append({
            "sample_id": canonical_id,
            "original_ids": "|".join(sorted(x.sample_id for x in members)),
            "label": labels[0],
            "source": "+".join(sorted({str(x.source) for x in members})),
            "family": family,
            "family_eligible": bool(labels[0] == 1 and not family_conflict),
            "strategy": (
                "curated_metabolic_pathway" if labels[0] == 1
                else "curated_nonmetabolic_control"
            ),
            "genes_list": list(genes),
            "genes": json.dumps(list(genes)),
            "n_collapsed_records": len(members),
        })
        decision = "collapsed_exact_duplicate" if len(members) > 1 else "retained_unique"
        if family_conflict:
            decision = "retained_stage1_excluded_stage2_family_conflict"
        for x in members:
            audit.append({
                "canonical_id": canonical_id,
                "sample_id": x.sample_id,
                "decision": decision,
                "family": x.family,
            })
    return pd.DataFrame(collapsed), pd.DataFrame(audit)


def assign_similarity_groups(
    data: pd.DataFrame,
    threshold: float = SIMILARITY_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sets = [set(x) for x in data.genes_list]
    uf = UnionFind(len(data))
    pair_rows = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            union = sets[i] | sets[j]
            value = len(sets[i] & sets[j]) / len(union) if union else 0.0
            if value >= threshold:
                uf.union(i, j)
                pair_rows.append({
                    "sample_id_a": data.iloc[i].sample_id,
                    "sample_id_b": data.iloc[j].sample_id,
                    "jaccard": value,
                })
    roots = [uf.find(i) for i in range(len(data))]
    ordered = {
        root: f"similarity_group_{k:04d}"
        for k, root in enumerate(sorted(set(roots)), 1)
    }
    result = data.copy()
    result["similarity_group"] = [ordered[root] for root in roots]
    return result, pd.DataFrame(pair_rows)
