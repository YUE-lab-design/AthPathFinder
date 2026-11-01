#!/usr/bin/env python3
"""Rice Algorithm 1 analog after pre-specified GO coverage ≥80%.

Does not change Arabidopsis Algorithm 1. Does not tune on any rice test fold.
KEGG dumps stay in code/.cache/external_rice and are not for git.

The confirmatory analog is pre-specified as all five StratifiedGroupKFold
outer folds (seed 42). Fold 5 is one of those folds; it is not re-selected
after seeing AUROC. Rebuilds the GO map from checksummed local caches or
from the frozen union TSV. No live mygene/KEGG/UniProt call.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from athpathfinder import paths
from athpathfinder.algorithms.family import JaccardFamily
from athpathfinder.algorithms.hypothesis_classes import deployed_lightgbm
from athpathfinder.algorithms.pathway_like import JaccardPathwayLike
from athpathfinder.algorithms.two_stage import (
    compose_end_to_end,
    threshold_by_balanced_accuracy,
)
from athpathfinder.embedding import select_and_feature
from athpathfinder.grouping import collapse_exact
from athpathfinder.problem import (
    FAMILIES,
    FAMILY_MIN,
    MIN_SET_SIZE,
    N_CONSTRUCTED_PRIMARY,
    NON_METABOLIC,
    SEED,
    XENOBIOTICS,
)

CACHE = paths.DATA_ROOT / "rice_sources"
OVERVIEWS = {"01100", "01110"}
COVERAGE_GATE = 0.80
GO_CACHE_FILES = (
    "ensembl_biomart_osativa.tsv",
    "kegg_conv_uniprot.txt",
    "uniprot_osa_all_go.tsv.gz",
    "mygene_osa_go.json",
    "link_pathway.txt",
    "list_pathway.txt",
)
FROZEN_MAP_NAME = "rice_union_gene_to_go.tsv.gz"
FROZEN_META_NAME = "rice_go_freeze.json"


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _require_cache() -> None:
    missing = [name for name in GO_CACHE_FILES if not (CACHE / name).exists()]
    if missing:
        raise SystemExit(
            "Rice analog rebuilds from checksummed local caches only. "
            f"Missing {missing} under {CACHE}. No live mygene/KEGG/UniProt call."
        )


def write_gene_to_go(gene_to_go: dict[str, set[str]], path: Path) -> None:
    lines = []
    for gene in sorted(gene_to_go):
        terms = ";".join(sorted(gene_to_go[gene]))
        if terms:
            lines.append(f"{gene}\t{terms}\n")
    with gzip.open(path, "wt") as handle:
        handle.write("gene\tgo_terms\n")
        handle.writelines(lines)


def load_gene_to_go(path: Path) -> dict[str, set[str]]:
    gene_to_go: dict[str, set[str]] = {}
    with gzip.open(path, "rt") as handle:
        next(handle)
        for line in handle:
            gene, terms = line.rstrip("\n").split("\t", 1)
            gene_to_go[gene] = {t for t in terms.split(";") if t.startswith("GO:")}
    return gene_to_go


def kegg_genes_and_pathways() -> tuple[set[str], dict[str, set[str]], dict[str, str]]:
    genes: set[str] = set()
    pathway_genes: dict[str, set[str]] = defaultdict(set)
    names: dict[str, str] = {}
    for line in (CACHE / "link_pathway.txt").read_text().splitlines():
        gene, pathway = line.split("\t")
        kid = gene.split(":", 1)[1]
        pid = pathway.split(":", 1)[1].replace("osa", "")
        genes.add(kid)
        pathway_genes[pid].add(kid)
    for line in (CACHE / "list_pathway.txt").read_text().splitlines():
        pid, name = line.split("\t", 1)
        names[pid.replace("osa", "")] = name
    return genes, pathway_genes, names


def build_gene_to_go_from_cache(kegg_genes: set[str]) -> tuple[dict[str, set[str]], dict]:
    gene_to_go: dict[str, set[str]] = defaultdict(set)
    sources = {
        "ensembl_biomart_entrez": set(),
        "uniprot_kegg_conv": set(),
        "mygene": set(),
    }

    with (CACHE / "ensembl_biomart_osativa.tsv").open() as handle:
        next(handle)
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            entrez, go = parts[1].strip(), parts[2]
            if entrez.isdigit() and go.startswith("GO:"):
                gene_to_go[entrez].add(go)
                if entrez in kegg_genes:
                    sources["ensembl_biomart_entrez"].add(entrez)

    kegg_to_up: dict[str, set[str]] = defaultdict(set)
    for line in (CACHE / "kegg_conv_uniprot.txt").read_text().splitlines():
        a, b = line.split("\t")
        kegg_to_up[a.split(":", 1)[1]].add(b.split(":", 1)[1])
    with gzip.open(CACHE / "uniprot_osa_all_go.tsv.gz", "rt") as handle:
        next(handle)
        acc_go: dict[str, set[str]] = defaultdict(set)
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            terms = {
                token.strip()
                for token in (parts[1] if len(parts) > 1 else "").replace(";", " ").split()
                if token.startswith("GO:")
            }
            if terms:
                acc_go[parts[0]] |= terms
    for gene, accessions in kegg_to_up.items():
        if gene not in kegg_genes:
            continue
        terms: set[str] = set()
        for acc in accessions:
            terms |= acc_go.get(acc, set())
        if terms:
            gene_to_go[gene] |= terms
            sources["uniprot_kegg_conv"].add(gene)

    mygene = json.loads((CACHE / "mygene_osa_go.json").read_text())
    for rec in mygene["genes"]:
        gene = str(rec["gene"])
        if gene in kegg_genes and rec.get("go"):
            gene_to_go[gene] |= set(rec["go"])
            sources["mygene"].add(gene)

    covered = set(gene_to_go) & kegg_genes
    stats = coverage_stats(kegg_genes, gene_to_go, sources)
    return {g: terms for g, terms in gene_to_go.items()}, stats


def coverage_stats(
    kegg_genes: set[str],
    gene_to_go: dict[str, set[str]],
    sources: dict[str, set[str]] | None = None,
) -> dict:
    covered = set(gene_to_go) & kegg_genes
    stats = {
        "kegg_pathway_genes": len(kegg_genes),
        "go_universe_n": len(gene_to_go),
        "covered_union": len(covered),
        "fraction_union": len(covered) / len(kegg_genes),
        "coverage_gate": COVERAGE_GATE,
        "passed_coverage_gate": len(covered) / len(kegg_genes) >= COVERAGE_GATE,
        "pre_registration": (
            "Union GO map (Ensembl Plants BioMart osativa_eg_gene entrez+GO, "
            "UniProt organism_id:39947 via KEGG conv/uniprot/osa, mygene.info v3 "
            "fields=go) locked before any rice AUROC. Pathway members without GO "
            "are dropped; sets with |G|<5 after that filter are excluded. "
            "Families come from Arabidopsis reviewed labels on the same KEGG map "
            "number. FAMILY_MIN=8. Gate and GO-term selection use rice development "
            "only. The confirmatory analog is all five grouped outer folds "
            "(StratifiedGroupKFold, seed 42), not a single fold chosen after AUROC."
        ),
    }
    if sources is not None:
        stats["per_source"] = {
            name: {"n": len(ids), "fraction": len(ids) / len(kegg_genes)}
            for name, ids in sources.items()
        }
    return stats


def freeze_or_load_gene_to_go(kegg_genes: set[str]) -> tuple[dict[str, set[str]], dict, dict]:
    out = paths.GROUPED_OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    map_path = out / FROZEN_MAP_NAME
    meta_path = out / FROZEN_META_NAME
    if map_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        digest = _sha256(map_path)
        if digest != meta.get("map_sha256"):
            raise SystemExit(
                f"Frozen rice GO map checksum mismatch: {digest} != {meta.get('map_sha256')}"
            )
        gene_to_go = load_gene_to_go(map_path)
        stats = coverage_stats(kegg_genes, gene_to_go)
        stats["per_source"] = meta.get("per_source", {})
        stats["loaded_from"] = str(map_path)
        return gene_to_go, stats, meta

    _require_cache()
    gene_to_go, stats = build_gene_to_go_from_cache(kegg_genes)
    write_gene_to_go(gene_to_go, map_path)
    meta = {
        "map_file": FROZEN_MAP_NAME,
        "map_sha256": _sha256(map_path),
        "n_genes": len(gene_to_go),
        "n_edges": int(sum(len(v) for v in gene_to_go.values())),
        "covered_union": stats["covered_union"],
        "fraction_union": stats["fraction_union"],
        "per_source": stats.get("per_source", {}),
        "license": (
            "GO term assignments are from Ensembl Plants BioMart, UniProt and "
            "mygene.info. KEGG pathway dumps are not redistributed; they remain "
            "in a local cache (https://www.kegg.jp/kegg/legal.html). The frozen "
            "map stores gene identifiers and GO terms only."
        ),
        "rebuild": (
            "From checksummed files in code/.cache/external_rice: "
            "python evaluate_rice_catalog.py. No live API."
        ),
        "input_sha256": {name: _sha256(CACHE / name) for name in GO_CACHE_FILES},
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    stats["loaded_from"] = "rebuilt_from_checksummed_cache"
    return gene_to_go, stats, meta


def arabidopsis_map_families() -> tuple[dict[str, str], set[str]]:
    labels = pd.read_csv(paths.PROCESSED / "reviewed_pathway_labels.csv")
    family_of: dict[str, str] = {}
    nonmet: set[str] = set()
    for rec in labels.itertuples():
        sid = str(rec.sample_id)
        if not sid.startswith("KEGG::path:ath"):
            continue
        pid = sid.rsplit("ath", 1)[-1]
        if rec.role == "curated_nonmetabolic":
            nonmet.add(pid)
        elif rec.role == "metabolic_pathway" and rec.family in FAMILIES:
            family_of[pid] = rec.family
    return family_of, nonmet


def build_rice_rows(
    pathway_genes: dict[str, set[str]],
    names: dict[str, str],
    gene_to_go: dict[str, set[str]],
    family_of: dict[str, str],
    nonmet: set[str],
) -> tuple[list[dict], list[dict], dict]:
    positives, curated = [], []
    dropped = {"overview": 0, "no_family": 0, "too_small": 0, "xenobiotics": 0}
    for pid, members in sorted(pathway_genes.items()):
        genes = sorted(g for g in members if g in gene_to_go)
        if pid in OVERVIEWS:
            dropped["overview"] += 1
            continue
        if len(genes) < MIN_SET_SIZE:
            dropped["too_small"] += 1
            continue
        if pid in nonmet:
            curated.append({
                "sample_id": f"KEGG::path:osa{pid}",
                "label": 0,
                "source": "KEGG",
                "family": "Rice curated non-metabolic",
                "strategy": "curated_nonmetabolic",
                "genes_list": genes,
                "genes": json.dumps(genes),
                "map_id": pid,
                "name": names.get(pid, ""),
            })
            continue
        family = family_of.get(pid)
        if family is None:
            dropped["no_family"] += 1
            continue
        if family == XENOBIOTICS or family not in FAMILIES:
            dropped["xenobiotics"] += 1
            continue
        positives.append({
            "sample_id": f"KEGG::path:osa{pid}",
            "label": 1,
            "source": "KEGG",
            "family": family,
            "strategy": "metabolic_pathway",
            "genes_list": genes,
            "genes": json.dumps(genes),
            "map_id": pid,
            "name": names.get(pid, ""),
            "family_eligible": True,
        })
    counts = pd.Series([r["family"] for r in positives]).value_counts()
    kept_families = set(counts[counts.ge(FAMILY_MIN)].index)
    # Rice deliberately retains all ten mapped families, unlike the current
    # Arabidopsis minimum-support cohort. The raw support counts are descriptive.
    for row in positives:
        row["family_eligible"] = row["family"] in FAMILIES
    audit = {
        "dropped": dropped,
        "n_positives_before_collapse": len(positives),
        "n_curated_before_collapse": len(curated),
        "family_counts_before_collapse": counts.to_dict(),
        "families_passing_FAMILY_MIN": sorted(kept_families),
        "family_support_policy": "all ten mapped families retained; FAMILY_MIN comparison is descriptive only",
    }
    return positives, curated, audit


