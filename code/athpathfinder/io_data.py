"""Readers, control generation and historical membership reconstruction."""

from __future__ import annotations

import ast
import hashlib
import json
import random
import re
import sqlite3
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd

from . import paths
from .distances import max_jaccard
from .problem import (
    MAX_CONTROL_JACCARD,
    MIN_SET_SIZE,
    SEED,
)

AGI_RE = re.compile(r"^AT[1-5MC]G\d{5}$", re.I)


def stable_rng(key: str) -> random.Random:
    digest = hashlib.sha256(f"{SEED}|{key}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def parse_gene_list(value) -> list[str]:
    if isinstance(value, list):
        raw = value
    else:
        raw = ast.literal_eval(str(value))
    return sorted({str(g).strip().upper() for g in raw if AGI_RE.match(str(g).strip())})


def parse_genes(value) -> list[str]:
    """Accept a list, JSON/Python-literal string, or whitespace/comma-separated AGI ids."""
    if isinstance(value, (list, tuple, set)):
        return parse_gene_list(list(value))
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") or text.startswith("("):
        return parse_gene_list(text)
    parts = re.split(r"[\s,;]+", text)
    return sorted({p.strip().upper() for p in parts if AGI_RE.match(p.strip())})


def read_current_go() -> dict[str, set[str]]:
    gene_to_go: dict[str, set[str]] = defaultdict(set)
    with (paths.DATA / "ATH_GO_GOSLIM.txt").open(errors="replace") as handle:
        for line in handle:
            if not line.strip() or line.startswith("!"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) >= 6 and AGI_RE.match(cols[0].strip()) and re.match(r"^GO:\d{7}$", cols[5].strip()):
                gene_to_go[cols[0].strip().upper()].add(cols[5].strip())
    return gene_to_go


def historical_sqlite_path() -> Path:
    candidates = [
        paths.DATA / "historical/AraCyc_2011_TAIR_GO_2012/org.At.tair.sqlite",
        Path("/private/tmp/org_at_271/org.At.tair.db/inst/extdata/org.At.tair.sqlite"),
        paths.OUT / "cache/org.At.tair.sqlite",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    cache = paths.OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "org.At.tair.db_2.7.1.tar.gz"
    urllib.request.urlretrieve(paths.BIOC_URL, archive)
    member_name = "org.At.tair.db/inst/extdata/org.At.tair.sqlite"
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.getmember(member_name)
        with tar.extractfile(member) as source, (cache / "org.At.tair.sqlite").open("wb") as target:
            target.write(source.read())
    return cache / "org.At.tair.sqlite"


def read_historical_sqlite():
    sqlite_path = historical_sqlite_path()
    con = sqlite3.connect(sqlite_path)
    genes = pd.read_sql_query("SELECT _id, gene_id AS agi FROM genes", con)
    entrez = pd.read_sql_query("SELECT _id, gene_id AS entrez FROM entrez_genes", con)
    aracyc = pd.read_sql_query(
        "SELECT genes.gene_id AS agi, aracyc.pathway_name FROM aracyc JOIN genes USING (_id)",
        con,
    )
    go_frames = []
    for table in ("go_bp", "go_cc", "go_mf"):
        go_frames.append(pd.read_sql_query(
            f"SELECT genes.gene_id AS agi, {table}.go_id FROM {table} JOIN genes USING (_id)",
            con,
        ))
    con.close()
    gene_to_go: dict[str, set[str]] = defaultdict(set)
    for agi, go_id in pd.concat(go_frames).drop_duplicates().itertuples(index=False):
        if AGI_RE.match(str(agi)):
            gene_to_go[str(agi).upper()].add(str(go_id))
    entrez_to_agi = {
        str(entrez_id): str(agi).upper()
        for _, agi, entrez_id in genes.merge(entrez, on="_id").itertuples(index=False)
        if pd.notna(entrez_id) and AGI_RE.match(str(agi))
    }
    return gene_to_go, aracyc, entrez_to_agi


def load_current_primary() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(paths.CURATED)
    df["genes_list"] = df["genes_list"].map(parse_gene_list)
    positives = df[df.label == 1][["sample_id", "label", "source", "family", "genes_list"]].copy()
    controls = df[(df.label == 0) & (df.strategy == "positive")][
        ["sample_id", "label", "source", "family", "genes_list"]
    ].copy()
    controls["strategy"] = "curated_nonmetabolic_control"
    positives["strategy"] = "curated_metabolic_pathway"
    for frame in (positives, controls):
        frame["genes"] = frame.genes_list.map(json.dumps)
    return positives.reset_index(drop=True), controls.reset_index(drop=True)


def load_canonical() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
    data = pd.read_csv(paths.PROCESSED / "current_canonical_pathways_and_controls.csv")
    data["genes_list"] = data.genes_list.map(lambda x: list(ast.literal_eval(str(x))))
    positives = data[data.label.eq(1)].copy().reset_index(drop=True)
    controls = data[data.label.eq(0)].copy().reset_index(drop=True)
    return positives, controls, read_current_go()


def generate_constructed_controls(
    positives: pd.DataFrame,
    gene_to_go: dict[str, set[str]],
    n_total: int,
    prefix: str,
) -> pd.DataFrame:
    """Generate six documented challenge-control types using actual GO mappings."""
    strategies = [
        "random_gene_set", "shuffled_pathway", "partial_pathway",
        "cross_pathway_mixture", "go_coherent_control", "annotation_sparse_control",
    ]
    allocation = {s: n_total // len(strategies) for s in strategies}
    for s in strategies[: n_total % len(strategies)]:
        allocation[s] += 1
    pos_sets = [set(x) for x in positives.genes_list]
    pos_union = set().union(*pos_sets)
    background = sorted(gene_to_go)
    nonpath_background = sorted(set(background) - pos_union)
    sparse_background = sorted(nonpath_background, key=lambda g: len(gene_to_go[g]))
    sparse_background = sparse_background[: max(500, len(sparse_background) // 3)]
    go_to_genes: dict[str, set[str]] = defaultdict(set)
    for gene in nonpath_background:
        for go_id in gene_to_go[gene]:
            go_to_genes[go_id].add(gene)
    valid_go = sorted(go_id for go_id, genes in go_to_genes.items() if len(genes) >= 30)
    sizes = [len(x) for x in positives.genes_list]
    rows = []

    def acceptable(genes: set[str]) -> bool:
        return len(genes) >= MIN_SET_SIZE and max_jaccard(genes, pos_sets) <= MAX_CONTROL_JACCARD

    for strategy in strategies:
        for i in range(allocation[strategy]):
            rng = stable_rng(f"{prefix}|{strategy}|{i}")
            target = max(5, min(60, rng.choice(sizes)))
            genes: set[str] | None = None
            for _ in range(1000):
                if strategy == "random_gene_set":
                    genes = set(rng.sample(nonpath_background, min(target, len(nonpath_background))))
                elif strategy == "shuffled_pathway":
                    genes = set(rng.sample(sorted(pos_union), min(target, len(pos_union))))
                elif strategy == "partial_pathway":
                    source = rng.choice(pos_sets)
                    keep = max(5, int(round(len(source) * rng.uniform(0.25, 0.55))))
                    genes = set(rng.sample(sorted(source), min(keep, len(source))))
                elif strategy == "cross_pathway_mixture":
                    a, b = rng.sample(pos_sets, 2)
                    ka, kb = max(2, target // 2), max(3, target - target // 2)
                    genes = set(rng.sample(sorted(a), min(ka, len(a)))) | set(
                        rng.sample(sorted(b), min(kb, len(b)))
                    )
                elif strategy == "go_coherent_control":
                    go_id = rng.choice(valid_go)
                    pool = sorted(go_to_genes[go_id])
                    genes = set(rng.sample(pool, min(target, len(pool))))
                else:
                    genes = set(rng.sample(sparse_background, min(target, len(sparse_background))))
                if acceptable(genes):
                    break
            if genes is None or not acceptable(genes):
                raise RuntimeError(f"Could not generate {strategy} control {i}")
            rows.append({
                "sample_id": f"{prefix}::{strategy}::{i + 1:04d}",
                "label": 0,
                "source": "constructed",
                "family": "Control",
                "strategy": strategy,
                "genes_list": sorted(genes),
                "genes": json.dumps(sorted(genes)),
            })
    return pd.DataFrame(rows)


def generate_size_matched_random_controls(
    reference: pd.DataFrame,
    gene_to_go: dict[str, set[str]],
    historical_positives: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    historical_sets = [set(x) for x in historical_positives.genes_list]
    historical_union = set().union(*historical_sets)
    background = sorted(set(gene_to_go) - historical_union)
    rows = []
    for i, ref in enumerate(reference.itertuples()):
        rng = stable_rng(f"{prefix}|{ref.sample_id}")
        target = len(ref.genes_list)
        genes = None
        for _ in range(1000):
            candidate = set(rng.sample(background, min(target, len(background))))
            if max_jaccard(candidate, historical_sets) <= MAX_CONTROL_JACCARD:
                genes = sorted(candidate)
                break
        if genes is None:
            raise RuntimeError(f"Could not create size-matched control for {ref.sample_id}")
        rows.append({
            "sample_id": f"{prefix}::{i + 1:04d}", "label": 0, "source": "constructed",
            "family": "Size-matched random control", "strategy": "size_matched_random_control",
            "genes_list": genes, "genes": json.dumps(genes), "matched_to": ref.sample_id,
        })
    return pd.DataFrame(rows)


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def build_historical_sets(current_positives):
    hist_go, old_aracyc, entrez_to_agi = read_historical_sqlite()
    old_kegg = pd.read_csv(paths.HIST / "04_2011_ath_pathway_gene_pairs_NCBI.csv")
    old_kegg["agi"] = old_kegg.gene_ncbi.astype(str).map(entrez_to_agi)
    old_kegg = old_kegg.dropna(subset=["agi"])
    current_aracyc = pd.read_csv(paths.DATA / "aracyc_pathways.20251021.txt", sep="\t")
    current_aracyc["Pathway-id"] = current_aracyc["Pathway-id"].astype(str)
    current_kegg_ids = {
        x.split("::", 1)[1].replace("path:", "")
        for x in current_positives.loc[current_positives.source == "KEGG", "sample_id"]
    }
    historical_rows = []
    for pathway_id, grp in old_kegg.groupby("pathway_id"):
        if pathway_id in current_kegg_ids:
            genes = sorted(set(grp.agi) & set(hist_go))
            if len(genes) >= MIN_SET_SIZE:
                historical_rows.append({
                    "sample_id": f"KEGG2011::{pathway_id}", "label": 1, "source": "KEGG2011",
                    "family": "Historical metabolic", "strategy": "historical_pathway",
                    "genes_list": genes, "genes": json.dumps(genes),
                })
    old_aracyc["normalized_name"] = old_aracyc.pathway_name.map(normalize_name)
    for name, grp in old_aracyc.groupby("normalized_name"):
        genes = sorted(set(grp.agi.astype(str).str.upper()) & set(hist_go))
        if len(genes) >= MIN_SET_SIZE:
            historical_rows.append({
                "sample_id": f"AraCyc2011::{name}", "label": 1, "source": "AraCyc2011",
                "family": "Historical metabolic", "strategy": "historical_pathway",
                "genes_list": genes, "genes": json.dumps(genes),
            })
    historical = pd.DataFrame(historical_rows)
    old_kegg_ids = set(old_kegg.pathway_id)
    later_rows = []
    for r in current_positives.loc[current_positives.source == "KEGG"].itertuples():
        pathway_id = r.sample_id.split("::", 1)[1].replace("path:", "")
        absent = pathway_id not in old_kegg_ids
        criterion = "KEGG identifier absent from 2011 snapshot"
        genes = sorted(set(r.genes_list) & set(hist_go))
        if absent and len(genes) >= MIN_SET_SIZE:
            later_rows.append({
                "sample_id": r.sample_id, "label": 1, "source": r.source, "family": r.family,
                "strategy": "later_catalogue_pathway", "eligibility_criterion": criterion,
                "genes_list": genes, "genes": json.dumps(genes),
                "historical_go_coverage": len(genes) / len(r.genes_list),
            })
    later = pd.DataFrame(later_rows)
    return hist_go, historical, later


def historical_family_lookup(current_positives: pd.DataFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for row in current_positives[current_positives.source.eq("KEGG")].itertuples():
        pathway_id = row.sample_id.split("::", 1)[1].replace("path:", "")
        lookup[f"KEGG2011::{pathway_id}"] = row.family
    current_aracyc = pd.read_csv(paths.DATA / "aracyc_pathways.20251021.txt", sep="\t", dtype=str)
    id_to_name = current_aracyc.drop_duplicates("Pathway-id").set_index("Pathway-id")["Pathway-name"]
    for row in current_positives[current_positives.source.eq("AraCyc")].itertuples():
        pathway_id = row.sample_id.split("::", 1)[1]
        if pathway_id in id_to_name.index:
            lookup[f"AraCyc2011::{normalize_name(id_to_name[pathway_id])}"] = row.family
    return lookup
