"""Problem objects for hierarchical gene-set annotation (F1/F2).

Shared objects: gene universe V, GO map α, catalog {(P_i, f_i)} with
f_i in F (|F|=10), query G ⊆ V. Output alphabet Y = {⊥} ∪ F, where ⊥ is
the non-metabolic label. Jaccard ≥ SIMILARITY_THRESHOLD groups are
indivisible evaluation atoms.
"""

from __future__ import annotations

NON_METABOLIC = "Non-metabolic"
"""⊥: the non-pathway-like / non-metabolic output of ŷ."""

FAMILIES: tuple[str, ...] = (
    "Amino acid metabolism",
    "Biosynthesis of other secondary metabolites",
    "Carbohydrate metabolism",
    "Energy metabolism",
    "Glycan biosynthesis and metabolism",
    "Lipid metabolism",
    "Metabolism of cofactors and vitamins",
    "Metabolism of other amino acids",
    "Metabolism of terpenoids and polyketides",
    "Nucleotide metabolism",
)

XENOBIOTICS = "Xenobiotics biodegradation and metabolism"

Y: tuple[str, ...] = (NON_METABOLIC,) + FAMILIES
"""Eleven-label end-to-end alphabet after gating."""

SEED = 42
FAMILY_MIN = 8
MIN_SET_SIZE = 5
SIMILARITY_THRESHOLD = 0.80
MAX_CONTROL_JACCARD = 0.50

MIN_GO_GENES = 20
MAX_GO_PERCENT = 0.30
VARIANCE_QUANTILE = 0.15
K_GO = 60
PAIRWISE_CAP = 15
OT_N_TERMS = 48
OT_SINKHORN_EPS = 0.08
OT_SINKHORN_ITERS = 20
STAGE1_CLEAR_DELTA = 0.01

N_CONSTRUCTED_PRIMARY = 360
STAGE1_HYPOTHESIS = "LightGBM"
STAGE2_HYPOTHESIS = "Extra Trees"
DEPLOYED_FAMILY_HEAD = "Jaccard 1-NN"
GATE_SELECTION = "maximum balanced accuracy on development OOF predictions"

SHORT_FAMILY = {
    "Amino acid metabolism": "Amino acid",
    "Biosynthesis of other secondary metabolites": "Secondary met.",
    "Carbohydrate metabolism": "Carbohydrate",
    "Energy metabolism": "Energy",
    "Glycan biosynthesis and metabolism": "Glycan",
    "Lipid metabolism": "Lipid",
    "Metabolism of cofactors and vitamins": "Cofactors",
    "Metabolism of other amino acids": "Other amino acids",
    "Metabolism of terpenoids and polyketides": "Terpenoids",
    "Nucleotide metabolism": "Nucleotide",
}


def short_family(value: str) -> str:
    return SHORT_FAMILY[value]


def hierarchical_label(is_control: bool, family: str) -> str:
    return NON_METABOLIC if is_control else family
