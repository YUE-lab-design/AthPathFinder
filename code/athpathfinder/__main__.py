"""CLI: python -m athpathfinder AT1G01040 AT1G01050 ..."""

from __future__ import annotations

import argparse
import json
import sys

from .predict import fit_publication_model, load_publication_model, predict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="AthPathFinder: hierarchical gene-set annotation (Algorithm 1).",
    )
    parser.add_argument("genes", nargs="*", help="AGI loci; at least 5 distinct GO-universe genes must remain after unknown/invalid IDs are discarded.")
    parser.add_argument("--fit", action="store_true", help="Fit and save a frozen development model.")
    parser.add_argument("--json", action="store_true", help="Print the prediction as JSON.")
    args = parser.parse_args(argv)
    if args.fit:
        model = fit_publication_model(save=True)
        print(f"saved fit with {len(model.selected_go)} GO terms, gate={model.gate:.6f}")
        if not args.genes:
            return 0
    if not args.genes:
        parser.error("provide AGI loci, or --fit")
    try:
        result = predict(args.genes)
    except ValueError as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"label={result['label']}  pathway_like={result['pathway_like']}  "
            f"stage1_score={result['stage1_score']:.4f}  family={result['family']}  "
            f"no_gene_overlap_support={result['no_gene_overlap_support']}  "
            f"retained_genes={len(result['genes'])}  discarded_genes={result['discarded_genes']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
