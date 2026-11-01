# AthPathFinder

AthPathFinder annotates gene sets using LightGBM pathway-like screening followed
by Jaccard 1-NN assignment to one of ten reference-catalog metabolic families.
The primary implementation uses Arabidopsis AGI identifiers. Rice experiments
retrain within rice; they are not direct transfer of the Arabidopsis model.

## Contents

- `code/athpathfinder/`: feature construction, inference, training, grouping and scoring.
- `code/fits/`: frozen deployed model and specification.
- `code/tests/`: software tests.
- `code/requirements.txt`: recorded runtime dependencies (Python 3.11).
- `code/evaluate_single_stage_11.py`: matched single-stage comparison driver.
- `data/processed_data/`: cohort and label/grouping audits.
- `data/grouped_analysis_results/`: archived fits, splits, predictions and analyses.
- `manuscript/lncs/case_study/`: descriptive expression-derived queries and outputs.
- `manuscript/lncs/figures/source_data/`: figure source data, without manuscript files.
- `documentation/verification_logs/`: archived execution evidence, not new test runs.

## Important data prerequisite

**This upload folder is not a self-contained reproducibility distribution.**
Third-party original snapshots are omitted pending redistribution confirmation.
Frozen inference also needs the exact GO annotation snapshot; models do not
embed the full GO lookup. Read `data/README.md` and `data/REQUIRED_INPUTS.json`.
Restore the required inputs at their listed relative paths before running the
inference or full reproduction commands. Current public downloads must not be
silently substituted for the recorded snapshots. Derived memberships and models
also retain applicable source terms; review those before public release.

## Install and run

From this project's root, using Python 3.11:

```sh
python -m pip install -r code/requirements.txt
PYTHONPATH=code python -m unittest discover -s code/tests -v
```

After restoring the required source data:

```sh
PYTHONPATH=code python -m athpathfinder --json AT1G17290 AT1G17300 AT1G23310 AT1G70560 AT1G70580
PYTHONPATH=code python code/verify_current_project.py
PYTHONPATH=code python code/verify_single_stage_11.py
```

At least five distinct known GO-universe genes must remain after normalization
and duplicate removal. Unknown identifiers are discarded and reported. Features
are invariant to query names, input order and duplicated genes. Stage 1 accepts
score >= the saved threshold. Jaccard ties use the first maximum in stored
reference order; zero overlap is flagged without changing family assignment.
A screening rejection does not establish absence of metabolic function.

## Reproduce the matched eleven-class comparator

After restoring the inputs, choose a new output directory:

```sh
PYTHONPATH=code python code/evaluate_single_stage_11.py --output /tmp/athpathfinder_single_stage_reproduction
```

This performs 12 fits. Both matched methods use the same 695 eligible development
sets, archived development folds and 169-set closed-catalog held-out endpoint.
The matched binary threshold is selected using development OOF scores. The
original deployed model and its threshold remain unchanged. The driver refuses
to overwrite an existing experiment directory.

`retrain_geneset_seed_v2.py` reconstructs the current, historical and rice fits;
`restore_diagnostics_geneset_v2.py` reconstructs development LOFO diagnostics.
These scripts write their named outputs: preserve archived results before reuse.
`plot_figures.py` writes figures under `manuscript/lncs/figures/` from saved results.

## Evaluation boundary

The earlier family-head selection consulted held-out results, so the reported
held-out comparisons remain exploratory. The single-stage comparison, refitting,
bootstrap and historical/rice analyses do not restore independent confirmation.
Stage 1 uses 170 sets / 167 groups; Stage 2 uses 84 eligible pathways; closed-catalog
end-to-end scoring uses 169 sets. Xenobiotics remains positive at recognition
and is excluded only from the ten-family and closed-catalog endpoints.
No unseen-family prediction or discovery of new biochemical pathways is claimed.

## Licence and publication information

The project software licence and author/copyright information have not yet been
confirmed. No licence grant or repository/DOI address is invented in this folder.
Source data remain subject to their original terms. Resolve these items before
public release. Original execution paths in provenance files are historical.

`UPLOAD_CHECKLIST.txt` lists the remaining release checks. `SHA256SUMS.txt` covers
this prepared folder. No experiment was rerun while preparing this upload copy.
