# Data and reproducibility boundaries

The source snapshots under `data/input_data/` are intentionally absent from this
GitHub upload folder. Public redistribution permission for KEGG, TAIR, PMN and
other source inputs has not been confirmed. Local possession is not permission
to redistribute. `REQUIRED_INPUTS.json` lists the exact expected paths, sizes and
SHA-256 hashes. Generic current downloads are not equivalent to these snapshots.

The original local complete project at `/Users/yuezhang/Documents/Playground/drzhang`
contains these inputs. For local reproduction, copy its `data/input_data/` directory
into the same relative location in this project. It is ignored by Git. Review
source permissions before distributing any source snapshots. No public input
archive URL or automated download of equivalent snapshots is currently supplied.

`processed_data/` contains the canonical cohort, reviewed family labels and
membership/grouping audits. `grouped_analysis_results/` contains frozen splits,
fitted artifacts, predictions, development OOF records, bootstrap summaries,
error analyses and the original family-head adoption decision. These retained
research outputs include derived source memberships; source terms continue to
apply and their redistribution status should be checked before making the
repository public. No data are anonymized or relicensed by this package.

The existing historical run manifests record the original execution environment,
including original local paths; they are provenance, not current download links.
`manuscript/lncs/case_study/` contains the descriptive expression query inputs and
outputs. `manuscript/lncs/figures/source_data/` supports the retained figure driver.
No manuscript text, PDF, old versions or backup archives are included.
