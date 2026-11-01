"""Historical membership helpers for the current fixed pipeline."""
import ast,json
from collections import defaultdict
import pandas as pd

def _parse_genes(value) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(x) for x in ast.literal_eval(str(value))]


def collapse_historical(historical: pd.DataFrame, family_lookup: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    historical = historical.copy()
    historical["family"] = historical.sample_id.map(family_lookup)
    buckets: dict[tuple, list] = defaultdict(list)
    for row in historical.itertuples():
        buckets[tuple(row.genes_list)].append(row)
    stage1, stage2 = [], []
    for i, (genes, members) in enumerate(buckets.items(), 1):
        record = {
            "sample_id": f"historical_canonical::{i:04d}",
            "original_ids": "|".join(sorted(x.sample_id for x in members)),
            "label": 1,
            "source": "historical",
            "family": "Historical metabolic",
            "strategy": "historical_pathway",
            "genes_list": list(genes),
            "genes": json.dumps(list(genes)),
        }
        stage1.append(record)
        fam = sorted({str(x.family) for x in members if pd.notna(x.family)})
        if len(fam) == 1:
            stage2.append({**record, "family": fam[0]})
    return pd.DataFrame(stage1), pd.DataFrame(stage2)


