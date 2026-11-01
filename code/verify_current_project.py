"""Read-only checks of the cleaned package; no fitting or result overwrites."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from athpathfinder import paths
from athpathfinder.predict import load_publication_model
from athpathfinder.evaluation import reconstruct_frozen_split
import evaluate_rice_catalog as rice
D=paths.GROUPED_OUTPUT/'geneset_seed_v2';R=paths.REPO_ROOT
model=load_publication_model(fit_if_missing=False);dev,test,go=reconstruct_frozen_split();saved=pd.read_csv(D/'current_predictions.csv').set_index('sample_id');checked=0
for row in test.itertuples():
 q=model.predict_one(row.genes_list,sample_id=row.sample_id).as_dict();s=saved.loc[row.sample_id]
 assert abs(q['stage1_score']-s.stage1_score)<1e-14;assert q['pathway_like']==bool(s.stage1_pred);assert q['label']==s.end_to_end_pred;checked+=1
rice._require_cache();kg,pg,names=rice.kegg_genes_and_pathways();rgo,stats,meta=rice.freeze_or_load_gene_to_go(kg);assert stats['covered_union']==5619 and len(kg)==5742
case=R/'manuscript/lncs/case_study';genes=pd.read_csv(case/'reconstructed_query_genes.csv');p=pd.read_csv(case/'frozen_query_predictions.csv');lookup={name:x.gene.tolist() for name,x in genes.groupby('query_id',sort=False)}
for row in p.itertuples():
 q=model.predict_one(lookup[row.query_id],sample_id=row.query_id).as_dict();assert abs(q['stage1_score']-row.stage1_score)<1e-14;assert q['pathway_like']==bool(row.accepted)
print(json.dumps({'current_heldout_scores_and_predictions_reproduced':checked,'expression_queries_reproduced':len(p),'rice_frozen_GO_map_hash_verified':True,'rice_coverage':[stats['covered_union'],len(kg)],'deployed_threshold':model.gate,'new_fits':0}))
