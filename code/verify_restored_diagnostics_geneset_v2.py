"""Check saved diagnostic outputs against row scores, IDs and exported fits."""
from pathlib import Path
import sys,json,pickle
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from athpathfinder import paths
from athpathfinder.evaluation import reconstruct_frozen_split
from athpathfinder.embedding import build_features
from athpathfinder.problem import FAMILIES
D=paths.GROUPED_OUTPUT/'geneset_seed_v2';O=D/'restored_diagnostics'
s=pd.read_csv(O/'stage1_lofo.csv');p=pd.read_csv(O/'stage1_lofo_predictions.csv');audits=json.loads((O/'stage1_lofo_audits.json').read_text());dev,outer,go=reconstruct_frozen_split();lookup=dev.set_index('sample_id',drop=False)
assert len(s)==10 and set(s.held_family)==set(FAMILIES) and len(audits)==10
for i,a in enumerate(audits,1):
 q=p[p.held_family.eq(a['held_family'])];row=s.set_index('held_family').loc[a['held_family']]
 assert q.sample_id.is_unique and set(q.sample_id)==set(a['test_ids'])
 assert not set(a['training_ids'])&set(a['test_ids'])
 assert not set(a['training_groups'])&set(a['test_groups'])
 assert not (set(a['training_ids'])|set(a['test_ids']))&set(outer.sample_id)
 assert set(lookup.loc[a['training_ids'],'family']) .isdisjoint([a['held_family']])
 assert q.loc[q.label.eq(1),'true_family'].eq(a['held_family']).all()
 assert int(q.label.sum())==row.n_pathways and int(q.label.eq(0).sum())==row.n_controls
 assert abs(roc_auc_score(q.label,q.score)-row.auroc)<1e-12
 assert abs(average_precision_score(q.label,q.score)-row.auprc)<1e-12
 # Rebuild only query features and verify exact exported-fit predictions.
 fit=pickle.loads((O/f'lofo_fold{i:02}_fit.pkl').read_bytes());frame=lookup.loc[q.sample_id].reset_index(drop=True)
 x,_=build_features(frame,fit['selected_go'],go)
 np.testing.assert_allclose(fit['stage1'].predict_proba(x)[:,1],q.score,rtol=0,atol=1e-14)
q=pd.read_csv(D/'current_predictions.csv');q=q[q.label.eq(1)&q.end_to_end_eligible];f=pd.read_csv(O/'current_family_f1.csv')
assert abs(f.f1.mean()-f1_score(q.true_family,q.stage2_pred,labels=list(FAMILIES),average='macro'))<1e-12
result={'all_passed':True,'lofo_fits_reloaded':10,'lofo_group_and_id_separation_verified':True,'outer_holdout_not_used_in_lofo':True,'all_lofo_metrics_recomputed_from_saved_scores':True,'all_exported_lofo_model_scores_reproduced':True,'current_family_f1_matches_current_macro_f1':True,'new_model_classes':0,'primary_fit_or_gate_changed':False}
(O/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
