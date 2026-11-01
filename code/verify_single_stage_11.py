"""Reload final comparator fits; verify predictions, groups, labels and OOF gate."""
from pathlib import Path
import json,pickle
import numpy as np
import pandas as pd
from athpathfinder.evaluation import reconstruct_frozen_split
from athpathfinder.training import archived_current_splits
from athpathfinder.scoring import closed_catalog_mask
from athpathfinder.embedding import build_features
from athpathfinder.algorithms.two_stage import threshold_by_balanced_accuracy
from athpathfinder import paths
D=paths.GROUPED_OUTPUT/'single_stage_11';s=json.loads((D/'summary.json').read_text());p=pd.read_csv(D/'heldout_predictions.csv');dev0,test0,go=reconstruct_frozen_split();dev=dev0[closed_catalog_mask(dev0)].reset_index(drop=True);test=test0[closed_catalog_mask(test0)].reset_index(drop=True)
assert list(test.sample_id)==list(p.sample_id) and len(test)==169 and len(dev)==695
assert not set(dev.similarity_group)&set(test.similarity_group)
foldmap={dev0.iloc[i].sample_id:f for f,(_,va) in enumerate(archived_current_splits(dev0),1) for i in va}
oof=pd.read_csv(D/'development_oof.csv');assert all(foldmap[x.sample_id]==x.fold for x in oof.itertuples())
assert abs(threshold_by_balanced_accuracy(oof.label.to_numpy(),oof.matched_stage1_score.to_numpy())-s['matched_screening_threshold'])<1e-14
for a in s['fold_audits']:
 assert not set(a['training_ids'])&set(a['validation_ids']);assert not set(a['training_groups'])&set(a['validation_groups']);assert not (set(a['training_ids'])|set(a['validation_ids']))&set(test.sample_id)
for name in ['single_stage_11','matched_binary']:
 z=pickle.loads((D/f'{name}_fit.pkl').read_bytes());x,_=build_features(test,z['selected_go'],go);prob=z['model'].predict_proba(x)
 if name=='single_stage_11':
  np.testing.assert_allclose(prob,p[[f'single_stage_probability_{i}' for i in range(11)]],rtol=0,atol=1e-14);pred=np.array(z['alphabet'])[prob.argmax(axis=1)];np.testing.assert_array_equal(pred,p.single_stage_11_pred)
 else:
  np.testing.assert_allclose(prob[:,1],p.matched_stage1_score,rtol=0,atol=1e-14);fam=z['family_head'].predict(test.genes_list.tolist());pred=np.where(prob[:,1]>=z['screening_threshold'],fam,z['alphabet'][0]);np.testing.assert_array_equal(pred,p.matched_two_stage_pred)
result={'all_passed':True,'reloaded_fits':2,'heldout_predictions_reproduced':169,'all_11_probabilities_reproduced':True,'archived_outer_and_development_fold_memberships_verified':True,'training_test_groups_disjoint':True,'development_only_threshold_reproduced':True,'new_model_fits_in_verification':0}
(D/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
