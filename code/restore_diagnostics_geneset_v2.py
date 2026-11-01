"""Current-version diagnostics: 10 fixed LightGBM LOFO fits; no model selection.

Only development rows enter LOFO. Saved holdout/historical predictions supply
other panels. No gate tuning, new family head, or independent cohort is created.
"""
from pathlib import Path
import sys,json,pickle,hashlib
import numpy as np,pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_fscore_support
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from athpathfinder import paths
from athpathfinder.evaluation import reconstruct_frozen_split
from athpathfinder.embedding import select_and_feature,FEATURE_VERSION
from athpathfinder.algorithms.hypothesis_classes import deployed_lightgbm
from athpathfinder.problem import FAMILIES,SEED
D=paths.GROUPED_OUTPUT/'geneset_seed_v2';O=D/'restored_diagnostics';O.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def lofo():
 dev,outer,go=reconstruct_frozen_split();positive=dev[dev.label.eq(1)&~dev.family.str.startswith('Ambiguous')];controls=dev[dev.label.eq(0)]
 summaries=[];predictions=[];audits=[]
 for index,family in enumerate(FAMILIES,1):
  held=positive[positive.family.eq(family)].copy();assert len(held)>=8
  gss=GroupShuffleSplit(n_splits=1,test_size=min(.45,len(held)/len(controls)+.02),random_state=SEED)
  _,idx=next(gss.split(controls,controls.label,controls.similarity_group))
  candidates=controls.iloc[idx];prefix=candidates.iloc[:len(held)]
  # Retain complete selected control groups, even if this slightly exceeds the target count.
  neg=controls[controls.similarity_group.isin(prefix.similarity_group)].copy()
  heldgroups=set(held.similarity_group)|set(neg.similarity_group)
  train=dev[(dev.label.eq(0)|(~dev.family.str.startswith('Ambiguous')))&~dev.similarity_group.isin(heldgroups)&~dev.family.eq(family)].reset_index(drop=True)
  test=pd.concat([held,neg],ignore_index=True)
  assert not set(train.similarity_group)&set(test.similarity_group)
  assert not train.family.eq(family).any()
  assert not set(train.sample_id)&set(outer.sample_id) and not set(test.sample_id)&set(outer.sample_id)
  xa,xb,names,featureaudit=select_and_feature(train,test,go)
  model=deployed_lightgbm().set_params(clf__n_jobs=2).fit(xa,train.label.to_numpy());score=model.predict_proba(xb)[:,1]
  record={'held_family':family,'method':'LightGBM','n_pathways':len(held),'n_controls':len(neg),'training_n':len(train),
   'auroc':float(roc_auc_score(test.label,score)),'auprc':float(average_precision_score(test.label,score))}
  summaries.append(record)
  for row,v in zip(test.itertuples(),score):predictions.append({'held_family':family,'sample_id':row.sample_id,'label':row.label,'true_family':row.family,'similarity_group':row.similarity_group,'score':float(v),'feature_version':FEATURE_VERSION})
  selected=[n.split('::',1)[1] for n in names if n.startswith('gofreq::')]
  assert len(selected)==60
  audits.append({'held_family':family,'training_ids':train.sample_id.tolist(),'test_ids':test.sample_id.tolist(),
    'training_groups':sorted(set(train.similarity_group)),'test_groups':sorted(set(test.similarity_group)),
    'selected_go':selected,'feature_selection':featureaudit,'model_parameters':model.get_params(deep=True)})
  (O/f'lofo_fold{index:02}_fit.pkl').write_bytes(pickle.dumps({'stage1':model,'selected_go':selected,'feature_version':FEATURE_VERSION,'held_family':family}))
  pd.DataFrame(summaries).to_csv(O/'stage1_lofo.csv',index=False)
  pd.DataFrame(predictions).to_csv(O/'stage1_lofo_predictions.csv',index=False)
  (O/'stage1_lofo_audits.json').write_text(json.dumps(audits,indent=2,default=str)+'\n')
  print('LOFO completed',index,record,flush=True)
 x=pd.DataFrame(summaries)
 summary={'feature_version':FEATURE_VERSION,'method':'fixed LightGBM; Stage-1 recognition only','fits_completed':len(x),'model_selection_performed':False,'threshold_selection_performed':False,
  'data_scope':'original Arabidopsis development cohort only; original outer holdout never used',
  'control_selection':'seed-42 GroupShuffleSplit with legacy test-size formula; candidate prefix targets held-family size, then complete control groups are retained',
  'group_policy':'remove entire union of held-family and selected-control similarity groups from training; unambiguous positives only',
  'family_mean_auroc':float(x.auroc.mean()),'family_mean_auprc':float(x.auprc.mean()),
  'lowest_auroc_family':str(x.loc[x.auroc.idxmin(),'held_family']),
  'no_claim':'not a new independent confirmatory cohort, not unseen-family label assignment'}
 (O/'stage1_lofo_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary,flush=True)

def existing_predictions():
 p=pd.read_csv(D/'current_predictions.csv');q=p[p.label.eq(1)&p.end_to_end_eligible];assert len(q)==84
 pr,re,f1,support=precision_recall_fscore_support(q.true_family,q.stage2_pred,labels=list(FAMILIES),zero_division=0)
 pd.DataFrame({'family':FAMILIES,'precision':pr,'recall':re,'f1':f1,'support':support}).to_csv(O/'current_family_f1.csv',index=False)
 a=pd.read_csv(D/'historical_harder_predictions.csv');a=a[a.label.eq(1)].copy();assert len(a)==20
 # Reconstruct the exact historical positive training membership used for the current fit.
 from athpathfinder.io_data import load_current_primary,build_historical_sets,historical_family_lookup
 from historical_support import collapse_historical
 pos,_=load_current_primary();go,hist,later=build_historical_sets(pos)
 hist,_=collapse_historical(hist,historical_family_lookup(pos));assert len(hist)==291
 lookup=later.set_index('sample_id');refs=[set(g) for g in hist.genes_list]
 assert set(a.sample_id)==set(lookup.index)
 a['max_training_jaccard']=[max(len(set(lookup.loc[sid,'genes_list'])&r)/len(set(lookup.loc[sid,'genes_list'])|r) for r in refs) for sid in a.sample_id]
 a['joint_correct']=a.stage1_pred.eq(1)&a.stage2_pred.eq(a.true_family)
 assert a.joint_correct.sum()==10
 a.to_csv(O/'historical_pathway_diagnostics.csv',index=False)
 grouped=a.groupby('true_family').agg(total=('sample_id','size'),joint_correct=('joint_correct','sum'));grouped['not_joint_correct']=grouped.total-grouped.joint_correct
 grouped.to_csv(O/'historical_family_counts.csv')
 summary={'n_pathways':20,'joint_correct':int(a.joint_correct.sum()),'lower_overlap_n':int(a.max_training_jaccard.lt(.5).sum()),'lower_overlap_joint_correct':int(a.loc[a.max_training_jaccard.lt(.5),'joint_correct'].sum()),'historical_reference_pathways':len(hist),'pathway_only':True,'controls_excluded_from_this_diagnostic_only':True}
 assert (summary['lower_overlap_n'],summary['lower_overlap_joint_correct'])==(16,9)
 (O/'historical_pathway_diagnostics_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print('Existing predictions summarized',summary,flush=True)

if __name__=='__main__':
 existing_predictions()
 lofo()
 (O/'run_completed.json').write_text(json.dumps({'feature_version':FEATURE_VERSION,'completed_utc':pd.Timestamp.now(tz='UTC').isoformat(),'new_fits':10,'models_added':0,'gates_changed':False,'input_sha256':{p.name:sha(p) for p in [D/'current_predictions.csv',D/'historical_harder_predictions.csv',D/'current_fit_specification.json']},'script_sha256':sha(Path(__file__))},indent=2)+'\n')
