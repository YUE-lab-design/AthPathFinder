"""Predeclared fixed LightGBM 11-class comparison; preserves the archived split.

Six multiclass fits and six matched binary fits. No holdout tuning.
The original deployed two-stage fit remains unchanged.
"""
from pathlib import Path
import json,pickle,time,hashlib,argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
from athpathfinder import paths
from athpathfinder.problem import FAMILIES,NON_METABOLIC
from athpathfinder.embedding import select_and_feature,FEATURE_VERSION
from athpathfinder.training import archived_current_splits
from athpathfinder.evaluation import reconstruct_frozen_split
from athpathfinder.algorithms.hypothesis_classes import deployed_lightgbm
from athpathfinder.algorithms.two_stage import threshold_by_balanced_accuracy
from athpathfinder.algorithms.family import JaccardFamily
from athpathfinder.scoring import closed_catalog_mask

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=paths.GROUPED_OUTPUT/'single_stage_11');args=parser.parse_args();OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
ALPHABET=[NON_METABOLIC]+list(FAMILIES);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol={'feature_version':FEATURE_VERSION,'seed':42,'single_stage':'LightGBM multiclass, 11 labels, argmax; no separate gate','parameters':'same 500 rounds/depth5/lr.03/colsample.7/L1.1/L2=1/class_weight balanced as original; 2 threads','matched_comparator':'binary LightGBM screening plus unchanged Jaccard 1-NN, on exactly the same closed-catalog development rows and fold-specific features','training_eligibility':'all controls plus unambiguous positives in the ten-family alphabet; exclude Xenobiotics/ambiguous positives, not relabel them as controls','split':'archived outer split and archived five development folds, intersected with eligible rows','feature_selection':'binary training-positive gene union, identical GO terms/features in both matched methods per fold; no heldout feature selection','threshold':'matched binary: original maximum-balanced-accuracy rule on grouped development OOF only; original deployed threshold unchanged','primary_endpoint':'11-label end-to-end macro-F1 on same 169 eligible heldout rows; accuracy/weighted-F1 complementary','uncertainty':'paired group-bootstrap, 2000 draws seed42, identical groups, fixed fits/decisions; zero for absent fixed labels','method_selection':'fixed protocol; do not adopt an architecture or threshold based on holdout outcome','limitations':'exploratory comparison on previously consulted holdout; multiclass objective produces more trees per boosting round; comparison is one specific baseline, not proof of universal architecture necessity'}
p=OUT/'protocol.json'
if p.exists():raise RuntimeError('Refusing to overwrite an existing experiment')
p.write_text(json.dumps(protocol,indent=2)+'\n')
original=paths.GROUPED_OUTPUT/'geneset_seed_v2';protected={str(p):sha(p) for p in original.rglob('*') if p.is_file()};protected[str(paths.FITS/'publication_athpathfinder_geneset_v2.pkl')]=sha(paths.FITS/'publication_athpathfinder_geneset_v2.pkl')
(OUT/'protected_original_sha256.json').write_text(json.dumps(protected,indent=2)+'\n')
dev0,test0,go=reconstruct_frozen_split();splits0=archived_current_splits(dev0)
keep=closed_catalog_mask(dev0).to_numpy();test=test0.loc[closed_catalog_mask(test0)].reset_index(drop=True);dev=dev0.loc[keep].reset_index(drop=True)
assert len(test)==169 and not set(dev.similarity_group)&set(test.similarity_group)
# Map the original fold memberships without moving any retained sample.
folds=np.full(len(dev0),-1)
for f,(_,va) in enumerate(splits0,1):folds[va]=f
folds=folds[keep];assert (folds>0).all()
y=np.where(dev.label.eq(0),NON_METABOLIC,dev.family);codes=np.array([ALPHABET.index(v) for v in y]);oof_multi=np.full((len(dev),11),np.nan);oof_bin=np.full(len(dev),np.nan);audits=[];timings=[]
for f in sorted(set(folds)):
 tr=np.flatnonzero(folds!=f);va=np.flatnonzero(folds==f);a,b=dev.iloc[tr],dev.iloc[va]
 assert not set(a.similarity_group)&set(b.similarity_group)
 xa,xb,names,audit=select_and_feature(a,b,go)
 for kind,target in [('single_stage_11',codes[tr]),('matched_binary',a.label.to_numpy())]:
  start=time.perf_counter();model=deployed_lightgbm().set_params(clf__n_jobs=2).fit(xa,target)
  if kind=='single_stage_11':
   assert list(model.classes_)==list(range(11));oof_multi[va]=model.predict_proba(xb)
  else:oof_bin[va]=model.predict_proba(xb)[:,1]
  timings.append({'kind':kind,'fold':int(f),'fit_seconds':time.perf_counter()-start});print(kind,'development fold',f,'complete',flush=True)
 audits.append({'fold':int(f),'training_ids':a.sample_id.tolist(),'validation_ids':b.sample_id.tolist(),'training_groups':sorted(set(a.similarity_group)),'validation_groups':sorted(set(b.similarity_group)),'feature_names':names,'feature_selection':audit})
assert np.isfinite(oof_multi).all() and np.isfinite(oof_bin).all();gate=threshold_by_balanced_accuracy(dev.label.to_numpy(),oof_bin)
x,xt,names,audit=select_and_feature(dev,test,go)
models={}
for kind,target in [('single_stage_11',codes),('matched_binary',dev.label.to_numpy())]:
 start=time.perf_counter();models[kind]=deployed_lightgbm().set_params(clf__n_jobs=2).fit(x,target);timings.append({'kind':kind,'fold':'full_development','fit_seconds':time.perf_counter()-start});print(kind,'full-development fit complete',flush=True)
family=JaccardFamily().fit(dev.loc[dev.label.eq(1),'genes_list'].tolist(),dev.loc[dev.label.eq(1),'family'].tolist())
refpred=family.predict(test.genes_list.tolist());bin_score=models['matched_binary'].predict_proba(xt)[:,1];prob=models['single_stage_11'].predict_proba(xt)
pred_single=np.array(ALPHABET)[np.argmax(prob,axis=1)];pred_matched=np.where(bin_score>=gate,refpred,NON_METABOLIC)
old=pd.read_csv(original/'current_predictions.csv').set_index('sample_id').loc[test.sample_id].reset_index();assert len(old)==169
pred_original=old.end_to_end_pred.to_numpy();true=np.where(test.label.eq(0),NON_METABOLIC,test.family)
result=test[['sample_id','similarity_group','label','family','family_eligible','source','strategy']].copy();result=result.rename(columns={'family':'true_family'});result['true_output']=true;result['single_stage_11_pred']=pred_single;result['matched_two_stage_pred']=pred_matched;result['original_two_stage_pred']=pred_original;result['matched_stage1_score']=bin_score;result['matched_screening_threshold']=gate
for i,label in enumerate(ALPHABET):result[f'single_stage_probability_{i}']=prob[:,i]
result.to_csv(OUT/'heldout_predictions.csv',index=False)
oof=dev[['sample_id','similarity_group','label','family']].copy();oof['fold']=folds;oof['single_stage_pred']=np.array(ALPHABET)[np.argmax(oof_multi,axis=1)];oof['matched_stage1_score']=oof_bin;oof.to_csv(OUT/'development_oof.csv',index=False)
for kind,model in models.items():
 payload={'feature_version':FEATURE_VERSION,'model':model,'selected_go':[v.split('::',1)[1] for v in names if v.startswith('gofreq::')],'alphabet':ALPHABET,'screening_threshold':gate if kind=='matched_binary' else None,'family_head':family if kind=='matched_binary' else None};(OUT/f'{kind}_fit.pkl').write_bytes(pickle.dumps(payload))
# Same test group draws for all three fixed models; confusion-count implementation.
preds={'single_stage_11':pred_single,'matched_two_stage':pred_matched,'original_two_stage':pred_original}
groups=result.similarity_group.unique();gidx={g:i for i,g in enumerate(groups)};ty=np.array([ALPHABET.index(v) for v in true]);cms=[]
for name,pred in preds.items():
 cm=np.zeros((len(groups),11,11),dtype=int)
 for g,t,v in zip(result.similarity_group,ty,pred):cm[gidx[g],t,ALPHABET.index(v)]+=1
 cms.append(cm)
cms=np.array(cms)
def metrics(cm):
 # cm shape (methods, labels, labels)
 support=cm.sum(axis=2);predsupport=cm.sum(axis=1);tp=np.diagonal(cm,axis1=1,axis2=2);den=support+predsupport;f=np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0);n=cm.sum(axis=(1,2));return np.stack([f.mean(axis=1),(f*support).sum(axis=1)/n,tp.sum(axis=1)/n],axis=1)
point=metrics(cms.sum(axis=1));rng=np.random.default_rng(42);boot=[]
for _ in range(2000):
 weights=np.bincount(rng.choice(len(groups),len(groups),replace=True),minlength=len(groups));boot.append(metrics(np.einsum('g,mgij->mij',weights,cms)))
boot=np.array(boot);keys=['macro_f1','weighted_f1','accuracy'];names_models=list(preds);summary={'protocol':protocol,'n_development_original':len(dev0),'n_development_matched':len(dev),'n_development_excluded':len(dev0)-len(dev),'n_test':len(test),'n_test_groups':len(groups),'n_development_controls':int(dev.label.eq(0).sum()),'n_development_pathways':int(dev.label.eq(1).sum()),'matched_screening_threshold':float(gate),'fixed_alphabet':ALPHABET,'models':{},'paired_differences':{},'new_model_fits':12,'timings':timings,'feature_selection':audit,'fold_audits':audits}
for i,name in enumerate(names_models):
 assert np.allclose(point[i],[f1_score(true,preds[name],labels=ALPHABET,average='macro',zero_division=0),f1_score(true,preds[name],labels=ALPHABET,average='weighted',zero_division=0),accuracy_score(true,preds[name])]);summary['models'][name]={key:{'point':float(point[i,k]),'low':float(np.quantile(boot[:,i,k],.025)),'high':float(np.quantile(boot[:,i,k],.975))} for k,key in enumerate(keys)}
for i in [1,2]:
 delta=boot[:,i]-boot[:,0];summary['paired_differences'][names_models[i]+'_minus_single_stage_11']={key:{'point':float(point[i,k]-point[0,k]),'low':float(np.quantile(delta[:,k],.025)),'high':float(np.quantile(delta[:,k],.975))} for k,key in enumerate(keys)}
assert all(sha(Path(p))==h for p,h in protected.items());summary['original_outputs_unchanged']=True
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(OUT/'status.json').write_text(json.dumps({'complete':True,'fits':12,'original_outputs_unchanged':True})+'\n')
print(json.dumps({k:summary[k] for k in ['n_development_matched','n_test','matched_screening_threshold','models','paired_differences']},indent=2),flush=True)
