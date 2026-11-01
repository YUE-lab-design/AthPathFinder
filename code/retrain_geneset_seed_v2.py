#!/usr/bin/env python3
"""Feature-rule repair: fixed architecture, fixed outer groups, dev-only gates.

All outputs use a new directory; original predictions and fitted models remain.
The original test-informed family-head decision remains an evaluation limitation.
"""
from pathlib import Path
import sys,json,pickle,warnings,hashlib
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score,average_precision_score
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from athpathfinder import paths
from athpathfinder.training import fit_current_model,fit_gate
from athpathfinder.embedding import build_features,FEATURE_VERSION
from athpathfinder.problem import FAMILIES,NON_METABOLIC
from athpathfinder.algorithms.family import JaccardFamily
from athpathfinder.algorithms.two_stage import compose_end_to_end
from athpathfinder.scoring import prediction_metrics,bootstrap_intervals,paired_group_differences,closed_catalog_mask
from athpathfinder.io_data import load_current_primary,build_historical_sets,historical_family_lookup,generate_size_matched_random_controls
from athpathfinder.grouping import assign_similarity_groups
from historical_support import collapse_historical,_parse_genes
OUT=paths.GROUPED_OUTPUT/'geneset_seed_v2';OUT.mkdir(exist_ok=True)

def write_json(name,x): (OUT/name).write_text(json.dumps(x,indent=2)+'\n')
def predictions(frame,model,selected,go,family,gate):
    x,_=build_features(frame,selected,go);s=model.predict_proba(x)[:,1]
    label,similarity=family.predict_with_support(frame.genes_list.tolist())
    cols=[c for c in ['sample_id','label','family','family_eligible','similarity_group','source','strategy','outer_fold'] if c in frame]
    p=frame[cols].rename(columns={'family':'true_family'}).copy()
    p['stage1_score']=s;p['stage1_pred']=(s>=gate).astype(int);p['stage2_pred']=label;p['end_to_end_pred']=compose_end_to_end(p.stage1_pred,label)
    p['family_jaccard']=similarity;p['no_gene_overlap_support']=similarity==0
    p['end_to_end_eligible']=closed_catalog_mask(p,list(FAMILIES));p['feature_version']=FEATURE_VERSION
    return p

def save(name,p,bootstrap=False):
    p.to_csv(OUT/(name+'_predictions.csv'),index=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');m=prediction_metrics(p,list(FAMILIES))
        result={'implementation':FEATURE_VERSION,'point':m}
        if bootstrap:result['ci95']=bootstrap_intervals(p,list(FAMILIES));result.update(n_resamples=2000,seed=42,unit='similarity_group')
    write_json(name+'_summary.json',result);print(name,m,flush=True)
    return result

def current():
    model,context=fit_current_model(OUT);model.save();dev,test,go=context['development'],context['test'],context['gene_to_go']
    p=predictions(test,model.stage1,model.selected_go,go,model.stage2,model.gate);save('current',p,True)
    from athpathfinder.algorithms.pathway_like import JaccardPathwayLike
    b=p.copy();baseline=JaccardPathwayLike().fit(dev.genes_list.tolist(),dev.label.to_numpy())
    b['stage1_score']=baseline.score(test.genes_list.tolist());bgate=float(pd.read_csv(paths.GROUPED_OUTPUT/'independent_metric_baselines.csv').query("method == 'Jaccard 1-NN'").iloc[0].stage1_gate)
    b['stage1_pred']=(b.stage1_score>=bgate).astype(int);b['end_to_end_pred']=compose_end_to_end(b.stage1_pred,b.stage2_pred)
    save('current_jaccard',b,True);write_json('current_paired_difference.json',paired_group_differences(p,b,list(FAMILIES)))
    # Canonical-query invariance checked against the actual newly fitted model.
    checked=[]
    for row in test[test.genes_list.map(len).gt(15)].itertuples():
        genes=list(row.genes_list);a=model.predict_one(genes,'original').as_dict();bb=model.predict_one(list(reversed(genes))+genes[:3],'renamed-file').as_dict()
        for key in ['stage1_score','pathway_like','family','label','family_jaccard','no_gene_overlap_support','genes']:assert a[key]==bb[key],(row.sample_id,key)
        np.testing.assert_array_equal(model.embed(genes,'a'),model.embed(list(reversed(genes))+genes,'b'))
        checked.append(row.sample_id)
    write_json('actual_model_invariance_checks.json',{'n_long_queries':len(checked),'all_passed':True,'sample_ids':checked,'checks':['renaming','reordering','duplicate insertion','exact features and predictions']})
    # Error categories; family errors are conditional on eligible accepted positives.
    z=p.copy();refs=[set(x) for x in dev.loc[dev.label.eq(1),'genes_list']]
    z['max_training_jaccard']=[max(len(set(g)&r)/len(set(g)|r) for r in refs) for g in test.genes_list]
    z['overlap_bin']=pd.cut(z.max_training_jaccard,[-.001,0,.25,.5,.8,1],labels=['0','(0,.25]','(.25,.50]','(.50,.80]','(.80,1]'])
    z['pathway_rejected']=z.label.eq(1)&z.stage1_pred.eq(0);z['control_accepted']=z.label.eq(0)&z.stage1_pred.eq(1)
    z['family_wrong_after_acceptance']=z.label.eq(1)&z.end_to_end_eligible&z.stage1_pred.eq(1)&z.stage2_pred.ne(z.true_family)
    z['family_wrong_before_gate']=z.label.eq(1)&z.end_to_end_eligible&z.stage2_pred.ne(z.true_family)
    cols=['pathway_rejected','control_accepted','family_wrong_after_acceptance','family_wrong_before_gate']
    z.to_csv(OUT/'current_errors_per_sample.csv',index=False)
    z.groupby(['label','source','strategy','overlap_bin'],observed=True).agg(n=('sample_id','size'),**{k:(k,'sum') for k in cols}).to_csv(OUT/'current_errors_stratified.csv')
    write_json('current_error_counts.json',{'counts':{k:int(z[k].sum()) for k in cols},'controls':z[z.label.eq(0)].groupby('strategy').agg(n=('sample_id','size'),accepted=('control_accepted','sum')).to_dict('index')})
    # Explicit boundary and support checks with the fitted API.
    from athpathfinder.predict import validate_query
    for n in [0,1,4]:
        try:model.predict_one(test.genes_list.iloc[0][:n]);raise AssertionError('short query accepted')
        except ValueError:pass
    query=list(test.genes_list.iloc[0]);unknown='AT9G99999';r=model.predict_one(query+[unknown]).as_dict();assert unknown in r['discarded_genes']
    write_json('current_api_checks.json',{'short_queries_rejected':True,'unknown_gene_discard_reported':True,'five_retained_required':True})

def historical():
    positives,_=load_current_primary();go,historical,later=build_historical_sets(positives)
    hist,fam=collapse_historical(historical,historical_family_lookup(positives));ft=fam[fam.family.isin(FAMILIES)]
    family=JaccardFamily().fit(ft.genes_list.tolist(),ft.family.tolist())
    controls=generate_size_matched_random_controls(pd.concat([hist,hist],ignore_index=True),go,hist,'historical_grouped_control')
    train=pd.concat([hist,controls],ignore_index=True,sort=False);train,_=assign_similarity_groups(train)
    split=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
    splits=list(split.split(train,train.label,train.similarity_group))
    model,selected,gate,oof,info=fit_gate(train,go,splits,'historical',OUT)
    (OUT/'historical_fit.pkl').write_bytes(pickle.dumps({'stage1':model,'selected_go':selected,'gate':gate,'feature_version':FEATURE_VERSION,'stage2':family}))
    hard=pd.read_csv(paths.GROUPED_OUTPUT/'historical_harder_controls.csv');hard['genes_list']=hard.genes_list.map(_parse_genes)
    easy=generate_size_matched_random_controls(later,go,hist,'later_grouped_control')
    for name,ctrl in [('historical_harder',hard),('historical_easy',easy)]:
        test=pd.concat([later,ctrl],ignore_index=True,sort=False);test['family_eligible']=test.label.eq(1)&test.family.isin(FAMILIES);test,_=assign_similarity_groups(test)
        save(name,predictions(test,model,selected,go,family,gate),True)
    # Name sensitivity uses the archived pathway IDs and original control recipe.
    archived=pd.read_csv(paths.GROUPED_OUTPUT/'aracyc_later_name_match_predictions.csv');raw=pd.read_csv(paths.DATA/'aracyc_pathways.20251021.txt',sep='\t',dtype=str)
    genes=defaultdict(set)
    for pid,g in zip(raw['Pathway-id'],raw['Gene-id']):
        if str(g).upper() in go:genes['AraCyc::'+str(pid)].add(str(g).upper())
    later=pd.DataFrame({'sample_id':archived.sample_id,'family':archived.family,'label':1,'family_eligible':True,'source':'AraCyc_names','genes_list':[sorted(genes[sid]) for sid in archived.sample_id]})
    ctrl=generate_size_matched_random_controls(later,go,hist,'aracyc_later_name_control');ctrl['family_eligible']=False
    test=pd.concat([later,ctrl],ignore_index=True,sort=False);test,_=assign_similarity_groups(test)
    save('historical_aracyc',predictions(test,model,selected,go,family,gate),True)

def rice():
    import evaluate_rice_catalog as r
    from athpathfinder.evaluation import primary_cohort,development_splits
    from athpathfinder.grouping import collapse_exact
    r._require_cache();kg,pg,names=r.kegg_genes_and_pathways();go,_,_=r.freeze_or_load_gene_to_go(kg)
    fam,nonmet=r.arabidopsis_map_families();pos,cur,audit=r.build_rice_rows(pg,names,go,fam,nonmet)
    pos,_=collapse_exact(pd.DataFrame(pos));cur,_=collapse_exact(pd.DataFrame(cur))
    # Explicitly preserve the ACTUAL ten-family rice experiment, distinct from Arabidopsis support filtering.
    pos['family_eligible']=pos.label.eq(1)&pos.family.isin(FAMILIES);cur['family_eligible']=False
    cohort,_=primary_cohort(pos,cur,go,360)
    archived=pd.read_csv(paths.GROUPED_OUTPUT/'rice_fivefold_predictions.csv');assert set(cohort.sample_id)==set(archived.sample_id)
    rows=[];allp=[]
    for fold in sorted(archived.outer_fold.unique()):
        ids=set(archived.loc[archived.outer_fold.eq(fold),'sample_id']);dev=cohort[~cohort.sample_id.isin(ids)].reset_index(drop=True);test=cohort[cohort.sample_id.isin(ids)].reset_index(drop=True)
        assert not set(dev.similarity_group)&set(test.similarity_group)
        model,selected,gate,oof,info=fit_gate(dev,go,development_splits(dev),f'rice_fold{fold}',OUT)
        train=dev[dev.label.eq(1)&dev.family_eligible];family=JaccardFamily().fit(train.genes_list.tolist(),train.family.tolist());assert set(family.families)==set(FAMILIES)
        test['outer_fold']=fold;p=predictions(test,model,selected,go,family,gate);result=save(f'rice_fold{fold}',p,False)
        (OUT/f'rice_fold{fold}_fit.pkl').write_bytes(pickle.dumps({'stage1':model,'selected_go':selected,'gate':gate,'stage2':family,'feature_version':FEATURE_VERSION}))
        rows.append({'outer_fold':int(fold),'gate':gate,**{k:v for k,v in result['point'].items() if k!='output_families'}});allp.append(p)
    pd.DataFrame(rows).to_csv(OUT/'rice_fivefold.csv',index=False);pd.concat(allp).to_csv(OUT/'rice_all_predictions.csv',index=False)
    keys=['stage1_auroc','stage1_auprc','stage2_macro_f1','end_to_end_accuracy','end_to_end_macro_f1','end_to_end_weighted_f1']
    write_json('rice_summary.json',{'implementation':FEATURE_VERSION,'outer_folds':'archived membership, unchanged','output_families':list(FAMILIES),'raw_min_support_audit':audit,'effective_policy':'all ten mapped unambiguous families; no FAMILY_MIN applied for rice','mean_sd':{k:{'mean':float(np.mean([q[k] for q in rows])),'sd':float(np.std([q[k] for q in rows],ddof=1))} for k in keys}})

def main():
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='y_pred contains classes not in y_true')
        for name,fn in [('current',current),('historical',historical),('rice',rice)]:
            if '--'+name in sys.argv or len(sys.argv)==1:fn()
    write_json('run_status.json',{'feature_version':FEATURE_VERSION,'completed_at_utc':pd.Timestamp.now(tz='UTC').isoformat(),'arguments':sys.argv[1:]})
if __name__=='__main__':main()
