"""Closed-catalog evaluation shared by fixed-prediction and training drivers.

Stage 1 includes every row. Stage 2 uses eligible positives before gating.
E2E includes those positives and ALL controls. Its fixed output alphabet is
rejection plus families represented by the fitted reference catalog.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from .problem import FAMILIES, NON_METABOLIC

ENDPOINT_VERSION = 'closed_catalog_v2'

def family_column(data):
    return 'true_family' if 'true_family' in data else 'family'

def eligible_positive(data, families=FAMILIES):
    mask = data.label.eq(1) & data[family_column(data)].isin(families)
    if 'family_eligible' in data:
        flags = data.family_eligible
        if flags.dtype != bool:
            flags = flags.map({True: True, False: False, 'True': True, 'False': False, 1: True, 0: False})
        if flags[data.label.eq(1)].isna().any():
            raise ValueError('Missing/invalid family eligibility for positive rows')
        mask &= flags.fillna(False).astype(bool)
    return mask

def closed_catalog_mask(data, families=FAMILIES):
    return data.label.eq(0) | eligible_positive(data, families)

def end_to_end_metrics(data, families=FAMILIES, pred_col='end_to_end_pred'):
    families = list(families)
    x = data.loc[closed_catalog_mask(data, families)]
    true = np.where(x.label.eq(0), NON_METABOLIC, x[family_column(x)])
    pred = x[pred_col].to_numpy()
    labels = [NON_METABOLIC] + families
    if not set(pred).issubset(labels):
        raise ValueError('Prediction outside the declared reference-catalog alphabet')
    if not len(x):
        raise ValueError('No eligible end-to-end rows')
    return {
        'end_to_end_n': int(len(x)), 'end_to_end_excluded_n': int(len(data)-len(x)),
        'end_to_end_accuracy': float(accuracy_score(true, pred)),
        'end_to_end_macro_f1': float(f1_score(true, pred, labels=labels, average='macro', zero_division=0)),
        'end_to_end_weighted_f1': float(f1_score(true, pred, labels=labels, average='weighted', zero_division=0)),
    }

def prediction_metrics(data, families=FAMILIES):
    families = list(families); pos = data.loc[eligible_positive(data, families)]
    y2 = pos[family_column(pos)]; p2 = pos.stage2_pred
    result = {'endpoint_version': ENDPOINT_VERSION, 'output_families': families,
              'stage1_test_n': int(len(data)), 'stage2_test_n': int(len(pos))}
    if 'stage1_score' in data:
        result.update(stage1_auroc=float(roc_auc_score(data.label, data.stage1_score)),
                      stage1_auprc=float(average_precision_score(data.label, data.stage1_score)))
    if 'stage1_pred' in data:
        result['stage1_balanced_accuracy'] = float(balanced_accuracy_score(data.label, data.stage1_pred))
    result.update(stage2_accuracy=float(accuracy_score(y2,p2)),
                  stage2_balanced_accuracy=float(balanced_accuracy_score(y2,p2)),
                  stage2_macro_f1=float(f1_score(y2,p2,labels=families,average='macro',zero_division=0)),
                  stage2_weighted_f1=float(f1_score(y2,p2,labels=families,average='weighted',zero_division=0)))
    result.update(end_to_end_metrics(data, families))
    return result

def bootstrap_intervals(data, families=FAMILIES, n_resamples=2000, seed=42, include_stage1=True):
    """S1/S2 sample original groups; E2E samples eligible endpoint groups.

    All labels are fixed across draws, including labels absent in a draw.
    Only test groups are resampled; models/gates are never refitted.
    """
    families=list(families); values={}
    def run(frame, kind):
        if frame.similarity_group.isna().any(): raise ValueError('Missing bootstrap groups')
        groups=frame.similarity_group.unique(); ids={g:np.flatnonzero(frame.similarity_group.eq(g)) for g in groups}
        rng=np.random.default_rng(seed)
        for _ in range(n_resamples):
            idx=np.concatenate([ids[g] for g in rng.choice(groups,len(groups),replace=True)])
            b=frame.iloc[idx]
            if kind=='e2e': m=end_to_end_metrics(b,families); m={k:v for k,v in m.items() if k not in ('end_to_end_n','end_to_end_excluded_n')}
            else:
                p=b.loc[eligible_positive(b,families)]; m={}
                if len(p):
                    y=p[family_column(p)]; pred=p.stage2_pred
                    m.update(stage2_accuracy=accuracy_score(y,pred),stage2_balanced_accuracy=balanced_accuracy_score(y,pred),
                             stage2_macro_f1=f1_score(y,pred,labels=families,average='macro',zero_division=0),
                             stage2_weighted_f1=f1_score(y,pred,labels=families,average='weighted',zero_division=0))
                if include_stage1 and b.label.nunique()==2:
                    m.update(stage1_auroc=roc_auc_score(b.label,b.stage1_score),stage1_auprc=average_precision_score(b.label,b.stage1_score))
            for k,v in m.items(): values.setdefault(k,[]).append(float(v))
    run(data.reset_index(drop=True),'s1s2')
    run(data.loc[closed_catalog_mask(data,families)].reset_index(drop=True),'e2e')
    return {k:{'low':float(np.quantile(v,.025)),'high':float(np.quantile(v,.975)),'n_successful':len(v)} for k,v in values.items()}

def paired_group_differences(a,b,families=FAMILIES,n_resamples=2000,seed=42):
    """Fixed predictions, shared group draws within each endpoint; A minus B."""
    a=a.reset_index(drop=True);b=b.set_index('sample_id').loc[a.sample_id].reset_index()
    for col in ['label',family_column(a),'similarity_group']:
        if not np.array_equal(a[col],b[col]):raise ValueError('Unpaired labels/groups')
    if not np.array_equal(closed_catalog_mask(a,families),closed_catalog_mask(b,families)):
        raise ValueError('Unpaired endpoint eligibility')
    result={}
    for scope,mask in [('stage1',np.ones(len(a),dtype=bool)),('end_to_end',closed_catalog_mask(a,families).to_numpy())]:
        aa=a.loc[mask].reset_index(drop=True);bb=b.loc[mask].reset_index(drop=True)
        groups=aa.similarity_group.unique();ids={g:np.flatnonzero(aa.similarity_group.eq(g)) for g in groups}
        rng=np.random.default_rng(seed)
        def measure(x):
            if scope=='stage1':return {'auroc':roc_auc_score(x.label,x.stage1_score),'auprc':average_precision_score(x.label,x.stage1_score)}
            m=end_to_end_metrics(x,families)
            return {k:v for k,v in m.items() if k not in ['end_to_end_n','end_to_end_excluded_n']}
        x,y=measure(aa),measure(bb);buckets={k:[] for k in x}
        for _ in range(n_resamples):
            ix=np.concatenate([ids[g] for g in rng.choice(groups,len(groups),replace=True)])
            if scope=='stage1' and aa.iloc[ix].label.nunique()<2:continue
            xx,yy=measure(aa.iloc[ix]),measure(bb.iloc[ix])
            for k in x:buckets[k].append(xx[k]-yy[k])
        result[scope]={'n':len(aa),'groups':len(groups),'difference':{k:{'point':float(x[k]-y[k]),'low':float(np.quantile(buckets[k],.025)),'high':float(np.quantile(buckets[k],.975)),'n_successful':len(buckets[k])} for k in x}}
    return {'n_resamples':n_resamples,'seed':seed,'direction':'A minus B','fixed_models_and_gates':True,'endpoints':result}
