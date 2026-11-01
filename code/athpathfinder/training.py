"""Fixed LightGBM/Jaccard architecture; all gates selected in development only."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import paths
from .embedding import select_and_feature,FEATURE_VERSION
from .algorithms.hypothesis_classes import deployed_lightgbm
from .algorithms.family import JaccardFamily
from .algorithms.two_stage import threshold_by_balanced_accuracy
from .evaluation import reconstruct_frozen_split,development_splits
from .problem import FAMILIES

def fit_gate(train,go,splits,tag,out=None):
    oof=np.full(len(train),np.nan);records=[];audits=[]
    for fold,(tr,va) in enumerate(splits,1):
        a,b=train.iloc[tr],train.iloc[va]
        assert not set(a.similarity_group)&set(b.similarity_group)
        xa,xb,names,audit=select_and_feature(a,b,go)
        model=deployed_lightgbm().set_params(clf__n_jobs=2).fit(xa,a.label.to_numpy())
        oof[va]=model.predict_proba(xb)[:,1]
        records.extend({'sample_id':sid,'fold':fold,'label':int(y),'score':float(v),'similarity_group':g} for sid,y,v,g in zip(b.sample_id,b.label,oof[va],b.similarity_group))
        audits.append({'fold':fold,'training_ids':a.sample_id.tolist(),'validation_ids':b.sample_id.tolist(),'feature_selection':audit})
        print(f'{tag} development fold {fold} completed',flush=True)
    assert np.isfinite(oof).all()
    gate=threshold_by_balanced_accuracy(train.label.to_numpy(),oof)
    x,_,names,audit=select_and_feature(train,train.iloc[:1],go)
    model=deployed_lightgbm().set_params(clf__n_jobs=2).fit(x,train.label.to_numpy())
    selected=[n.split('::',1)[1] for n in names if n.startswith('gofreq::')]
    info={'feature_version':FEATURE_VERSION,'gate':float(gate),'gate_selection':'maximum balanced accuracy on development-only grouped OOF','selected_go':selected,'feature_selection':audit,'fold_audits':audits}
    if out:
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(records).to_csv(out/(tag+'_development_oof.csv'),index=False)
        (out/(tag+'_fit_specification.json')).write_text(json.dumps(info,indent=2)+'\n')
    return model,selected,gate,pd.DataFrame(records),info

def archived_current_splits(dev):
    archived=pd.read_csv(paths.GROUPED_OUTPUT/'stage1_grouped_cv_oof.csv')
    archived=archived[archived.method.eq('LightGBM')]
    assert archived.sample_id.is_unique and set(archived.sample_id)==set(dev.sample_id)
    folds=archived.set_index('sample_id').loc[dev.sample_id,'fold'].to_numpy()
    return [(np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)) for fold in sorted(set(folds))]

def fit_current_model(out=None):
    from .predict import AthPathFinder
    dev,test,go=reconstruct_frozen_split()
    out=Path(out) if out else paths.GROUPED_OUTPUT/'geneset_seed_v2'
    s1,selected,gate,oof,info=fit_gate(dev,go,archived_current_splits(dev),'current',out)
    eligible=dev.label.eq(1)&dev.family_eligible&dev.family.isin(FAMILIES)
    family=JaccardFamily().fit(dev.loc[eligible,'genes_list'].tolist(),dev.loc[eligible,'family'].tolist())
    model=AthPathFinder(selected,go,s1,family,gate,list(FAMILIES),stage2_name='Jaccard 1-NN',family_kind='jaccard')
    return model,{'development':dev,'test':test,'gene_to_go':go,'oof':oof,'specification':info}
