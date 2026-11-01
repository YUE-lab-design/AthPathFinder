"""Integration tests of the exported API and real CLI with the newly fitted model."""
from pathlib import Path
import sys,json,subprocess
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from athpathfinder.predict import load_publication_model,predict
from athpathfinder.evaluation import reconstruct_frozen_split
from athpathfinder import paths
D=paths.GROUPED_OUTPUT/'geneset_seed_v2';model=load_publication_model(fit_if_missing=False);_,test,_=reconstruct_frozen_split();checked=0
for row in test[test.genes_list.map(len).gt(15)].itertuples():
 a=model.predict_one(row.genes_list,'a').as_dict();b=model.predict_one(row.genes_list[::-1]+row.genes_list,'file-b').as_dict()
 assert a==b
 np.testing.assert_array_equal(model.embed(row.genes_list,'one'),model.embed(row.genes_list[::-1]+row.genes_list,'two'))
 checked+=1
known=test.genes_list.iloc[0][:5];assert len(set(known))==5
unknown='AT1G99999';assert unknown not in model.gene_to_go
base=predict(known,model);aug=predict(known[::-1]+known+[unknown],model)
for k in ['stage1_score','pathway_like','family','label','genes','family_jaccard','no_gene_overlap_support']:assert base[k]==aug[k]
assert aug['discarded_genes']==[unknown]
def cli(args):
 p=subprocess.run([sys.executable,'-m','athpathfinder','--json',*args],cwd=HERE,text=True,capture_output=True)
 return p
valid=cli(known);assert valid.returncode==0 and json.loads(valid.stdout)['genes']==known
invalid=cli(known[:4]*2+[unknown]);assert invalid.returncode==2 and 'At least 5 distinct retained' in invalid.stderr
augmented=cli(known+[unknown]);assert augmented.returncode==0 and json.loads(augmented.stdout)['discarded_genes']==[unknown]
points=pd.read_csv(D/'current_predictions.csv');sid=points.loc[points.label.eq(1)&points.end_to_end_eligible&points.stage1_pred.eq(1)&points.no_gene_overlap_support,'sample_id'].iloc[0]
g=test.set_index('sample_id').loc[sid,'genes_list'];zero=predict(g,model);assert zero['family'] is not None and zero['no_gene_overlap_support'] and zero['family_jaccard']==0
result={'feature_version':'gene_set_seed_v2','all_passed':True,'real_long_query_invariance_n':checked,'valid_cli_exit':valid.returncode,'invalid_duplicate_unknown_cli_exit':invalid.returncode,'unknown_cli_reported':True,'unknown_does_not_change_retained_query_prediction':True,'real_zero_overlap_flag_and_unchanged_family':True,'fit_if_missing':False}
(D/'api_cli_integration_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
