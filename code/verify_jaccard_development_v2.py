import sys,json
from pathlib import Path
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from athpathfinder.training import archived_current_splits
from athpathfinder.evaluation import reconstruct_frozen_split
from athpathfinder.algorithms.pathway_like import JaccardPathwayLike
from athpathfinder.algorithms.two_stage import threshold_by_balanced_accuracy
from sklearn.metrics import roc_auc_score,average_precision_score
D=HERE.parent/'data/grouped_analysis_results/geneset_seed_v2';dev,test,_=reconstruct_frozen_split();scores=np.zeros(len(dev));foldids=np.zeros(len(dev),dtype=int)
for fold,(tr,va) in enumerate(archived_current_splits(dev),1):
 m=JaccardPathwayLike().fit(dev.iloc[tr].genes_list.tolist(),dev.iloc[tr].label.to_numpy());scores[va]=m.score(dev.iloc[va].genes_list.tolist());foldids[va]=fold
newgate=threshold_by_balanced_accuracy(dev.label.to_numpy(),scores);old=float(pd.read_csv(HERE.parent/'data/grouped_analysis_results/independent_metric_baselines.csv').query("method == 'Jaccard 1-NN'").iloc[0].stage1_gate)
assert abs(newgate-old)<1e-12,(newgate,old)
pd.DataFrame({'sample_id':dev.sample_id,'fold':foldids,'label':dev.label,'score':scores}).to_csv(D/'jaccard_development_oof.csv',index=False)
(D/'jaccard_gate_verification.json').write_text(json.dumps({'gate':newgate,'original_gate':old,'development_only':True,'same_archived_folds':True,'auroc':roc_auc_score(dev.label,scores),'auprc':average_precision_score(dev.label,scores)},indent=2)+'\n');print('Jaccard development gate verified:',newgate)
