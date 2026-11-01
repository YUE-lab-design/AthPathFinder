"""RECOMB-format figures from existing current-version results; no refitting."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import roc_curve,roc_auc_score,confusion_matrix
import fitz
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from athpathfinder.problem import FAMILIES,short_family
R=HERE.parent;D=R/'data/grouped_analysis_results/geneset_seed_v2';O=D/'restored_diagnostics';P=R/'manuscript/lncs/figures';P.mkdir(exist_ok=True)
BLUE,ORANGE,GREEN,INK,GREY='#0072B2','#D55E00','#009E73','#233343','#667684'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':11,'axes.labelsize':11,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':11,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#9AA5AE','text.color':INK,'savefig.facecolor':'white'})
checks=[]
def save(fig,name):
 for ext in ['pdf','svg','png']:fig.savefig(P/(name+'.'+ext),dpi=300,bbox_inches='tight',pad_inches=.05)
 plt.close(fig)
 doc=fitz.open(P/(name+'.pdf'));page=doc[0];sizes=[s['size'] for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans'] if s['text'].strip()]
 minimum=min(sizes)*468/page.rect.width
 assert minimum>=10,(name,minimum,page.rect.width)
 checks.append({'figure':name,'native_width_pt':page.rect.width,'minimum_font_at_6_5in_width_pt':minimum})
def panel(ax,label):ax.set_title(label,loc='left',fontweight='bold',pad=10)
# Full-width method schematic; all text remains at least 10 pt when embedded.
fig,ax=plt.subplots(figsize=(6.4,3.2));ax.axis('off');ax.set_position([.01,.01,.98,.98])
def box(x,y,w,h,title,body,face,edge=BLUE,color=INK):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008,rounding_size=.015',facecolor=face,edgecolor=edge,lw=1.1,transform=ax.transAxes))
 ax.text(x+w/2,y+h-.035,title,ha='center',va='top',fontweight='bold',color=color,transform=ax.transAxes)
 if body:ax.text(x+w/2,y+.035,body,ha='center',va='bottom',linespacing=1.3,transform=ax.transAxes,color=color)
def arrow(start,end,label='',where=None):
 ax.annotate('',xy=end,xytext=start,xycoords='axes fraction',arrowprops={'arrowstyle':'-|>','color':GREY,'lw':1.2})
 if label:ax.text(*where,label,ha='center',transform=ax.transAxes,color=GREY)
box(.02,.62,.40,.34,'1  Prepare input','Normalize and deduplicate\nDiscard unknown identifiers\nRequire ≥5 retained genes','#EDF4FA')
box(.55,.62,.43,.34,'2  Represent the set','60 GO frequencies\n4 GO-Jaccard summaries\n3 size/information features','#EDF7F1',GREEN)
box(.02,.09,.40,.33,'3  Recognize','LightGBM score, s\nScreening threshold, τ','#FFF4E9',ORANGE)
box(.55,.33,.43,.20,'s < τ','Not predicted pathway-like','#F3F4F5',GREY,'#B40000')
box(.55,.02,.43,.25,'4  Assign family','Jaccard 1-NN (10 families)\nGene-overlap support flag','#EDF4FA')
arrow((.42,.79),(.55,.79))
ax.plot([.765,.765,.22],[.62,.565,.565],color=GREY,lw=1.2,transform=ax.transAxes);arrow((.22,.565),(.22,.42))
arrow((.42,.34),(.55,.43),'s < τ',(.485,.44));arrow((.42,.18),(.55,.15),'s ≥ τ',(.485,.09))
save(fig,'fig_framework_geneset_v2')
p=pd.read_csv(D/'current_predictions.csv');positive=p[p.label.eq(1)&p.end_to_end_eligible]
# Compact, explicit family codes keep the matrix readable at >=10 pt.
codes=['AA','SM','CH','EN','GL','LI','CO','OA','TE','NU']
fig,axes=plt.subplots(1,2,figsize=(6.4,3.15),gridspec_kw={'width_ratios':[1,1.4]})
fpr,tpr,_=roc_curve(p.label,p.stage1_score);axes[0].plot(fpr,tpr,color=BLUE,lw=1.6);axes[0].plot([0,1],[0,1],'--',color='#B9C0C6');axes[0].text(.98,.04,f'AUROC {roc_auc_score(p.label,p.stage1_score):.3f}',ha='right',transform=axes[0].transAxes)
axes[0].set(xlabel='False-positive rate',ylabel='True-positive rate',xlim=(-.02,1.02),ylim=(-.02,1.02),xticks=[0,.5,1],yticks=[0,.5,1]);axes[0].set_aspect('equal');axes[0].set_anchor('N');panel(axes[0],'A  Recognition')
cm=confusion_matrix(positive.true_family,positive.stage2_pred,labels=list(FAMILIES));prop=cm/cm.sum(axis=1,keepdims=True);im=axes[1].imshow(prop,cmap='Blues',vmin=0,vmax=1)
for i in range(10):
 for j in range(10):
  if cm[i,j]:axes[1].text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if prop[i,j]>.6 else INK)
axes[1].set_xticks(range(10),codes,rotation=90);axes[1].set_yticks(range(10),[f'{c} ({n})' for c,n in zip(codes,cm.sum(axis=1))]);axes[1].set(xlabel='Predicted family',ylabel='Reference (support)');axes[1].set_anchor('N');panel(axes[1],'B  Family assignment')
fig.colorbar(im,ax=axes[1],fraction=.05,pad=.03,ticks=[0,.5,1]);fig.tight_layout(w_pad=1.2);save(fig,'fig_current_geneset_v2')
x=pd.read_csv(O/'stage1_lofo.csv').sort_values('auroc',ascending=False).reset_index(drop=True);y=np.arange(10)
fig,ax=plt.subplots(figsize=(6.4,3.55));ax.barh(y-.17,x.auroc,height=.31,color=BLUE,label='AUROC');ax.barh(y+.17,x.auprc,height=.31,color=ORANGE,label='AUPRC');ax.set_yticks(y,[f'{short_family(r.held_family)} ({r.n_pathways}/{r.n_controls})' for r in x.itertuples()]);ax.invert_yaxis();ax.set(xlim=(0,1.03),xlabel='Stage-1 recognition performance',xticks=[0,.25,.5,.75,1]);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.19),ncol=2,frameon=False);ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False);fig.tight_layout();save(fig,'fig_stage1_lofo_geneset_v2')
x=pd.read_csv(O/'current_family_f1.csv').sort_values('f1',ascending=False).reset_index(drop=True);y=np.arange(10)
fig,ax=plt.subplots(figsize=(6.4,3.35));ax.barh(y,x.f1,color=BLUE,height=.6)
for i,r in enumerate(x.itertuples()):ax.text(r.f1+.02,i,f'{r.f1:.2f} (n={r.support})',va='center')
ax.set_yticks(y,[short_family(f) for f in x.family]);ax.invert_yaxis();ax.set(xlim=(0,1.3),xticks=[0,.25,.5,.75,1],xlabel='Family F1 before screening');ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False);fig.tight_layout();save(fig,'fig_family_f1_geneset_v2')
rows=pd.read_csv(R/'manuscript/lncs/figures/source_data/current_control_acceptance.csv');fig,ax=plt.subplots(figsize=(6.4,2.9));y=np.arange(len(rows));labels=rows.label.replace({'Positive-universe recombination':'Positive-universe\nrecombination'})
ax.barh(y,rows.acceptance_fraction,color=[ORANGE]*2+[BLUE]*5,height=.58)
for i,r in enumerate(rows.itertuples()):ax.text(r.acceptance_fraction+.02,i,f'{r.accepted}/{r.n} ({r.acceptance_fraction:.0%})',va='center')
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set(xlim=(0,1.03),xlabel='Fraction accepted as pathway-like',xticks=[0,.25,.5,.75,1]);ax.xaxis.set_major_formatter(PercentFormatter(1));ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False);fig.tight_layout();save(fig,'fig_controls_geneset_v2')
fig,axes=plt.subplots(1,2,figsize=(6.4,2.85),gridspec_kw={'width_ratios':[1,1.25]})
for i,(tag,label,color) in enumerate([('historical_harder','Harder',ORANGE),('historical_easy','Random',BLUE)]):
 q=pd.read_csv(D/(tag+'_predictions.csv'));s=json.loads((D/(tag+'_summary.json')).read_text());f,t,_=roc_curve(q.label,q.stage1_score);axes[0].plot(f,t,color=color,lw=1.6,ls='-' if i==0 else '--',label=f'{label}: {s["point"]["stage1_auroc"]:.3f}');v=s['point']['end_to_end_macro_f1'];c=s['ci95']['end_to_end_macro_f1'];axes[1].errorbar(v,i,xerr=[[v-c['low']],[c['high']-v]],fmt='o',color=color,capsize=4);axes[1].text(.49,i,f'{v:.3f}\n[{c["low"]:.3f}, {c["high"]:.3f}]',va='center')
axes[0].plot([0,1],[0,1],':',color='#B9C0C6');axes[0].set(xlabel='False-positive rate',ylabel='True-positive rate',xlim=(-.02,1.02),ylim=(-.02,1.02),xticks=[0,.5,1],yticks=[0,.5,1]);axes[0].legend(loc='lower right',frameon=False,title='AUROC');panel(axes[0],'A  Recognition')
axes[1].set_yticks([0,1],['Harder','Random']);axes[1].set(xlim=(0,.83),ylim=(1.6,-.6),xlabel='End-to-end macro-F1',xticks=[0,.2,.4]);panel(axes[1],'B  Full pipeline');fig.tight_layout(w_pad=1.3);save(fig,'fig_historical_geneset_v2')
x=pd.read_csv(O/'historical_pathway_diagnostics.csv');c=pd.read_csv(O/'historical_family_counts.csv').sort_values(['total','true_family'],ascending=[False,True]).reset_index(drop=True);y=np.arange(len(c));fig,axes=plt.subplots(1,2,figsize=(6.4,3.15),gridspec_kw={'width_ratios':[1.1,1]})
axes[0].barh(y,c.joint_correct,color=GREEN,height=.6);axes[0].barh(y,c.not_joint_correct,left=c.joint_correct,color='#D5E4F0',height=.6)
for i,r in enumerate(c.itertuples()):axes[0].text(r.total+.1,i,f'{r.joint_correct}/{r.total}',va='center')
axes[0].set_yticks(y,[short_family(f) for f in c.true_family]);axes[0].invert_yaxis();axes[0].set(xlim=(0,6.2),xlabel='Later KEGG pathways',xticks=[0,2,4,6]);panel(axes[0],'A  Joint correctness')
for yes,color,marker,label in [(True,GREEN,'o','Correct'),(False,ORANGE,'x','Incorrect')]:
 q=x[x.joint_correct.eq(yes)];axes[1].scatter(q.max_training_jaccard,q.stage1_score,color=color,marker=marker,s=32,label=label)
gate=json.loads((D/'historical_fit_specification.json').read_text())['gate'];axes[1].axhline(gate,ls='--',color=GREY);axes[1].axvline(.5,ls=':',color=GREY);axes[1].set(xlim=(-.02,.65),ylim=(0,1.04),xlabel='Maximum Jaccard to\nhistorical training',ylabel='Stage-1 score',xticks=[0,.3,.6],yticks=[0,.5,1]);axes[1].legend(loc='lower left',frameon=False);panel(axes[1],'B  Score and overlap');fig.tight_layout(w_pad=1.2);save(fig,'fig_historical_diagnostics_geneset_v2')
r=pd.read_csv(D/'rice_fivefold.csv');fig,axes=plt.subplots(1,3,figsize=(6.4,2.8),sharey=True)
for ax,(key,title,color) in zip(axes,[('stage1_auroc','Recognition\nAUROC',BLUE),('stage2_macro_f1','Family\nmacro-F1',GREEN),('end_to_end_macro_f1','End-to-end\nmacro-F1',ORANGE)]):
 a=r[key];mean=a.mean();sd=a.std(ddof=1);ax.scatter(range(1,6),a,color=color,s=25);ax.axhline(mean,color=color,ls='--');ax.axhspan(mean-sd,mean+sd,color=color,alpha=.1);ax.set(xticks=[1,3,5],xlabel='Outer fold',ylim=(0,1.04));ax.set_title(title,fontweight='bold');ax.text(.5,.025,f'{mean:.3f}\n±{sd:.3f}',ha='center',transform=ax.transAxes,color=color)
axes[0].set_ylabel('Score');fig.tight_layout(w_pad=.8);save(fig,'fig_rice_geneset_v2')
(P/'font_checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(checks)
