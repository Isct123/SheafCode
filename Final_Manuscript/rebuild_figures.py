                                                                         

                                                  
                              
                                                                   
   
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data';FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
COLORS=['#176b7a','#ce8542','#73798b','#9b536a']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220,'font.family':'DejaVu Sans'})
def short(label):return label.replace('Raw replication ','Raw ')
def save(f,name):
    f.savefig(FIG/f'{name}.pdf',bbox_inches='tight');f.savefig(FIG/f'{name}.png',bbox_inches='tight');plt.close(f)
annotations=pd.read_csv(DATA/'figure_annotations.csv')
def ptext(p):
    if pd.isna(p):return 'not tested'
    return '<0.001' if p<.001 else f'{p:.3f}'
for name in ['snapshot','channel','prolonged']:
    d=pd.read_csv(DATA/f'{name}.csv');networks=list(d.network.unique());methods=list(d.method.unique());y=np.arange(len(networks))
    f,ax=plt.subplots(figsize=(10,5.6))
    for i,m in enumerate(methods):
        t=d[d.method==m].set_index('network').loc[networks];v=t['mean'].to_numpy();ci=t[['ci_low','ci_high']].to_numpy()
        ax.errorbar(v,y+(i-(len(methods)-1)/2)*.17,xerr=np.maximum(0,[v-ci[:,0],ci[:,1]-v]),fmt='o',ms=4.8,capsize=2,color=COLORS[i],label=m)
    meta=annotations[annotations['plot']==name].set_index('network').loc[networks]
    labels=[]
    for j,(network,t) in enumerate(meta.iterrows()):
        count=f"n={int(t['n'])}" + (f", {int(t.runs)} runs" if name=='prolonged' else '')
        labels.append(f'{short(network)}\n{count}')
        gain=f'{t.gain_percent:+.1f}%'
        ax.text(1.035,j,f'{gain}\n$p_H$ {ptext(t.p_holm)}',transform=ax.get_yaxis_transform(),va='center',fontsize=9,color=COLORS[0] if t.gain_percent>0 else '#333333')
    ax.text(1.035,1.025,'Gain vs '+('Kalman' if name=='prolonged' else 'Gaussian')+'\n'+('Run test' if name=='prolonged' else 'Paired target test'),transform=ax.transAxes,fontsize=9,va='bottom')
    ax.set_yticks(y,labels,fontsize=9);ax.invert_yaxis();ax.set_xlim(left=0);ax.set_xlabel('Mean standardized RMSE (lower is better)');ax.grid(axis='x',alpha=.2);ax.legend(loc='upper center',bbox_to_anchor=(.5,1.15),ncol=2,frameon=False,fontsize=9);f.tight_layout(rect=[0,0,.85,1])
    titles={'snapshot':'Single-station snapshot outages','channel':'Single-channel outages','prolonged':'First 24 hours: retrospective outage reconstruction'}
    f.text(.5,1.035,titles[name],ha='center',fontsize=11);save(f,name)
d=pd.read_csv(DATA/'certificate_deciles.csv');networks=list(d.network.unique());y=np.arange(len(networks))
f,(ax,detail)=plt.subplots(1,2,figsize=(10.5,5.5),sharey=True,gridspec_kw={'width_ratios':[1.15,1]})
meta=annotations[annotations['plot']=='certificates'].set_index('network').loc[networks]
for i,g in enumerate(['safe','risky']):
    t=d[d.group==g].set_index('network').loc[networks];v=t['mean'].to_numpy();ci=t[['ci_low','ci_high']].to_numpy()
    for panel in [ax,detail]:
        keep=np.ones(len(t),dtype=bool) if panel is ax else np.array([x!='National' for x in networks])
        panel.errorbar(v[keep],(y+(i-.5)*.22)[keep],xerr=np.maximum(0,[v[keep]-ci[keep,0],ci[keep,1]-v[keep]]),fmt='o',ms=4,capsize=2,color=COLORS[i],label=['Lowest C2* decile','Highest C2* decile'][i])
labels=[f"{short(network)}\nn={int(t['n'])}/{int(t.n_high)}" for network,t in meta.iterrows()]
ax.set_yticks(y,labels,fontsize=9);ax.invert_yaxis();ax.set_xlim(left=0);detail.set_xlim(.28,1.06)
ax.set_title('All networks: full intervals',fontsize=10);detail.set_title('Detail: non-national networks',fontsize=10)
for panel in [ax,detail]:panel.grid(axis='x',alpha=.2);panel.set_xlabel('Mean standardized RMSE',fontsize=9)
detail.text(.32,networks.index('National'),'National interval shown at left',va='center',fontsize=8,color='#555555')
for j,(network,t) in enumerate(meta.iterrows()):
    detail.text(1.045,j,f'{t.ratio:.2f}x\n$p_H$ {ptext(t.p_holm)}',transform=detail.get_yaxis_transform(),va='center',fontsize=8)
detail.text(1.045,1.025,'High / low\nHolm test',transform=detail.transAxes,fontsize=8,va='bottom')
f.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.96),ncol=2,frameon=False,fontsize=9)
f.tight_layout(rect=[0,0,.88,.92]);f.text(.5,1.035,'Pre-ranked held-out sensitivity groups',ha='center',fontsize=11);save(f,'certificates')
d=pd.read_csv(DATA/'thinning.csv');f,ax=plt.subplots(figsize=(8.5,4.4))
for i,(network,t) in enumerate(d.groupby('network',sort=False)):
    ax.plot(100*t.edge_retention,100*t.identifiable_mean,marker=['o','s','^'][i],color=COLORS[i],label=network)
    ax.fill_between(100*t.edge_retention,100*t.identifiable_min,100*t.identifiable_max,color=COLORS[i],alpha=.1)
ax.set_xlabel('Retained spatial edges (%)');ax.set_ylabel('Artificial outage patterns passing C1 (%)');ax.set_ylim(0,104);ax.set_xlim(10,102);ax.legend(frameon=False);ax.grid(alpha=.2);f.tight_layout();save(f,'thinning')
d=pd.read_csv(DATA/'horizons.csv');networks=list(d.network.unique());hours=['1-3','4-6','7-12','13-24','25-48','49-72'];f,axes=plt.subplots(3,3,figsize=(11,8.5),sharex=True)
for ax,network in zip(axes.flat,networks):
    for i,m in enumerate(['st_sheaf','kalman_var','hybrid']):
        v=d[(d.network==network)&(d.method==m)].set_index('hours').reindex(hours).pooled_rmse
        ax.plot(range(6),v,marker=['o','s','^'][i],ms=3,color=COLORS[i],label=['Space-time sheaf','Kalman/VAR','Hybrid'][i])
    ax.set_title(short(network),fontsize=10);ax.set_ylim(bottom=0);ax.grid(alpha=.15);ax.set_xticks(range(6),hours,rotation=35,fontsize=8)
for ax in axes[:,0]:ax.set_ylabel('Pooled standardized RMSE')
f.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False);f.supxlabel('Hours after outage (bins)');f.tight_layout(rect=[0,.02,1,.96]);save(f,'horizons')
faults=pd.read_csv(DATA/'fault_severity.csv');graph=pd.read_csv(DATA/'national_graph_study.csv');f,axes=plt.subplots(1,2,figsize=(10,4.3))
for i,(network,t) in enumerate(faults.groupby('network',sort=False)):axes[0].plot(t.severity,t.top1,marker=['o','s','^'][i],color=COLORS[i],label=network)
axes[0].set_xlabel('Synthetic corruption severity (model SD)');axes[0].set_ylabel('Syndrome top-1 localization accuracy');axes[0].set_ylim(0,1);axes[0].legend(frameon=False);axes[0].grid(alpha=.2)
t=graph[graph.variant.isin(['geo_knn_k8','corr_knn_k8','union_geo_corr_k8','geo_knn_k8_dmax25km'])]
axes[1].bar(range(len(t)),t.test_rmse,color=[COLORS[0],COLORS[2],COLORS[1],COLORS[3]]);axes[1].set_xticks(range(len(t)),['Frozen geo','Correlation','Geo + corr\n(validation choice)','25 km cap'],rotation=18,fontsize=8);axes[1].set_ylabel('Mean single-station RMSE');axes[1].set_ylim(0,.85);axes[1].grid(axis='y',alpha=.2);f.tight_layout();save(f,'secondary')
d=pd.read_csv(DATA/'risk_association.csv');f,axes=plt.subplots(2,2,figsize=(9.5,6.5))
for ax,(network,t) in zip(axes.flat,d.groupby('network',sort=False)):
    points=ax.scatter(t.c2,t.rmse,c=t['size'],cmap='viridis',vmin=1,vmax=12,s=19,alpha=.7,edgecolors='none');rho=t[['c2','rmse']].corr(method='spearman').iloc[0,1]
    ax.set_title(f'{network} (Spearman {rho:.2f})',fontsize=11);ax.set_xlabel('Static artificial-outage C2*');ax.set_ylabel('Pattern mean standardized RMSE');ax.grid(alpha=.15)
f.tight_layout(rect=[0,0,.9,1]);bar=f.colorbar(points,ax=list(axes.flat),fraction=.026,pad=.025);bar.set_label('Erased stations');save(f,'risk_association')
print('Rebuilt eight figures from bundled plot CSVs.')
