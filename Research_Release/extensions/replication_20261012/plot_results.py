                                                                           
from pathlib import Path
import json
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[3]/'Research_Release/matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'Research_Release/replication_20261012_results'

def main():
    comparisons=json.loads((OUT/'comparisons.json').read_text())
    table=pd.read_csv(OUT/'ALL_COHORT_RESULTS.csv')
    primary=table[(table.status=='completed')&(table.scope=='primary_every_third_day')].sort_values(['provider','cohort'])
    names=primary.cohort.tolist()
    labels=[name.replace('uk_zone_','UK: ').replace('nsw_','NSW: ').replace('epa_pmno2_','EPA PM/NO2: ').replace('epa_','EPA: ').replace('_',' ') for name in names]
    fig,axes=plt.subplots(1,2,figsize=(12,max(7,len(names)*.31)),sharey=True,layout='constrained')
    colors=['#0072b2','#d55e00']
    for ax,baseline,title,color in zip(axes,['var_causal','random_forest_causal'],['Against causal VAR','Against causal random forest'],colors):
        for y,name in enumerate(names):
            value=comparisons[name]['primary_every_third_day'][baseline]['7']['mse']
            if not value.get('cases'):continue
            gain=value['gain_percent'];ci=value.get('ci95')
            if ci:ax.plot(ci,[y,y],color=color,lw=1.5)
            ax.scatter([gain],[y],color=color,s=28,zorder=3)
        ax.axvline(0,color='#333333',lw=.8)
        ax.grid(axis='x',alpha=.2)
        ax.set_title(title)
        ax.set_xlabel('Normalized RMSE gain (%)\nPositive values favour sheaf')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(np.arange(len(names)),labels)
    axes[0].invert_yaxis()
    fig.suptitle(f'{len(names)} additional recent-data cohorts: primary 24-hour outage comparisons',fontsize=13)
    unavailable=int((table.status!='completed').sum())
    fig.supxlabel('Bars: paired seven-day 95% bootstrap intervals; omitted below five calendar bins. '
                  f'{unavailable} additional acquired cohort(s) had no eligible test cases.',fontsize=8)
    fig.savefig(OUT/'ALL_COHORT_RESULTS.pdf',bbox_inches='tight')
    fig.savefig(OUT/'ALL_COHORT_RESULTS.png',dpi=160,bbox_inches='tight')
    plt.close(fig)
    if (OUT/'CCA_IDENTITY_RESULTS.csv').exists():
        table=pd.read_csv(OUT/'CCA_IDENTITY_RESULTS.csv')
        primary=table[table.scope=='primary_every_third_day'].sort_values('cohort')
        fig,ax=plt.subplots(figsize=(9,max(7,len(primary)*.31)),layout='constrained')
        y=np.arange(len(primary))
        ax.scatter(primary.gain_vs_fixed_identity,y-.12,label='Fixed component ablation',color='#0072b2',s=25)
        ax.scatter(primary.gain_vs_tuned_identity,y+.12,label='Equally tuned identity-spatial model',color='#d55e00',s=25)
        ax.set_yticks(y,primary.cohort.str.replace('_',' '))
        ax.invert_yaxis();ax.axvline(0,color='#333333',lw=.8);ax.grid(axis='x',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
        ax.set_xlabel('Spatial CCA normalized RMSE gain (%)\nPositive values favour the CCA setting; temporal CCA retained in both controls')
        ax.set_title('Spatial CCA comparisons on all additional cohorts')
        ax.legend(loc='best',fontsize=8)
        fig.savefig(OUT/'CCA_IDENTITY_RESULTS.pdf',bbox_inches='tight')
        fig.savefig(OUT/'CCA_IDENTITY_RESULTS.png',dpi=160,bbox_inches='tight')
        plt.close(fig)
    print('Exported standalone plots retaining all cohorts and losses.')

if __name__=='__main__':main()
