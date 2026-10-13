                                                                                
from pathlib import Path
from fractions import Fraction as Q
import hashlib
import json
import os
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'audit_tools/mpl_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
import numpy as np

WORK = ROOT / 'SUBMISSION_WORKSPACE'
OUT = WORK / 'Source/figures'
BLUE, ORANGE, TEAL, GREY = '#245681', '#c16c26', '#237b79', '#62717c'
plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def box(ax, x, y, width, height, text, color=BLUE, size=10):
    ax.add_patch(FancyBboxPatch((x,y),width,height,boxstyle='round,pad=0.025',
                              facecolor='white',edgecolor=color,linewidth=1.4))
    ax.text(x+width/2,y+height/2,text,ha='center',va='center',fontsize=size,color=color)

def arrow(ax, start, end, color=GREY):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=12,
                               linewidth=1.3,color=color))

def save(fig, name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.08)
    fig.savefig(ROOT/'audit_tools'/(name+'_preview.png'),dpi=150,bbox_inches='tight')
    plt.close(fig)

def main():
    folder=ROOT/'Research_Release/recent_data_results/delhi'
    sys.path.insert(0,str(ROOT/'Research_Release/extensions/online_validation'))
    with (folder/'models.pkl').open('rb') as f:
        model=pickle.load(f)
    metadata=json.loads((folder/'dataset_manifest.json').read_text())
    station_ids=metadata['stations'];edges=[tuple(map(int,e)) for e in model['edges']]
    n=len(station_ids);p=len(metadata['channels']);D=model['sheaf'].D
    assert n==5 and p==4 and len(edges)==9 and D.shape==(len(edges)*p,n*p)
    for j,(u,v) in enumerate(edges):
        columns=np.flatnonzero(np.any(D[j*p:(j+1)*p]!=0,axis=0))
        assert set(columns//p)=={u,v}

    fig,axes=plt.subplots(1,2,figsize=(10,3.75),gridspec_kw={'width_ratios':[1,1.25]})
    ax=axes[0];ax.set_aspect('equal');ax.axis('off')
    angle=np.linspace(np.pi/2,np.pi/2+2*np.pi,n,endpoint=False)
    positions=np.c_[np.cos(angle),np.sin(angle)];target=2
    for u,v in edges:
        a,b=positions[u],positions[v]
        ax.plot([a[0],b[0]],[a[1],b[1]],color='#c5cdd3',lw=1.5,zorder=1)
    for v,(x,y) in enumerate(positions):
        color=ORANGE if v==target else BLUE
        ax.add_patch(Circle((x,y),.21,facecolor='white',edgecolor=color,lw=2,zorder=2))
        ax.text(x,y,station_ids[v],ha='center',va='center',color=color,fontweight='bold',fontsize=10,zorder=3)
    ax.set_xlim(-1.38,1.38);ax.set_ylim(-1.25,1.37)
    ax.set_title('A. Actual learned Indian graph',loc='left',fontweight='bold',fontsize=11)
    ax.text(0,-1.1,'Blue: available   Orange: illustrative outage',ha='center',fontsize=8.5)
    ax.text(0,-1.25,'Layout is schematic; edges come from the fitted model.',ha='center',fontsize=8)
    ax=axes[1];ax.axis('off');ax.set_xlim(0,10);ax.set_ylim(0,5)
    ax.set_title('B. What one edge means',loc='left',fontweight='bold',fontsize=11)
    box(ax,.2,3.4,3.6,.95,'Station 17\nPollutant vector '+r'$x_u$',BLUE)
    box(ax,6.2,3.4,3.6,.95,'Station 50\nPollutant vector '+r'$x_v$',BLUE)
    arrow(ax,(2,3.35),(2,2.7));arrow(ax,(8,3.35),(8,2.7))
    box(ax,.2,1.7,3.6,1,'Learned translation\n'+r'$M_{uv}x_u$',TEAL)
    box(ax,6.2,1.7,3.6,1,'Learned translation\n'+r'$N_{uv}x_v$',TEAL)
    arrow(ax,(3.85,2.2),(4.45,2.2));arrow(ax,(6.15,2.2),(5.55,2.2))
    ax.text(5,2.2,'compare',ha='center',va='center',fontsize=9)
    ax.text(5,.95,r'Penalty uses $M_{uv}x_u-N_{uv}x_v$.',ha='center',fontsize=11)
    ax.text(5,.25,'Compare translated readings; raw concentrations need not match.',ha='center',fontsize=8.7)
    fig.subplots_adjust(wspace=.2);save(fig,'network_structure')

    fig,ax=plt.subplots(figsize=(10,3.1));ax.set_xlim(0,10);ax.set_ylim(0,4.1);ax.axis('off')
    ax.text(0,3.88,'Prediction at the first outage hour t',fontsize=11,fontweight='bold')
    labels=['Earlier history\n(t minus 24 to t minus 1)','Current hour\n(t)','Future hours']
    for x,label in zip([3.1,5.9,8.6],labels):ax.text(x,3.15,label,ha='center',va='center',fontsize=9)
    ax.text(0,2.5,'Target station',va='center',fontsize=10)
    ax.text(0,1.78,'Other stations',va='center',fontsize=10)
    for y in [2.18,1.46]:
        box(ax,1.7,y,2.8,.59,'Observed history',BLUE,9)
        box(ax,7.25,y,2.7,.59,'Excluded from causal input',GREY,8.5)
    box(ax,4.72,2.18,2.35,.59,'Hidden target',ORANGE,9)
    box(ax,4.72,1.46,2.35,.59,'Available readings',BLUE,9)
    ax.plot([7.16,7.16],[1.35,3.5],color=GREY,ls='--',lw=1)
    box(ax,.1,.1,3.05,.78,'Training-fitted graph\nand within-station prior',TEAL,9.5)
    box(ax,3.7,.1,2.65,.78,'Combine available data\ninto an estimate',BLUE,9.5)
    box(ax,6.9,.1,3,.78,'Decision bounds + queries\nwhen ambiguity remains',ORANGE,9.5)
    arrow(ax,(3.2,.49),(3.64,.49));arrow(ax,(6.4,.49),(6.85,.49))
    save(fig,'causal_workflow')

                                                                         
    eta=Q(1);radius=abs(Q(0)-Q(10))/2-eta
    assert radius==4 and Q(39,10)<radius
    assert Q(0)+(radius+eta)==Q(10)-(radius+eta)==5
    assert max(Q(0),Q(4))<=5 and max(Q(4),Q(6))>5
    fig,axes=plt.subplots(1,2,figsize=(10,3.9),gridspec_kw={'width_ratios':[1,1.35]})
    ax=axes[0];ax.set_xlim(-.4,3.8);ax.set_ylim(-.1,3.2);ax.axis('off')
    ax.set_title('A. Query the station that separates both pairs',loc='left',fontsize=10,fontweight='bold')
    for y,label in [(2.3,'Opposite pair 1: separating set {A, B}'),(1.55,'Opposite pair 2: separating set {B, C}')]:
        ax.text(1.65,y+.42,label,ha='center',fontsize=8,zorder=4,bbox=dict(facecolor='white',edgecolor='none',pad=1))
    for x,label,color in [(.45,'A',BLUE),(1.65,'B',ORANGE),(2.85,'C',BLUE)]:
        ax.add_patch(Circle((x,.65),.23,facecolor='white',edgecolor=color,lw=2))
        ax.text(x,.65,label,ha='center',va='center',color=color,fontweight='bold')
    for xs,y in [([.45,1.65],2.3),([1.65,2.85],1.55)]:
        ax.plot(xs,[y,y],color=TEAL,lw=2)
        for x in xs:ax.plot([x,x],[.95,y],color=TEAL,lw=1.2)
    ax.text(1.65,.05,'No query faults: B alone resolves both.\nOne query fault: each pair needs 3 separators.',ha='center',va='top',fontsize=8.5)
    ax=axes[1];ax.set_title('B. State uncertainty enlarges reading intervals',loc='left',fontsize=10,fontweight='bold')
    for y,rho in zip([2,1,0],[0,3.9,4]):
        width=float(Q(str(rho))+eta)
        for centre,color,offset in [(0,BLUE,.10),(10,ORANGE,-.10)]:
            lo,hi=centre-width,centre+width
            ax.plot([lo,hi],[y+offset,y+offset],color=color,lw=3,solid_capstyle='butt')
            ax.plot([lo,hi],[y+offset,y+offset],ls='',marker='|',color=color,ms=9)
        if rho==4:
            ax.plot([5,5],[y-.10,y+.10],color='#111111',lw=1,zorder=3)
            ax.plot(5,y,'o',color='#111111',ms=5,zorder=4)
    ax.set_xlim(-6,16);ax.set_ylim(-.6,2.65)
    ax.set_yticks([2,1,0],['rho = 0','rho = 3.9','rho = 4'])
    ax.set_xlabel('Possible second-station reading (illustrative units)',fontsize=8.5)
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,labelsize=9)
    ax.text(5,2.53,'Blue centre 0; orange centre 10; clean error 1.',ha='center',fontsize=8)
    ax.text(5,-.48,'At rho = 4, both explanations can report 5.',ha='center',fontsize=8)
    fig.subplots_adjust(wspace=.27,bottom=.19);save(fig,'repair_intuition')

    report={'status':'passed','scope':'Explanatory diagrams; only graph topology/IDs are data-derived. Outage display, workflow and repair examples are illustrative, not additional measured results.',
            'network':{'stations':station_ids,'channels':metadata['channels'],'edges':edges,'operator_shape':list(D.shape),'illustrative_hidden_station':station_ids[target],
                       'model_sha256':sha(folder/'models.pkl'),'metadata_sha256':sha(folder/'dataset_manifest.json')},
            'worked_example':{'reading_error':1,'separation_radius':4,'semantic_margin':5,'boundary_shared_reading':5},
            'generator_sha256':sha(Path(__file__)),
            'figures':{name:sha(OUT/(name+'.pdf')) for name in ['network_structure','causal_workflow','repair_intuition']}}
    (WORK/'Guides/DIAGRAM_CHECK.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Three vector diagrams generated; saved network topology and exact illustrative interval checks passed.')

if __name__=='__main__':
    main()
