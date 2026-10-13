                                                                                   
import os, sys, json, hashlib, zipfile, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'joint_decision_results'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(ROOT/'extensions/validation'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT.parent/'audit_tools/mpl_cache'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from joint import joint_envelope
from result_sensitivity import pair_cases,cluster_sensitivity,leave_cluster_out


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_frozen():
    audit=json.loads((ROOT/'validation_results/audit.json').read_text())
    if audit['status']!='complete': raise RuntimeError('Earlier supplemental suite is not complete')
    for name,digest in audit['source_sha256'].items():
        if sha(ROOT/'extensions/validation'/name)!=digest: raise RuntimeError('Frozen source changed: '+name)
    for name,digest in audit['completed_result_sha256'].items():
        if sha(ROOT/'validation_results'/name)!=digest: raise RuntimeError('Frozen result changed: '+name)


def main():
    verify_frozen()
    config={'random_seed':719,'synthetic_coupled_cases':50,'learned_map_synthetic_cases_per_region':12,
            'comparisons':10,'cluster_days':[1,7,14],'bootstrap_draws':20000,
            'scope':'Frozen-result sensitivity; no new fitting or test-driven selection. Joint-decision cases are synthetic, not field effectiveness.'}
    (OUT/'PROTOCOL.json').write_text(json.dumps(config,indent=2))
    input_hashes={};checks=0;rows=[];rng=np.random.default_rng(719)
                                                                             
                                                                          
    for i in range(50):
        total=float(rng.uniform(1.5,3));threshold=.375*total
        r=joint_envelope([[1,1]],[-total],np.eye(2)/threshold,[0,0],[],0,0,0,[0,0],[total,total])
        assert abs(r['lower']-4/3)<1e-7 and abs(r['independent_lower'])<1e-7
        rows.append({'region':'coupled_example','case':i,'joint_lower':r['lower'],
                     'independent_lower':r['independent_lower'],'upper':r['upper'],
                     'decision':r['decision'],'seconds':None,'scope':'Synthetic coupled state; no observed pollution outcome'})
        checks+=2
    from common import cohort
    witnesses=[]
    for region in ['delhi2025','bengaluru_raw']:
        ds,m,folder=cohort(region)
        D=np.load(folder/'model.npz')['delta'];input_hashes[str((folder/'model.npz').relative_to(ROOT))]=sha(folder/'model.npz')
        channels=['PM2.5','PM10' if 'PM10' in ds.pollutants else 'NO2']
        H=np.eye(ds.p)[[ds.pollutants.index(channel) for channel in channels]]
        for i in range(12):
            v=i%ds.n;U=np.arange(v*ds.p,(v+1)*ds.p);A=D[:,U]
            blocks=[D[:,j*ds.p:(j+1)*ds.p] for j in range(ds.n) if j!=v]
            z=rng.uniform(-1,1,ds.p);e=rng.uniform(-1,1,ds.p)
            b=-A@z-blocks[i%len(blocks)]@e+rng.uniform(-.03,.03,len(D))
                                                                                
            offsets=np.ones(2)
            start=time.perf_counter()
            r=joint_envelope(A,b,H,offsets,blocks,1,.03,1,[-3]*ds.p,[3]*ds.p)
            elapsed=time.perf_counter()-start;truth=float(np.max(H@z+offsets))
            assert r['lower']-1e-6<=truth<=r['upper']+1e-6
            assert r['lower']>=r['independent_lower']-1e-6
            checks+=2
            for end in ['lower','upper']:
                w=r[end+'_witness'];x=np.array(w['state']);fault=np.array(w['fault'])
                B=np.concatenate([blocks[j] for j in w['support']],axis=1) if w['support'] else np.empty((len(D),0))
                assert np.max(abs(A@x+b+B@fault))<=.03+1e-6
                assert abs(float(np.max(H@x+offsets))-r[end])<1e-6
                checks+=2
            rows.append({'region':region,'channels':','.join(channels),'case':i,'joint_lower':r['lower'],
                         'independent_lower':r['independent_lower'],'upper':r['upper'],
                         'decision':r['decision'],'quantity_truth':truth,'seconds':elapsed,
                         'scope':'Actual learned maps, synthetic feasible states and faults'})
            witnesses.append({'region':region,'case':i,'envelope':r})
    pd.DataFrame(rows).to_csv(OUT/'joint_cases.csv',index=False)
    (OUT/'joint_witnesses.json').write_text(json.dumps(witnesses,indent=2))

    comparisons=[];seed_rows=[]
    def load(path):
        input_hashes[str(path.relative_to(ROOT))]=sha(path)
        return pd.read_csv(path)
    def add(label,region,mode,frame,method,reference):
        comparisons.append((label,region,mode,pair_cases(frame,method,reference)))
    G=load(ROOT/'validation_results/graph_cases.csv')
    for region in ['delhi2025','bengaluru_raw']:
        B=load(ROOT/'extension_results/strong_baselines'/f'{region}_cases.csv')
        add('Causal sheaf vs VAR',region,'causal',B[B['mode']=='causal'],'temporal_sheaf','full_covariance_var')
        hybrid=load(ROOT/'validation_results/hybrid_cases.csv')
        add('Hybrid vs sheaf',region,'causal',hybrid[hybrid.region==region],'hybrid','temporal_sheaf')
        if region=='bengaluru_raw':
            add('Hybrid vs seasonal',region,'causal',hybrid[hybrid.region==region],'hybrid','seasonal')
        graph=G[G.region==region]
        if set(graph.seed.unique())!={0,1,2}:raise RuntimeError('Incomplete neural seeds')
        if graph.duplicated(['seed','day','station']).any():raise RuntimeError('Duplicate neural cases')
        all_sets=[set(zip(f.day,f.station)) for _,f in graph.groupby('seed')]
        if not all(s==all_sets[0] for s in all_sets):raise RuntimeError('Seed case-universe mismatch')
        baseline=B[(B['mode']=='retrospective') & (B.method=='temporal_sheaf')]
        average=graph.groupby(['day','station','method'],as_index=False).mse.mean()
        add('GRIN mean loss vs sheaf',region,'retrospective',pd.concat([average,baseline]),'official_grin','temporal_sheaf')
        for seed,f in graph.groupby('seed'):
            p=pair_cases(pd.concat([f[['day','station','method','mse']],baseline]),'official_grin','temporal_sheaf')
            seed_rows.append({'region':region,'seed':int(seed),'station_days':len(p),
                              'grin_gain_percent':float(100*(1-np.sqrt(p.method_mse.sum()/p.reference_mse.sum())))})
    external=load(ROOT/'validation_results/external/cases.csv')
    for method in ['full_covariance_var','seasonal','city_mean']:
        add('Later sheaf vs '+method,'delhi_later','causal',external,'temporal_sheaf',method)
    if len(comparisons)!=10:raise AssertionError('Declared contrast family changed')
    results=[];leave=[]
    for label,region,mode,pair in comparisons:
        for days in [1,7,14]:
            r=cluster_sensitivity(pair,days,family_size=10,draws=20000)
            results.append({'comparison':label,'region':region,'mode':mode,**r})
            assert r['family_sensitivity_low']<=r['nominal_95_low']<=r['nominal_95_high']<=r['family_sensitivity_high']
            checks+=1
        leave.extend({'comparison':label,'region':region,'mode':mode,**r} for r in leave_cluster_out(pair))
    S=pd.DataFrame(results);S.to_csv(OUT/'comparison_sensitivity.csv',index=False)
    pd.DataFrame(leave).to_csv(OUT/'leave_week_out.csv',index=False)
    pd.DataFrame(seed_rows).to_csv(OUT/'neural_seed_spread.csv',index=False)

    with PdfPages(OUT/'Additional_Results_Figures.pdf') as pdf:
        fig,ax=plt.subplots(figsize=(8,5))
        region_labels={'delhi2025':'Delhi learned maps','bengaluru_raw':'Bengaluru learned maps',
                       'coupled_example':'Designed coupled examples'}
        for label,group in pd.DataFrame(rows).groupby('region'):
            ax.scatter(group.independent_lower,group.joint_lower,label=region_labels[label],alpha=.7)
        limits=ax.get_xlim();lo=min(limits[0],ax.get_ylim()[0]);hi=max(limits[1],ax.get_ylim()[1])
        ax.plot([lo,hi],[lo,hi],color='gray',linestyle='--')
        ax.axhline(1,color='gray',linestyle=':',linewidth=.8)
        ax.axvline(1,color='gray',linestyle=':',linewidth=.8)
        ax.set(xlabel='Maximum separate lower score',ylabel='Joint lower score',
               title='Joint thresholds retain pollutant dependence')
        ax.legend();fig.text(.5,.015,'Synthetic feasible states; actual network maps in two groups. Conditional mathematical bounds.',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.04,1,1]);pdf.savefig(fig);fig.savefig(OUT/'joint_thresholds.png',dpi=180);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(15,7),sharey=True)
        display_regions={'delhi2025':'Delhi','bengaluru_raw':'Bengaluru','delhi_later':'Later Delhi period'}
        labels=[f'{display_regions[region]}\n{label.replace("full_covariance_var","VAR").replace("city_mean","city mean")}' for label,region,_,_ in comparisons]
        for ax,days in zip(axes,[1,7,14]):
            part=S[S.cluster_days==days]
            for i,r in enumerate(part.itertuples()):
                color='darkorange' if r.small_cluster_warning else 'navy'
                ax.plot([r.family_sensitivity_low,r.family_sensitivity_high],[i,i],color='lightgray',linewidth=4)
                ax.plot([r.nominal_95_low,r.nominal_95_high],[i,i],color=color,linewidth=2)
                ax.scatter([r.gain_percent],[i],color=color,s=18)
            ax.axvline(0,color='black',linestyle=':');ax.set(title=f'{days}-day calendar clusters',xlabel='RMSE improvement (%)')
        axes[0].set_yticks(range(len(labels)),labels);axes[0].invert_yaxis()
        fig.suptitle('Frozen comparison sensitivity; positive favours the first method')
        fig.text(.5,.015,'Dark: nominal 95%; pale: 10-contrast sensitivity. Orange: fewer than 8 bins. Exploratory; sparse clusters limit inference.',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.05,1,.96]);pdf.savefig(fig);fig.savefig(OUT/'comparison_sensitivity.png',dpi=180);plt.close(fig)
    joint_table=pd.DataFrame(rows).groupby('region').agg(cases=('case','size'),
        extra_joint_certificates=('decision',lambda s:int((s=='exceedance').sum())))
                                                                               
    f=pd.DataFrame(rows)
    f['additional']=(f.joint_lower>1+1e-6)&(f.independent_lower<=1+1e-6)
    joint_table['extra_joint_certificates']=f.groupby('region').additional.sum()
    lines=['# Additional joint-decision and sensitivity results','',
           'Completed on fixed saved results without refitting any predictor. Earlier releases remain intact.','',
           '## Joint multi-pollutant theorem','',
           'The max-affine score has a sharp lower endpoint found by an epigraph LP over each allowed fault support. This can certify that at least one pollutant exceeds a declared threshold even when neither pollutant is individually forced above it. The upper endpoint is already determined by the separate maxima.','',
           joint_table.to_string(),'',
           'The 50 coupled examples deliberately demonstrate this strict improvement; they are not independent observed environmental events. The two learned-network groups use synthetic planted states and faults. Neither learned-network group produced additional exceedance certificates in this study. Any benefit is therefore mathematical and demonstrated on designed examples, not established for actual monitoring decisions or avoided exposure. Delhi channels are PM2.5/PM10; Bengaluru has no PM10 in this cohort and uses PM2.5/NO2. Scores use shifted standardized coordinates, not official raw concentration AQI.','',
           '## Frozen comparison sensitivity','',S.to_string(index=False),'',
           'Intervals resample whole calendar bins and retain every station-day within a selected bin. The 7/14-day analyses avoid treating separated sampled days as consecutive weeks. They still assume clusters can be resampled and are sensitivity analyses, not arbitrary-dependence confidence guarantees. All three cluster scales are shown; none is selected for a favorable outcome. Fewer than eight occupied bins is explicitly flagged.','',
           'The pale bounds use percentile levels 0.05/(2*10) and 1-0.05/(2*10) for the declared ten contrasts. This is a Bonferroni sensitivity calculation, not exact multiple-testing control: bootstrap approximation, temporal dependence and prior test access remain. The three cluster scales are alternative diagnostics, not thirty independent confirmatory tests.','',
           '## Neural training seeds','',pd.DataFrame(seed_rows).to_string(index=False),'',
           'Aggregate GRIN loss averages each station-day MSE across all three seeds before computing RMSE. It estimates the mean loss of these trained models; it is not the error of an ensemble of predictions. Seed minima/maxima are observed spread, not population confidence bounds. No seed was selected by test performance.','',
           '## Freeze boundary','',
           'Freeze the computational release after this audit. Further work requiring a new outcome should be defined before collecting untouched data. Independent proof and priority review, independent-instrument validation and a measured maintenance/intervention pilot remain external evidence opportunities. No first-ever theorem, official AQI decision or actual pollution reduction is established.','',
           'References: Boyd and Vandenberghe, Convex Optimization, https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf; Politis and Romano, The Stationary Bootstrap, https://doi.org/10.1080/01621459.1994.10476870. The implemented fixed calendar-cluster resampling is not their stationary-bootstrap algorithm.']
    (OUT/'ADDITIONAL_RESULTS_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    from render_proof import main as render_proof
    render_proof()
    verify_frozen()
    audit={'status':'complete','case_checks':checks,'input_sha256':input_hashes,
           'source_sha256':{p.name:sha(p) for p in HERE.glob('*') if p.is_file()},
           'result_sha256':{p.name:sha(p) for p in OUT.glob('*') if p.is_file() and p.name not in ['audit.json','Additional_Results.zip']},
           'scope':config['scope']}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    with zipfile.ZipFile(OUT/'Additional_Results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.glob('*'):
            if p.is_file() and p.suffix!='.zip':z.write(p,'results/'+p.name)
        for p in HERE.glob('*'):
            if p.is_file():z.write(p,'source/'+p.name)
    print('Completed:',checks,'checks;',len(rows),'joint cases;',len(S),'sensitivity rows',flush=True)


if __name__=='__main__': main()
