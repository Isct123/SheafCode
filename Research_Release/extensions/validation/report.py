                                                                           
import sys,ast,zipfile,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
os.environ.setdefault('MPLCONFIGDIR',str(ROOT.parent/'audit_tools/mpl_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
def main():
    figures=OUT/'figures';figures.mkdir(exist_ok=True);checks=0;pages=[]
    from render_theory import main as render_theory
    render_theory()
    input_paths={}
    for region in REGIONS:
        m=json.loads((ROOT/'results_expanded'/region/'manifest.json').read_text())
        input_paths.update(m['data_sha256'])
    for name,sha in input_paths.items():
        if digest(ROOT/name)!=sha:raise AssertionError('Frozen archive changed: '+name)
        checks+=1
    required=['hybrid_summary.json','fault_scope.json','refinement_scope.json','operator_scope.json','alert_scope.json']
    scripts=['hybrid.py','faults.py','bounds_study.py','operator_study.py','alerts.py']
    for filename,script in zip(required,scripts):
        if json.loads((OUT/filename).read_text())['source_sha256']!=digest(HERE/script):raise RuntimeError('Stale study: '+script)
        checks+=1
    H=pd.read_csv(OUT/'hybrid_cases.csv');hour=pd.read_csv(OUT/'hybrid_hourly.csv');expected={}
    for region in REGIONS:
        ds,m,folder=cohort(region);ds.set_transform('none')
        lookup={(s,p):ds.std[v,q] for v,s in enumerate(ds.stations) for q,p in enumerate(ds.pollutants)}
        g=hour[hour.region==region].copy();g['error']=[((r.prediction-r.truth)/lookup[(r.station,r.pollutant)])**2 for r in g.itertuples()]
        calc=g.groupby(['day','station','method']).error.mean();saved=H[H.region==region].set_index(['day','station','method']).mse
        if not np.allclose(calc.sort_index(),saved.sort_index(),rtol=1e-8,atol=1e-10):raise AssertionError('Hybrid per-cell arithmetic drift')
        checks+=len(calc)
    F=pd.read_csv(OUT/'fault_cases.csv')
    correct=np.array([ast.literal_eval(r.fault_support)==ast.literal_eval(r.predicted_support) for r in F.itertuples()])
    if not np.array_equal(correct,F.exact.to_numpy(bool)):raise AssertionError('Fault exact-support arithmetic drift')
    checks+=len(F)
    summary=F.groupby(['region','fault_count','direction','magnitude','method']).exact.mean()
    saved=pd.read_csv(OUT/'fault_summary.csv').set_index(['region','fault_count','direction','magnitude','method']).exact_end_to_end
    if not np.allclose(summary.sort_index(),saved.sort_index()):raise AssertionError('Fault summary drift')
    A=pd.read_csv(OUT/'alert_cases.csv');S=pd.read_csv(OUT/'alert_summary.csv')
    for r in S.itertuples():
        g=A[(A.region==r.region)&(A.horizon==r.horizon)&(A.missing_rate==r.missing_rate)&(A.method==r.method)];pos=g[g.positive];neg=g[~g.positive]
        if len(g)!=r.cases or len(pos)!=r.positive_cases or len(neg)!=r.negative_cases:raise AssertionError('Alert sample count mismatch')
        if len(pos) and not np.isclose(pos.alert.mean(),r.sensitivity):raise AssertionError('Alert sensitivity mismatch')
        if len(neg) and not np.isclose(neg.alert.mean(),r.false_positive_rate):raise AssertionError('Alert FPR mismatch')
        checks+=1
    refine=pd.read_csv(OUT/'refinement_cases.csv');op=pd.read_csv(OUT/'operator_uncertainty_cases.csv')
    if (refine.width+1e-6<refine.exact_width).any() or not op.robust_covered.all():raise AssertionError('Conditional certificate checks failed')
    checks+=len(refine)+len(op)
    external=json.loads((OUT/'external_summary.json').read_text()) if (OUT/'external_summary.json').exists() else {'status':'awaiting_compatible_observations'}
    if external['status']=='evaluated':
        lock=json.loads((OUT/'external/evaluation_lock.json').read_text())
        for filename,sha in [('observations.csv',lock['data_sha256'])]:
            if digest(OUT/'external'/filename)!=sha:raise AssertionError('External input changed after lock')
        if lock['source_sha256']!=digest(HERE/'external_evaluate.py'):raise AssertionError('External source changed')
        e=pd.read_csv(OUT/'external/cases.csv');p=e.pivot(index=['day','station'],columns='method',values='mse')
        for method,r in external['comparison'].items():
            gain=100*(1-np.sqrt(p.temporal_sheaf.mean()/p[method].mean()))
            if not np.isclose(gain,r['rmse_improvement_percent']):raise AssertionError('External headline arithmetic drift')
            checks+=1
    graph={'status':'pending'}
    if (OUT/'graph_summary.json').exists():
        graph=json.loads((OUT/'graph_summary.json').read_text())
        if graph['source_sha256']!=digest(HERE/'graph_benchmark.py'):raise AssertionError('Graph source drift')
        gf=pd.read_csv(OUT/'graph_cases.csv')
        for region,seeds in graph['comparisons'].items():
            ref=pd.read_csv(ROOT/'extension_results/strong_baselines'/f'{region}_cases.csv');ref=ref[ref['mode']=='retrospective'].copy();ref['method']=ref.method.replace({'seasonal_persistence':'seasonal'})
            for seed,comparisons in seeds.items():
                f=pd.concat([gf[(gf.region==region)&(gf.seed==int(seed))],ref]);p=f.pivot(index=['day','station'],columns='method',values='mse')
                for method,r in comparisons.items():
                    common=p[['official_grin',method]].dropna();gain=100*(1-np.sqrt(common.official_grin.mean()/common[method].mean()))
                    if not np.isclose(gain,r['rmse_improvement_percent']):raise AssertionError('Graph summary drift')
                    checks+=1
        graph['status']='complete'
    def figure(name,fig):
        fig.tight_layout();fig.savefig(figures/(name+'.pdf'));fig.savefig(figures/(name+'.png'),dpi=200);pages.append(fig)
    hybrid=json.loads((OUT/'hybrid_summary.json').read_text());fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,region in zip(axes,REGIONS):
        methods=['temporal_sheaf','full_covariance_var','seasonal'];data=[hybrid['comparisons'][region][method] for method in methods];values=np.array([r['rmse_improvement_percent'] for r in data]);ci=np.array([r['paired_day_ci'] for r in data])
        ax.errorbar(values,np.arange(3),xerr=np.maximum(0,np.c_[values-ci[:,0],ci[:,1]-values].T),fmt='o',capsize=4);ax.axvline(0,color='grey');ax.set_yticks(range(3),['Sheaf','Full-covariance VAR','Seasonal']);ax.set_xlabel('Hybrid normalized RMSE improvement (%)');ax.set_title(region+'; '+str(data[0]['cases'])+' paired cases')
    fig.suptitle('Validation-selected causal hybrid; exploratory day-bootstrap intervals');figure('01_hybrid',fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,region in zip(axes,REGIONS):
        g=pd.read_csv(OUT/'fault_summary.csv');g=g[(g.region==region)&(g.fault_count==2)&(g.magnitude==2)];directions=['positive','negative','random','unequal','gradual'];xx=np.arange(5)
        for offset,method in [(-.18,'calibrated_classifier'),(.18,'full_covariance_penalized_ls')]:
            q=g[g.method==method].set_index('direction').reindex(directions);ax.bar(xx+offset,100*q.exact_end_to_end,.36,label=method.replace('_',' '))
        ax.set_xticks(xx,directions,rotation=30);ax.set_ylim(0,100);ax.set_ylabel('Unknown-count exact support recovery (%)');ax.set_title(region+'; '+str(int(g.cases.iloc[0]))+' hours per scenario');ax.legend(fontsize=7)
    fig.suptitle('Injected two-fault magnitude-2 scenarios; unequal/ramp amplitudes differ');figure('02_unknown_count_faults',fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,region in zip(axes,REGIONS):
        for method in ['sheaf','city_mean']:
            g=S[(S.region==region)&(S.method==method)];ax.scatter(100*g.false_positive_rate,100*g.sensitivity,label=method)
        ax.set_xlabel('Actual test false-positive rate (%)');ax.set_ylabel('Hourly threshold sensitivity (%)');ax.set_ylim(0,100);ax.set_title(region);ax.legend()
    fig.suptitle('Validation-frozen cutoffs; all 24/72/168h context-loss scenarios');figure('03_frozen_alerts',fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    r=pd.read_csv(OUT/'refinement_summary.csv');o=pd.read_csv(OUT/'operator_uncertainty_summary.csv')
    for region in REGIONS:
        g=r[r.region==region];axes[0].plot(g.node_budget,g.median_width,'o-',label=region);g=o[o.region==region];axes[1].plot(100*g.relative_entry_radius,g.median_width,'o-',label=region)
    axes[0].set_xscale('log');axes[0].set_xlabel('Maximum LP nodes per endpoint');axes[0].set_ylabel('Median standardized quantity interval width');axes[0].set_title('Support refinement');axes[1].set_xlabel('Declared entry perturbation radius (%)');axes[1].set_ylabel('Median standardized interval width');axes[1].set_title('Operator uncertainty');axes[0].legend();axes[1].legend();fig.suptitle('Synthetic feasible states on actual learned maps; conditional bounds');figure('04_certificate_refinement',fig)
    if external['status']=='evaluated':
        fig,ax=plt.subplots(figsize=(8,4));methods=list(external['comparison']);r=[external['comparison'][m] for m in methods];v=np.array([x['rmse_improvement_percent'] for x in r]);ci=np.array([x['paired_day_ci'] for x in r]);ax.errorbar(v,np.arange(len(v)),xerr=np.maximum(0,np.c_[v-ci[:,0],ci[:,1]-v].T),fmt='o',capsize=4);ax.axvline(0,color='grey');ax.set_yticks(range(len(v)),[m.replace('_',' ') for m in methods]);ax.set_xlabel('Temporal sheaf normalized PM2.5 RMSE improvement (%)');ax.set_title(f"New same-network time holdout: {external['cases']} station-days, {external['days']} days, PM-only inputs");figure('05_new_period',fig)
    if graph['status']=='complete':
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,region in zip(axes,REGIONS):
            for seed,comps in graph['comparisons'][region].items():
                r=comps['temporal_sheaf'];value=r['rmse_improvement_percent'];ci=r['paired_day_ci'];ax.errorbar(value,int(seed),xerr=[[max(0,value-ci[0])],[max(0,ci[1]-value)]],fmt='o',capsize=3)
            ax.axvline(0,color='grey');ax.set_xlabel('Official GRIN RMSE improvement over sheaf (%)');ax.set_ylabel('Training seed');ax.set_title(region)
        fig.suptitle('Retrospective matched outages; official architecture, project training');figure('06_official_grin',fig)
    with PdfPages(OUT/'Validation_Figures.pdf') as pdf:
        for fig in pages:pdf.savefig(fig);plt.close(fig)
    lines=['# Supplemental technical validation','',f'Case/provenance arithmetic checks: {checks}. Official graph benchmark status: **{graph["status"]}**.','',
        'The completed release remains the primary historical evidence. These added studies preserve negative findings and protocol distinctions. No first-ever priority, emissions reduction, health benefit or official AQI/GRAP decision is asserted.','',
        '## Causal hybrid','',json.dumps(hybrid['comparisons'],indent=2),'',
        'Delhi blending degraded performance against the original sheaf. Bengaluru blending improved performance against the tested components. Weights were selected on validation; test archives had previously been examined. These are exploratory paired calendar-day intervals.','',
        '## Unknown-count faults','',pd.read_csv(OUT/'fault_summary.csv').to_string(index=False),'',
        'Faults are synthetic shifts in transformed standardized channels. The classifier learns from synthetic validation faults, searches all eligible stations and receives no true fault count at test time. Retain every magnitude/direction and clean false alarm rate. Paired calendar-week comparisons are in fault_paired_week_intervals.csv. Separate contiguous ramp episodes retain misses, exact supports and timestamps.','',
        '## Pollution alerts','',S.to_string(index=False),'',
        'Thresholds were frozen on validation negatives. The full-stream weekly intervals include zero for several sensitivity differences despite numerical gains. Case counts are pollutant hours, not independent observations. Episode delays are conditional on detection; missing truth breaks episodes and boundary truncations are counted. Persistent context-channel loss differs from independent cell masking.','',
        '## Later-period frozen-model evaluation','',json.dumps(external,indent=2),'',
        'Public OpenAQ data sources, geographic mapping, rejected units, averaged duplicates and hashes are retained under external/. The new period uses the same instruments and partial network, with PM-only inputs. Prior user access cannot be independently verified. No parameters or alert cutoffs were tuned on these outcomes.','',
        '## Mathematical/computational additions','',
        'REFINEMENT_THEOREM.md proves anytime outer validity and finite exactness under exhaustive support branching. OPERATOR_UNCERTAINTY_THEOREM.md proves containment after explicit residual inflation for bounded map errors. Both specialize established mathematics; numerical tests do not establish historical priority or independent proof review.','',
        r.to_string(index=False) if isinstance(r,pd.DataFrame) else pd.read_csv(OUT/'refinement_summary.csv').to_string(index=False),'',o.to_string(index=False),'',
        'Synthetic feasible states and declared perturbation radii are separate from observed environmental effectiveness. Wider robust bounds are a cost of protection; tighter optimization does not eliminate model/assumption misspecification.','',
        '## Published graph comparator','',json.dumps(graph,indent=2),'',
        'The pinned official architecture is bidirectional. Its outcomes are compared against retrospective methods only. Project-specific outage training and validation selection do not reproduce the original paper benchmark scores. Seed-level outcomes must all be retained.','',
        '## Remaining evidence boundaries','',
        'The later period is short and partially observed. Field deployment, independent-instrument replication, physical map-error bounds and measured pollution/exposure reduction remain unestablished. Internal follow-ups and their many nominal intervals are exploratory. Mathematical priority needs a focused literature review and independent proof assessment.','',
        'The root Run_Validation_Research.cmd finishes the configured multi-seed graph study and automatically refreshes this report, figures and archive. Current graph checkpoints resume after interruption.']
    (OUT/'TECHNICAL_VALIDATION_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    manifest={'status':'complete' if graph['status']=='complete' else 'non_neural_complete_graph_pending','arithmetic_checks':checks,'external_status':external['status'],'source_sha256':{p.name:digest(p) for p in HERE.glob('*.py')},'completed_result_sha256':{str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.csv','.pdf','.png','.md']},'scope':'Supplemental audit; preserves earlier results; graph pending is never represented as complete'}
    save('audit.json',manifest)
    with zipfile.ZipFile(OUT/'Supplemental_Research_Results.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in OUT.rglob('*'):
            if p.is_file() and p.suffix not in ['.zip','.pt','.tmp']:archive.write(p,'results/'+str(p.relative_to(OUT)))
        for p in HERE.glob('*'):
            if p.is_file() and p.suffix in ['.py','.md','.ps1']:archive.write(p,'source/'+p.name)
        archive.write(HERE/'vendor/grin/SOURCE.json','source/official_grin_SOURCE.json')
        primary=ROOT/'Final_Research_Results.zip'
        if primary.exists():archive.write(primary,'primary_evidence/Final_Research_Results.zip')
    print('Supplemental report:',manifest['status'],checks,'arithmetic checks',flush=True)
if __name__=='__main__':main()
