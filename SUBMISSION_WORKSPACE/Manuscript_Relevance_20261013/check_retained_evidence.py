                                                                                    
from pathlib import Path
import json,re,ast,hashlib,math
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
tex=(OUT/'SheafCode_Focused_Manuscript.tex').read_text(encoding='utf-8')
compact=' '.join(tex.split());checks=[];inputs={}
def track(p):
    p=ROOT/p;inputs[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def js(p):return json.loads(track(p).read_text(encoding='utf-8'))
def frame(p):return pd.read_csv(track(p))
def check(name,value,detail=None):
    checks.append({'check':name,'passed':bool(value),'detail':detail})
    if not value:print('FAILED:',name,detail)
def present(name,phrase):check(name,' '.join(phrase.split()) in compact,phrase)
def close(name,a,b,tol=1e-9):check(name,np.isclose(a,b,atol=tol,rtol=tol),{'computed':float(a),'reported':float(b)})
def gains(f,a,b,value='mse'):
    p=f.pivot(index=['day','station'],columns='method',values=value)[[a,b]].dropna()
    return 100*(1-np.sqrt(p[a].sum()/p[b].sum())),len(p)

originals=[('India','Research_Release/recent_data_results/delhi'),('US','Research_Release/recent_data_results/epa_la'),('UK','Research_Release/uk_validation_results/uk_aurn')]
for region,folder in originals:
    summary=js(folder+'/reconstruction_summary.json');f=frame(folder+'/outage_cases.csv')
    dataset=track(folder+'/dataset.npz')
    with np.load(dataset) as d:
        check(region+'/original_start',str(d['times'][0]).startswith('2026-01-01'))
        lengths=[int((d['split']==k).sum()) for k in [0,1,3,2]]
        present(region+'/split_lengths',' & '.join(map(str,lengths)))
    primary=f[f.primary.astype(str)=='True']
    for method,record in summary['primary_every_third_day'].items():
        if method in ['pm25_physical','retrospective_vs_causal']:continue
        gain,n=gains(primary,'sheaf_causal',method)
        close(region+'/'+method+'/raw_primary_gain',gain,record['gain_percent'])
        check(region+'/'+method+'/primary_cases',n==record['cases'])
    gain,n=gains(primary,'sheaf_causal','var_causal','pm25_raw_mse')
    close(region+'/physical_pm25',gain,summary['primary_every_third_day']['pm25_physical']['var_causal']['gain_percent'])
    gain,n=gains(f,'sheaf_causal','var_causal');rec=summary['all_eligible_test_days']['var_causal']
    close(region+'/raw_all_day_gain',gain,rec['gain_percent']);check(region+'/all_day_cases',n==rec['cases'])
    present(region+'/headline',f"{summary['primary_every_third_day']['var_causal']['gain_percent']:.2f}")
                                                               
    if region!='UK':
        faults=frame(folder+'/available_fault_summary.csv')
        for r in faults.itertuples():
            if r.true_count in [1,2]:present(region+'/localization/'+r.kind+'/'+str(r.true_count),f'{100*r.exact_support:.1f}')
        grid=frame(folder+'/history_horizon_context_cases.csv')
        for key,g in grid.groupby(['history','horizon','context_loss']):
            gain,n=gains(g,'sheaf_causal','var_causal')
            displayed=f'{abs(gain):.1f}'
            present(region+'/history_horizon/'+str(key),displayed)

external=frame('Research_Release/expanded_validation_results/COHORT_RESULTS.csv')
rf=frame('Research_Release/weakness_resolution_results/NONLINEAR_SUMMARY.csv').set_index('cohort')
ext=external[external.provider!='OpenAQ']
for col,total in [('cases',2114),('stations',73),('observed_pollutant_hour_cells',549302)]:check('external/'+col,int(ext[col].sum())==total)
check('external/VAR_wins',int((ext.gain>0).sum())==13)
check('external/RF_wins',sum(float(rf.loc[r.cohort,'gain'])>0 for r in ext.itertuples())==18)
for r in ext.itertuples():
    other=rf.loc[r.cohort];ci=ast.literal_eval(other.ci7) if pd.notna(other.ci7) else []
    varci=f'[{r.ci_low:.2f}, {r.ci_high:.2f}]' if np.isfinite(r.ci_low) else '---'
    rfci=f'[{ci[0]:.2f}, {ci[1]:.2f}]' if ci else '---'
    row=f'{r.cases} & {r.gain:.2f} {varci} & {other.gain:.2f} {rfci} & {r.pm25_gain:.2f}'
    present('external/row/'+r.cohort,row)

neural=frame('Research_Release/weakness_resolution_results/NEURAL_SUMMARY.csv')
for r in neural.itertuples():
    ci=ast.literal_eval(r.ci7);present('GRIN/'+r.code+'/'+str(r.seed),f'{r.code} & {r.seed} & {r.cases} & {r.gain:.2f} & {ci[0]:.2f}--{ci[1]:.2f}')
f=frame('Research_Release/recent_data_results/delhi/outage_cases.csv');g=frame('Research_Release/recent_data_results/delhi/grin_cases.csv')
for seed,rows in g.groupby('seed'):
    one=rows.copy();one['method']='grin'
    combined=pd.concat([f[f.method=='sheaf_retrospective'],one],ignore_index=True)
    gain,n=gains(combined,'sheaf_retrospective','grin');present('GRIN/India/'+str(seed),f'A & {seed} & {n} & {gain:.2f}')

ablation=js('Research_Release/recent_data_results/delhi/ablation_summary.json')
present('identity_ablation',f"{abs(ablation['identity_spatial_maps']['gain_percent']):.2f}")
intervals=frame('Research_Release/weakness_resolution_results/RISK_INTERVAL_AGGREGATES.csv')
for r in intervals[(intervals.loss==0)&intervals.method.isin(['station_fixed','risk_station'])].itertuples():
    label='All' if r.group=='all' else 'Extreme';method='Fixed' if r.method=='station_fixed' else 'Risk grouped'
    present('interval/'+r.group+'/'+r.method,f'{label} & {method} & {r.cells} & {100*r.macro_coverage:.1f} & {100*r.pooled_coverage:.1f} & {r.macro_finite_width:.2f} & {100*r.full_fraction:.3f}')
for folder in ['delhi','epa_la']:
    f=frame('Research_Release/recent_data_results/'+folder+'/uncertainty_summary.csv')
                                                                                    
    method='pooled_fixed' if folder=='delhi' else 'station_fixed'
    rec=f[(f.context_loss==0)&(f.method==method)&(f.group=='all')].iloc[0]
    present('original_interval/'+folder,f'{100*rec.coverage:.1f}')

configs=frame('Research_Release/decision_repair_results/REPAIR_CONFIGURATIONS.csv')
check('repair/414_total',len(configs)==414)
check('repair/90_empty',int((configs.sampled_scenarios==0).sum())==90)
check('repair/18_nonempty_banks',len(configs[configs.sampled_scenarios>0].cohort.unique())==18)
for f in [0,1]:
    active=configs[(configs.query_faults==f)&(configs.safe>0)&(configs.exceedance>0)]
    good=active[active.decision_status=='optimal'];both=good[good.pattern_status=='optimal']
    row=f'{f} & {len(active)} & {len(good)} & {int((good.pattern_status=="infeasible").sum())} & {len(both)} & {int(both.decision_queries.sum())}/{int(both.pattern_queries.sum())}'
    present('repair/table/'+str(f),row)
adapt=js('Research_Release/decision_repair_results/ADAPTIVE_SUMMARY.json')
radius=js('Research_Release/decision_repair_results/RADIUS_SUMMARY.json')
for f in ['0','1']:
    rec=adapt['summaries'][f];check('adaptive/'+f,rec['search_limit']==0)
    check('radius/'+f,radius['summaries'][f]['feasible']==rec['optimal'])

joint=frame('Research_Release/joint_certification_results/CASES.csv')
check('joint/272',len(joint)==272);check('joint/17_roles',joint.cohort.nunique()==17)
check('joint/two_variants',joint.groupby(['cohort','timestamp']).size().eq(2).all())
check('joint/variant_counts',joint.scenario.value_counts().to_dict()=={'clean':136,'unit_shift':136})
check('joint/40_positive_abstentions',int(joint.positive_truth.sum())==40 and joint[joint.positive_truth].decision.eq('abstain').all())
check('joint/155_issued',int(joint.issued.sum())==155 and int(joint.wrong.sum())==0)
mathfacts=js('Research_Release/math_integration_results/REPORT_FACTS.json')
check('math/1032_rank_checks',mathfacts['anchor_subset_checks']==1032 and mathfacts['anchor_rank_disagreements']==0)

ops=js('Research_Release/operational_validation_results/SUMMARY.json')['cohorts']
ops=[c for c in ops if c['status']=='complete']
check('operational/18_roles',len(ops)==18)
check('operational/box_members',sum(c['test_box_members'] for c in ops)==10292)
check('operational/test_vectors',sum(c['complete_hours']['2'] for c in ops)==13187)
for f in ['0','1']:
    totals={k:sum(c['metrics'][f]['direct'][k] for c in ops) for k in ['trials','issued','wrong','abstained']}
    present('guard/table/'+f,f"{f} & {totals['trials']:,} & {totals['issued']:,} / {totals['wrong']}")
    check('guard/denominators/'+f,totals['issued']+totals['abstained']==totals['trials'])
    mean_queries=sum(c['metrics'][f]['direct']['mean_queries']*c['metrics'][f]['direct']['trials'] for c in ops)
    full_queries=sum(c['metrics'][f]['direct']['trials']*c['stations'] for c in ops)
    present('guard/saving/'+f,f'{100*(1-mean_queries/full_queries):.2f}')
    for model in ['box']:
        present('archive_wrong/'+f,f"{sum(c['metrics'][f][model]['wrong'] for c in ops):,}")
query=js('Research_Release/query_policy_results/SUMMARY.json')['cohorts']
for f in [0,1]:
    rows=[r for r in query if r['faults']==f]
    box=sum(r['box_first'] for r in rows);freq=sum(r['training'] for r in rows);val=sum(r['validated'] for r in rows)
    present('ordering/simple_query_difference/'+str(f),str(box-freq))
    present('ordering/validation_gain/'+str(f),f'{100*(1-val/freq):.3f}')

alerts=frame('Research_Release/recent_data_results/delhi/alert_summary.csv')
for method in ['sheaf_causal','var_causal']:
    r=alerts[alerts.method==method].iloc[0]
    present('alerts/'+method+'/sensitivity',f'{100*r.sensitivity:.1f}')
    present('alerts/'+method+'/false_alarm',f'{100*r.false_alarm_rate:.1f}')
    check('alerts/counts/'+method,(r.positive_cases,r.negative_cases)==(1177,4012))
mask=js('Research_Release/recent_data_results/delhi/matched_missing_mask_summary.json')
check('missing_masks/counts',mask['cases']==719 and mask['source_patterns']==155)
present('missing_masks/gain',f"{mask['gain_percent']:.2f}")
daily=frame('Research_Release/recent_data_results/delhi/pm25_daily_summary.csv')
for method in ['sheaf_causal','city_mean']:
    r=daily[daily.method==method].iloc[0];check('daily/'+method,r.common_station_days==190 and np.isclose(r.category_accuracy,140/190) and r.missed_poor_rate==1)

                                                  
graph=js('SUBMISSION_WORKSPACE/IRIS_Video_Kit_Environmental_Focus/GRAPH_INFLUENCE.json')
check('graph/actual_edges',len(graph['edges'])==9 and ['8235','10485'] in graph['edges'] and ['17','8235'] not in graph['edges'])
check('counterexample/shared_weights_fail',2/1.5>1 and .5*2/1.5<=1)
labels=re.findall(r'\\label\{([^}]+)\}',tex);refs=re.findall(r'\\(?:ref|eqref|pageref)\{([^}]+)\}',tex)
check('TeX/unique_labels',len(labels)==len(set(labels)));check('TeX/defined_references',set(refs)<=set(labels))
entries=re.findall(r'\\bibitem[\s\S]*?(?=\n\\bibitem|\n\\end\{thebibliography\})',tex)
keys={re.search(r'\]\{([^}]+)\}',e)[1] for e in entries}
cited=set(','.join(re.findall(r'\\cite\w*\{([^}]+)\}',tex)).split(','))
check('TeX/defined_citations',cited<=keys);check('TeX/no_unused_entries',keys<=cited)
check('TeX/proof_count',tex.count(r'\begin{proposition}')==6 and tex.count(r'\begin{theorem}')==1 and tex.count(r'\begin{proof}')==7)
report={'status':'passed' if all(c['passed'] for c in checks) else 'failed','checks':checks,'input_sha256':inputs,'check_count':len(checks),
    'scope':'Reaggregation and displayed-value checks for the saved current benchmark and main follow-up evidence. Separate exact mathematical replays are logged. This does not refit models or establish physical assumptions or mathematical priority.'}
(OUT/'FOCUSED_EVIDENCE_AUDIT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'status':report['status'],'checks':len(checks),'failures':[c for c in checks if not c['passed']]},indent=2))
if report['status']!='passed':raise SystemExit(1)
