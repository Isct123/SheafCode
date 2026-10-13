                                                                                
import hashlib,json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'matplotlib_cache'))
from sheafcode.reliable import paired_bootstrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'extension_results';DOC=ROOT/'docs';F=DOC/'figures'
S=json.loads((OUT/'strong_baselines/summary.json').read_text())
H=json.loads((OUT/'horizon_stress/summary.json').read_text())
R=json.loads((OUT/'worst_case_repair/summary.json').read_text())
assert S['source_sha256']==hashlib.sha256((HERE/'run.py').read_bytes()).hexdigest()
assert H['source_sha256']==hashlib.sha256((HERE/'horizon_stress.py').read_bytes()).hexdigest()
assert R['source_sha256']==hashlib.sha256((ROOT/'extensions/functional/evaluate_repair.py').read_bytes()).hexdigest()
table=[]
for region,g in S['comparisons'].items():
    cases=pd.read_csv(OUT/'strong_baselines'/(region+'_cases.csv'))
    for mode,rec in g.items():
        sub=cases[cases['mode']==mode]
        t=sub.pivot_table(index=['day','station'],columns='method',values='mse').dropna(subset=['temporal_sheaf','full_covariance_var']).reset_index()
        t['delta']=t.full_covariance_var-t.temporal_sheaf
        t['week']=((pd.to_datetime(t.day)-pd.Timestamp('1970-01-01'))//pd.Timedelta(days=7)).astype(int)
        weeks=t.groupby('week').delta.agg(['sum','count']);rng=np.random.default_rng(399)
        ids=rng.integers(0,len(weeks),(2000,len(weeks)))
        samples=weeks['sum'].to_numpy()[ids].sum(1)/weeks['count'].to_numpy()[ids].sum(1)
        table.append({'region':region,'mode':mode,**rec,'seven_calendar_day_cluster_ci95':np.quantile(samples,[.025,.975]).tolist(),
            'station_cluster_ci95':paired_bootstrap(sub,'temporal_sheaf','full_covariance_var',unit='station')['difference_ci95']})
pd.DataFrame(table).to_csv(DOC/'stronger_baseline_dependence.csv',index=False)
lines=['# Ambitious technical validation: completed extensions','',
 'Code and experiment additions completed 6 October 2026; manuscript unchanged. The expanded national benchmark remains a separate running job. These analyses are internal and exploratory on archives already examined.','',
 '## Stronger comparator','',
 'The comparator includes correlated full process-noise covariance, covariance shrinkage, raw versus log/diurnal transforms, ridge strength and observation noise. Twenty-four configurations per mode are selected on eight validation days and up to six stations. It is a stronger classical state-space comparison, not a neural state-of-the-art claim. Tests verify positive covariance, correlated innovations, preserved observed training cells and causal prefix invariance.','',
 '| Cohort | Mode | Paired cases / days | Sheaf RMSE gain | Day MSE-difference CI | Seven-calendar-day CI |','|---|---|---:|---:|---|---|']
for r in table:
    lines.append(f"| {r['region']} | {r['mode']} | {r['n']} / {r['clusters']} | {100*r['rmse_relative_gain']:.1f}% | {r['difference_ci95']} | {r['seven_calendar_day_cluster_ci95']} |")
lines+=['','Full covariance is selected for Delhi; a partially shrunk full covariance is selected for Bengaluru. Positive results therefore are not merely comparisons to the diagonal-noise model. These supplementary intervals are not new multiplicity-adjusted tests.','',
 '## Outage duration and target-history stress','',
 'Parameters selected for the 24-hour task are transferred unchanged to 72-hour outages and zero-history conditions. Twelve deterministic starting days per raw cohort, all selected stations, checkpoint hour 1 then every six hours. Eligibility requires at least half of full-outage target cells observed. Only the observed checkpoint cells are scored. These gains are checkpoint metrics, not full-outage-hour RMSE and not interchangeable with the main table.','',
 '| Cohort | Prior target history / outage hours | Cases | Gain vs full-covariance VAR | Gain vs seasonal persistence |','|---|---:|---:|---:|']
for key,g in H['comparisons'].items():
    region,history,duration=key.split(':');a=g['full_covariance_var'];b=g['seasonal_persistence']
    lines.append(f"| {region} | {history.removeprefix('history')} / {duration.removeprefix('outage')} | {a['n']} | {100*a['rmse_relative_gain']:.1f}% | {100*b['rmse_relative_gain']:.1f}% |")
lines+=['','Pre-outage target history matters. Delhi retains favorable numerical gains at 72 hours with 24 hours of history; zero-history cases show little advantage over static sheaf reconstruction and can lose to seasonal fallback. Bengaluru with history can beat VAR but lose to seasonal persistence. All conditions are retained rather than selecting only favorable horizons. Exact paired confidence intervals are in the saved horizon summary.','',
 '## New derived proposition: sharp repair uncertainty','',
 'The exact worst-case functional-width proposition is implemented and proved in `extensions/functional/REPAIR_PROPOSITION.md`. Six functional/repair extension tests pass, including 100 randomized attainment systems and 500 feasible revealed-reading comparisons. Together with 29 core/repair and two stronger-comparator tests, 37 tests have passed in their respective suites. The repair experiment additionally verifies '+str(R['conditional_actual_reading_theorem_checks'])+' conditional actual-reading theorem cases.','',
 'A rigorous bound does not guarantee the best empirical point prediction. The new worst-case policy was evaluated on 716 feasible Delhi station pairs and 540 Bengaluru pairs. Four of 720 attempted Delhi pairs had an inconsistent residual budget and were explicitly excluded. Its aggregate standardized-model PM error did not improve over random expected repair; Bengaluru was worse. The target is an average standardized model PM coordinate, which under a log transform is not mean physical concentration. No superior policy or environmental outcome is asserted.','',
 '## What remains genuinely ambitious','',
 'The strongest remaining additions would be a modern neural imputation comparator with matched causal information and an independently withheld archive period or physical deployment. More tests on the same previously seen data cannot supply external validation or establish mathematical priority. The existing functional and repair propositions are project-specific derivations with established mathematical antecedents. The environmental objective remains monitoring continuity; actual pollution reduction, official action recovery and health effects require additional evidence.','',
 'All extension sources and result summaries are hashed in `ambitious_extension_manifest.json`. The expanded core sources were not edited.','']
(DOC/'AMBITIOUS_TECHNICAL_VALIDATION.md').write_text('\n'.join(lines),encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(11,5),sharey=True)
for ax,region in zip(axes,['delhi2025','bengaluru_raw']):
    data=[]
    for key,g in H['comparisons'].items():
        if key.startswith(region+':'):
            _,h,d=key.split(':');data.append((h+' / '+d,100*g['full_covariance_var']['rmse_relative_gain'],100*g['seasonal_persistence']['rmse_relative_gain']))
    y=np.arange(len(data));ax.barh(y-.15,[x[1] for x in data],height=.3,label='vs full-covariance VAR');ax.barh(y+.15,[x[2] for x in data],height=.3,label='vs seasonal persistence')
    ax.set_yticks(y,[x[0] for x in data]);ax.axvline(0,color='black',lw=.8);ax.set_title(region);ax.set_xlabel('Relative checkpoint RMSE gain (%)');ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(F/'horizon_history_stress.pdf');fig.savefig(F/'horizon_history_stress.png',dpi=160);plt.close(fig)
sources=[p for p in (ROOT/'extensions').rglob('*.py') if '__pycache__' not in str(p)]
sources += [DOC/'AMBITIOUS_TECHNICAL_VALIDATION.md',OUT/'strong_baselines/summary.json',OUT/'horizon_stress/summary.json',OUT/'worst_case_repair/summary.json']
(DOC/'ambitious_extension_manifest.json').write_text(json.dumps({'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},indent=2))
print(DOC/'AMBITIOUS_TECHNICAL_VALIDATION.md',flush=True)
