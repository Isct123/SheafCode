                                                                                    
import argparse,hashlib,json,sys,os
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sheafcode.reliable import paired_bootstrap
ap=argparse.ArgumentParser();ap.add_argument('--output',default='results');args=ap.parse_args()
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'matplotlib_cache'))
RESULTS=ROOT/args.output;DOCS=ROOT/('docs' if args.output=='results' else 'smoke_docs');FIGS=DOCS/'figures';FIGS.mkdir(parents=True,exist_ok=True)
REGIONS=['delhi','ncr','mumbai','kolkata','hyderabad','bengaluru','india','delhi2025','bengaluru_raw']
summaries={};statuses={};comparisons=[];environments=[];faults=[];intervals=[];overlaps=[]
for r in REGIONS:
    folder=RESULTS/r
    if not (folder/'manifest.json').exists():statuses[r]='not run';continue
    manifest=json.loads((folder/'manifest.json').read_text());statuses[r]=manifest['status']
    if manifest['status']!='complete':continue
    if not (folder/'summary.json').exists():statuses[r]='missing summary';continue
    s=json.loads((folder/'summary.json').read_text());summaries[r]=s
    for mode,table in s['prolonged'].items():
        for reference,cmp in table.items():
            if cmp.get('n',0):comparisons.append({'region':r,'mode':mode,'reference':reference,**cmp})
    for method,rec in s['intervals'].items():intervals.append({'region':r,'method':method,**rec})
    if (folder/'fault_structure_summary.json').exists():
        f=json.loads((folder/'fault_structure_summary.json').read_text())
        faults.extend({'region':r,**x} for x in f['results'])
                                                                 
    if (folder/'environment_common.csv').exists():
        A=pd.read_csv(folder/'environment_common.csv')
        A['miss_indicator']=((A.true_exceed==1)&(A.pred_exceed==0)).astype(float)
        A['wrong_indicator']=(A.true_exceed!=A.pred_exceed).astype(float)
        for metric in ['absolute_error','wrong_indicator','miss_indicator']:
            for ref in ['gaussian','kriging','idw','city_mean']:
                cmp=paired_bootstrap(A,'sheaf',ref,value=metric)
                if cmp.get('n',0):environments.append({'region':r,'metric':metric,'reference':ref,**cmp})
pd.DataFrame(comparisons).to_csv(DOCS/'prolonged_comparisons.csv',index=False)
pd.DataFrame(environments).to_csv(DOCS/'environment_comparisons.csv',index=False)
pd.DataFrame(faults).to_csv(DOCS/'fault_summary.csv',index=False)
pd.DataFrame(intervals).to_csv(DOCS/'interval_summary.csv',index=False)
for i,a in enumerate(summaries):
    ca=json.loads((RESULTS/a/'cohort.json').read_text())
    for b in list(summaries)[i+1:]:
        cb=json.loads((RESULTS/b/'cohort.json').read_text())
        shared=sorted(set(ca['stations'])&set(cb['stations']))
        source_a='archive' if a not in ['delhi2025','bengaluru_raw'] else a
        source_b='archive' if b not in ['delhi2025','bengaluru_raw'] else b
        start=max(pd.Timestamp(ca['timeline_start']),pd.Timestamp(cb['timeline_start']))
        end=min(pd.Timestamp(ca['timeline_end']),pd.Timestamp(cb['timeline_end']))
        overlaps.append({'a':a,'b':b,'shared_ids':shared,'same_source':source_a==source_b,
                         'calendar_overlap_hours':max(0,int((end-start).total_seconds()/3600)+1),
                         'interpretation':'calendar overlap does not prove duplicate values; raw acquisition paths may share instruments'})
(DOCS/'overlap.json').write_text(json.dumps(overlaps,indent=2),encoding='utf-8')
lines=['# Corrected research experiment report','',
       'Research reference only. Generated from corrected, training-only-cohort runs. This is a rerun of previously examined data, not an untouched holdout.',
       '', '## Completion and exclusions','']
for r in REGIONS:
    line=f'- {r}: {statuses.get(r,"not run")}'
    if r in summaries:line+=f'; {summaries[r]["n_stations"]} stations, {summaries[r]["n_pollutants"]} pollutants, {summaries[r]["sampled_target_stations"]} evaluated target stations'
    elif (RESULTS/r/'manifest.json').exists():line+='; '+json.loads((RESULTS/r/'manifest.json').read_text()).get('reason','')
    lines.append(line)
lines+=['','## Primary single-station reconstruction','',
        '| Network | Paired cases | Sheaf / Gaussian RMSE ratio | Day-cluster difference CI, MSE |','|---|---:|---:|---|']
for r,s in summaries.items():
    c=s['snapshot'].get('gaussian',{})
    if c.get('n',0):lines.append(f'| {r} | {c["n"]} | {1-c["rmse_relative_gain"]:.3f} | {c["difference_ci95"]} |')
lines+=['','## Prolonged reconstruction: tuned VAR comparison','',
        'Positive gain favors sheaf. Retrospective models access later surviving readings; causal models use prefixes only. Intervals resample days and do not treat overlapping regions as independent.',
        '', '| Network | Mode | Paired cases / days | Relative RMSE gain | MSE difference CI |','|---|---|---:|---:|---|']
for c in comparisons:
    if c['reference']=='stable_var':lines.append(f'| {c["region"]} | {c["mode"]} | {c["n"]} / {c["clusters"]} | {100*c["rmse_relative_gain"]:.1f}% | {c["difference_ci95"]} |')
lines+=['','## Headline eligibility','',
        'No ISEF selection or pollution-reduction guarantee follows from these experiments. A defensible reconstruction headline requires improvement over the tuned baseline with a positive paired interval on the direct-concentration cohorts, not just the easiest fallback.']
raw=[c for c in comparisons if c['region'] in ['delhi2025','bengaluru_raw'] and c['reference']=='stable_var' and c['mode']=='causal']
supported=[c for c in raw if c['difference_ci95'][0]>0]
if supported:
    lines.append('Candidate finding, subject to final audit: causal reconstruction improves over the tuned ridge VAR filter in '+', '.join(f'{c["region"]} ({100*c["rmse_relative_gain"]:.1f}% RMSE gain)' for c in supported)+'.')
else:lines.append('The completed direct-concentration causal comparisons do not establish superiority to the tuned VAR filter with a positive day-cluster interval. Do not claim that superiority.')
lines+=['','## Environmental threshold endpoint','',
        'PM2.5 daily mean above 60 micrograms per cubic metre is a threshold proxy. Truth requires 16 observed hours; predictions are scored on those same hours. It is not GRAP invocation, measured human exposure, or demonstrated pollution reduction.',
        '', '| Network | Method | Station-days | Positive days | Accuracy | Miss rate | False-alarm rate |','|---|---|---:|---:|---:|---:|---:|']
for r,s in summaries.items():
    for method,g in s['environment'].items():
        fmt=lambda x:'undefined' if x is None else f'{100*x:.1f}%'
        lines.append(f'| {r} | {method} | {g["station_days"]} | {g["true_exceedances"]} | {fmt(g["accuracy"])} | {fmt(g["miss_rate"])} | {fmt(g["false_alarm_rate"])} |')
lines+=['','Paired error and missed-exceedance differences are in `environment_comparisons.csv`. Repair ranking and unverified natural-gap estimates are retained per network. Direct raw measurements and inverted sub-index archives must remain distinct in interpretation.',
        '', '## Intervals','',
        '| Network | Setting | Cells | Coverage | Lowest station quarter | Full / empty sets |','|---|---|---:|---:|---:|---:|']
for x in intervals:lines.append(f'| {x["region"]} | {x["method"]} | {x["n_cells"]} | {100*x["pooled_coverage"]:.2f}% | {100*x["lowest_station_quarter"]:.2f}% | {x["full_sets"]} / {x["empty_sets"]} |')
lines+=['','Adaptive guarantees apply to the revealed-label score stream; fixed intervals are the no-label outage setting. Constant-scale control must exactly reproduce unit-scale coverage. Dynamic-scale benefit is an empirical comparison, not assumed.',
        '', '## Fault alarms and structural validation','',
        'See `fault_summary.csv`: distinguish unknown-count end-to-end exact support and alarm rate from known-count localization. Fault directions include positive, negative and random vectors. Clean alarms are reported. All fully observed candidates use the same universe; incomplete syndrome equations are excluded without imputed truth.',
        '', '## Remaining scientific limits','',
        '- Existing data have already been examined; an independently blinded new data period is still required for a pristine holdout claim.',
        '- GRIN, meteorological covariates, actual policy decisions and field deployment are not established by this runner.',
        '- Default runs sample target stations and days; they are not exhaustive replay of every archive observation.',
        '- Day clusters preserve same-day dependence but not longer serial dependence; station and moving-block sensitivities remain for the final analysis.',
        '- Generic structural rank does not prove general-position CCA; the report compares actual support and numerical rank.',
        '- Final manuscript integration and a human-readable claim-by-claim audit follow review of these outputs.',
        '', 'References: [CPCB](https://cpcb.nic.in/displaypdf.php?id=bWFudWFsLW1vbml0b3JpbmcvQVFJX05BTVBfUmVwX1NlcHRlbWJlcjIwMTYucGRm), [ACI](https://papers.nips.cc/paper/2021/file/0d441de75945e5acbc865406fc9a2559-Paper.pdf), [sparse uniqueness](https://pmc.ncbi.nlm.nih.gov/articles/PMC153464/).']
(DOCS/'OVERNIGHT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'figure.dpi':150})
    C=pd.DataFrame(comparisons)
    if len(C):
        fig,axes=plt.subplots(1,2,figsize=(11,5),sharey=True)
        for ax,mode in zip(axes,['retrospective','causal']):
            g=C[(C.reference=='stable_var')&(C['mode']==mode)]
            ax.barh(g.region,100*g.rmse_relative_gain,color=['#208080' if x>0 else '#a44' for x in g.rmse_relative_gain])
            ax.axvline(0,color='black',lw=.8);ax.set_xlabel('Relative RMSE gain (%)');ax.set_title(mode.capitalize())
        fig.suptitle('Whole-station 24-hour outages vs validation-tuned ridge VAR')
        fig.tight_layout();fig.savefig(FIGS/'prolonged.pdf');fig.savefig(FIGS/'prolonged.png');plt.close(fig)
    I=pd.DataFrame(intervals)
    if len(I):
        fig,ax=plt.subplots(figsize=(10,5))
        for name,g in I.groupby('method'):ax.plot(g.region,100*g.pooled_coverage,marker='o',label=name.replace('_',' '))
        ax.axhline(90,color='black',ls='--',lw=.8);ax.set_ylabel('Pooled observed-cell coverage (%)')
        ax.tick_params(axis='x',rotation=35);ax.legend(fontsize=8);fig.tight_layout()
        fig.savefig(FIGS/'intervals.pdf');fig.savefig(FIGS/'intervals.png');plt.close(fig)
except ImportError as e:
    print('Figure generation unavailable:',str(e),flush=True)
                                                                           
import zipfile
with zipfile.ZipFile(ROOT/('Overnight_Results.zip' if args.output=='results' else 'Smoke_Test_Results.zip'),'w',zipfile.ZIP_DEFLATED) as z:
    for folder in [RESULTS,DOCS]:
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix!='.npz':z.write(p,p.relative_to(ROOT))
print('Report:',DOCS/'OVERNIGHT_REPORT.md',flush=True)
