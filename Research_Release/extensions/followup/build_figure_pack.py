                                                                                
import hashlib,json,shutil,sys,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[2];WORKSPACE=ROOT.parent
OUT=ROOT/'Results_Figure_Pack';OUT.mkdir(exist_ok=True);DATA=OUT/'source_tables';DATA.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
captions=[];sources=set();pages=PdfPages(OUT/'Research_Figures.pdf')

def read(relative):
    p=ROOT/relative;sources.add(p);return pd.read_csv(p)

def save(name,fig,caption,table=None):
    fig.tight_layout();fig.savefig(OUT/(name+'.png'),dpi=240,bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');pages.savefig(fig,bbox_inches='tight');plt.close(fig)
    if table is not None:table.to_csv(DATA/(name+'.csv'),index=False)
    captions.append((name,caption))

def gain_interval(frame,method,reference,keys):
    p=frame.pivot_table(index=keys,columns='method',values='mse').dropna(subset=[method,reference]).reset_index()
    p['day']=pd.to_datetime(p.day).dt.floor('D');g=p.groupby('day')[[method,reference]].agg(['sum','count'])
    rng=np.random.default_rng(901);ix=rng.integers(0,len(g),(2000,len(g)))
    sums=g[(method,'sum')].to_numpy()[ix].sum(1);refs=g[(reference,'sum')].to_numpy()[ix].sum(1)
    draws=100*(1-np.sqrt(sums/refs));point=100*(1-np.sqrt(p[method].mean()/p[reference].mean()))
    ci=np.quantile(draws,[.025,.975]);return {'gain':point,'low':ci[0],'high':ci[1],'cases':len(p),'days':len(g)}

                                                                            
rows=[]
for region in ['delhi2025','bengaluru_raw']:
    f=read(f'results_expanded/{region}/prolonged_with_gaussian.csv');f=f[f['mode']=='causal']
    for ref in ['stable_var','gaussian','seasonal_persistence']:
        rows.append({'region':region,'reference':ref,**gain_interval(f,'temporal_sheaf',ref,['day','station'])})
    f=read(f'extension_results/strong_baselines/{region}_cases.csv');f=f[f['mode']=='causal']
    rows.append({'region':region,'reference':'full_covariance_var',**gain_interval(f,'temporal_sheaf','full_covariance_var',['day','station'])})
t=pd.DataFrame(rows);fig,axes=plt.subplots(1,2,figsize=(11,4.4))
labels={'stable_var':'Diagonal-noise VAR','gaussian':'Gaussian conditioning','seasonal_persistence':'Seasonal persistence','full_covariance_var':'Full-covariance VAR'}
for ax,region,title in zip(axes,['delhi2025','bengaluru_raw'],['Raw Delhi','Raw Bengaluru']):
    g=t[t.region==region].reset_index(drop=True);y=np.arange(len(g))
    ax.errorbar(g.gain,y,xerr=np.array([g.gain-g.low,g.high-g.gain]),fmt='o',capsize=4,color='#235a83');ax.axvline(0,color='gray',lw=1)
    ax.set_yticks(y,[labels[x] for x in g.reference]);ax.invert_yaxis();ax.set_xlabel('RMSE reduction (%)');ax.set_title(title)
    ax.set_xlim(-15,45)
    for j,r in g.iterrows():ax.annotate(f'{r.gain:.1f}%',(r.gain,j),xytext=(4,7),textcoords='offset points')
save('01_causal_reconstruction',fig,'Causal 24-hour artificial station outages in the two raw-concentration cohorts. RMSE reduction is 100(1-sqrt(mean case MSE sheaf / mean case MSE comparator)). Bars are paired day-cluster bootstrap 95% intervals on this ratio, recalculated from common cases; they are not multiplicity-adjusted tests. Delhi has 399 cases/58 days; Bengaluru 171/60. The full-covariance control is a separately tuned stronger comparator. No hidden target readings are used during the outage. Bengaluru does not outperform seasonal persistence.',t)

                                                                               
t=read('expanded_docs/reviewed_comparisons.csv');t=t[t['mode']=='causal'];fig,ax=plt.subplots(figsize=(10,5))
networks=list(t.region.unique());refs=['stable_var','gaussian','seasonal_persistence']
for j,ref in enumerate(refs):
    g=t[t.reference==ref].set_index('region').loc[networks];ax.scatter(g.rmse_gain*100,np.arange(len(g))+(j-1)*.18,label=labels[ref])
ax.axvline(0,color='gray',lw=1);ax.set_yticks(np.arange(len(networks)),networks);ax.set_xlabel('Causal RMSE reduction (%)');ax.legend(loc='lower right');ax.set_title('Expanded reconstruction: all eligible networks')
save('02_all_networks',fig,'Causal reconstruction point estimates on the seven eligible networks. These are overlapping networks, not seven independent replications. Delhi2025 and Bengaluru_raw are raw-concentration cohorts; other cohorts are inverted-sub-index archives. Mumbai and Kolkata were excluded by the fixed training-coverage rule. Point estimates only; dependence intervals are in reviewed_comparisons.csv.',t)

                                                                 
t=read('constrained_results/matched_false_alarm_comparisons.csv');t=t[t.target_fpr==.05];fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,region,title in zip(axes,['delhi2025','bengaluru_raw'],['Raw Delhi','Raw Bengaluru']):
    g=t[t.region==region].sort_values('missing_rate');x=g.missing_rate*100
    ax.plot(x,g.sheaf_sensitivity*100,'o-',label='Sheaf');ax.plot(x,g.city_mean_sensitivity*100,'s--',label='City mean')
    ax.set_ylim(0,100);ax.set_xlabel('Additional pollutant-cell missingness (%)');ax.set_ylabel('Exceedance sensitivity (%)');ax.set_title(title+' at 5% false positives');ax.legend()
save('03_matched_false_alarms',fig,'Descriptive empirical sensitivity at a matched 5% false-positive rate on common eligible hourly PM-threshold cases. Interpolated ROC points may correspond to randomized adjacent cutoffs; they are not deployment thresholds chosen on validation. At zero added missingness, Bengaluru has 80 positives and Delhi 633. Dependence-aware paired-day difference intervals are saved in the source table and are not multiplicity adjusted. Bengaluru advantage disappears at 50% added missingness. These are illustrative hourly thresholds, not official AQI/GRAP decisions or measured pollution reduction.',t)
fig,ax=plt.subplots(figsize=(8,4))
for j,region in enumerate(['delhi2025','bengaluru_raw']):
    g=t[t.region==region].sort_values('missing_rate');y=np.arange(3)+(j-.5)*.18
    ax.errorbar(g.difference*100,y,xerr=np.array([(g.difference-g.day_bootstrap_lower)*100,(g.day_bootstrap_upper-g.difference)*100]),fmt='o',capsize=3,label=region)
ax.axvline(0,color='gray');ax.set_yticks(np.arange(3),['0% added missing','20% added missing','50% added missing']);ax.set_xlabel('Sensitivity difference: sheaf minus city mean (percentage points)');ax.legend()
save('04_alert_difference_intervals',fig,'Paired calendar-day bootstrap 95% intervals for empirical sensitivity differences at matched 5% false-positive rate, 500 bootstrap replicates. These exploratory intervals are not adjusted for the complete 18-comparison family. Crossing zero indicates uncertainty; absence of crossing is not a preregistered confirmatory result.',t)

                                                                        
t=read('followup_results/interval_summary.csv');fig,axes=plt.subplots(2,2,figsize=(10,7))
for col,region in enumerate(['delhi2025','bengaluru_raw']):
    for method,label in [('fixed_no_feedback','Fixed width'),('availability_scaled_no_feedback','Availability scale')]:
        g=t[(t.region==region)&(t.method==method)].sort_values('missing_rate');x=g.missing_rate*100
        axes[0,col].plot(x,g.coverage*100,'o-',label=label);axes[1,col].plot(x,g.mean_width,'o-',label=label)
    axes[0,col].axhline(90,color='gray',ls=':');axes[0,col].set_ylim(0,100);axes[0,col].set_title(region);axes[0,col].set_ylabel('Coverage (%)');axes[0,col].legend()
    axes[1,col].set_xlabel('Additional cell missingness (%)');axes[1,col].set_ylabel('Mean PM2.5 interval width\n(concentration units)')
save('05_no_feedback_uncertainty',fig,'Fixed calibration without hidden test-target feedback. Both empirical coverage and mean interval width are shown at nominal 90% calibration. Calibration was performed in the supplied validation period after model selection, so coverage is descriptive rather than an independent split-conformal guarantee. Availability scaling is not a consistently better control. Cases are 2,904 per Delhi setting and 1,338 per Bengaluru setting; dependent cells/stations are not independent trials.',t)

                                                                   
rows=[]
for region in ['delhi2025','bengaluru_raw']:
    path=ROOT/f'results_expanded/{region}/fault_structure_summary.json';sources.add(path);d=json.loads(path.read_text())
    for r in d['results']:
        if r.get('fault_count')==2 and r.get('magnitude')==2:rows.append({'region':region,**r})
t=pd.DataFrame(rows);fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,region in zip(axes,['delhi2025','bengaluru_raw']):
    g=t[t.region==region].sort_values('direction');x=np.arange(len(g))
    for j,(field,label) in enumerate([('ls_known_count_exact','Given true count'),('ls_alarm','Alarm'),('ls_exact_end_to_end','Exact end to end')]):ax.bar(x+(j-1)*.24,g[field]*100,width=.23,label=label)
    ax.set_xticks(x,g.direction);ax.set_ylim(0,105);ax.set_ylabel('Success / alarm frequency (%)');ax.set_title(region);ax.legend(fontsize=8)
save('06_fault_detection_and_localization',fig,'Two injected faults at magnitude two in transformed standardized coordinates. Same-sign and random directions are distinct scenarios. Given-count support localization, alarm rate and exact end-to-end success are different metrics. The decoder searches zero/one/two faults without receiving the true count for end-to-end evaluation. Sampled hour counts are in the source table; do not claim performance on every hour. High conditional localization does not establish a reliable fault alarm.',t)

                                                       
f=read('constrained_results/cases.csv');t=read('constrained_results/summary.csv');fig,axes=plt.subplots(2,2,figsize=(11,7));rows=[]
for col,region in enumerate(['delhi2025','bengaluru_raw']):
    for horizon in [24,72]:
        g=f[(f.region==region)&(f.horizon==horizon)&(f.scenario=='clean')]
        for method in ['joint_budget_no_history','joint_temporal']:
            z=g[g.method==method];rows.append({'region':region,'horizon':horizon,'method':method,'cases':len(z),'coverage':z.covered.mean(),'decision_yield':(z.decision!='abstain').mean(),'median_finite_width':(z.upper-z.lower).replace([np.inf,-np.inf],np.nan).median()})
    part=pd.DataFrame(rows);part=part[part.region==region]
    for method,label in [('joint_budget_no_history','No history'),('joint_temporal','Causal history')]:
        z=part[part.method==method];axes[0,col].plot(z.horizon,z.median_finite_width,'o-',label=label);axes[1,col].plot(z.horizon,z.decision_yield*100,'o-',label=label)
    axes[0,col].set_ylabel('Median finite PM2.5 width\n(concentration units)');axes[0,col].set_title(region);axes[0,col].legend()
    axes[0,col].set_ylim(bottom=0);axes[0,col].set_xticks([24,72])
    axes[1,col].set_ylim(0,100);axes[1,col].set_xlabel('Outage horizon (hours)');axes[1,col].set_ylabel('Decided cases (%)')
    axes[1,col].set_xticks([24,72])
save('07_temporal_bounds_and_abstention',fig,'Clean-scenario sampled endpoints under an unknown zero-or-one bounded surviving-station fault. Compare methods with the SAME jointly calibrated residual budget. Temporal information is the last trusted target reading before a 24/72-hour outage. Calibration uses an earlier archive-test period, evaluation its later period; previous archive access is disclosed. Smaller intervals did not translate into substantial threshold-decision yield. All injected-fault scenarios, coverage and inconsistent cases remain in summary.csv.',pd.DataFrame(rows))

                                                                   
t=read('constrained_results/budget_dual_benchmark.csv');g=t.groupby(['fault_stations','budget']).median(numeric_only=True).reset_index();fig,axes=plt.subplots(1,2,figsize=(10,4));x=np.arange(len(g));ticks=[f'{int(r.fault_stations)} stations\nt={int(r.budget)}' for _,r in g.iterrows()]
axes[0].plot(x,g.exact_seconds,'o-',label='Exact support LPs');axes[0].plot(x,g.dual_seconds,'s--',label='Dual outer certificate');axes[0].set_yscale('log');axes[0].set_ylabel('Median elapsed time (seconds, log scale)');axes[0].legend(fontsize=8)
axes[1].plot(x,g.exact_width,'o-',label='Exact support envelope');axes[1].plot(x,g.dual_width,'s--',label='Dual outer envelope');axes[1].set_ylabel('Median functional interval width');axes[1].legend(fontsize=8)
for ax in axes:ax.set_xticks(x,ticks)
save('08_computation_tightness_tradeoff',fig,'Synthetic benchmark: three repeated timings per station-count/fault-budget setting. The support-free dual certificate is faster but conservative; wider intervals and the strict-gap counterexample must accompany the runtime claim. Station count and fault budget both change. These timings are not real-network alert throughput or evidence of first-ever priority.',t)

                                                                              
rows=[]
for region in ['delhi2025','bengaluru_raw']:
    f=read(f'followup_results/neural_control/{region}_cases.csv');rows.append({'region':region,**gain_interval(f,'sheaf','graph_nonlinear_control',['day','station','pattern'])})
t=pd.DataFrame(rows);fig,ax=plt.subplots(figsize=(7,3.8));ax.errorbar(t.gain,np.arange(2),xerr=np.array([t.gain-t.low,t.high-t.gain]),fmt='o',capsize=4);ax.axvline(0,color='gray');ax.set_yticks(np.arange(2),['Raw Delhi','Raw Bengaluru']);ax.set_xlabel('Snapshot RMSE reduction against custom neural control (%)')
save('09_nonlinear_snapshot_control',fig,'Matched single-station snapshot comparisons with a custom graph-conditioned MLP. It is not GRIN or a faithful published temporal graph benchmark. Bars are paired day-cluster bootstrap intervals; one optimization seed and previously examined archives limit inference. Delhi is essentially tied. Cases/days and exact ratio arithmetic are in the source table.',t)

                                                                                   
path=ROOT/'extension_results/worst_case_repair/summary.json';sources.add(path);d=json.loads(path.read_text());rows=[]
for region,comparisons in d['comparisons'].items():
    for ref,value in comparisons.items():rows.append({'region':region,'reference':ref,'rmse_gain':value['rmse_relative_gain'],'cases':value['n'],'days':value['clusters']})
t=pd.DataFrame(rows);fig,ax=plt.subplots(figsize=(8,4));y=np.arange(len(t));ax.scatter(t.rmse_gain*100,y);ax.axvline(0,color='gray');ax.set_yticks(y,[r.region+' / '+r.reference for _,r in t.iterrows()]);ax.set_xlabel('RMSE reduction of worst-case repair policy (%)');ax.set_title('Repair policy: empirical mean-error tradeoffs')
save('10_repair_policy_tradeoffs',fig,'Worst-case functional-width repair policy versus Gaussian trace, variance and expected random-repair controls on paired artificial two-station outages. Point estimates only; the original paired uncertainty is retained in the source JSON. Minimizing a worst-case bound does not guarantee lower average reconstruction MSE. The policy did not establish a consistent empirical advantage; no physical repair intervention occurred.',t)

pages.close()
                                                                           
prior=[]
for base,label in [(ROOT/'docs/figures','supplementary'),(WORKSPACE/'Final_Manuscript/figures','earlier_evidence')]:
    target=OUT/label;target.mkdir(exist_ok=True)
    for p in base.glob('*'):
        if p.suffix not in ['.pdf','.png'] or 'preview' in p.stem:continue
        shutil.copy2(p,target/p.name);sources.add(p);prior.append(str((target/p.name).relative_to(OUT)))
for p in (WORKSPACE/'Final_Manuscript/sections').glob('*.tex'):
    if '\\includegraphics' in p.read_text(encoding='utf-8'):shutil.copy2(p,OUT/'earlier_evidence'/('captions_'+p.name))
lines=['# Results figure guide','','New figures are exported as vector PDF and 240-dpi PNG. Research_Figures.pdf collects the ten new figures. Each source table is preserved. Earlier figures are retained in separate folders with original manuscript captions; do not mix their protocols or statistical units with the expanded studies.','',
    '## Main evidence to lead with','',
    'Lead with 01_causal_reconstruction. Use 03/04 for potential monitoring usefulness, with the false-alarm, missingness and multiplicity limitations. Use 08 for the theorem-to-computation tradeoff, clearly labeled synthetic. The remaining figures provide controls, failure modes and supplementary evidence. No graph establishes first-ever novelty, reduced pollution or a guaranteed competition outcome.','']
for name,caption in captions:lines+=['## '+name,'',caption,'']
lines+=['## Preserved earlier evidence','']+[ '- '+p for p in prior]
(OUT/'FIGURE_GUIDE.md').write_text('\n'.join(lines),encoding='utf-8')
manifest={'new_figures':len(captions),'preserved_figure_files':len(prior),'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)},
    'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'limitations':['Historical figures retain original protocols and require their associated captions','Bootstrap intervals are exploratory; no new multiplicity-adjusted claims','First-ever originality and field environmental benefit unestablished']}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(ROOT/'Results_Figure_Pack.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        if p.is_file():z.write(p,p.relative_to(ROOT))
print('Created',len(captions),'new figures, captions, source tables and preserved',len(prior),'earlier figure files',flush=True)
