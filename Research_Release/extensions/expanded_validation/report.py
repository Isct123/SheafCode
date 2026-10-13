                                                                             
from pathlib import Path
import sys,json,shutil,importlib.util,re,hashlib,os
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=ROOT/'Research_Release/expanded_validation_results';PACKAGE=ROOT/'IRIS_Submission_Recent_Data';SRC=PACKAGE/'Expanded_Evidence_Source';REVIEW=PACKAGE/'Review_Notes'
sys.path.insert(0,str(ROOT/'Research_Release/extensions/recent_data'))
import common
common.OUT=OUT
from common import paired,sha,save
def js(folder,name):return json.loads((folder/name).read_text())
def esc(s):return str(s).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&').replace('#',r'\#')
def number(x):return 'n/a' if x is None or not np.isfinite(x) else f'{x:.2f}'
def interval(x):return 'suppressed' if x is None else f'{x[0]:.2f} to {x[1]:.2f}'
def table(caption,heads,rows):
    return '\\begin{table}[htbp]\\centering\\small\\caption{'+caption+'}\\resizebox{\\linewidth}{!}{\\begin{tabular}{'+'l'*len(heads)+'}\\toprule\n'+' & '.join(heads)+r'\\\midrule'+'\n'+'\n'.join(' & '.join(esc(x) for x in row)+r'\\' for row in rows)+'\n\\bottomrule\\end{tabular}}\\end{table}\n'
def main():
    names=sorted(p.parent.name for p in OUT.glob('*/dataset.npz'))
    if not names:raise RuntimeError('No expansion evidence')
    for name in names:
        folder=OUT/name
        if js(folder,'run_status.json')['status'] not in ['completed','unavailable_data'] or js(folder,'AUDIT.json')['status']!='passed':raise RuntimeError('Unaudited/incomplete cohort: '+name)
    SRC.mkdir(exist_ok=True);REVIEW.mkdir(exist_ok=True)
    for name in ['complete_theory.tex','references.bib','uk_section.tex','uk_catalogue.tex']:shutil.copy2(PACKAGE/'Three_Network_Source'/name,SRC/name)
    uk_section=SRC/'uk_section.tex'
    uk_text=uk_section.read_text(encoding='utf-8').replace('No UK neural experiment, official-bulletin/DAQI validation, physical repair, emissions reduction or measured health benefit is claimed. The existing three-seed neural comparison remains Indian-only.',
        'No neural experiment on this original UK cohort, official-bulletin/DAQI validation, physical repair, emissions reduction or measured health benefit is claimed. The original three-seed neural comparison concerns India; a separate UK regional neural refit is reported with the external follow-up below.')
    uk_section.write_text(uk_text,encoding='utf-8')
    shutil.copytree(PACKAGE/'Three_Network_Source/figures',SRC/'figures',dirs_exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(OUT/'plot_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[];coverage=[];faults=[];mathrows=[];stressrows=[];catalog=[];comparisons=[];availability=[];operation=[];unavailable=[];environmental=[]
    counts={};codes={}
    for name in names:
        folder=OUT/name;meta=js(folder,'dataset_manifest.json');provider=meta['provider'];counts[provider]=counts.get(provider,0)+1;code={'EPA':'D','AURN':'E','NSW':'F','OpenAQ':'A-sensitivity'}[provider]+(str(counts[provider]) if provider!='OpenAQ' else '');codes[name]=code
        ds=np.load(folder/'dataset.npz',allow_pickle=False)
        if js(folder,'run_status.json')['status']=='unavailable_data':
            unavailable.append({'cohort':name,'code':code,'provider':provider,'stations':len(ds['stations']),'reason':js(folder,'run_status.json')['reason'],'audit_checks':js(folder,'AUDIT.json')['check_count']});continue
        cases=pd.read_csv(folder/'outage_cases.csv',dtype={'station':str});dep=js(folder,'dependence_sensitivity.json')
        allresult=dep['all_eligible_test_days']['7'];primary=dep['primary_every_third_day']['7'];pm=paired(cases,'sheaf_causal','var_causal',value='pm25_raw_mse');city=paired(cases,'sheaf_causal','city_mean')
        rows.append({'cohort':name,'code':code,'provider':provider,'stations':len(ds['stations']),'channels':len(ds['channels']),'train_hours':int((ds['split']==0).sum()),'select_hours':int((ds['split']==1).sum()),'calibration_hours':int((ds['split']==3).sum()),'test_hours':int((ds['split']==2).sum()),'observed_pollutant_hour_cells':int(np.isfinite(ds['X']).sum()),'start':str(pd.Timestamp(ds['times'][0]).date()),'end':str(pd.Timestamp(ds['times'][-1]).date()),'cases':allresult['cases'],'dates':allresult['dates'],'weekly_bins':allresult['calendar_bins'],'gain':allresult['gain_percent'],'ci_low':allresult['ci95'][0] if allresult['ci95'] else np.nan,'ci_high':allresult['ci95'][1] if allresult['ci95'] else np.nan,'ci_14':dep['all_eligible_test_days']['14']['ci95'],'primary_gain':primary['gain_percent'],'primary_cases':primary['cases'],'pm25_gain':pm['gain_percent'],'city_gain':city['gain_percent'],'audit_checks':js(folder,'AUDIT.json')['check_count']})
        comparisons.append({'code':code,**{k:paired(cases,'sheaf_causal',k)['gain_percent'] for k in ['var_causal','city_mean','seasonal','gaussian','spatial_sheaf']}})
        t=pd.DatetimeIndex(ds['times']);daily=pd.DataFrame({'day':t.floor('D'),'fraction':np.isfinite(ds['X'][:,:,0]).mean(1)}).groupby('day').fraction.mean();availability.append((code,daily))
        u=pd.read_csv(folder/'uncertainty_summary.csv')
        for group in ['all','extreme']:
            g=u[(u.context_loss==0)&(u.group==group)]
            for _,r in g.iterrows():coverage.append({'cohort':name,'code':code,'group':group,**r.to_dict()})
        f=pd.read_csv(folder/'available_fault_summary.csv') if (folder/'available_fault_summary.csv').exists() else pd.DataFrame()
        for _,r in f.iterrows():faults.append({'cohort':name,'code':code,**r.to_dict()})
        fs=js(folder,'fault_status.json');av=js(folder,'available_fault_status.json');hy=js(folder,'hybrid_summary.json');repair=js(folder,'repair_summary.json');ab=js(folder,'ablation_summary.json');env=pd.read_csv(folder/'alert_summary.csv')
        operation.append({'code':code,'strict_fault_status':fs['status'],'available_fault_status':av['status'],'hybrid_vs_VAR':hy.get('var_causal',{}).get('gain_percent'),'repair_cases':repair.get('trace_policy',{}).get('cases',0),'repair_gain_vs_trace':repair.get('trace_policy',{}).get('gain_percent'),'identity_ablation_gain':ab.get('identity_spatial_maps',{}).get('gain_percent')})
        clean=f[(f.kind=='clean')&(f.true_count==0)] if len(f) else pd.DataFrame()
        hours=pd.read_csv(folder/'available_fault_hour_summary.csv');hc=hours[(hours.kind=='clean')&(hours.true_count==0)];ho=hours[(hours.kind=='opposite')&(hours.true_count==2)]
        operation[-1].update({'sampled_fault_episodes':av['sampled_episodes'],'episode_clean_hours':int(clean.cases.iloc[0]) if len(clean) else 0,'episode_false_alarm':float(clean.alarm_rate.iloc[0]) if len(clean) else None,'hour_clean_cases':int(hc.cases.iloc[0]) if len(hc) else 0,'hour_false_alarm':float(hc.alarm_rate.iloc[0]) if len(hc) else None,'hour_two_opposite_exact':float(ho.exact_support.iloc[0]) if len(ho) else None})
        alert=env[env.method=='sheaf_causal'].iloc[0];baseline=env[env.method=='var_causal'].iloc[0];masked=js(folder,'matched_missing_mask_summary.json')
        environmental.append({'code':code,'hours':int(alert.cases),'positive_hours':int(alert.positive_cases),'sensitivity':alert.sensitivity,'false_alarm_rate':alert.false_alarm_rate,'var_false_alarm_rate':baseline.false_alarm_rate,'mask_cells':masked['cases'],'mask_gain':masked.get('gain_percent')})
        status=js(folder,'mathematics_status.json');joint=pd.read_csv(folder/'functional_joint_bounded_cases.csv');mathrows.append({'code':code,'sets':status['rank_subsets'],'agree':status['all_structured_rank_agree'],'thinning_disagreements':status['thinning_disagreements'],'actual_states':len(joint),'abstentions':int(joint.joint_decision.eq('abstain').sum()),'inconsistent':int(joint.joint_status.eq('inconsistent').sum()),'planted':status['synthetic_certificate_cases']})
        stress=pd.read_csv(folder/'history_horizon_context_cases.csv')
        for (history,horizon,loss),g in stress.groupby(['history','horizon','context_loss']):
            result=paired(g,'sheaf_causal','var_causal');stressrows.append({'code':code,'history':history,'horizon':horizon,'loss':loss,'cases':result.get('cases',0),'gain':result.get('gain_percent')})
        for path in sorted(folder.glob('*.csv')):
            try:n=len(pd.read_csv(path))
            except pd.errors.EmptyDataError:n=0
            catalog.append({'cohort':name,'table':path.name,'rows':n,'sha256':sha(path)})
    results=pd.DataFrame(rows);cov=pd.DataFrame(coverage);fault=pd.DataFrame(faults);stress=pd.DataFrame(stressrows)
    results.to_csv(OUT/'COHORT_RESULTS.csv',index=False);cov.to_csv(OUT/'COVERAGE_RESULTS.csv',index=False);fault.to_csv(OUT/'FAULT_RESULTS.csv',index=False);pd.DataFrame(catalog).to_csv(OUT/'RESULT_TABLE_INVENTORY.csv',index=False);stress.to_csv(OUT/'STRESS_RESULTS.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'ALL_COMPARATOR_RESULTS.csv',index=False);pd.DataFrame(operation).to_csv(OUT/'OPERATIONAL_RESULTS.csv',index=False)
    pd.DataFrame(unavailable).to_csv(OUT/'UNAVAILABLE_COHORTS.csv',index=False)
    pd.DataFrame(environmental).to_csv(OUT/'ENVIRONMENTAL_RESULTS.csv',index=False)
    def put(label,fig):
        fig.tight_layout();fig.savefig(SRC/'figures'/('expanded_'+label+'.pdf'),bbox_inches='tight');fig.savefig(SRC/'figures'/('expanded_'+label+'.png'),dpi=170,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,max(4,len(rows)*.27)))
    yy=np.arange(len(rows))
    for i,r in results.iterrows():
        color='steelblue' if r.gain>=0 else 'darkorange';axs[0].plot(r.gain,i,'o',color=color)
        if np.isfinite(r.ci_low):axs[0].plot([r.ci_low,r.ci_high],[i,i],color=color)
        axs[1].plot(r.pm25_gain,i,'o',color='steelblue' if r.pm25_gain>=0 else 'darkorange')
    for ax in axs:ax.set_yticks(yy,results.code);ax.invert_yaxis();ax.axvline(0,color='black',linewidth=.7);ax.set_xlabel('Causal sheaf RMSE reduction versus VAR (%)')
    axs[0].set_title('Normalized error; seven-day bin interval');axs[1].set_title('Physical PM2.5 error; point estimates');put('reconstruction',fig)
    fig,axs=plt.subplots(1,2,figsize=(10,max(4,len(rows)*.24)))
    for ax,group,title in zip(axs,['all','extreme'],['Overall no-feedback coverage','Training-threshold extreme coverage']):
        c=cov[(cov.group==group)&cov.method.eq('pooled_fixed')]
        ax.barh(c.code,c.coverage*100);ax.axvline(90,color='black',linestyle='--');ax.invert_yaxis();ax.set_xlim(0,105);ax.set_xlabel('Observed coverage (%)');ax.set_title(title)
        for i,r in enumerate(c.itertuples()):ax.text(min(101,r.coverage*100+1) if np.isfinite(r.coverage) else 2,i,'n='+str(r.cases) if r.cases else 'n=0; coverage unavailable',va='center',fontsize=6)
    put('coverage',fig)
    days=pd.date_range('2025-10-14','2026-10-05',freq='D');image=np.full((len(rows),len(days)),np.nan)
    for i,(code,day) in enumerate(availability):image[i]=day.reindex(days).to_numpy()
    fig,ax=plt.subplots(figsize=(11,max(4,len(rows)*.24)));im=ax.imshow(image,vmin=0,vmax=1,cmap='Blues',aspect='auto');ax.set_yticks(yy,results.code);ticks=np.linspace(0,len(days)-1,7).astype(int);ax.set_xticks(ticks,[str(days[x].date()) for x in ticks],rotation=25,ha='right');fig.colorbar(im,ax=ax,label='Observed PM2.5 fraction across selected sites');ax.set_title('Complete calendars; blank areas are outside each acquisition window');put('availability',fig)
    configurations=[(h,w,l) for l in [0.,.2,.5] for h in [0,24] for w in [24,72,168]];image=np.full((len(rows),len(configurations)),np.nan)
    for i,r in results.iterrows():
        for j,(h,w,l) in enumerate(configurations):
            g=stress[(stress.code==r.code)&stress.history.eq(h)&stress.horizon.eq(w)&stress.loss.eq(l)]
            if len(g):image[i,j]=g.gain.iloc[0]
    fig,ax=plt.subplots(figsize=(11,max(4,len(rows)*.25)));im=ax.imshow(image,vmin=-30,vmax=30,cmap='RdBu',aspect='auto');ax.set_yticks(yy,results.code);ax.set_xticks(range(len(configurations)),[f'{h}/{w}/{l:g}' for h,w,l in configurations],rotation=65,ha='right',fontsize=7);fig.colorbar(im,ax=ax,label='Causal RMSE reduction vs VAR (%)');ax.set_title('History / outage hours / surviving-context loss; color saturates at +/-30%');put('stress',fig)
    families=sorted(fault[fault.true_count==2].kind.unique());matrix=np.full((len(rows),len(families)),np.nan)
    for i,r in results.iterrows():
        for j,k in enumerate(families):
            g=fault[(fault.cohort==r.cohort)&(fault.kind==k)&(fault.true_count==2)]
            if len(g):matrix[i,j]=100*g.exact_support.iloc[0]
    fig,ax=plt.subplots(figsize=(10,max(4,len(rows)*.25)));image=ax.imshow(matrix,vmin=0,vmax=100,cmap='viridis',aspect='auto');ax.set_yticks(yy,results.code);ax.set_xticks(range(len(families)),families,rotation=35,ha='right');fig.colorbar(image,ax=ax,label='Alarm and exact support (%)');ax.set_title('Two-fault injected episodes; dependent hours, not field faults');put('faults',fig)
    independent=results[results.provider!='OpenAQ'];total=int(independent.cases.sum());positive=int((independent.gain>0).sum());cells=int(independent.observed_pollutant_hour_cells.sum());sites=int(independent.stations.sum())
    md=['# Broad empirical expansion: complete results and remaining weaknesses','',f'Added {len(independent)} regional cohorts from {independent.provider.nunique()} official providers, {sites} selected station memberships and {cells:,} observed pollutant-hour cells. New all-day outage evaluation: {total:,} paired station-days. These are dependent cases, not independent experiments. The longer Indian resplit is separately labelled; it reuses previously accessed observations.','',f'Causal sheaf beats tuned VAR on normalized all-day RMSE in {positive}/{len(independent)} new cohorts. Favorable and unfavorable cohorts are retained.','', '| Cohort | Provider | Sites | Station-days | Dates | RMSE gain vs VAR | 7-day interval | Physical PM gain |','|---|---|---:|---:|---:|---:|---|---:|']
    for r in results.itertuples():md.append(f'| {r.code} ({r.cohort}) | {r.provider} | {r.stations} | {r.cases} | {r.dates} | {r.gain:.2f}% | {interval(None if not np.isfinite(r.ci_low) else [r.ci_low,r.ci_high])} | {r.pm25_gain:.2f}% |')
    md+=['','## Weaknesses addressed','- Geographic and sample coverage: deterministic training-eligible US counties, UK regional boxes and Australian regions; full calendars, raw records, units, query bodies and hashes preserved.','- Short original Indian endpoint: original primary result preserved; longer-test sensitivity added without counting reused observations as independent.','- Incomplete evidence reporting: all compatible task families repeated, every cohort included, failures and uncertainty retained.','- Dependence and unequal units: station-day paired comparisons, seven- and fourteen-day block checks, physical PM2.5 reported separately; no naive pooled international percentage.','- Traceability: raw-source tensor reconstruction, independent seven-method inference replay, theorem assumption checks and table arithmetic audit.','','## Limitations retained explicitly','- More observations do not prove universal dominance. Compare all tested cohorts and controls; do not select only favorable results.','- Public sensors are not independent laboratories; provider, site, instrument, period and missingness dependencies remain.','- Cohort selection is training-only but not random sampling; coarse geographic boxes are stress tests, not guaranteed homogeneous air sheds.','- Expansion was specified after earlier outcomes and is exploratory, not preregistered or blinded.','- Neural comparison remains Indian-only. Extra non-neural replications do not establish international neural superiority.','- No-feedback empirical coverage is not a guarantee during arbitrary unobserved outages. Show extreme-cell failures, interval width, full/empty sets and feedback assumptions.','- Fault injections are synthetic; field-fault truth is unavailable. Environmental threshold proxies are not official AQI/GRAP decisions or measured emissions/health improvements.','- Conditional certificate proofs rely on budgets, boxes and fixed operators; numerical tests do not establish first-ever mathematical priority or replace independent proof review.','- Only retrieved recent observations enter the expanded empirical chapter. Documentation-only candidates (EEA, Canada NAPS, Hong Kong) are not claimed as tested sources.','- Observation-age eligibility expires with the actual submission date. The oldest new records are 14 October 2025; reconfirm if submission is after 13 October 2026.']
    if unavailable:
        md+=['','## Fixed cohorts without test records']+[f"- {r['code']} ({r['cohort']}): {r['reason']}. Raw-source audit passed; not counted as an empirical win or loss." for r in unavailable]
    (REVIEW/'EXPANDED_EVIDENCE_REVIEW.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    chapter='\\clearpage\\section{Broad external empirical validation}\n'
    chapter+=f'To address the limited size and breadth of the initial endpoint, a fixed expansion protocol selected {len(independent)} additional regional cohorts from three official monitoring providers before expansion model outcomes. They contain {sites} selected station memberships, {cells:,} observed pollutant-hour cells and {total:,} paired eligible outage station-days. These memberships and cases are not independent trials. The original 77-case Indian primary result is unchanged. A longer Indian resplit is reported separately as reused-observation sensitivity. The expansion is exploratory because it follows the earlier study; it is not a blinded or externally preregistered test.\n'
    chapter+=r'\subsection{Source selection and information control} EPA supplies native PM2.5 and ozone from 14 October 2025--31 March 2026. Eligible counties are sorted by state/county identifiers; the first twelve qualifying groups excluding the initial US county are retained, with at most four sites. Within a site/channel, the lowest training-eligible parameter code and monitor POC are used. Duplicate records are averaged only within that monitor identity; native units and qualifiers remain documented. UK AURN supplies January--June PM2.5 and PM10 in four predefined nonoverlapping regional boxes outside the initial UK selection box. NSW supplies January--June hourly PM2.5 and PM10 from the first six lexically ordered qualifying metadata regions, subject to availability. NSW Hour 1 denotes 00:00--01:00 in fixed AEST (UTC+10), including daylight-saving months. Only true hourly averages are used, not rolling indices. Public preliminary/ratification quality is documented per source; no undocumented final ratification is asserted. Official sources describe the archive and timing conventions \citep{eparecent,aurn2026,nswapi2026,nswtime2026}.'+'\n'
    chapter+=r'Every channel needs at least 50\% training coverage and 720 training hours, with at least three eligible stations. Selection uses training availability only. New external timelines use 60/10/10/20\% train/select/calibrate/test splits; the longer Indian sensitivity uses January--March training, April selection, May calibration and June--early October testing. Models are refitted within each cohort, not transferred unchanged. All eligible days form the expansion endpoint; every-third-day results remain separately reported. No expansion outcome is used to retune the primary pipeline. Region identifiers below are anonymous analysis codes; exact sites and coordinates remain in source manifests.'+'\n'
    for start in range(0,len(rows),10):
        chunk=rows[start:start+10]
        chapter+=table('Additional cohort calendars and evaluation denominators; longer Indian resplit is not independent.',['Code','Provider','Sites/channels','Calendar','Test h','Days/cases'],[[r['code'],r['provider'],f"{r['stations']}/{r['channels']}",r['start']+'--'+r['end'],r['test_hours'],f"{r['dates']}/{r['cases']}"] for r in chunk])
    chapter+=r'\begin{figure}[p]\centering\includegraphics[width=\linewidth,height=.8\textheight,keepaspectratio]{figures/expanded_availability}\caption{Daily selected-site PM2.5 availability across complete intended calendars. A-sensitivity contains reused Indian observations. Blank areas fall outside a cohort calendar; low availability is not zero pollution.}\end{figure}'+'\n'
    if unavailable:
        chapter+=table('Training-eligible cohorts without recorded test observations; these are retained as unavailable, not replaced with favorable networks.',['Code','Provider','Sites','Observed test cells'],[[r['code'],r['provider'],r['stations'],0] for r in unavailable])
        chapter+=f'The fixed acquisition selected {len(independent)+len(unavailable)} external cohorts. {len(unavailable)} had no recorded observations in the intended test period, so only {len(independent)} contribute empirical test results. Their raw-source and normalization checks passed. No missing test truth is imputed into an evaluation, and no unavailable cohort is counted as a win or loss.\n'
    chapter+='\\subsection{Matched reconstruction and dependence sensitivity}\n'
    chapter+=f'The causal sheaf has positive normalized all-day RMSE gain over tuned VAR in {positive}/{len(independent)} additional external cohorts. This is a descriptive sign count, not independent-binomial evidence. Every negative result is included. Improvements over city mean or seasonal persistence do not substitute for the strongest tuned comparator. Aggregate station-day MSE and native physical PM2.5 error remain separate. Fourteen-day intervals are often suppressed because the test calendar contains fewer than five such bins; seven-day precision can therefore overstate certainty about longer-scale dependence.\n'
    for start in range(0,len(rows),10):
        chunk=rows[start:start+10]
        chapter+=table('All-day expansion results. Positive values favor sheaf; negative values favor VAR. Bootstrap intervals are exploratory.',['Code','Gain (\%)','7-day interval','14-day interval','PM2.5 gain (\%)','Primary gain (\%)'],[[r['code'],number(r['gain']),interval(None if not np.isfinite(r['ci_low']) else [r['ci_low'],r['ci_high']]),interval(r['ci_14']),number(r['pm25_gain']),number(r['primary_gain'])] for r in chunk])
    chapter+=r'\begin{figure}[p]\centering\includegraphics[width=\linewidth,height=.8\textheight,keepaspectratio]{figures/expanded_reconstruction}\caption{All added cohorts, including unfavorable results. Left: normalized all-day paired gains with seven-day calendar-bin intervals where supported. Right: physical PM2.5 point estimates. A-sensitivity reuses Indian observations. Cohorts share providers and protocols; they are not independent laboratories.}\end{figure}'+'\n'
    for start in range(0,len(comparisons),10):chapter+=table('All-day causal sheaf gains over every causal control (percent); retrospective sheaf remains separately tabulated in the underlying cases.',['Code','VAR','City mean','Seasonal','Gaussian','Spatial sheaf'],[[r['code']]+[number(r[k]) for k in ['var_causal','city_mean','seasonal','gaussian','spatial_sheaf']] for r in comparisons[start:start+10]])
    chapter+=r'\begin{figure}[p]\centering\includegraphics[width=\linewidth,height=.8\textheight,keepaspectratio]{figures/expanded_stress}\caption{Eighteen prespecified history, outage-horizon and additional context-loss configurations for every expansion cohort. Colors saturate at 30\% magnitude for readability; exact values and paired case counts are preserved in the stress-result table. Positive values favor sheaf, negative values favor VAR.}\end{figure}'+'\n'
    chapter+='\\subsection{Uncertainty, faults and mathematical verification}\n'
    for start in range(0,len(rows),10):
        values=[]
        for r in rows[start:start+10]:
            g=cov[(cov.cohort==r['cohort'])&cov.method.eq('pooled_fixed')]
            overall=g[g.group=='all'].iloc[0];extreme=g[g.group=='extreme'].iloc[0]
            values.append([r['code'],int(overall.cases),number(100*overall.coverage),number(overall.finite_mean_width),number(100*overall.full_set_fraction),int(extreme.cases),number(100*extreme.coverage)])
        chapter+=table('No-feedback pooled PM2.5 intervals. Width is a full interval width in micrograms per cubic metre; extremes use training-only station thresholds.',['Code','Cells','Coverage (\\%)','Finite width','Full sets (\\%)','Extreme cells','Extreme coverage (\\%)'],values)
    chapter+=r'Each compatible task family is repeated: seven reconstruction methods, partial-channel/snapshot tests, eighteen outage/history/context combinations, strict complete-network and available-candidate fault studies, no-feedback and separately labelled revealed-label intervals, block intervals, structured rank and exhaustive four-block margins, functional and joint certificates, nominal dual refinements, operator uncertainty, shared-setting ablations, validation-selected hybrids, repair and illustrative environmental thresholds. Complete CSV tables preserve denominators and configuration fields. Fault-support success includes the alarm requirement; clean-state false alarms and unknown fault-count selection remain distinct from localization. These are injected faults, not labelled operational malfunctions. Thresholds are illustrative monitoring proxies; they do not claim official regulatory validity.'+'\n'
    chapter+=r'\begin{figure}[p]\centering\includegraphics[width=\linewidth,height=.8\textheight,keepaspectratio]{figures/expanded_coverage}\caption{No-feedback PM2.5 intervals: overall versus training-threshold extreme coverage, with cell denominators. The dashed line marks nominal 90\%; correlated cell counts are not independent replications.}\end{figure}'+'\n'
    chapter+=r'\begin{figure}[p]\centering\includegraphics[width=\linewidth,height=.8\textheight,keepaspectratio]{figures/expanded_faults}\caption{Alarm and exact-support rates for two injected faults. White rows indicate no eligible 24-hour episode, not zero success; the separate sampled-hour table reports compatible observations. Weak, stuck or multiplicative faults limit robustness. Episode hours are dependent; these are not field-fault success probabilities.}\end{figure}'+'\n'
    for start in range(0,len(mathrows),10):chapter+=table('Actual-map theorem checks and joint-certificate decision availability. Planted cases check implementation, not empirical pollution.',['Code','Rank sets','Agree','Thin mismatch','Actual states','Abstain','Inconsistent'],[[r['code'],r['sets'],str(r['agree']),r['thinning_disagreements'],r['actual_states'],r['abstentions'],r['inconsistent']] for r in mathrows[start:start+10]])
    chapter+=r'Strict fault studies require complete training, selection and calibration vectors; missing selection vectors or insufficient calibration produce an explicit unavailable status, rather than an estimated success rate. A finite 95\% split-conformal order statistic requires at least nineteen calibration scores. Real-state residual certificates require a finite complete-vector calibration budget; where none exists, rank enumeration and planted algorithm checks remain applicable but actual-state certificates are unavailable. No test-label-derived budget is substituted.'+'\n'
    for start in range(0,len(operation),10):chapter+=table('Exploratory validation-selected hybrid, shared-setting identity ablation and artificial repair; negative gains are retained. The identity column compares original sheaf with identity maps, not VAR. Repair compares a selected station repair with trace-policy repair.',['Code','Hybrid vs VAR (\\%)','Original vs identity (\\%)','Repair cases','Repair vs trace (\\%)'],[[r['code'],number(r['hybrid_vs_VAR']),number(r['identity_ablation_gain']),r['repair_cases'],number(r['repair_gain_vs_trace'])] for r in operation[start:start+10]])
    for start in range(0,len(operation),10):chapter+=table('Fault alarms and exact recovery: episode and sampled-hour studies have different eligibility and cannot be pooled. n/a episode FPR means no eligible episode.',['Code','Episodes','Clean episode h','Episode FPR (\\%)','Clean sampled h','Sampled FPR (\\%)','Two opposite exact (\\%)'],[[r['code'],r['sampled_fault_episodes'],r['episode_clean_hours'],number(None if r['episode_false_alarm'] is None else 100*r['episode_false_alarm']),r['hour_clean_cases'],number(None if r['hour_false_alarm'] is None else 100*r['hour_false_alarm']),number(None if r['hour_two_opposite_exact'] is None else 100*r['hour_two_opposite_exact'])] for r in operation[start:start+10]])
    chapter+='\\subsection{Environmental proxies and real-missingness validation}\n'
    chapter+=r'The illustrative hourly PM2.5 threshold is 60 micrograms per cubic metre; it is not an official hourly AQI rule. Many additional networks have few or zero positive observed hours. Sensitivity is undefined with no positives, and a high false-alarm rate can make a method operationally unsuitable despite lower RMSE. Additional missing-context masks are transplanted to a different fully observed station-day for validation; they do not reveal the original missing pollution, verify personal exposure or establish emissions reduction. Sparse mask cells are a selection-biased exploratory check, not a substitute for the outage endpoint.'+'\n'
    for start in range(0,len(environmental),10):chapter+=table('Illustrative hourly threshold and actual-mask transplantation. n/a sensitivity means zero positives. Mask gain is physical PM2.5 RMSE reduction against city mean.',['Code','Hours/positive','Sens. (\\%)','Sheaf FPR (\\%)','VAR FPR (\\%)','Mask cells','Mask gain (\\%)'],[[r['code'],f"{r['hours']}/{r['positive_hours']}",number(100*r['sensitivity']),number(100*r['false_alarm_rate']),number(100*r['var_false_alarm_rate']),r['mask_cells'],number(r['mask_gain'])] for r in environmental[start:start+10]])
    chapter+='\\subsection{What the expansion resolves and what remains}\n'
    chapter+=r'The expansion addresses narrow geographic coverage, a small original endpoint, short secondary testing, missing physical-error comparisons, dependence sensitivity and incomplete reporting of failure regions. It does not prove that the method wins in every network. Sparse extremes, synthetic fault truth, absence of operational outage labels, limited neural benchmarking, nonrandom site selection and unproven mathematical priority remain explicit limitations. The environmental evidence concerns reconstructed monitoring continuity and simulated decision proxies, not demonstrated emissions reduction or health benefit. More test rows strengthen an audit only when their provenance, information sets and dependencies are preserved.'+'\n'
    chapter+=f'Independent inference replay verifies all seven method losses for {int(results.cases.sum()):,} expansion station-days; source/tensor, frozen hashes, interval arithmetic and conditional mathematical checks passed {int(results.audit_checks.sum()):,} checks. This is an internal reproducibility audit, not independent refitting, external proof review or guaranteed submission eligibility. The oldest additional records are 14 October 2025; the twelve-month window must be rechecked against the actual submission date.\n'
    chapter=chapter.replace(r'\\%',r'\%')
    chapter=chapter.replace(f'protocol selected {len(independent)} additional regional cohorts',f'protocol selected {len(independent)+len(unavailable)} additional regional cohorts, of which {len(independent)} had usable test observations')
    (SRC/'expanded_section.tex').write_text(chapter,encoding='utf-8')
    bib=(SRC/'references.bib').read_text()+r'''
@misc{nswapi2026,author={{NSW Government}},title={Air quality API: official data services and hourly observation schema},year={2026},url={https://www.airquality.nsw.gov.au/air-quality-data-services/air-quality-api},note={Original API requests and responses preserved; accessed 7 October 2026}}
@misc{nswtime2026,author={{NSW Government}},title={Air quality data explorer: Australian Eastern Standard Time convention},year={2026},url={https://www.airquality.nsw.gov.au/air-quality-data-services/data-explorer},note={The reporting clock remains AEST during daylight-saving months}}
@article{breiman2001,author={Leo Breiman},title={Random Forests},journal={Machine Learning},volume={45},number={1},pages={5--32},year={2001},doi={10.1023/A:1010933404324}}
'''
    (SRC/'references.bib').write_text(bib,encoding='utf-8')
    followup_path=ROOT/'Research_Release/extensions/weakness_resolution/report.py'
    followup_spec=importlib.util.spec_from_file_location('weakness_report',followup_path)
    followup=importlib.util.module_from_spec(followup_spec);followup_spec.loader.exec_module(followup)
    followup_results=followup.compose(SRC)
    review_file=REVIEW/'EXPANDED_EVIDENCE_REVIEW.md'
    review_text=review_file.read_text(encoding='utf-8')
    review_text=review_text.replace('Neural comparison remains Indian-only. Extra non-neural replications do not establish international neural superiority.',
        'Three fixed external provider cohorts now supplement the original Indian neural comparison, with all seeds retained. This does not establish international neural superiority. See WEAKNESS_RESOLUTION_REVIEW.md for the complete follow-up.')
    review_file.write_text(review_text,encoding='utf-8')
    text=(PACKAGE/'Three_Network_Source/three_network_full.tex').read_text(encoding='utf-8')
                                                                         
    marker=r'\subsection{Generic structural rank and actual structured maps}'
    continuity=(HERE/'theory_continuity.tex').read_text(encoding='utf-8')
    if text.count(marker)!=1:raise RuntimeError('Theory insertion anchor changed')
    text=text.replace(marker,continuity+'\n'+marker,1)
    theory=(SRC/'complete_theory.tex').read_text(encoding='utf-8')
    if theory.count(marker)!=1:raise RuntimeError('Proof-source insertion anchor changed')
    theory=theory.replace(marker,continuity+'\n'+marker,1)
    progression_spec=importlib.util.spec_from_file_location('theory_progression',HERE/'theory_progression.py')
    progression=importlib.util.module_from_spec(progression_spec);progression_spec.loader.exec_module(progression)
    theory=progression.reorganize(theory)
    theory=theory.replace('The two recent fitted operators have zero numerical kernel',
        'The two initial recent fitted operators have zero numerical kernel')
    (SRC/'complete_theory.tex').write_text(theory,encoding='utf-8')
    start=text.index(r'\section{Mathematical results}')
    end=text.index('The preserved earlier supplementary acquisition',start)
    text=text[:start]+progression.main_narrative()+'\n'+text[end:]
    start=text.index(r'\section{Complete conditional mathematical statements and proofs}')
    end=text.index(r'\section{Detailed recent numerical supplement}',start)
    text=text[:start]+theory+'\n'+r'\clearpage'+'\n'+text[end:]
    text=text.replace('The central mathematical contribution is a coupled pollutant-threshold certificate over unknown bounded-fault explanations, accompanied by an exact nominal correction for approximate dual vectors.',
        'The mathematical treatment develops six linked propositions: actual-map graph recoverability, quantity identification under unknown faults, equivalent-support search compression, joint threshold certification, interruption-safe search refinement and exact nominal correction of approximate dual bounds.')
    text=text.replace('Research paper: three recent networks','Research paper: broad recent-data validation')
    text=text.replace(r'{6 October 2026\par}',r'{7 October 2026\par}')
    text=text.replace('All empirical fitting, model selection, calibration and testing in this paper use observations recorded during 2026.','Empirical observations fall between 14 October 2025 and early October 2026. Original 2026 benchmarks are preserved; additional recent external cohorts and a reused-observation Indian sensitivity are identified separately.')
    abstract=f'''Air-quality outages interrupt local pollution monitoring. SheafCode learns vector-valued spatial and temporal consistency maps for causal reconstruction, with conditional certificates for pollutant quantities and unknown sensor faults. Original recent benchmarks are preserved, and a fixed exploratory expansion adds {len(independent)} regional cohorts from official US, UK and Australian archives: {sites} selected station memberships and {total:,} paired artificial-outage station-days. Cohorts are selected on training availability, with separate chronological fitting, selection, calibration and test blocks. All added observations begin on or after 14 October 2025. The original Indian primary result remains 16.5\\% lower normalized RMSE than tuned full-covariance causal VAR (77 station-days; exploratory weekly interval 12.6--20.2\\%). Across the added external cohorts, causal sheaf has positive normalized all-day gains in {positive}/{len(independent)} networks; every unfavorable comparison is retained. Physical PM2.5 errors and seven-/fourteen-day dependence sensitivities are reported separately. Compatible fault, interval, stress, ablation, repair and mathematical checks are repeated on the new fitted operators. Independent implementations replay all seven method losses, while original records and hashes support source audits. Extreme-cell undercoverage, weak-fault failures and conditional-certificate abstentions limit operational claims. The mathematical contribution specializes established sheaf, sparse-uniqueness and convex-dual theory; first-ever priority is not established. Demonstrated environmental value concerns monitoring continuity and simulated decision proxies, rather than measured emissions reduction. This is exploratory refitting, not blinded transfer or guaranteed universal dominance.'''
    longgain=results[results.provider=='OpenAQ'].gain.iloc[0]
    abstract=abstract.replace('Original recent benchmarks are preserved,',f'A longer reused-observation Indian sensitivity gives {longgain:.1f}\\% normalized improvement. Original recent benchmarks are preserved,')
    abstract+=f" Sheaf improves over a causal forest in {followup_results['external_forest_wins']}/{followup_results['external_cohorts']} added cohorts; three fixed external retrospective GRIN comparisons retain every seed."
    text=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _:r'\begin{abstract}'+'\n'+abstract+'\n'+r'\end{abstract}',text,count=1,flags=re.S)
    text=text.replace('\\section{Discussion and limitations}','\\input{expanded_section.tex}\n\\section{Discussion and limitations}',1)
    text=text.replace('\\section{Discussion and limitations}','\\input{weakness_section.tex}\n\\section{Discussion and limitations}',1)
    text=text.replace('This study refits and evaluates the method using only 2026 observations','The initial benchmarks refit and evaluate the method using 2026 observations')
    text=text.replace('The study evaluates this hypothesis on entirely recent public observations.','The study evaluates this hypothesis on recent public observations; the broad external expansion below retains all outcomes rather than selecting only gains.')
    text=text.replace('All recent empirical fitting and evaluation use 2026 observations, within twelve months of this study.',
        'Recent empirical fitting and evaluation use observations recorded on or after 14 October 2025. Observation-age eligibility must be rechecked against the actual submission date.')
    text=text.replace('\\section{Discussion and limitations}',
        '\\section{Discussion and limitations}\nThe broader exploratory comparison adds twenty evaluable external cohorts. Causal sheaf has numerical normalized gains in thirteen of twenty against VAR and eighteen of twenty against the new forest. Retrospective neural comparisons favor sheaf in the fixed US and UK cohorts but slightly favor GRIN in Australia; every seed remains visible. The larger evidence base mitigates narrow baseline and geographic coverage without proving uniform dominance. Risk grouping improves extreme coverage but remains far below nominal, and temporal fault sensitivity can rise alongside false alarms. These follow-ups do not replace the original primary endpoint.\n',1)
    (SRC/'expanded_full_manuscript.tex').write_text(text,encoding='utf-8')
    spec=importlib.util.spec_from_file_location('builder',ROOT/'audit_tools/build_iris_submission.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder);builder.SRC=SRC;builder.OUT=PACKAGE
    target=PACKAGE/'Upload_Files/SheafCode_Full_Empirical_Manuscript.pdf';builder.compile_pdf('expanded_full_manuscript',target)
    synopsis=r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=1in]{geometry}\usepackage{amsmath,booktabs,hyperref}
\hypersetup{pdfauthor={},pdftitle={SheafCode: research synopsis}}
\setlength{\parindent}{0pt}\setlength{\parskip}{.5em}
\begin{document}\begin{center}{\Large\bfseries SheafCode}\par Causal air-quality reconstruction and conditional pollution certificates\end{center}
\textbf{Research question.} Can training-fitted vector-valued spatial and temporal consistency maps reconstruct a dark monitoring station more accurately than matched causal controls, and when can uncertainty support a pollution-threshold decision?

\textbf{Method and mathematical contribution.} Chronological training, model selection, interval calibration and test blocks are separate. A quadratic sheaf decoder includes the full-state prior cross term and uses only the prediction-time prefix. Controls include tuned full-covariance VAR, city mean, seasonal persistence, Gaussian conditioning and spatial sheaf. The mathematical work specializes actual-map kernel reduction, quantity-specific uncertainty, unknown bounded-fault feasible sets, sharp joint pollutant-threshold envelopes and nominal dual corrections. Written proofs and computational checks supplement one another; first-ever general mathematical priority is not established.

\textbf{Original evidence.} The fixed Indian primary test reports a 16.5\% normalized RMSE reduction versus tuned VAR on 77 station-days (16 dates; exploratory seven-day interval 12.6--20.2\%). US and UK original recent benchmarks are preserved in the full manuscript. A longer Indian resplit is a sensitivity using previously accessed observations, not an independent dataset.
'''
    synopsis+=f'\\textbf{{Broader validation.}} A fixed exploratory expansion selected {len(independent)+len(unavailable)} further regional cohorts from official EPA, AURN and NSW monitoring archives. {len(independent)} have usable test observations, providing {sites} selected station memberships, {cells:,} observed pollutant-hour cells and {total:,} paired eligible outage station-days; {len(unavailable)} are explicitly unavailable. Causal sheaf has positive normalized all-day gain versus VAR in {positive}/{len(independent)} evaluable new cohorts. Negative comparisons remain visible. Station-days are dependent, and separate seven-/fourteen-day sensitivities and physical PM2.5 errors are reported. All new records begin on or after 14 October 2025.\n'
    synopsis+=r'''\textbf{Additional experiments and checks.} Compatible partial-channel, snapshot, eighteen outage-stress, strict/available fault, no-feedback and revealed-label interval, ablation, hybrid, repair and environmental-proxy studies are repeated. Source identities, units, complete calendars, raw request/response records, hashes and independent seven-method inference replay support reproducibility. Inadequate calibration or absent test truth produces an unavailable status rather than a manufactured result.

\textbf{Environmental relevance and limitations.} The evidence concerns monitoring continuity and simulated threshold decisions. It does not measure emissions reduction or health improvement. Extreme-cell interval undercoverage, weak and multiplicative fault failures, certificate abstentions, public-data selection bias and temporal dependence remain explicit. The neural comparison is confined to the original Indian benchmark; the expansion is exploratory rather than blinded or externally preregistered.

\textbf{Submission scope and assistance.} The full manuscript contains the recent empirical evidence and complete mathematical treatment in one document. Observation-age eligibility must be checked at the actual submission date. AI assistance in code, mathematical derivation/checking, audits and writing is disclosed; student understanding and contributions must be represented accurately.
\end{document}'''
    synopsis=synopsis.replace('\\textbf{Broader validation.}', '\n\n\\textbf{Broader validation.}').replace('\\textbf{Additional experiments and checks.}', '\n\n\\textbf{Additional experiments and checks.}')
    synopsis=synopsis.replace('The mathematical work specializes actual-map kernel reduction, quantity-specific uncertainty, unknown bounded-fault feasible sets, sharp joint pollutant-threshold envelopes and nominal dual corrections.',
        'Six linked propositions address graph recoverability, quantity identification under faults, equivalent-support BFS compression, joint pollutant thresholds, interrupted search and nominal numerical bounds. Assumption changes between unrestricted and bounded faults are explicit.')
    synopsis=synopsis.replace('The neural comparison is confined to the original Indian benchmark;',
        'Added controls include a causal forest and three fixed external retrospective GRIN cohorts with all seeds retained;')
    (SRC/'expanded_synopsis.tex').write_text(synopsis,encoding='utf-8');builder.compile_pdf('expanded_synopsis',PACKAGE/'Upload_Files/SheafCode_Full_Empirical_Synopsis.pdf')
    (PACKAGE/'Upload_Files/Project_Abstract_Expanded.txt').write_text(re.sub(r'\\%', '%',abstract),encoding='utf-8')
    save('publication_status.json',{'status':'generated_pending_document_audit','manuscript':str(target.relative_to(ROOT)),'manuscript_sha256':sha(target),'new_external_cohorts':len(independent),'new_external_station_days':total,'new_external_observed_cells':cells,'positive_var_gain_cohorts':positive,'all_cohort_audit_checks':int(results.audit_checks.sum())})
    print('Expanded manuscript generated',len(names),'cohorts',total,'new external cases',flush=True)
if __name__=='__main__':main()
