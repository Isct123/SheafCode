                                                                               
from pathlib import Path
import sys,json,importlib.util,os
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'Research_Release/weakness_resolution_results';BASE=ROOT/'Research_Release/expanded_validation_results'
PACKAGE=ROOT/'IRIS_Submission_Recent_Data';REVIEW=PACKAGE/'Review_Notes'
sys.path.insert(0,str(ROOT/'Research_Release/extensions/recent_data'))
import common
import numpy as np,pandas as pd

def number(x):return 'undefined' if x is None or not np.isfinite(x) else f'{x:.2f}'
def ci(x):return 'suppressed' if x is None else f'{x[0]:.2f} to {x[1]:.2f}'
def esc(s):return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def table(caption,heads,rows):
    return '\\begin{table}[htbp]\\centering\\small\\caption{'+caption+'}\\resizebox{\\linewidth}{!}{\\begin{tabular}{'+'l'*len(heads)+'}\\toprule\n'+' & '.join(heads)+r'\\\midrule'+'\n'+'\n'.join(' & '.join(esc(x) for x in row)+r'\\' for row in rows)+'\n\\bottomrule\\end{tabular}}\\end{table}\n'
def figure(name,caption):return '\\begin{figure}[htbp]\\centering\\includegraphics[width=.96\\linewidth]{figures/'+name+'.pdf}\\caption{'+caption+'}\\end{figure}\n'

def compose(src,allow_pending=False):
    REVIEW.mkdir(exist_ok=True);(src/'figures').mkdir(exist_ok=True)
    if not allow_pending:
        audit=json.loads((OUT/'AUDIT.json').read_text())
        if audit['status']!='passed':raise RuntimeError('Follow-up audit incomplete')
        for path,digest in audit['implementation_sha256'].items():
            if common.sha(ROOT/path)!=digest:raise RuntimeError('Audited follow-up source changed: '+path)
    catalog=pd.read_csv(BASE/'COHORT_RESULTS.csv');codes=dict(zip(catalog.cohort,catalog.code))
    grid_audit=json.loads((OUT/'threshold_grid/AUDIT.json').read_text())
    if grid_audit['status']!='passed' or grid_audit['source_sha256']!=common.sha(HERE/'threshold_sensitivity.py'):raise RuntimeError('Concentration-grid audit incomplete')
    for path,digest in grid_audit['input_sha256'].items():
        if common.sha(ROOT/path)!=digest:raise RuntimeError('Concentration-grid input changed')
    external=set(catalog[catalog.provider!='OpenAQ'].cohort)
    nonlinear=json.loads((OUT/'nonlinear/status.json').read_text())['cohorts']
    rows=[]
    for r in nonlinear:
        a=r['summary']['7']['mse'];b=r['summary']['7']['pm25_raw_mse']
        rows.append({'cohort':r['cohort'],'code':codes[r['cohort']], 'external':r['cohort'] in external,
                     'cases':r['test_station_days'],'leaf':r['leaf'],'gain':a['gain_percent'],'ci7':a['ci95'],
                     'ci14':r['summary']['14']['mse']['ci95'],'physical_gain':b['gain_percent']})
    nf=pd.DataFrame(rows);nf.to_csv(OUT/'NONLINEAR_SUMMARY.csv',index=False)
    ext=nf[nf.external];wins=int((ext.gain>0).sum())
    section='\\clearpage\\section{Exploratory follow-up on remaining empirical weaknesses}\n'
    section+=f'The following analyses use the same frozen expanded datasets and add no independent observations or cohorts. Their protocols were fixed before each follow-up analysis, after earlier test outcomes had been examined; they are exploratory, not prospectively preregistered confirmation. The original primary endpoint and all preceding results are retained. A stronger causal nonlinear control, three-provider neural replication, prediction-risk intervals, equal-false-alarm alert comparisons, validation-only model choice and temporal fault detection directly test specific weaknesses. The follow-up audit separately checks checkpoints, eligibility, causal inputs, source identities and numerical summaries.\n'
    section+='\\subsection{A matched causal nonlinear control}\n'
    section+='The nonlinear benchmark uses the established random-forest method~\\cite{breiman2001}, not a new model invention.\n'
    section+=r'Random forests use current surviving readings, one- and twenty-four-hour lags, the last pre-outage target reading, missingness indicators, station identity, calendar phase and outage duration. The hidden station is removed before feature construction; every feature at an outage hour uses only that hour and its past. Channel-specific forests contain 64 trees, maximum depth 12 and minimum leaf size chosen from 5 or 20 on the original selection cases. Up to 80 training days and twelve selection days are evenly sampled using the existing rule. The control shares every test station-day and target cell with causal sheaf. This is a finite computational comparison, not a claim of optimal forest tuning.'+'\n'
    section+=f'Causal sheaf has a positive normalized RMSE gain versus this control in {wins}/{len(ext)} added external cohorts. This is a numerical win count, not a significance count or proof of universal superiority. In particular, unfavorable forests comparisons are retained. Fourteen-day intervals are suppressed where fewer than five calendar bins are available. The Indian sensitivity remains reused evidence.\n'
    for start in range(0,len(nf),11):
        section+=table('All nonlinear-control comparisons. Positive gain means lower sheaf RMSE; normalized and physical concentration gains differ. Intervals are exploratory, unadjusted paired calendar-bin sensitivities.',
                       ['Code','Cases','Leaf','Norm. gain (\\%)','7-day interval','14-day interval','PM gain (\\%)'],
                       [[r.code,r.cases,r.leaf,number(r.gain),ci(r.ci7),ci(r.ci14),number(r.physical_gain)] for r in nf.iloc[start:start+11].itertuples()])
    section+='\\subsection{Three fixed external neural comparisons}\n'
    section+=r'The official pinned GRIN architecture~\cite{cini2022grin} is refitted in one training-mask-eligible cohort from each added provider. Cohorts are the first lexicographic provider member with at least ten eligible training cases and three selection cases, without using test performance: EPA 04-013, AURN central and NSW Illawarra. Each uses seeds 0, 1 and 2, widths 32 and 64, at most thirty epochs, seven-epoch early stopping and selection-half validation. Every seed is shown. GRIN is bidirectional and uses the complete surviving-observation window; its matched comparator is retrospective sheaf, not the causal headline method. This extends a narrow original Indian benchmark without establishing superiority on every external cohort or equivalence to published GRIN benchmark scores.'+'\n'
    neural=[]
    for item in json.loads((OUT/'neural/SELECTION.json').read_text())['selections']:
        folder=OUT/'neural'/item['cohort'];file=folder/'grin_summary.json'
        if not file.exists():
            if allow_pending:continue
            raise RuntimeError('Neural training incomplete: '+item['cohort'])
        reference=pd.read_csv(folder/'outage_cases.csv',dtype={'station':str})
        predictions=pd.read_csv(folder/'grin_cases.csv',dtype={'station':str})
        choices=json.loads((folder/'grin_choices.json').read_text())
        for choice in choices:
            seed=choice['seed'];joined=pd.concat([predictions[predictions.seed.eq(seed)],reference])
            a=common.paired(joined,'sheaf_retrospective','grin_retrospective')
            b=common.paired(joined,'sheaf_retrospective','grin_retrospective',days=14)
            neural.append({'cohort':item['cohort'],'code':codes[item['cohort']],'seed':seed,
                           'width':choice['width'],'epoch':choice['epoch'],'cases':a['cases'],
                           'gain':a['gain_percent'],'ci7':a['ci95'],'ci14':b['ci95']})
    neuralf=pd.DataFrame(neural);neuralf.to_csv(OUT/'NEURAL_SUMMARY.csv',index=False)
    if len(neuralf):section+=table('Retrospective sheaf versus official refitted GRIN. Positive gain favors sheaf. All selected seeds, selection widths and best epochs remain visible; these are dependent refits on three fixed cohorts.',
                                 ['Code','Seed','Width','Epoch','Cases','Gain (\\%)','7-day interval','14-day interval'],
                                 [[r.code,r.seed,r.width,r.epoch,r.cases,number(r.gain),ci(r.ci7),ci(r.ci14)] for r in neuralf.itertuples()])
    if allow_pending and len(neuralf)!=9:section+='External neural training is unfinished; this preview is not a publication or final audit.\n'
    section+='\\subsection{No-feedback prediction-risk intervals}\n'
    section+=r'To address extreme-cell undercoverage, a prediction-time risk group is defined by whether the maximum of sheaf and city PM2.5 predictions exceeds the station training 75th percentile. Calibration is split by this covariate and either by station (risk-station) or pooled after dividing absolute error by the training station standard deviation (risk-pooled-scaled). The fixed quantile convention returns a full interval when insufficient calibration scores support the nominal 90\% rank. No hidden test labels update intervals. A training 95th-percentile truth threshold defines extreme cells for evaluation only. The snapshot information set is retained; these are not demonstrated intervals for an uninterrupted twenty-four-hour causal outage. Mondrian-style grouping does not guarantee coverage under arbitrary temporal shift.'+'\n'
    ints=pd.read_csv(OUT/'uncertainty_decisions/INTERVAL_SUMMARY.csv')
    intervalrows=[]
    for loss in [0.,.2,.5]:
        for group in ['all','extreme']:
            for method in ['pooled_fixed','station_fixed','availability_scaled','risk_station','risk_pooled_scaled']:
                g=ints[ints.cohort.isin(external)&ints.context_loss.eq(loss)&ints.group.eq(group)&ints.method.eq(method)]
                nonempty=g[g.cells>0]
                cells=int(g.cells.sum());pooled=float((nonempty.coverage*nonempty.cells).sum()/cells) if cells else np.nan
                intervalrows.append({'loss':loss,'group':group,'method':method,'cohorts':len(nonempty),'cells':cells,
                                     'macro_coverage':nonempty.coverage.mean(),'pooled_coverage':pooled,
                                     'macro_finite_width':nonempty.finite_mean_width.mean(),
                                     'full_fraction':float((nonempty.full_set_fraction*nonempty.cells).sum()/cells) if cells else np.nan,
                                     'worst_station_quarter':nonempty.minimum_station_quarter_coverage.min()})
    aggregated=pd.DataFrame(intervalrows);aggregated.to_csv(OUT/'RISK_INTERVAL_AGGREGATES.csv',index=False)
    for loss in [0.,.2,.5]:
        g=aggregated[aggregated.loss.eq(loss)]
        labels={'pooled_fixed':'Pooled','station_fixed':'Station','availability_scaled':'Availability','risk_station':'Risk station','risk_pooled_scaled':'Risk pooled'}
        section+=table(f'External snapshot intervals with additional context loss {100*loss:.0f}\\%. Cohort-macro averages omit empty groups; cell-pooled coverage weights observations. Width is an unweighted mean of cohort finite widths in micrograms per cubic metre, with full intervals reported separately. These distinct summaries do not form a single independent pooled experiment.',
                       ['Group/method','Cohorts','Cells','Macro (\\%)','Pooled (\\%)','Width','Full (\\%)','Worst (\\%)'],
                       [[r.group+'/'+labels[r.method],r.cohorts,r.cells,number(100*r.macro_coverage),number(100*r.pooled_coverage),number(r.macro_finite_width),number(100*r.full_fraction),number(100*r.worst_station_quarter)] for r in g.itertuples()])
    original=aggregated[(aggregated.loss==0)&aggregated.group.eq('extreme')]
    fixed=original[original.method.eq('station_fixed')].iloc[0];risk=original[original.method.eq('risk_station')].iloc[0]
    section+=f'For no additional context loss, extreme-cell cohort-macro coverage changes from {100*fixed.macro_coverage:.1f}\\% for station-fixed intervals to {100*risk.macro_coverage:.1f}\\% for risk-station intervals ({risk.cells:,} extreme cells across {risk.cohorts} nonempty external cohorts). This is a partial empirical improvement, still far below 90\\%; it does not resolve the extreme-event weakness. Per-cohort seven-/fourteen-day coverage sensitivities, actual minimum station-quarter counts and all full-set frequencies are retained in the case and summary archive. No average of regional worst quarters is described as the global worst quarter.\n'
    riskrows=[]
    for name in sorted(external):
        subset=ints[ints.cohort.eq(name)&ints.context_loss.eq(0)]
        fixedall=subset[subset.method.eq('station_fixed')&subset.group.eq('all')].iloc[0]
        riskall=subset[subset.method.eq('risk_station')&subset.group.eq('all')].iloc[0]
        fixedext=subset[subset.method.eq('station_fixed')&subset.group.eq('extreme')].iloc[0]
        riskext=subset[subset.method.eq('risk_station')&subset.group.eq('extreme')].iloc[0]
        riskrows.append([codes[name],int(fixedall.cells),number(100*fixedall.coverage)+'/'+number(100*riskall.coverage),
                         int(fixedext.cells),number(100*fixedext.coverage)+'/'+number(100*riskext.coverage),
                         number(100*riskall.full_set_fraction),
                         number(100*riskall.minimum_station_quarter_coverage)+'/'+str(int(riskall.minimum_station_quarter_cells))])
    for start in range(0,len(riskrows),10):section+=table('Per-cohort snapshot interval denominators, without extra context loss. Coverage pairs are station-fixed / risk-station. Extreme cells exceed that station training 95th percentile. Worst-quarter coverage/count is the actual minimum station-quarter for risk-station intervals, not an averaged worst statistic.',
         ['Code','All cells','All coverage (\\%)','Extreme cells','Extreme coverage (\\%)','Full sets (\\%)','Worst quarter (\\%)/n'],riskrows[start:start+10])
    widths=pd.read_csv(OUT/'width_control/SUMMARY.csv');widths=widths[widths.cohort.isin(external)]
    excluded=json.loads((OUT/'width_control/EXCLUSIONS.json').read_text())
    excluded_external=[r for r in excluded if r['cohort'] in external]
    section+=f'An additional amendment tests a uniform-radius control equal to the risk-station average calibration radius, separately for each station/context condition. No test truth sets its width. It matches mean calibration width, not mean test width: changing risk frequencies under shift can change the test widths. Both risk groups must have finite calibration quantiles; {len(excluded_external)} external station/context combinations are unavailable under this calibration-only rule, with all excluded cells listed in the archive. Risk and control coverage are compared only on the resulting identical eligible rows.\n'
    widthrows=[]
    for (loss,group),g in widths.groupby(['loss','group']):
        good=g[g.cells>0]
        widthrows.append([f'{int(100*loss)}%/'+group,len(good),int(good.cells.sum()),
                          number(100*good.risk_coverage.mean())+'/'+number(100*good.control_coverage.mean()),
                          number(good.risk_width.mean())+'/'+number(good.control_width.mean())])
    section+=table('Calibration-mean-width control on matched calibration-supported rows. Coverage and mean finite test-width pairs are risk-station / uniform control, averaged across nonempty external cohorts. Width is in micrograms per cubic metre. The control is not matched to the test width; paired seven-/fourteen-day coverage-difference sensitivities are archived.',
                   ['Context loss/group','Cohorts','Cells','Coverage pair (\\%)','Test-width pair'],widthrows)
    section+='\\subsection{Alert quality at matched false-alarm limits}\n'
    section+=r'For the previously declared PM2.5 threshold proxy (60 micrograms per cubic metre), complete ROC curves and average precision compare methods on common observed hours. Tied prediction scores enter atomically. Reported sensitivity is the maximum attainable deterministic threshold sensitivity at or below each false-positive cap. Thresholds here use test labels to describe the curve and are not deployable calibration-selected rules. AUC summarizes ranking; it does not prove clinical, regulatory or operational impact. Cohorts without both positive and negative hours have undefined discrimination rather than perfect scores. Paired seven-/fourteen-day calendar-bin bootstrap differences are suppressed with fewer than five bins or when any bootstrap draw lacks either class.'+'\n'
    section+='The original calibration-selected alerts remain separate: each method uses a clean calibration 95th-percentile cutoff and its actual test false-positive rate is reported in the preceding environmental section. Equal calibration targets do not imply equal test false-positive rates after temporal shift. The added ROC analysis compares attained tradeoffs without rewriting those deployment-like rules.\n'
    alerts=json.loads((OUT/'uncertainty_decisions/ALERT_SUMMARY.json').read_text())
    uncertainty=json.loads((OUT/'uncertainty_decisions/PAIRED_ROC_UNCERTAINTY.json').read_text())
    rocrows=[]
    for name in sorted(external):
        a=next(r for r in alerts if r['cohort']==name and r['method']=='sheaf_causal')
        b=next(r for r in alerts if r['cohort']==name and r['method']=='var_causal')
        u=next(r for r in uncertainty if r['cohort']==name and r['bin_days']==7)
        valid=a['status']=='evaluated'
        rocrows.append({'cohort':name,'code':codes[name],'positives':a['positive_hours'],'negatives':a['negative_hours'],
                        'sheaf_auc':a.get('auc'),'var_auc':b.get('auc'),
                        'sheaf_sens05':a['sensitivity_under_fpr_cap']['0.05'] if valid else None,
                        'var_sens05':b['sensitivity_under_fpr_cap']['0.05'] if valid else None,
                        'ci7':u['fpr05_sensitivity_ci95'],'ci14':next(r for r in uncertainty if r['cohort']==name and r['bin_days']==14)['fpr05_sensitivity_ci95']})
    pd.DataFrame(rocrows).to_csv(OUT/'MATCHED_ALERT_SUMMARY.csv',index=False)
    for start in range(0,len(rocrows),10):
        section+=table('Matched alert ranking and deterministic sensitivity at FPR no greater than 5\\%. Sensitivity-difference intervals are in percentage points (sheaf minus VAR); undefined classes and suppressed uncertainty are retained.',
                       ['Code','Pos./neg. hours','AUC sheaf/VAR','Sens. sheaf/VAR (\\%)','Difference interval 7d','Difference interval 14d'],
                       [[r['code'],f"{r['positives']}/{r['negatives']}",number(r['sheaf_auc'])+'/'+number(r['var_auc']),
                         number(None if r['sheaf_sens05'] is None else 100*r['sheaf_sens05'])+'/'+number(None if r['var_sens05'] is None else 100*r['var_sens05']),
                         ci(None if r['ci7'] is None else [100*x for x in r['ci7']]),ci(None if r['ci14'] is None else [100*x for x in r['ci14']])] for r in rocrows[start:start+10]])
    grid=pd.read_csv(OUT/'threshold_grid/SUMMARY.csv');grid=grid[grid.cohort.isin(external)]
    section+='Because severe-hour counts are sparse, a further fixed exploratory sensitivity retains every threshold in the concentration grid 15, 25, 35, 60 and 90 micrograms per cubic metre. These remain hourly concentration proxies; the grid does not substitute hourly readings for official daily limits. No threshold is chosen as a new primary endpoint. Every method AUC, average precision, positive/negative count and descriptive false-positive-constrained sensitivity is archived, including undefined classes. More positives at a lower threshold do not establish severe-event performance.\n'
    gridrows=[]
    for name in sorted(external):
        row=[codes[name]]
        for threshold in [15,25,35,60,90]:
            a=grid[grid.cohort.eq(name)&grid.threshold.eq(threshold)&grid.method.eq('sheaf_causal')].iloc[0]
            b=grid[grid.cohort.eq(name)&grid.threshold.eq(threshold)&grid.method.eq('var_causal')].iloc[0]
            difference=100*(a.sensitivity_fpr05-b.sensitivity_fpr05)
            row.append(number(difference)+'/'+str(int(a.positive_hours)))
        gridrows.append(row)
    for start in range(0,len(gridrows),10):section+=table('Fixed concentration-grid robustness: sheaf-minus-VAR sensitivity difference in percentage points at FPR no greater than 5\\%, followed by the positive-hour count. Every threshold uses the same observed test predictions; undefined class results remain labelled undefined. Cutoffs are descriptive, not deployed calibration choices.',
           ['Code','15: diff./n','25: diff./n','35: diff./n','60: diff./n','90: diff./n'],gridrows[start:start+10])
    section+='\\subsection{Validation-only model selection is not a universal safeguard}\n'
    guard=json.loads((OUT/'selection_guard/SUMMARY.json').read_text());guardext=[r for r in guard if r['cohort'] in external]
    gaincount=sum(r['summary']['gain_percent']>0 for r in guardext)
    section+=f'A cohort-level selector chooses the lowest original selection MSE among sheaf, VAR and the new forest, with deterministic lexical tie-breaking. Test outcomes never choose its branch. The chosen method obtains positive test gains versus VAR in {gaincount}/{len(guardext)} external cohorts; it is identical to VAR where that branch is selected and can lose after distribution shift. It therefore does not turn a validation win into guaranteed test improvement. Every branch and test result is archived.\n'
    for start in range(0,len(guard),11):section+=table('Validation-only model choice. Positive gain favors the selected branch against VAR; neither adverse test outcomes nor reused Indian evidence are discarded.',
        ['Code','Selected branch','Cases','Gain (\\%)','7-day interval','PM gain (\\%)'],
        [[codes[r['cohort']],r['chosen_method'],r['case_count'],number(r['summary']['gain_percent']),ci(r['summary']['ci95']),number(r['physical_summary']['gain_percent'])] for r in guard[start:start+11]])
    section+='\\subsection{Temporal evidence for synthetic weak-fault detection}\n'
    section+=r'The original unknown-count alarm is compared with thresholded pointwise whitened syndrome norm and its causal four-hour mean. Both new cutoffs use a separate clean calibration block at a nominal 5\% upper tail. Four-hour calibration windows must be consecutive, fully within that block and have at least two complete candidate stations at each hour. Test comparisons use identical eligible rows and include clean states. The temporal detector starts with the preceding three surviving clean observations of the simulated episode, as an observed-history warm start, and subsequently uses only current and previous syndrome scores. Calendar shift can move clean false-positive rates away from 5\%; a temporal gain at higher test false alarms is not a free improvement. This is detection, not support localization, and the injected trajectories remain synthetic.'+'\n'
    temporal=pd.read_csv(OUT/'fault_temporal/SUMMARY.csv');temp_rows=[];latencies=[]
    for name in sorted(external):
        for method in ['original_count_gate','point_norm','four_hour_norm']:
            g=temporal[temporal.cohort.eq(name)&temporal.method.eq(method)]
            clean=g[g.kind.eq('clean')];drift=g[g.kind.eq('drift')&g.true_count.eq(1)]
            if clean.empty or drift.empty:continue
            temp_rows.append({'cohort':name,'code':codes[name],'method':method,'clean_hours':int(clean.hours.sum()),
                             'clean_fpr':float((clean.alarm_rate*clean.hours).sum()/clean.hours.sum()),
                             'drift_hours':int(drift.hours.sum()),'drift_sensitivity':float((drift.alarm_rate*drift.hours).sum()/drift.hours.sum())})
        file=OUT/'fault_temporal'/name/'cases.csv'
        if not file.exists():continue
        f=pd.read_csv(file)
        for (method,kind,count,episode),g in f[f.kind.ne('clean')].groupby(['method','kind','true_count','episode']):
            g=g.sort_values('timestamp');alarmed=g[g.alarm]
            first=(pd.Timestamp(alarmed.timestamp.iloc[0])-pd.Timestamp(g.timestamp.iloc[0])).total_seconds()/3600 if len(alarmed) else np.nan
            latencies.append({'cohort':name,'method':method,'kind':kind,'true_count':count,'episode':episode,
                              'detected':bool(len(alarmed)),'first_alarm_hours_from_first_eligible_row':first,
                              'eligible_hours':len(g),'scope':'Latency starts at first eligible evaluated row; nondetections retained, not imputed as zero'})
    tf=pd.DataFrame(temp_rows);tf.to_csv(OUT/'TEMPORAL_COMPARISON.csv',index=False)
    pd.DataFrame(latencies).to_csv(OUT/'TEMPORAL_EPISODE_LATENCIES.csv',index=False)
    for start in range(0,len(external),10):
        names=sorted(external)[start:start+10];tab=[]
        for name in names:
            g=tf[tf.cohort.eq(name)]
            if g.empty:tab.append([codes[name],'unavailable','unavailable','unavailable']);continue
            pairs=[]
            for method in ['original_count_gate','point_norm','four_hour_norm']:
                r=g[g.method.eq(method)].iloc[0];pairs.append(f'{100*r.drift_sensitivity:.1f}/{100*r.clean_fpr:.1f}')
            tab.append([codes[name],*pairs])
        section+=table('One-fault drift sensitivity / clean false-positive rate (percent) on identical eligible synthetic hours. Full fault families, sample sizes, episodes and first-alarm latency with nondetections are archived. These cutoffs have equal nominal calibration targets, not necessarily equal test false alarms.',
                       ['Code','Original count gate','Point norm','Four-hour norm'],tab)
    section+=r'\paragraph{Why weak faults may remain undetectable.} In the relaxed syndrome model, let clean uncertainty be a Euclidean ball $B(0,\varepsilon)$ and let a fault shift the nominal syndrome by $a$. Clean and faulty explanation balls intersect if and only if $\|a\|_2\leq 2\varepsilon$. If they intersect, the triangle inequality gives this bound; conversely their midpoint lies in both balls. A deterministic observation-only detector must assign the same answer at an intersecting observation and therefore cannot be uniformly correct for both explanations. This is the classical bounded-noise separation obstruction, not an additional novelty claim. It does not assert that every ball residual is attainable under the structured model. Temporal repetition helps only with additional assumptions on residual variation and fault persistence; it cannot eliminate this worst-case ambiguity.'+'\n'
    section+='\\subsection{Novelty and environmental claims after these checks}\n'
    section+=r'The six-proposition progression remains intact. Spanning-tree elimination and holonomy specialize sheaf consistency~\cite{hansen2019spectral,ghrist2011network}; the unknown-support union argument specializes sparse uniqueness and functional observability~\cite{donoho2003,functional2022}; support-span compression exploits equivalence of nuisance explanations and is not claimed to be fixed-parameter tractable. Joint threshold envelopes specialize bounded-error feasible-set reasoning~\cite{bounded1990,boyd2004}; anytime refinement uses established branch-and-bound~\cite{land1960}; nominal dual correction connects to rigorous optimization bounds~\cite{jansson2004}. The monitoring-specific formulation and integration may be a project contribution, but experiments and renamed propositions cannot establish first-ever mathematical priority. No new numbered theorem is added merely to increase a count. External proof review and a broader priority search remain separate evidence needs.'+'\n'
    section+=r'The stronger forest comparison and added neural refits mitigate limited baseline coverage. Risk grouping partially improves extreme coverage; unequal false-alarm tradeoffs are now exposed rather than hidden. Validation-only selection and temporal fault detection are evaluated without promising uniform success. Artificial outages, synthetic fault labels, sparse extreme events, nonrandom cohort selection, temporal dependence and the absence of blinded external replication remain. None of these computations measures a reduction in emissions, personal exposure or disease, and regulatory AQI/GRAP decisions are not established. Environmental value supported here is reconstruction of monitoring information and conditional threshold reasoning. Those operational and priority limitations cannot be eliminated by adding more internal tests.'+'\n'
    os.environ['MPLCONFIGDIR']=str(OUT/'plot_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,6));y=np.arange(len(ext));ax.barh(y,ext.gain,color=np.where(ext.gain>0,'#24788a','#b55243'))
    for k,r in enumerate(ext.itertuples()):
        if r.ci7:ax.plot(r.ci7,[k,k],color='black',lw=1)
    ax.set_yticks(y,ext.code);ax.axvline(0,color='black',lw=.8);ax.set_xlabel('Sheaf normalized RMSE gain versus causal forest (%)');ax.invert_yaxis();fig.tight_layout();fig.savefig(src/'figures/followup_forest.pdf');plt.close(fig)
    section+=figure('followup_forest','Every external causal forest comparison with unadjusted seven-day exploratory intervals where supported. Three-case cohorts and suppressed intervals remain in the table; bars are not statistical significance labels.')
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,group in zip(axes,['all','extreme']):
        for method,label in [('station_fixed','Station fixed'),('risk_station','Risk station')]:
            g=ints[ints.cohort.isin(external)&ints.context_loss.eq(0)&ints.group.eq(group)&ints.method.eq(method)].set_index('cohort').reindex(sorted(external))
            ax.plot(np.arange(len(g)),100*g.coverage,'o-',ms=3,label=label)
        ax.axhline(90,color='grey',ls='--',lw=.7);ax.set_xticks(np.arange(len(external)),[codes[n] for n in sorted(external)],rotation=90);ax.set_title(group+' snapshot cells');ax.set_ylim(0,102);ax.legend(fontsize=8)
    axes[0].set_ylabel('Empirical coverage (%)');fig.tight_layout();fig.savefig(src/'figures/followup_risk.pdf');plt.close(fig)
    section+=figure('followup_risk','Overall and extreme snapshot coverage for each external cohort, without additional context loss. Missing extreme groups appear as gaps. The table reports denominators, finite widths and full sets; extreme coverage remains substantially below nominal.')
    if len(neuralf):
        fig,ax=plt.subplots(figsize=(7,3))
        for seed in [0,1,2]:
            g=neuralf[neuralf.seed.eq(seed)];ax.plot(np.arange(len(g))+(seed-1)*.09,g.gain,'o',label='Seed '+str(seed))
        names=list(neuralf.cohort.drop_duplicates());ax.set_xticks(np.arange(len(names)),[codes[n] for n in names]);ax.axhline(0,color='black',lw=.7);ax.set_ylabel('Retrospective sheaf gain vs GRIN (%)');ax.legend();fig.tight_layout();fig.savefig(src/'figures/followup_neural.pdf');plt.close(fig)
        section+=figure('followup_neural','All three selected training seeds in each fixed external neural cohort. Positive gains favor retrospective sheaf. Seeds are dependent model refits and not additional independent datasets.')
    (src/'weakness_section.tex').write_text(section,encoding='utf-8')
    md=['# Remaining-weakness follow-up: complete results','',
        'These are exploratory follow-ups on frozen datasets. No new independent cohorts are counted, no previous primary result is replaced, and no first-ever theorem or field environmental effect is certified.','',
        f'Causal sheaf has lower normalized RMSE than the causal forest in {wins}/{len(ext)} added external cohorts. A numerical gain is not necessarily significant.','',
        '| Cohort | Cases | Forest leaf | Sheaf gain | Weekly interval | Physical PM gain |','|---|---:|---:|---:|---|---:|']
    for r in nf.itertuples():md.append(f'| {r.code} ({r.cohort}) | {r.cases} | {r.leaf} | {r.gain:.2f}% | {ci(r.ci7)} | {r.physical_gain:.2f}% |')
    md+=['','## External neural benchmark','Retrospective comparison only; each provider representative is selected lexicographically using training masks, and every seed remains visible.','',
         '| Cohort | Seed | Selected width/epoch | Cases | Sheaf gain vs GRIN | Weekly interval |','|---|---:|---|---:|---:|---|']
    for r in neuralf.itertuples():md.append(f'| {r.code} ({r.cohort}) | {r.seed} | {r.width}/{r.epoch} | {r.cases} | {r.gain:.2f}% | {ci(r.ci7)} |')
    md+=['','## Interval and decision weaknesses',f'Extreme snapshot coverage, cohort-macro: station fixed {100*fixed.macro_coverage:.1f}%, risk station {100*risk.macro_coverage:.1f}%, on {risk.cells:,} extreme cells in {risk.cohorts} nonempty external cohorts. The improvement remains far below 90%. Snapshot intervals do not establish uninterrupted causal-outage intervals.',
         'Full risk/width/full-set/station-quarter summaries and matched-FPR alert ranking are in the manuscript and machine-readable tables. ROC thresholds are test-label descriptive, not deployable choices. Empty positive classes remain undefined.',
         'The fixed hourly concentration grid 15/25/35/60/90 retains every threshold and cohort, exposing sensitivity to sparse severe-event positives. These thresholds are illustrative hourly proxies, not official daily standards or new primary endpoints.',
         'A uniform-radius control matches each station risk interval mean calibration width, without test-label tuning. Both groups must support finite calibration quantiles. Identical-row coverage and differing test widths under shift are reported; this is not a test-width-matched comparison.',
         f'Validation-only selection obtains positive gains versus VAR in {gaincount}/{len(guardext)} external cohorts; it can lose after distribution shift.',
         'Temporal fault detectors are compared on matched rows with clean false alarms and synthetic fault families. Four-hour evidence can change both sensitivity and false alarms. Episode latency retains nondetections rather than treating them as zero delay. Detection is distinct from localization.','',
         '## What remains unresolved',
         '- Extreme-event coverage and weak/multiplicative-fault performance remain imperfect; negative results are retained.',
         '- Baseline coverage is broader, but three neural cohorts do not establish international neural dominance.',
         '- Artificial outages and injected faults do not substitute for labelled operational outages and real sensor failures.',
         '- Public sensors, dates and model seeds are dependent; exploratory temporal sensitivities are not blinded external replication.',
         '- Six connected mathematical propositions retain their proofs and assumptions. Established ingredients are credited; first-ever priority and external proof review remain unestablished.',
         '- Monitoring continuity is a demonstrated computational target. Reduced emissions, health effects and actual regulatory decisions are not measured.',
         '- Competition eligibility, student ownership, guide records and submission-day observation ages require accurate external documentation.','',
         '## Evidence locations','Research_Release/weakness_resolution_results contains frozen selection, cases, checkpoints, protocols, replay records, summary tables and the final AUDIT.json. The designated manuscript includes all these follow-up findings alongside the original evidence.']
    if allow_pending:md.insert(2,'**Preview only: external neural training and the final follow-up audit are unfinished.**')
    (REVIEW/'WEAKNESS_RESOLUTION_REVIEW.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    (OUT/'REPORT_STATUS.json').write_text(json.dumps({'status':'preview' if allow_pending else 'generated','external_forest_wins':wins,
        'external_cohorts':len(ext),'neural_seed_comparisons':len(neuralf),'extreme_fixed_macro':fixed.macro_coverage,
        'extreme_risk_macro':risk.macro_coverage,'extreme_cells':int(risk.cells),'external_selector_positive':gaincount},indent=2))
    return {'external_forest_wins':wins,'external_cohorts':len(ext),'neural_seed_comparisons':len(neuralf)}

if __name__=='__main__':compose(PACKAGE/'Expanded_Evidence_Source',allow_pending='--preview' in sys.argv)
