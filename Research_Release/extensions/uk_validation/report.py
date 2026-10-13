                                                                                 
from pathlib import Path
import sys, json, shutil, importlib.util, re, hashlib, zipfile
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
OUT=ROOT/'Research_Release/uk_validation_results'; DATA=OUT/'uk_aurn'
PACKAGE=ROOT/'IRIS_Submission_Recent_Data'; SRC=PACKAGE/'Three_Network_Source'; REVIEW=PACKAGE/'Review_Notes'

def js(name): return json.loads((DATA/name).read_text())
def csv(name): return pd.read_csv(DATA/name)
def esc(s): return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def num(x): return 'undefined' if x is None or not np.isfinite(x) else f'{x:.2f}'
def pct(x): return num(100*x) if x is not None else 'undefined'
def ci(value): return 'suppressed' if value is None else f'[{value[0]:.2f}, {value[1]:.2f}]'
def table(caption, headers, rows):
    return '\\begin{table}[htbp]\\centering\\small\\caption{'+caption+'}\n\\begin{tabular}{'+'l'*len(headers)+'}\\toprule\n'+' & '.join(headers)+r'\\\midrule'+'\n'+'\n'.join(' & '.join(esc(x) for x in row)+r'\\' for row in rows)+'\n\\bottomrule\\end{tabular}\\end{table}\n'

def main():
    audit=json.loads((OUT/'AUDIT.json').read_text())
    if audit['status']!='passed': raise RuntimeError('UK audit must pass before manuscript integration')
    SRC.mkdir(exist_ok=True); REVIEW.mkdir(exist_ok=True)
    for name in ['complete_theory.tex','references.bib']:
        shutil.copy2(PACKAGE/'Editable_Source'/name,SRC/name)
    shutil.copytree(PACKAGE/'Editable_Source/figures',SRC/'figures',dirs_exist_ok=True)
    osenv=__import__('os').environ; osenv['MPLCONFIGDIR']=str(OUT/'plot_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest=js('dataset_manifest.json'); reconstruction=js('reconstruction_summary.json')
    primary=reconstruction['primary_every_third_day']['var_causal']; allcase=reconstruction['all_eligible_test_days']['var_causal']
    uncertainty=csv('uncertainty_summary.csv'); faults=csv('available_fault_summary.csv'); alerts=csv('alert_summary.csv'); ranks=csv('structured_rank_cases.csv'); margins=csv('exhaustive_block_margins.csv'); joint=csv('functional_joint_bounded_cases.csv'); blocks=csv('block_interval_day_summary.csv')
    ablations=js('ablation_summary.json'); hybrid=js('hybrid_summary.json'); masked=js('matched_missing_mask_summary.json')
    def put(name,figure):
        figure.tight_layout();figure.savefig(SRC/'figures'/('uk_'+name+'.pdf'),bbox_inches='tight');figure.savefig(SRC/'figures'/('uk_'+name+'.png'),dpi=160,bbox_inches='tight');plt.close(figure)
    figure,axes=plt.subplots(1,2,figsize=(10,3.8))
    for ax,scope,title in zip(axes,['primary_every_third_day','all_eligible_test_days'],['Fixed primary days','All eligible test days']):
        methods=['var_causal','seasonal','city_mean','gaussian','spatial_sheaf'];values=[reconstruction[scope][k]['gain_percent'] for k in methods]
        ax.barh(['Causal VAR','Seasonal','City mean','Gaussian','Spatial sheaf'],values);ax.axvline(0,color='black',linewidth=.7);ax.set_xlabel('Causal sheaf RMSE reduction (%)');ax.set_title('Network C (UK): '+title)
    put('reconstruction',figure)
    figure,axes=plt.subplots(1,2,figsize=(10,4))
    u=uncertainty[(uncertainty.context_loss==0)&(uncertainty.group=='all')]
    axes[0].bar(u.method,u.coverage*100);axes[0].axhline(90,color='black',linestyle='--');axes[0].tick_params(axis='x',rotation=20);axes[0].set_ylabel('Observed coverage (%)');axes[0].set_title('No-feedback hourly intervals')
    uu=uncertainty[(uncertainty.context_loss==0)&(uncertainty.group=='extreme')]
    axes[1].bar(uu.method,uu.coverage*100);axes[1].axhline(90,color='black',linestyle='--');axes[1].tick_params(axis='x',rotation=20);axes[1].set_title('Training-threshold extreme cells');axes[1].set_ylabel('Observed coverage (%)')
    put('uncertainty',figure)
    figure,ax=plt.subplots(figsize=(9,4));f=faults[faults.true_count==2];ax.bar(f.kind,f.exact_support*100);ax.set_ylim(0,105);ax.tick_params(axis='x',rotation=30);ax.set_ylabel('Alarm and exact support (%)');ax.set_title('Network C: two-fault episode hours; dependent trials');put('faults',figure)
    figure,ax=plt.subplots(figsize=(9,3.8));ax.barh(list(ablations),[v['gain_percent'] for v in ablations.values()]);ax.axvline(0,color='black',linewidth=.7);ax.set_xlabel('Original causal sheaf RMSE reduction over ablation (%)');ax.set_title('Shared settings; variants not individually retuned');put('ablations',figure)
    figure,axes=plt.subplots(1,3,figsize=(10,3.4));stress=csv('history_horizon_context_cases.csv')
    for ax,loss in zip(axes,[0.,.2,.5]):
        image=np.full((2,3),np.nan)
        for i,history in enumerate([0,24]):
            for j,horizon in enumerate([24,72,168]):
                q=stress[(stress.history==history)&(stress.horizon==horizon)&(stress.context_loss==loss)]
                p=q.pivot(index=['day','station'],columns='method',values='mse')[['sheaf_causal','var_causal']].dropna()
                if len(p):image[i,j]=100*(1-np.sqrt(p.sheaf_causal.sum()/p.var_causal.sum()))
        ax.imshow(image,cmap='RdBu',vmin=-30,vmax=30,aspect='auto')
        for i,j in np.ndindex(image.shape):ax.text(j,i,'n/a' if not np.isfinite(image[i,j]) else f'{image[i,j]:.1f}',ha='center',va='center')
        ax.set_xticks(range(3),['24 h','72 h','168 h']);ax.set_yticks(range(2),['No history','24 h history']);ax.set_title(f'Context loss {loss:.0%}')
    figure.suptitle('Network C: causal RMSE reduction against VAR (%)');put('stress',figure)
    figure,ax=plt.subplots(figsize=(9,3.6));feasible=joint[joint.joint_status=='bounded'].reset_index(drop=True)
    ax.plot(feasible.index,feasible.joint_lower,label='Joint lower bound');ax.plot(feasible.index,feasible.joint_upper,label='Joint upper bound');ax.axhline(1,color='black',linestyle='--',label='Illustrative threshold');ax.set_xlabel('Feasible held-out state (inconsistent models excluded)');ax.set_ylabel('Dimensionless maximum pollutant score');ax.legend();put('joint',figure)
    rows=[]
    for scope in ['primary_every_third_day','all_eligible_test_days']:
        r=reconstruction[scope]['var_causal'];rows.append([scope.replace('_',' '),r['cases'],r['dates'],num(r['gain_percent']),ci(r['ci95'])])
    section=r'''\section{Additional recent external validation: Network C (UK)}\label{sec:uk}
This exploratory external-network extension was fixed after the Indian/US outcomes, before UK model fitting or test evaluation. It is algorithm transfer with newly fitted UK maps, not zero-shot transfer of fitted Indian operators, external preregistration or proof of no prior user access. The original Indian primary endpoint and its denominator remain unchanged. The UK result is reported regardless of direction.

Official DEFRA AURN yearly hourly files and metadata supply the observations \citep{aurn2026,openair2026}. The complete calendar is January 1--June 30, 2026. A fixed geographic box and metadata-active PM2.5, PM10 and ozone define candidates. Lexicographic site-code order and training-only 50\%/720-hour eligibility select the first five qualifying sites. All three channels retain native micrograms per cubic metre. Modelled weather variables are excluded. Missing hours remain missing, negative readings are excluded, and duplicate source timestamps trigger an error. Hour-beginning UTC timestamps follow the source convention; per-channel metadata mark the selected period ratified through June 30. Original source files, metadata, acquisition/parser history and hashes are preserved. Ratification status may subsequently be revised.

The same 60/10/10/20 training/selection/calibration/test split and recent-study point-model grids are used. Selection and calibration are disjoint; train-only normalization, correlation graph, CCA and covariance fitting precede testing. The fixed primary outage has 24 hours of hidden station observations after 24 hours of target history, evaluated every third eligible test day. All-day, physical-unit and retrospective comparisons remain distinct. Exploratory calendar-week intervals are suppressed below five bins.
'''
    section+=table('UK matched causal reconstruction. Positive gain means lower sheaf RMSE; no pooling with other networks.',['Scope','Station-days','Dates','Gain vs VAR (\%)','95\% interval'],rows)
    section+=r'\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{figures/uk_reconstruction.pdf}\caption{UK matched causal point comparisons under the fixed primary and all-day scopes. Simple controls and the tuned VAR use the same observed target cells.}\end{figure}'+'\n'
    physical=reconstruction['primary_every_third_day']['pm25_physical']['var_causal']
    section+=f"Physical PM2.5 primary gain against VAR is {num(physical['gain_percent'])}\\% across {physical['cases']} station-days, with interval {ci(physical['ci95'])}. This differs from the all-channel normalized endpoint.\\par\n"
    intervalrows=[]
    for _,r in uncertainty[(uncertainty.context_loss==0)&uncertainty.group.isin(['all','extreme'])].iterrows():intervalrows.append([r['method'],r['group'],int(r.cases),pct(r.coverage),num(r.finite_mean_width),pct(r.full_set_fraction)])
    section+=table('UK no-feedback intervals at nominal 90\%, without extra context loss. Width is full finite width in native PM2.5 units. Empty groups have undefined coverage; full sets are reported separately.',['Method','Group','Cells','Coverage (\%)','Width','Full sets (\%)'],intervalrows)
    section+=r'\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{figures/uk_uncertainty.pdf}\caption{UK interval coverage, including training-threshold extreme cells. No hidden target feedback is used by these fixed interval methods; temporal exchangeability is not established.}\end{figure}'+'\n'
    blockrows=[[r['method'],int(r.station_days),pct(r.simultaneous_coverage),pct(r.full_set_fraction)] for _,r in blocks[blocks.context_loss==0].iterrows()]
    section+=table('UK simultaneous coverage over each observed station-day, distinguished from hourly coverage.',['Calibration','Station-days','All-hour coverage (\%)','Full sets (\%)'],blockrows)
    faultrows=[[r.kind,int(r.true_count),int(r.cases),pct(r.alarm_rate),pct(r.exact_support)] for _,r in faults[(faults.true_count==2)|(faults.kind=='clean')].iterrows()]
    section+=table('UK availability-aware synthetic fault episodes. Exact support success requires an alarm; the detector is not given the true fault count. Episode hours are dependent.',['Family','Faults','Hours','Alarm (\%)','Exact (\%)'],faultrows)
    section+=r'\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{figures/uk_faults.pdf}\caption{UK two-fault end-to-end recovery. Count output is capped at two; three injected faults cannot be recovered exactly. Natural field faults are not independently labelled.}\end{figure}'+'\n'
    section+=table('UK hourly PM2.5 illustrative 60 micrograms per cubic metre proxy; not an official UK DAQI or regulatory test.',['Method','Truth hours','Positive','Sensitivity (\%)','False alarms (\%)'],[[r.method,int(r.cases),int(r.positive_cases),pct(r.sensitivity),pct(r.false_alarm_rate)] for _,r in alerts.iterrows()])
    daily=csv('pm25_daily_summary.csv')
    section+=table('UK common-day PM2.5-only category proxy; not official AQI/DAQI.',['Method','Days','Accuracy (\%)','Poor-or-worse','Missed (\%)'],[[r.method,int(r.common_station_days),pct(r.category_accuracy),int(r.poor_or_worse_truth_days),pct(r.missed_poor_rate)] for _,r in daily.iterrows()])
    feasible=joint[joint.joint_status=='bounded'];decisions=feasible.joint_decision.value_counts().to_dict()
    section+=f"Actual-map structured ranks agree on {int(ranks.agree.sum())}/{len(ranks)} exhaustively enumerated nonempty outage sets; {len(margins)} one- through four-block margins are enumerated. Generic-pattern thinning discrepancies, if any, are retained in the detailed tables rather than treated as a guaranteed Hall match. UK bounded joint models are feasible in {len(feasible)}/{len(joint)} evaluated states; decision counts are {esc(str(decisions))}. Feasible abstention and infeasible model explanations are separate. These checks apply the existing conditional propositions, without establishing a new priority claim.\\par\n"
    section+=table('UK shared-settings ablations. Negative gain favors the ablation; these variants were not individually retuned.',['Variant','Station-days','Gain (\%)','95\% interval'],[[k,v['cases'],num(v['gain_percent']),ci(v['ci95'])] for k,v in ablations.items()])
    section+=r'\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{figures/uk_ablations.pdf}\caption{UK shared-settings ablations retain both favorable and unfavorable results.}\end{figure}'+'\n'
    section+=table('Exploratory UK validation-selected convex hybrid. It does not replace the fixed original-model headline.',['Comparator','Station-days','Gain (\%)','95\% interval'],[[k,v['cases'],num(v['gain_percent']),ci(v['ci95'])] for k,v in hybrid.items()])
    section+=f"Transplanted natural masks yield {masked['cases']} hidden PM2.5 cells from {masked['evaluated_patterns']} evaluated patterns; gain against city averaging is {num(masked.get('gain_percent'))}\\%. This artificial-gap test does not verify the original naturally missing truth and retains complete-day selection bias. Simulated repair, map perturbation, exact nominal dual/refinement, history/horizon/context stress, partial-channel and revealed-label interval records are retained in the evidence catalogue. Revealed target labels are unavailable during an operational dark outage. No UK neural experiment, official-bulletin/DAQI validation, physical repair, emissions reduction or measured health benefit is claimed. The existing three-seed neural comparison remains Indian-only.\\par\n"
    section+=r'\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{figures/uk_stress.pdf}\caption{UK history, outage-horizon and context-loss stress tests. All selected parameters remain fixed; these exploratory task results do not replace the primary endpoint.}\end{figure}'+'\n'
    section+=r'\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{figures/uk_joint.pdf}\caption{UK joint bounds on feasible models only. States violating a model assumption and inconsistent explanations are not silently counted as successful certification. The concentration references are illustrative.}\end{figure}'+'\n'
    repair=js('repair_summary.json')
    section+=table('UK simulated station repair against the random-repair comparator. These are model-based experiments, not physical interventions.',['Policy','Cases','Gain (\%)','95\% interval'],[[k,v['cases'],num(v['gain_percent']),ci(v['ci95'])] for k,v in repair.items()])
    section+=f"An independent incidence-matrix objective and observation-matrix Kalman replay reproduced all seven point methods across {audit['station_days_replayed']} eligible UK station-days. The audit of sources, chronology, normalization, ranks and conditional containment passed {audit['check_count']} checks. This checks saved-model inference rather than independently refitting the models; finite checks do not replace written conditional proofs.\\par\n"
    (SRC/'uk_section.tex').write_text(section,encoding='utf-8')
    appendix='\\clearpage\\section{UK complete evidence-table catalogue}\n'
    inventory=[]
    for path in sorted(DATA.glob('*.csv')):
        try:count=len(pd.read_csv(path))
        except pd.errors.EmptyDataError:count=0
        inventory.append({'table':path.name,'rows':count,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        appendix+='\\par '+esc(path.name)+': '+str(count)+' rows.\n'
    (SRC/'uk_catalogue.tex').write_text(appendix,encoding='utf-8')
    pd.DataFrame(inventory).to_csv(OUT/'RESULT_TABLE_INVENTORY.csv',index=False)
    extra=r'''
@misc{aurn2026,author={{Department for Environment, Food and Rural Affairs}},title={Automatic Urban and Rural Network: hourly observations and quality verification},year={2026},url={https://www.gov.uk/guidance/air-pollution-monitoring-automatic-urban-and-rural-network-aurn},note={Official 2026 archive and metadata accessed 7 October 2026}}
@misc{openair2026,author={{openair project}},title={UK air quality data import: timestamps, units and ratification metadata},year={2026},url={https://openair-project.github.io/openair/reference/importUKAQ-wrapper.html}}
'''
    with (SRC/'references.bib').open('a',encoding='utf-8') as f:f.write(extra)
    def integrate(text):
        text=text.replace('\\section{Discussion and limitations}', '\\input{uk_section.tex}\n\\section{Discussion and limitations}',1)
        text=text.replace('\\bibliography{references}', '\\input{uk_catalogue.tex}\n\\bibliography{references}')
        text=text.replace('Research paper\\par}', 'Research paper: three recent networks\\par}')
        sentence=f" An additional protocol-fixed UK AURN test used five stations with three mass-concentration channels from January--June 2026; its primary VAR-relative RMSE gain was {num(primary['gain_percent'])}\\% across {primary['cases']} station-days."
        text=text.replace('\\end{abstract}',sentence+'\n\\end{abstract}',1)
        text=text.replace('Public monitoring networks are coded A (India) and B (USA) for anonymous review.', 'Public monitoring networks are coded A (India), B (USA) and C (UK) for anonymous review.')
        return text
    spec=importlib.util.spec_from_file_location('builder',ROOT/'audit_tools/build_iris_submission.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder);builder.SRC=SRC
    outputs=[]
    for source,name,target in [('research_paper.tex','three_network_paper',PACKAGE/'Upload_Files/SheafCode_Three_Network_Research_Paper.pdf'),('technical_reference.tex','three_network_full',PACKAGE/'Upload_Files/SheafCode_Three_Network_Full_Manuscript.pdf')]:
        (SRC/(name+'.tex')).write_text(integrate((PACKAGE/'Editable_Source'/source).read_text(encoding='utf-8')),encoding='utf-8')
        builder.compile_pdf(name,target);outputs.append(target)
    synopsis=(PACKAGE/'Editable_Source/synopsis.tex').read_text(encoding='utf-8')
    synopsis=synopsis.replace('EPA hourly files supply four Network B stations and three channels from January to March.', 'EPA hourly files supply four Network B stations and three channels from January to March. A protocol-fixed external UK AURN extension supplies five Network C (UK) stations and three mass-concentration channels from January to June, with source ratification dates preserved.')
    synopsis=synopsis.replace('Identity spatial maps outperform',f"The UK primary gain is {primary['gain_percent']:.1f}\\% on {primary['cases']} station-days; its exploratory interval is {primary['ci95'][0]:.1f} to {primary['ci95'][1]:.1f}\\%. Identity spatial maps outperform",1)
    synopsis=synopsis.replace('United States Environmental Protection Agency.', 'United States Environmental Protection Agency, and DEFRA AURN with openair documentation.')
    (SRC/'three_network_synopsis.tex').write_text(synopsis,encoding='utf-8')
    target=PACKAGE/'Upload_Files/SheafCode_Three_Network_Synopsis.pdf';builder.compile_pdf('three_network_synopsis',target);outputs.append(target)
    abstract=f'''Air-quality station outages interrupt local pollution monitoring. SheafCode learns vector-valued spatial and temporal consistency maps to reconstruct missing pollutant channels causally, with conditional bounds for pollutant quantities and uncertain sensor faults. The study uses only 2026 observations for fitting, selection, calibration and testing: five Indian stations, four US stations and five UK stations. Chronological model-selection and interval-calibration blocks are separate. UK source metadata identify the selected January-June observations as ratified.

The fixed primary test hides one station for twenty-four hours after twenty-four hours of history, comparing identical observed target cells. Against validation-tuned full-covariance causal VAR, normalized RMSE reductions were 16.5% in India (77 station-days; exploratory weekly interval 12.6-20.2%), 8.7% in the US (24 station-days; insufficient bins for an interval), and {primary['gain_percent']:.1f}% in the UK ({primary['cases']} station-days; {primary['ci95'][0]:.1f}-{primary['ci95'][1]:.1f}%). The UK extension was specified before its fitting outcomes, after the original study.

Actual-map recoverability, functional and joint-threshold envelopes, exact nominal dual correction and conditional containment checks accompany full proofs. Ablations, injected faults, interval coverage, monitoring proxies and negative results are reported separately. Shared prior observations and exploratory analyses limit independence claims. Environmental value is evaluated through monitoring continuity; emissions reduction, health benefits and verified regulatory decisions are not demonstrated. Mathematical specialization is credited without asserting first-ever priority.'''
    words=len(abstract.split())
    if words>250:raise RuntimeError('Three-network abstract exceeds target')
    (PACKAGE/'Upload_Files/Project_Abstract_Three_Network.txt').write_text(abstract,encoding='utf-8')
                                                                                                  
    oldsrc=PACKAGE/'Full_Manuscript_Source'
    if (oldsrc/'full_manuscript.tex').exists():
        for p in SRC.glob('uk_*.tex'):shutil.copy2(p,oldsrc/p.name)
        for p in (SRC/'figures').glob('uk_*'):shutil.copy2(p,oldsrc/'figures'/p.name)
        shutil.copy2(SRC/'references.bib',oldsrc/'references.bib')
        (oldsrc/'full_manuscript_uk.tex').write_text(integrate((oldsrc/'full_manuscript.tex').read_text(encoding='utf-8')),encoding='utf-8')
        builder.SRC=oldsrc;target=PACKAGE/'Supporting_Reference/SheafCode_Full_Manuscript_With_UK.pdf';builder.compile_pdf('full_manuscript_uk',target);outputs.append(target);builder.SRC=SRC
    import pymupdf as fitz
    documents=[]
    for target in outputs:
        with fitz.open(target) as doc:
            text='\n'.join(p.get_text() for p in doc)
            if 'Network C (UK)' not in ' '.join(text.split()):raise RuntimeError('UK section absent: '+target.name)
            if target.stat().st_size>=20_000_000:raise RuntimeError('IRIS paper size exceeded')
            documents.append({'path':str(target.relative_to(ROOT)),'pages':len(doc),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    report=f'''# UK external validation: completed study

The UK study is an exploratory external-network extension using five stations, three native mass-concentration channels and January--June 2026 observations. It uses training-only eligibility and a fixed 60/10/10/20 timeline. Source metadata mark all selected channels ratified through the evaluated period. The original Indian/US point results remain unchanged.

- Primary UK gain versus tuned causal VAR: **{primary['gain_percent']:.2f}%**, {primary['cases']} paired station-days, {primary['dates']} dates; exploratory weekly interval {ci(primary['ci95'])}.
- All eligible UK days: **{allcase['gain_percent']:.2f}%**, {allcase['cases']} station-days.
- Physical PM2.5 primary gain: **{physical['gain_percent']:.2f}%**; {physical['cases']} station-days.
- Source/tensor and independent saved-model inference audit: **{audit['check_count']} checks passed**, all seven methods replayed on {audit['station_days_replayed']} station-days.
- {len(inventory)} detailed case tables, six new figures, all compatible non-neural task families, all negative/undefined outcomes retained. Strict-complete and available-candidate faults are separate. Neural results remain Indian-only; no UK GRIN or actual DAQI/intervention claim is added.

## One-file manuscript choices

Use `Upload_Files/SheafCode_Three_Network_Full_Manuscript.pdf` for the single recent-only paper with full conditional proofs and detailed evidence. `SheafCode_Three_Network_Research_Paper.pdf` is the shorter main-paper rendering of the same current evidence. The complete historical reading manuscript is `Supporting_Reference/SheafCode_Full_Manuscript_With_UK.pdf`, whose older empirical observations remain unconfirmed for IRIS acceptance. These are alternatives, not files that all need to be uploaded.

## What carries forward and what changes

The older and current studies are **not identical except for dataset**. The core vector-valued sheaf framework and valid conditional mathematical ideas carry forward. The audited historical manuscript already includes repaired causal, cohort-selection and baseline analyses; these repairs are not inventions of the UK study. The current recent-data experiments change observation periods, eligible cohorts, channel/unit combinations, chronological blocks, selected fitted parameters and evaluation denominators. Precise primary sampling and declared study grids are documented separately. Across the supplied earlier drafts, false robust-support statements and invalid adaptive boundary guarantees were corrected, while generic structural theory was separated from actual structured-map checks. The current proofs further clarify their mathematical premises. The prior empirical results retain their own protocols and cannot be copied as UK or recent results. The UK study introduces external evidence, not a new theorem or proof of first-ever priority.

The UK operators reuse current conditional mathematical statements. All numerical results are recomputed. Environmental utility remains monitoring continuity and decision-proxy evaluation; no demonstrated emissions reduction, health benefit or weather-forecast improvement is claimed.
'''
    (REVIEW/'UK_VALIDATION_REPORT.md').write_text(report,encoding='utf-8')
    (REVIEW/'THREE_NETWORK_DOCUMENT_CHECKS.json').write_text(json.dumps({'status':'passed','documents':documents,'audit_checks':audit['check_count'],'figures':6,'abstract_words':words,'table_inventory':inventory},indent=2),encoding='utf-8')
    docspec=importlib.util.spec_from_file_location('uk_document_checker',HERE/'check_documents.py');checker=importlib.util.module_from_spec(docspec);docspec.loader.exec_module(checker);checker.main()
                                                                                            
    files=[p for p in OUT.rglob('*') if p.is_file() and p.suffix not in ['.zip','.pyc','.tmp'] and p.name!='status.json' and '__pycache__' not in p.parts and 'plot_cache' not in p.parts]
    files += [p for p in HERE.rglob('*') if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts]
    files += [p for p in SRC.rglob('*') if p.is_file()]
    files += outputs+[REVIEW/'UK_VALIDATION_REPORT.md',REVIEW/'THREE_NETWORK_DOCUMENT_CHECKS.json',REVIEW/'UK_FINAL_DOCUMENT_AUDIT.json',REVIEW/'UK_DOCUMENT_CONTACT.png',PACKAGE/'Upload_Files/Project_Abstract_Three_Network.txt',ROOT/'Run_UK_Validation.cmd']
                                                                                 
    for directory in ['recent_data','online_validation','structured','functional','followup','joint_decisions','validation']:
        files += list((ROOT/'Research_Release/extensions'/directory).glob('*.py'))
    entries=[{'path':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(files))]
    archive=PACKAGE/'Supporting_Reference/SheafCode_UK_Validation_Evidence.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):z.write(p,str(p.relative_to(ROOT)))
        z.writestr('UK_FILE_MANIFEST.json',json.dumps(entries,indent=2))
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise RuntimeError('UK ZIP CRC failure')
        for item in entries:
            if hashlib.sha256(z.read(item['path'])).hexdigest()!=item['sha256']:raise RuntimeError('UK ZIP member hash mismatch')
    (REVIEW/'UK_PACKAGE_CHECKS.json').write_text(json.dumps({'status':'passed','hashes_checked':len(entries),'crc_checked':True,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'documents':documents},indent=2),encoding='utf-8')
    print('UK manuscript integration and evidence archive complete',json.dumps(documents),flush=True)

if __name__=='__main__': main()
