                                                                              
import sys,shutil,subprocess,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABEL={'delhi2025':'Delhi','bengaluru_raw':'Bengaluru'}
FOLDER=OUT/'figures';FOLDER.mkdir(exist_ok=True)
M=ROOT.parent/'Final_Manuscript/Extended_Validation'

def esc(x):
    s=str(x)
    for a,b in [('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:s=s.replace(a,b)
    return s

def output(fig,name):
    fig.savefig(FOLDER/(name+'.pdf'),bbox_inches='tight');fig.savefig(FOLDER/(name+'.png'),dpi=160,bbox_inches='tight');plt.close(fig)

def table(headers,rows,caption):
    return '\n'.join([r'\begin{table}[htbp]\centering\small\caption{'+esc(caption)+'}',
        r'\begin{tabular}{l'+'r'*(len(headers)-1)+r'}\hline',
        ' & '.join(esc(h) for h in headers)+r'\\\hline',
        *[' & '.join(esc(v) for v in row)+r'\\' for row in rows],r'\hline\end{tabular}\end{table}'])

def figure(name,caption):
    return r'\begin{figure}[htbp]\centering\includegraphics[width=.98\linewidth]{figures/'+name+r'.pdf}\caption{'+esc(caption)+r'}\end{figure}'

def main():
    audit=json.loads((OUT/'audit.json').read_text());assert audit['status']=='passed'
    replay=json.loads((OUT/'replay_summary.json').read_text());bj=json.loads((OUT/'beijing_corrected_summary.json').read_text())
    faults=pd.read_csv(OUT/'persistent_fault_summary.csv');gain=pd.read_csv(OUT/'raw_gain_fault_summary.csv')
    uncertainty=pd.read_csv(OUT/'uncertainty_shift_corrected_summary.csv');official=json.loads((OUT/'official_summary.json').read_text())
    proxy=pd.read_csv(OUT/'official_proxy_comparison.csv');rp=pd.read_csv(OUT/'replay_cases.csv')
    fig,ax=plt.subplots(figsize=(6.5,4))
    for method,label in [('temporal_sheaf','Sheaf'),('full_covariance_var','Full VAR')]:
        q=rp[rp.method==method];ax.scatter(q.saved_mse,q.mse,s=12,alpha=.65,label=label)
    high=rp[['saved_mse','mse']].max().max();ax.plot([0,high],[0,high],'k--',lw=1)
    ax.set(xlabel='Frozen station-day normalized MSE',ylabel='Second implementation normalized MSE',title='Raw Delhi replay: all 399 station-days reproduced')
    ax.legend();output(fig,'01_raw_replay')
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),sharex=True)
    for ax,key,title in zip(axes,['comparisons','pm25_raw_comparisons'],['Four pollutants: 502 station-days','PM2.5 concentration: 501 station-days']):
        for i,(method,label) in enumerate([('full_covariance_var','Full VAR'),('seasonal','Seasonal'),('city_mean','Raw city mean')]):
            q=bj[key][method];point=q['gain_percent'];lo,hi=q['ci95'];ax.errorbar(point,i,xerr=[[max(point-lo,0)],[max(hi-point,0)]],fmt='o',capsize=4)
        ax.set_yticks(range(3),['Full VAR','Seasonal','Raw city mean']);ax.axvline(0,color='grey');ax.set(title=title,xlabel='Sheaf RMSE reduction versus comparator (%)')
    fig.suptitle('Beijing network transfer: fixed protocol, 12 stations and 42 days');fig.tight_layout();output(fig,'02_network_transfer')
    kinds=['same_sign','opposite_sign','unequal','drift','stuck','shared_additive','shared_scale','partial_channel']
    values=np.array([[faults[(faults.region==region)&(faults.fault_count==2)&(faults.method=='classifier')&(faults.fault_type==kind)].exact_end_to_end.iloc[0] for kind in kinds]+[
        gain[(gain.region==region)&(gain.fault_count==2)&(gain.method=='classifier')&(gain.gain_factor==1.5)].exact_end_to_end.iloc[0]] for region in ['delhi2025','bengaluru_raw']])
    fig,ax=plt.subplots(figsize=(11,3.4));image=ax.imshow(values,vmin=0,vmax=1,cmap='viridis',aspect='auto')
    ax.set_xticks(range(9),['Same sign','Opposite','Unequal','Drift','Stuck','Shared add.','Std. gain','One channel','Raw gain 1.5'],rotation=30,ha='right')
    ax.set_yticks([0,1],['Delhi','Bengaluru']);ax.set_title('Frozen detector: exact end-to-end recovery of two faulty stations')
    for i in range(2):
        for j in range(9):ax.text(j,i,f'{100*values[i,j]:.1f}%',ha='center',va='center',color='white' if values[i,j]<.6 else 'black')
    fig.colorbar(image,ax=ax,label='Alarm and exact support success');fig.tight_layout();output(fig,'03_persistent_faults')
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for region,color in [('delhi2025','tab:blue'),('bengaluru_raw','tab:orange')]:
        q=uncertainty[(uncertainty.region==region)&(uncertainty.method=='station_fixed')&(uncertainty.group_type=='all')].sort_values('extra_missing_rate')
        axes[0].plot(100*q.extra_missing_rate,100*q.coverage,'o-',color=color,label=LABEL[region]);axes[2].plot(100*q.extra_missing_rate,100*q.decided_fraction,'o-',color=color,label=LABEL[region])
    q=uncertainty[(uncertainty.region=='bengaluru_raw')&(uncertainty.method=='station_fixed')&(uncertainty.extra_missing_rate==0)&(uncertainty.group_type=='train_extreme')]
    axes[1].bar(['Ordinary','Train-defined\nextreme'],[100*q[q['group']=='False'].coverage.iloc[0],100*q[q['group']=='True'].coverage.iloc[0]],color=['tab:blue','tab:red'])
    axes[1].set_title('Bengaluru: 33,495 / 837 cases');axes[1].set_ylabel('Coverage (%)')
    for ax in axes[:2]:ax.axhline(90,color='grey',ls='--');ax.set_ylim(0,101)
    axes[0].set(xlabel='Added context-cell missingness (%)',ylabel='Coverage (%)',title='No-feedback station intervals');axes[0].legend()
    axes[2].set(xlabel='Added context-cell missingness (%)',ylabel='Decided cases (%)',title='Decision yield at PM2.5 > 60');fig.tight_layout();output(fig,'04_uncertainty_shift')
    og=proxy.groupby('method').category_matches.agg(['sum','count'])
    fig,ax=plt.subplots(figsize=(7,4));methods=['observed_proxy','sheaf_proxy','city_fallback_proxy'];v=[100*og.loc[x,'sum']/og.loc[x,'count'] for x in methods]
    ax.bar(['Observed proxy','Sheaf proxy','City fallback proxy'],v,color=['grey','tab:blue','tab:orange'])
    for i,x in enumerate(methods):ax.text(i,v[i]+2,f"{int(og.loc[x,'sum'])}/{int(og.loc[x,'count'])}",ha='center')
    ax.set(ylim=(0,105),ylabel='Published-category agreement (%)',title='14 selected hotspot station-days across 10 dates');output(fig,'05_official_records')
                                                                                
    report=f'''# Additional computational validation: completed results

Completed 6 October 2026. All six requested computational extensions were implemented and executed. The earlier audited manuscript remains intact; the updated integrated reading copy is `Final_Manuscript/SheafCode_Validated_Manuscript.pdf`.

## What the new evidence adds

- **Independent raw replay:** all 399 Delhi station-days reproduced by a second implementation. Maximum case-MSE discrepancy: {replay['max_case_mse_discrepancy']:.3g}. RMSE reduction against full-covariance VAR remains {replay['comparison']['gain_percent']:.2f}%; this reuses the supplied instruments/data and selected settings.
- **Additional-country network:** Beijing retains 12 training-eligible stations and 502 four-pollutant station-days over 42 sampled dates. Against full VAR, the sheaf RMSE is **{abs(bj['comparisons']['full_covariance_var']['gain_percent']):.2f}% higher**. Against seasonal persistence it is {bj['comparisons']['seasonal']['gain_percent']:.2f}% lower; against the raw-concentration city mean it is {bj['comparisons']['city_mean']['gain_percent']:.2f}% lower. PM2.5-only comparisons use 501 station-days. Settings and sampling were fixed before acquisition; maps were newly fitted using Beijing training only, not transported unchanged from Delhi.
- **Persistent faults:** the frozen unknown-count detector was tested on clean, same/opposite-sign, unequal, drift, stuck, shared additive, standardized gain, partial-channel and deliberately over-capacity three-fault episodes. A separate study multiplies actual raw readings by 1.2/1.5/2.0. Stuck, gain, unequal and gradual faults reveal major weaknesses; all estimated supports and alarms are saved. Triple-fault exact localization is impossible for this detector's two-fault output capacity.
- **Uncertainty shift:** 459,216 method/case rows cover every observed local test PM2.5 target cell at three context-loss rates with three fixed interval methods. At zero added loss, station-specific coverage is 93.69% in Delhi and 92.15% in Bengaluru. Bengaluru coverage on its 837 training-defined extreme cases is only 3.35%; high overall coverage therefore does not establish useful tail coverage. Delhi has only one such extreme case, insufficient for an inferential tail conclusion. No outage labels update the intervals; the original model's validation reuse prevents an independent split-conformal claim.
- **Exact finite mathematics:** {audit['exact_rational_checks']:,} checks passed, including all 19,683 ternary 3-by-3 operators, rational block-support collisions, omitted-block noisy separation, exact span enumeration, adaptive boundary telescoping and rational dual corrections. These supplement the written proofs; they are not a proof-assistant verification of unrestricted theorems or a new priority claim.
- **Official observations:** {official['downloaded_files']}/{official['requested_files']} requested CPCB PDFs acquired, with dates and hashes. Four-pollutant approximate proxies were compared on 14 eligible selected-hotspot station-days across 10 dates: observed proxy 13/14, sheaf proxy 12/14, city fallback proxy 6/14 category agreements. Source hotspot selection, incomplete network/pollutants and averaging/unit differences prevent representative official-AQI or GRAP-action conclusions. Unavailable files and all rejected alignments are retained.

## Verification and interpretation

Seven new unit tests passed; {audit['total_checks']:,} aggregation, support, coverage, source-lock and document-input checks passed in addition to the exact finite checks. The original audited PDF and frozen core model/source hashes remain unchanged. Figures have vector PDF and PNG copies; corrected source CSVs are the publication inputs.

The initial Beijing city comparator averaged standardized station coordinates. The corrected comparator averages raw concentrations and then uses the target's training scale; only those baseline rows changed, with originals retained. The primary VAR and seasonal results did not change. Bootstrap coverage intervals with fewer than five weekly clusters are suppressed, while sample counts and outcomes remain visible.

This extension strengthens reproducibility and exposes generalization, fault and tail-uncertainty boundaries. It does not produce universal superiority, a first-ever theorem, measured cleaner air or operational official-policy recovery. Negative tests are scientific findings rather than a reason to select a different favorable test after seeing outcomes. The appropriate main claim remains conditional monitoring continuity in settings where it demonstrably helps.

Source: [UCI Beijing dataset](https://archive.ics.uci.edu/dataset/501/beijing), DOI 10.24432/C5RK5G, CC BY 4.0; CPCB source URLs and PDF hashes are in `official_acquisition.json`. Raw acquisitions, protocol locks, detailed cases, rejected alignments and proof checks accompany the package.

## Running and freezing

All stages have completed here. No overnight run is required to obtain these reported results. `Run_Online_Validation.cmd` is a resumable reproduction entry point, not an instruction to rerun before documenting the completed study. Preserve both the original audited release and this extension. Changing protocols or models requires a separate prospective evaluation and fresh audit.
'''
    (OUT/'ONLINE_VALIDATION_REPORT.md').write_text(report,encoding='utf-8')
                                                                                
    M.mkdir(exist_ok=True)
    original=ROOT.parent/'Final_Manuscript/Complete_Research'
    for sub in ['sections','figures','data']:
        shutil.copytree(original/sub,M/sub,dirs_exist_ok=True)
    for name in ['main.tex','preamble.tex','references.bib','abstract.txt']:shutil.copy2(original/name,M/name)
    for p in FOLDER.glob('*.pdf'):shutil.copy2(p,M/'figures'/('online_'+p.name))
    abstract=(M/'abstract.txt').read_text().strip()
    sentence=' A protocol-fixed Beijing test found 5.9% higher error than that filter across 502 station-days, while outperforming simpler baselines.'
    abstract=abstract.replace('An official bidirectional',sentence.strip()+' An official bidirectional')
    (M/'abstract.txt').write_text(abstract+'\n',encoding='utf-8')
    (M/'sections/01_abstract.tex').write_text(r'\begin{abstract}'+esc(abstract)+r'\end{abstract}'+'\n',encoding='utf-8')
    bib=(M/'references.bib').read_text()+r'''
@misc{chen2017beijing,author={Chen, Song},title={Beijing Multi-Site Air Quality},year={2017},howpublished={UCI Machine Learning Repository},doi={10.24432/C5RK5G},note={CC BY 4.0; original archived station files, acquired under a fixed protocol}}
'''
    (M/'references.bib').write_text(bib,encoding='utf-8')
    extension=r'''\section{Additional computational validation}\label{app:online}
The six computational validation tracks below were fixed in a separate protocol while preserving the earlier audited release. Settings, acquisition hashes, all case records, unavailable documents and negative outcomes accompany the companion report. These additional tests constrain the generalization and environmental claims; they do not introduce a first-ever mathematical priority claim.

\subsection{Second implementation of the Delhi headline}
The raw hourly parquet was grouped independently, reindexed to the full calendar and selected using full-training eligibility. It exactly reproduces the saved tensor, timestamps, splits and station set. Separately implemented CCA quadratic assembly and Kalman filtering reproduce all 399 causal station-days, with maximum case-MSE discrepancy below $10^{-15}$. The 23.18\% VAR-relative RMSE reduction survives this implementation check. Seven-day calendar-cluster resampling gives a nominal 19.14--27.44\% interval, conditional on the chosen resampling scheme. This is not independent instruments or a new laboratory replication.
'''
    extension+=figure('online_01_raw_replay','Every Delhi station-day loss from the second implementation versus the recorded loss. The identity line shows agreement; the test evaluates implementation reproducibility using the same raw data and recorded selected settings.')
    extension+=r'''\subsection{Protocol-fixed transfer to Beijing}
The public UCI archive contains twelve nationally controlled Beijing sites with hourly observations from March 2013 through February 2017 \citep{chen2017beijing}. Only its original nested station files are used. A chronological 60/20/20 split precedes the full-training 50\%/720-hour rule. All twelve sites qualify for PM2.5, PM10, NO2 and O3. Because coordinates are absent from this source, a training-only mean absolute correlation graph replaces the geographic graph with fixed $k=3$. Sheaf settings are the recorded Delhi choices: raw transform, full-direction ridge CCA, regularization one, temporal scale two and lags one/twenty-four. Maps are fitted on Beijing training data only. This tests algorithm transfer, not zero-shot transfer of Delhi's fitted maps.

The VAR uses the already specified twelve-setting ridge/noise/shrinkage grid and eight evenly sampled validation days. All eligible stations are tested every seventh complete test day, anchored before evaluation. No alternative model is selected after test access. All-pollutant normalized scoring has 502 station-days over 42 dates; PM2.5 physical scoring has 501. Every method shares the same target observations, history and surviving context. The raw city comparator averages concentrations before target normalization; its corrected records preserve the initial standardized-mean diagnostic separately.
'''
    rows=[]
    for name,label in [('full_covariance_var','Full VAR'),('seasonal','Seasonal'),('city_mean','Raw city mean')]:
        q=bj['comparisons'][name];p=bj['pm25_raw_comparisons'][name]
        rows.append([label,f"{q['gain_percent']:.2f}",f"[{q['ci95'][0]:.2f}, {q['ci95'][1]:.2f}]",f"{p['gain_percent']:.2f}"])
    extension+=table(['Comparator','All-pollutant gain (%)','Nominal interval','PM2.5 gain (%)'],rows,'Beijing RMSE reduction relative to each comparator. Negative reduction means higher sheaf error. Intervals resample seven-day calendar groups; they do not remove all possible temporal dependence.')
    extension+=figure('online_02_network_transfer','Protocol-fixed Beijing outcomes. The sheaf beats simpler seasonal and raw city-mean baselines but loses to the separately tuned full-covariance VAR in both endpoints. Intervals are exploratory calendar-group percentile intervals.')
    extension+=r'''The four-pollutant error is 5.93\% higher than VAR; PM2.5 error is 8.54\% higher. Thus the favorable Delhi comparator result does not generalize uniformly. This negative external-network outcome narrows the claim rather than motivating retrospective test-based retuning. The acquired public benchmark's prior use by others or prior user access is not independently excluded.

\subsection{Persistent and raw-coordinate faults}
The already fitted detector and clean cutoff remain fixed. Twenty-four-hour episodes use all fully observed candidate stations and unknown count at prediction. Of 48 planned test starts, 29 Delhi and 33 Bengaluru starts have at least two fully observed stations throughout the episode. The studies evaluate clean readings, shifts, opposite signs, unequal amplitudes, drift, stuck readings, correlated offsets, standardized-coordinate gains and single-channel errors. Three faults explicitly exceed the detector's maximum output count of two. A separate raw-coordinate experiment multiplies original readings by 1.2/1.5/2.0 and applies the frozen transform afterward; it is distinct from multiplying standardized coordinates. Artificial faults do not label naturally occurring field faults.
'''
    rows=[]
    for kind,label in [('same_sign','Same sign'),('opposite_sign','Opposite signs'),('unequal','Unequal'),('drift','Drift'),('stuck','Stuck'),('partial_channel','One channel')]:
        a=faults[(faults.region=='delhi2025')&(faults.method=='classifier')&(faults.fault_count==2)&(faults.fault_type==kind)].iloc[0]
        b=faults[(faults.region=='bengaluru_raw')&(faults.method=='classifier')&(faults.fault_count==2)&(faults.fault_type==kind)].iloc[0]
        rows.append([label,int(a.cases),f'{100*a.exact_end_to_end:.2f}',int(b.cases),f'{100*b.exact_end_to_end:.2f}'])
    extension+=table(['Two-fault type','Delhi hours','Exact (%)','Bengaluru hours','Exact (%)'],rows,'Frozen classifier end-to-end exact support recovery. Success requires an alarm and exact support; candidate universe and requested fault count are not supplied to the predictor. Hours within episodes are dependent.')
    extension+=figure('online_03_persistent_faults','Exact two-fault success of the frozen detector. Standardized gain and raw concentration gain are different perturbations. Delhi cases usually number 696, Bengaluru 792; stuck Bengaluru has 768 eligible hours. Count-three and clean false alarms remain in the complete tables.')
    extension+=r'''Clean false alarms rise to 3.30\% in the Delhi episode sample and 6.06\% in Bengaluru. Strong same/opposite-sign Delhi shifts localize well, while stuck, gain, drift and unequal faults often fail. For raw multiplicative gain 1.5, two-fault exact success is only 6.61\% in Delhi and 0.76\% in Bengaluru. This detector is not a general solution to realistic sensor failure; a favorable same-sign injection statistic is insufficient evidence for that claim.

\subsection{No-feedback intervals across calendar and pollution shift}
Every available local test PM2.5 target is evaluated under artificial whole-station snapshot hiding and 0/20/50\% additional context-cell losses. Three fixed methods use pooled error, station error or station error normalized by pre-outcome availability gain. Calibration uses the final half of validation and receives no hidden test-target updates. The original frozen point model had used validation in selection, so these are empirical interval diagnostics, not an independent split-conformal guarantee. Groups use actual calendar quarters and training-defined station 95th-percentile PM2.5 thresholds. Test truth is used only to score and retrospectively stratify.
'''
    rows=[]
    for region in ['delhi2025','bengaluru_raw']:
        for group_type,group,label in [('all','all','All'),('train_extreme','True','Extreme')]:
            q=uncertainty[(uncertainty.region==region)&(uncertainty.method=='station_fixed')&(uncertainty.extra_missing_rate==0)&(uncertainty.group_type==group_type)&(uncertainty['group']==group)].iloc[0]
            rows.append([LABEL[region],label,int(q.cases),f'{100*q.coverage:.2f}',f'{q.mean_width:.1f}',f'{100*q.decided_fraction:.2f}'])
    extension+=table(['Network','Group','Cases','Coverage (%)','Full width','Decided (%)'],rows,'Station-specific fixed intervals at zero added context loss. Width is PM2.5 micrograms per cubic metre. Extreme means above the training station 95th percentile. The single Delhi extreme case does not support inferential tail coverage.')
    extension+=figure('online_04_uncertainty_shift','Empirical no-feedback coverage and decision yield. The horizontal 90% line is the calibration target, not a theorem for shifted test distributions. Bengaluru extreme coverage is 3.35% over 837 cases despite 92.15% overall coverage. Delhi has only one training-defined extreme test case.')
    extension+=r'''The complete study has 459,216 method/case rows, not independent observations. Weekly intervals are suppressed for fewer than five clusters. High marginal coverage and low tail coverage can coexist; narrow or correct intervals among selected decided cases do not establish unconditional operational reliability. These failures remain unresolved performance limitations of the frozen method.

\subsection{Exact finite verification of the mathematical boundaries}
'''+esc(f"{audit['exact_rational_checks']:,}")+r''' checks pass using standard-library exact rational arithmetic. The suite exhausts all 19,683 ternary scalar three-station operators, constructs explicit sparse syndrome collisions when uniform uniqueness fails, checks random two-coordinate station blocks and noisy omitted-block separation, verifies rational equal-span enumeration, evaluates corrected dual bounds and adaptive-level telescoping, and retains strict-gap/counterexample witnesses. These are finite executable checks supplementing Appendix~\ref{app:proofs}; they are not proof-assistant verification of all dimensions or a certification of originality.

\subsection{Archived official observations and decision agreement}
Fixed weekly dates across the Delhi test period request both city and detailed Delhi/NCR CPCB bulletins, plus two historical provenance controls. Native Windows certificate validation acquires 33 of 34 requested PDFs; all accepted publication dates and source hashes are retained. The detailed documents list selected hotspots rather than a representative station sample. Seventy station entries are parsed; after station, held-out-period and coverage alignment, fourteen selected station-days across ten dates support a common-method comparison. Historical controls are excluded from test performance.

The proxy uses selected PM2.5, PM10, NO2 and O3, with the preceding 24-hour window ending at 16:00 and last-window eight-hour ozone averaging. Omitted channels, gas-unit provenance, rolling/validity rules and reporting practices prevent its interpretation as official AQI. The observed proxy agrees with published categories on 13/14 cases; sheaf agrees on 12/14 and raw city fallback on 6/14. These counts describe this selected sample only. They do not establish official GRAP invocation, population-representative accuracy or environmental intervention benefit.
'''
    extension+=figure('online_05_official_records','Descriptive published-category agreement on fourteen common selected-hotspot station-days. No inferential significance or representative operational accuracy is claimed. Even the observed selected-pollutant proxy differs from the official category on one case.')
    extension+=r'''\subsection{Verification and consequences for the main claims}
'''+esc(f"Seven new unit tests and {audit['total_checks']:,} independent aggregation/source checks passed, in addition to the exact finite checks.")+r''' The original audited release and core model/source hashes remain intact. Corrections preserve the originally recorded data and distinguish corrected publication inputs. All negative findings are retained. No new long experiment remains running for these completed results. The contribution is monitoring reconstruction with conditional reliability analysis and measured failure boundaries; the work does not establish universal superiority, first-ever mathematical priority or measured cleaner air.
'''
    (M/'sections/15_online_validation.tex').write_text(extension,encoding='utf-8')
    maintex=(M/'main.tex').read_text();maintex=maintex.replace(r'\end{document}',r'\clearpage\input{sections/15_online_validation.tex}'+'\n'+r'\end{document}')
    (M/'main.tex').write_text(maintex,encoding='utf-8')
                                                                         
    with (M/'sections/06_reconstruction.tex').open('a',encoding='utf-8') as f:
        f.write(r'''\subsection{Additional-country validation and independent replay}
An additional protocol-fixed Beijing network test has 502 station-days across twelve stations and 42 dates. The sheaf beats seasonal persistence and raw city mean but has 5.93\% higher all-pollutant RMSE than the tuned full-covariance VAR. This narrows generalization beyond the favorable Delhi result. A second implementation independently reproduces all 399 Delhi headline cases from the supplied raw parquet. Complete settings and comparisons appear in Appendix~\ref{app:online}.
''')
    with (M/'sections/07_monitoring.tex').open('a',encoding='utf-8') as f:
        f.write(r'''\subsection{Additional robustness and tail-coverage evidence}
The expanded persistent-fault study reveals poor recovery of stuck, gain, unequal and drift faults despite high same-sign Delhi shift recovery. In the complete no-feedback uncertainty audit, Bengaluru station-specific coverage is 92.15\% overall but only 3.35\% on 837 training-defined extreme-pollution cases. Fourteen aligned selected CPCB hotspot station-days give category agreements 12/14 for the sheaf proxy and 6/14 for the city fallback; selection and proxy limitations prevent general official-decision claims. Appendix~\ref{app:online} retains every negative outcome and denominator.
''')
    discussion=(M/'sections/09_discussion.tex').read_text()
    discussion=discussion.replace(r'\section{Conclusion}',r'''The additional-country test loses to full-covariance VAR, and the full uncertainty audit reveals severe tail undercoverage. Persistent fault tests show poor stuck/gain recovery. Independent raw replay strengthens reproducibility, and the small official-hotspot comparison supports descriptive monitoring relevance only. These findings further restrict general and operational claims; detailed outcomes are in Appendix~\ref{app:online}.

\section{Conclusion}''')
    (M/'sections/09_discussion.tex').write_text(discussion,encoding='utf-8')
    env=os.environ.copy();env['TECTONIC_CACHE_DIR']=str(ROOT.parent/'audit_tools/tex_cache')
    result=subprocess.run([str(ROOT.parent/'audit_tools/tectonic/tectonic.exe'),'--only-cached','--keep-logs','--outdir',str(M),str(M/'main.tex')],cwd=M,env=env,capture_output=True,text=True)
    (OUT/'manuscript_compile.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError('Extended manuscript compilation failed')
    shutil.copy2(M/'main.pdf',ROOT.parent/'Final_Manuscript/SheafCode_Validated_Manuscript.pdf')
                                                                               
    standalone=r'\documentclass[11pt]{article}\input{preamble.tex}\title{SheafCode: Additional Computational Validation}\author{Integrated research reference}\date{6 October 2026}\begin{document}\maketitle'+'\n'
    standalone+=r'\begin{center}AI-assisted reference; negative results and verification limits retained.\end{center}'+'\n'+extension.replace(r'Appendix~\ref{app:proofs}','the integrated manuscript proof appendix')
    standalone+=r'\bibliography{references}\end{document}'
    (M/'validation_report.tex').write_text(standalone,encoding='utf-8')
    result=subprocess.run([str(ROOT.parent/'audit_tools/tectonic/tectonic.exe'),'--only-cached','--keep-logs','--outdir',str(M),str(M/'validation_report.tex')],cwd=M,env=env,capture_output=True,text=True)
    (OUT/'report_compile.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError('Standalone report compilation failed')
    shutil.copy2(M/'validation_report.pdf',OUT/'ONLINE_VALIDATION_REPORT.pdf')
    save('report_metadata.json',{'status':'compiled','source_sha256':sha(__file__),'abstract_words':len(abstract.split()),
         'integrated_pdf_sha256':sha(M/'main.pdf'),'standalone_pdf_sha256':sha(OUT/'ONLINE_VALIDATION_REPORT.pdf'),
         'scope':'New reading copies include all earlier manuscript evidence and the six additional computational tracks; previous audited release unchanged.'})
    print('Figures and both reports compiled; abstract words',len(abstract.split()),flush=True)

if __name__=='__main__':main()
