                                                                                    
import sys,re,shutil,importlib.util,subprocess,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
os.environ['MPLCONFIGDIR']=str(OUT/'plot_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitz

PACKAGE=ROOT.parent/'IRIS_Submission_Recent_Data'
SRC=PACKAGE/'Editable_Source';FIG=SRC/'figures';REVIEW=PACKAGE/'Review_Notes'
REGIONS=['delhi','epa_la'];NAMES={'delhi':'Network A (India)','epa_la':'Network B (USA)'}
LABELS={'var_causal':'Causal VAR','seasonal':'Seasonal','city_mean':'City mean','gaussian':'Gaussian','spatial_sheaf':'Spatial sheaf','sheaf_causal':'Causal sheaf','sheaf_retrospective':'Retrospective sheaf'}

def js(region,name):return json.loads((OUT/region/name).read_text())
def csv(region,name):return pd.read_csv(OUT/region/name,dtype={'station':str})
def pct(x):return 'undefined' if x is None or not np.isfinite(x) else f'{100*x:.1f}'
def tex(x):return str(x).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&').replace('#',r'\#')
def anonymous(s):return s.replace('Los Angeles County','Network B').replace('Los Angeles','Network B').replace('Delhi','Network A')
def putfigure(name,fig):
    fig.tight_layout();fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');fig.savefig(FIG/(name+'.png'),dpi=160,bbox_inches='tight');plt.close(fig)
def figure(name,caption):return '\n'+r'\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{figures/'+name+r'}\caption{'+caption+r'}\end{figure}'+'\n'
def table(headers,rows,caption):
    return '\n'+r'\begin{table}[htbp]\centering\small\caption{'+caption+'}\n'+r'\resizebox{\linewidth}{!}{\begin{tabular}{'+'l'*len(headers)+r'}\toprule'+'\n'+' & '.join(headers)+r'\\\midrule'+'\n'+'\n'.join(' & '.join(map(str,r))+r'\\' for r in rows)+'\n'+r'\bottomrule\end{tabular}}\end{table}'+'\n'

def plots():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for ax,r in zip(axes,REGIONS):
        s=js(r,'reconstruction_summary.json')['primary_every_third_day'];methods=['var_causal','seasonal','city_mean','gaussian','spatial_sheaf']
        for i,k in enumerate(methods):
            x=s[k]['gain_percent'];ci=s[k]['ci95'];ax.plot(x,i,'o',color='#166b91')
            if ci:ax.hlines(i,*ci,color='#166b91',lw=2)
        ax.axvline(0,color='gray',lw=.8);ax.set_yticks(range(5),[LABELS[k] for k in methods]);ax.set_xlabel('Sheaf RMSE gain (%)');ax.set_title(NAMES[r]+f"\n{s['var_causal']['cases']} station-days; {s['var_causal']['dates']} dates")
    putfigure('primary_reconstruction',fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for ax,r in zip(axes,REGIONS):
        s=js(r,'ablation_summary.json')
        for i,(k,z) in enumerate(s.items()):
            ax.plot(z['gain_percent'],i,'o',color='#166b91' if z['gain_percent']>=0 else '#b84532')
            if z['ci95']:ax.hlines(i,*z['ci95'],color='gray')
        ax.axvline(0,color='gray');ax.set_yticks(range(len(s)),[k.replace('_',' ') for k in s]);ax.set_xlabel('Original sheaf gain over ablation (%)');ax.set_title(NAMES[r])
    putfigure('ablations',fig)
    fig,axes=plt.subplots(2,3,figsize=(10,5.7),sharex=True,sharey=True)
    for ri,r in enumerate(REGIONS):
        f=csv(r,'history_horizon_context_cases.csv')
        for ci,loss in enumerate([0,.2,.5]):
            a=np.full((2,3),np.nan)
            for hi,hist in enumerate([0,24]):
                for ti,horizon in enumerate([24,72,168]):
                    q=f[(f.history==hist)&(f.horizon==horizon)&(f.context_loss==loss)];p=q.pivot(index=['day','station'],columns='method',values='mse').dropna()
                    if len(p):a[hi,ti]=100*(1-np.sqrt(p.sheaf_causal.sum()/p.var_causal.sum()))
            ax=axes[ri,ci];im=ax.imshow(a,cmap='RdBu',vmin=-30,vmax=30,aspect='auto')
            for i,j in np.ndindex(a.shape):ax.text(j,i,'n/a' if not np.isfinite(a[i,j]) else f'{a[i,j]:.1f}',ha='center',va='center')
            ax.set_xticks(range(3),['24 h','72 h','168 h']);ax.set_yticks(range(2),['0 h history','24 h history']);ax.set_title(NAMES[r]+f'\nContext loss {loss:.0%}',fontsize=9)
    fig.subplots_adjust(right=.87);fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.02,label='RMSE gain over VAR (%)');fig.savefig(FIG/'stress.pdf',bbox_inches='tight');fig.savefig(FIG/'stress.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,r in zip(axes,REGIONS):
        f=csv(r,'available_fault_summary.csv');p=f.pivot(index='kind',columns='true_count',values='exact_support').reindex(['same','opposite','random','unequal','drift','stuck','raw_gain','one_channel']);a=100*p.to_numpy();ax.imshow(a,vmin=0,vmax=100,cmap='Blues',aspect='auto')
        for i,j in np.ndindex(a.shape):ax.text(j,i,f'{a[i,j]:.1f}' if np.isfinite(a[i,j]) else 'n/a',ha='center',va='center',color='white' if a[i,j]>65 else 'black')
        ax.set_xticks(range(len(p.columns)),[f'{k} faults' for k in p.columns]);ax.set_yticks(range(len(p)),[k.replace('_',' ') for k in p.index]);ax.set_title(NAMES[r]+'; end-to-end exact support (%)')
    putfigure('faults',fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for ax,r in zip(axes,REGIONS):
        f=csv(r,'uncertainty_summary.csv');q=f[(f.group=='all')&(f.context_loss==0)]
        for k,g in q.groupby('method'):ax.scatter(g.finite_mean_width,100*g.coverage,label=k.replace('_',' '))
        b=csv(r,'block_interval_summary.csv');z=b[(b.context_loss==0)&(b.method=='day_max')].iloc[0];ax.scatter(z.width,100*z.coverage,marker='s',label='day-max hourly')
        ax.axhline(90,color='gray',ls='--');ax.set_xlabel('Mean full interval width (PM2.5, ug/m3)');ax.set_ylabel('Hourly empirical coverage (%)');ax.set_title(NAMES[r]);ax.legend(fontsize=8)
    putfigure('uncertainty',fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for ax,r in zip(axes,REGIONS):
        f=csv(r,'alert_summary.csv')
        for row in f.itertuples():ax.scatter(100*row.false_alarm_rate,100*row.sensitivity if np.isfinite(row.sensitivity) else 0,label=LABELS[row.method])
        ax.set_xlabel('False alarm rate (%)');ax.set_ylabel('Sensitivity (%)');ax.set_title(NAMES[r]+(' (no positive hours)' if f.positive_cases.max()==0 else ''));ax.legend(fontsize=8)
        if f.positive_cases.max()==0:ax.set_ylabel('No sensitivity estimate');ax.set_yticks([])
    putfigure('alerts',fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for ax,r in zip(axes,REGIONS):
        f=csv(r,'exhaustive_block_margins.csv');col=next(k for k in f.columns if 'singular' in k or 'sigma' in k or 'margin' in k);group=next(k for k in f.columns if k in ['size','count','blocks','support_size'])
        for k,g in f.groupby(group):ax.scatter(np.repeat(k,len(g)),g[col],alpha=.6,s=15)
        ax.set_xlabel('Number of erased station blocks');ax.set_ylabel('Minimum singular value');ax.set_title(NAMES[r]);ax.set_xticks(range(1,5))
    putfigure('margins',fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for ax,r in zip(axes,REGIONS):
        f=csv(r,'verified_refinement_cases.csv');f['width']=f.upper-f.lower
        for k,g in f.groupby('node_budget'):ax.plot(k,g.width.mean(),'o',color='#166b91')
        ax.set_xlabel('Support-tree node budget');ax.set_ylabel('Mean certificate width (normalized target)');ax.set_title(NAMES[r]+'; 12 synthetic feasible states')
    putfigure('refinement',fig)
    fig,ax=plt.subplots(figsize=(8,3.3));s=js('delhi','grin_summary.json')['comparisons']
    for i,(seed,z) in enumerate(s.items()):
        ax.plot(z['gain_percent'],i,'o',color='#166b91');ax.hlines(i,*z['ci95'],color='#166b91')
    ax.axvline(0,color='gray');ax.set_yticks(range(3),[f'GRIN seed {k}' for k in s]);ax.set_xlabel('GRIN gain over retrospective sheaf (%)');ax.set_title('Matched retrospective information; all 242 Network A station-days');putfigure('neural',fig)
    fig,axes=plt.subplots(2,1,figsize=(9,5))
    for ax,r in zip(axes,REGIONS):
        d=load(r);f=pd.DataFrame(np.isfinite(d['X'][:,:,0]),index=d['times']).resample('D').mean();ax.imshow(f.T,aspect='auto',vmin=0,vmax=1,cmap='Blues',extent=[0,len(f),len(f.columns)-.5,-.5]);ax.set_yticks(range(len(f.columns)),d['stations']);ticks=np.linspace(0,len(f)-1,5).astype(int);ax.set_xticks(ticks,[str(f.index[i].date()) for i in ticks]);ax.set_ylabel('Site ID');ax.set_title(NAMES[r]+'; daily PM2.5 observation availability')
        for k,color in [(1,'#b84532'),(3,'#ab7212'),(2,'#166b91')]:ax.axvline((d['times'][d['split']==k].min().floor('D')-f.index.min()).days,color=color,lw=1)
    putfigure('availability',fig)

def evidence_tables():
    rows=[]
    for r in REGIONS:
        for p in sorted((OUT/r).glob('*.csv')):
            try:f=pd.read_csv(p)
            except pd.errors.EmptyDataError:f=pd.DataFrame()
            rows.append({'region':r,'table':p.name,'rows':len(f),'columns':len(f.columns),'sha256':sha(p),'scope':'See originating script and protocol; table rows are not independent sample counts; zero-column CSV indicates no eligible records'})
    frame_save('RESULT_TABLE_INVENTORY.csv',pd.DataFrame(rows))
    return rows

def documents():
    spec=importlib.util.spec_from_file_location('iris_builder',ROOT.parent/'audit_tools/build_iris_submission.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b);b.OUT=PACKAGE;b.SRC=SRC
    shutil.copy2(ROOT.parent/'Final_Manuscript/Extended_Validation/references.bib',SRC/'references.bib')
    with (SRC/'references.bib').open('a',encoding='utf-8') as f:f.write(r'''
@misc{openaqrecent,author={{OpenAQ}},title={Original measurement archive and public AWS access},year={2026},howpublished={\url{https://docs.openaq.org/aws/about}},note={2026 station measurements acquired 6 October 2026; native units and original compressed records retained}}
@misc{eparecent,author={{United States Environmental Protection Agency}},title={AirData 2026 hourly concentration files and data formats},year={2026},howpublished={\url{https://aqs.epa.gov/aqsweb/airdata/download_files.html}},note={Parameters 88101, 44201 and 42602; format documentation at https://aqs.epa.gov/aqsweb/airdata/FileFormats.html; preliminary reporting status possible}}
@misc{cpcbrecent,author={{Central Pollution Control Board}},title={Selected daily air-quality detail bulletins},year={2026},howpublished={\url{https://cpcb.nic.in/}},note={Eight weekly-dated original bulletins, 14 August to 2 October 2026; source URLs, exact document dates, hashes and parsed rows retained}}
''')
    ds={r:load(r) for r in REGIONS};rec={r:js(r,'reconstruction_summary.json') for r in REGIONS};a=rec['delhi']['primary_every_third_day']['var_causal'];us=rec['epa_la']['primary_every_third_day']['var_causal']
    abstract=f'''Air-quality monitoring gaps can hide local pollution while neighbouring stations remain active. SheafCode learns vector-valued spatial and temporal consistency maps and reconstructs missing channels causally, with conditional mathematical bounds for pollution quantities and unknown sensor faults. This study refits and evaluates the method using only 2026 observations: five Delhi stations from January to early October and four Los Angeles County stations from January to March. Training, model selection, interval calibration and testing occupy separate chronological blocks. In the fixed primary artificial-outage test, hiding a station for 24 hours after 24 hours of history, the sheaf achieved {a['gain_percent']:.1f}\\% lower normalized root mean squared error than a validation-tuned full-covariance vector autoregressive filter ({a['cases']} paired station-days, {a['dates']} dates; seven-day block interval {a['ci95'][0]:.1f}--{a['ci95'][1]:.1f}\\%). The US secondary test showed {us['gain_percent']:.1f}\\% lower error across {us['cases']} station-days, with uncertainty suppressed because only three weekly bins were available. Actual-map rank identities, functional envelopes, joint-threshold optimization and exact nominal dual corrections are checked on the new fitted operators. Countervailing results remain visible: identity spatial maps outperform the chosen learned maps under shared settings, weak or multiplicative faults often evade recovery, and no-feedback intervals under-cover extreme US observations. Simulated alert improvements incur additional false alarms. The demonstrated environmental value is better monitoring continuity and quantified uncertainty, not measured emissions reduction. Historical experiments remain separate supporting research; novelty priority and competition eligibility are not certified.'''
    body=r'''\section{Introduction and objective}
Missing readings interrupt air-quality surveillance. A dark station can have a different pollutant level from its neighbours, so city averaging may discard useful local information. Reconstruction accuracy, algebraic recoverability, uncertainty and decision usefulness are distinct questions.

The primary hypothesis is that training-fitted spatial and temporal sheaf maps improve causal whole-station reconstruction against a validation-tuned full-covariance vector autoregressive (VAR) filter under matched information. The study evaluates this hypothesis on entirely recent public observations. Secondary tasks examine quantity-specific certificates, unknown-count faults, longer outages, additional missing context, no-feedback intervals, alert proxies, artificial repair and computational verification. Naturally missing truth cannot establish reconstruction accuracy.

\section{Literature and mathematical contribution}
Cellular sheaves and their spectral theory predate this work \citep{hansen2019spectral,ghrist2011network}. Sheaf-based air-quality filtering is also prior work \citep{pham2022airquality}; CCA is established \citep{hotelling1936relations}. Sparse uniqueness, functional observability, bounded-error estimation and convex dual verification have substantial prior theory \citep{donoho2003,functional2022,bounded1990,boyd2004,jansson2004}. The project claims a specialized derivation and integrated monitoring implementation, rather than proven first-ever mathematical priority.

The central mathematical contribution is a coupled pollutant-threshold certificate over unknown bounded-fault explanations, accompanied by an exact nominal correction for approximate dual vectors. A joint feasible set can certify that at least one pollutant exceeds its reference without certifying a particular pollutant individually. Additional results analyze the actual structured maps rather than assuming generic Hall rank, separate residual sensitivity from prior bias, and permit safe support-tree refinement under limited computation. Written assumptions, proofs and counterexamples are provided; finite checks supplement, rather than replace, proofs.

\section{Recent data and frozen methodology}
\subsection{Provenance, dates and units}
Delhi observations were acquired from the OpenAQ public original-measurement archive \citep{openaqrecent}. A fixed identifier/geographic search found six candidate locations. A January 15 probe intersected a reporting gap; the documented availability amendment expanded the probe to January 1 before model outcomes. Training-only eligibility retained five stations and four channels; one location failed channel coverage. Original compressed daily records, acquisition attempts, timestamps, units and hashes are preserved. Actual valid observations run from January 1 to October 2, while the intended calendar ends October 5; missing end days remain missing.

An independent geographic source uses EPA 2026 hourly files \citep{eparecent}. The secondary acquisition protocol fixed Los Angeles County, January--March, three native channels, increasing site identifiers and a maximum six sites before fitting that network. Training-only eligibility retained four. The lowest training-eligible monitor POC is selected per site/channel; records are not pooled across POCs. Hourly file format was checked against the official schema. The archive may be preliminary and reporting-delayed. These are two networks, not nine independent regional replications or zero-shot transfer.
'''
    rows=[]
    for r in REGIONS:
        d=ds[r];mf=js(r,'dataset_manifest.json');rows.append([NAMES[r],len(d['stations']),len(d['channels']),int((d['split']==0).sum()),int((d['split']==1).sum()),int((d['split']==3).sum()),int((d['split']==2).sum())])
    body+=table(['Network','Sites','Channels','Train h','Select h','Cal. h','Test h'],rows,'Chronological timeline positions. Individual observations and station-days are dependent.')
    body+=figure('availability','Daily PM2.5 availability on the complete intended timeline. Vertical lines mark selection, calibration and test starts. Light cells indicate absent observations, not zero pollution; availability differs across native channels.')
    body+=r'''Delhi uses PM2.5, PM10 and ozone in $\mu$g/m$^3$, and NO2 in ppb. EPA uses PM2.5 in $\mu$g/m$^3$, ozone in ppm and NO2 in ppb. Gas-volume units are preserved and normalized within their own station/channel; they are not silently converted to mass. Delhi uses India local time; EPA uses UTC. Duplicate compatible Delhi station-hour/channel readings are averaged explicitly. Negative reported concentrations are excluded by the stated ingestion rule, which can affect low-level summaries.

\subsection{Training, model selection and information sets}
The complete intended calendar is divided into 60\% training, 20\% validation and 20\% test. Validation is split into equal model-selection and calibration blocks. Station/channel eligibility, means, standard deviations, correlation graph, CCA maps and covariance fitting use training only. Eligibility requires at least 50\% training coverage and 720 observed training hours per channel, at least three stations and two channels including PM2.5. No historical data train or calibrate these models.

Station vectors $x_v\in\mathbb R^p$ are normalized in their native channel units. Weighted CCA spatial maps and elapsed one-/twenty-four-hour temporal maps define discrepancies $Dx$. With observed coordinates $O$, unknown coordinates $U$, block precision $P\succ0$ and $\lambda>0$, decoding solves
\begin{align}
\widehat z&=\arg\min_z\|D_Uz+D_Ox_O\|_2^2+\lambda x(z)^TPx(z),\label{eq:objective}\\
\widehat z&=-(D_U^TD_U+\lambda P_{UU})^{-1}(D_U^TD_O+\lambda P_{UO})x_O.\label{eq:decode}
\end{align}
The full-state prior includes $P_{UO}x_O$, allowing partial-channel conditioning. Missing cells remain unknown. In a temporal window $D$ includes temporal rows. The causal estimate uses only the prefix ending at its prediction hour; the retrospective estimate can use future surviving context, never hidden targets. Artificial-outage truth is used only for evaluation. Hidden-target and future perturbation tests check causal invariance.

Sheaf selection searches $\lambda\in\{0.1,1\}$ and temporal scale $\{0.5,2\}$. VAR searches ridge $\{1,100\}$, innovation-noise scale $\{0.05,0.5\}$ and covariance shrinkage $\{0,0.2,1\}$. Training-only iterative completion and correlated innovations strengthen the VAR control. Selection uses the first validation half; no-feedback intervals use the second. Controls include raw city mean (averaging concentration before target normalization), seasonal persistence, static Gaussian conditioning and spatial sheaf reconstruction. Every method uses the same eligible target cells.

The primary task hides one whole station for 24 hours after 24 hours of history, selecting every third eligible test day with a fixed anchor. Each case needs at least $12p$ observed target cells. Aggregate RMSE is the square root of mean station-day MSE, not a pooled-cell RMSE or average of daily RMSE. Gain is $100(1-\sqrt{\sum E_{\rm sheaf}/\sum E_B})$. Physical PM2.5 error is separate. Paired seven-day calendar-bin bootstrap intervals use 2,000 draws; intervals are suppressed with fewer than five bins and remain exploratory under temporal dependence. All eligible days and eighteen history/horizon/context-loss settings are reported as sensitivities, not alternative primary choices.

The official pinned GRIN implementation \citep{cini2022grin} is trained on Delhi with all three seeds and widths 32/64, at most thirty epochs and seven-epoch early stopping. Each seed chooses its width/checkpoint on selection validation only. Its bidirectional results are compared with the retrospective sheaf. This additional benchmark, the 66-weight convex hybrid and conservative block intervals were specified after the primary outcomes, so their status is exploratory. Every attempted seed and negative result is retained.
'''
    math=b.BODY[b.BODY.index(r'\section{Mathematical results}'):b.BODY.index(r'\section{Results and numerical validation}')]
    math=math.replace('The fitted seven-network operators have zero numerical kernel at their stated tolerance','Both recently fitted operators have zero numerical kernel at their stated tolerance').replace(r'$\square$',' QED')
    body+=math
    body+=r'''\section{Recent reconstruction results}
'''
    rows=[]
    for r in REGIONS:
        for scope,key in [('Primary','primary_every_third_day'),('All days','all_eligible_test_days')]:
            z=rec[r][key]['var_causal'];ci=z['ci95'];rows.append([NAMES[r],scope,z['cases'],z['dates'],f"{z['gain_percent']:.2f}",f"{ci[0]:.2f}--{ci[1]:.2f}" if ci else 'Suppressed'])
    body+=table(['Network','Task','Cases','Dates','Gain (\%)','Block interval'],rows,'Causal sheaf gain over validation-tuned causal full-covariance VAR. Positive favors the sheaf; US intervals are suppressed because only three weekly bins exist.')
    body+=figure('primary_reconstruction','Fixed primary task against five matched controls. Delhi intervals resample seven-day calendar bins; US points have insufficient bins for interval estimation.')
    body+=f"The Delhi primary PM2.5-only physical gain over VAR is {rec['delhi']['primary_every_third_day']['pm25_physical']['var_causal']['gain_percent']:.2f}\\%, distinct from the multichannel normalized headline. EPA physical PM2.5 has 23 eligible paired cases rather than 24, because one multichannel case has no observed PM2.5 truth. All-case losses, exclusions and eligible cells are retained. Both recent networks support improved reconstruction under this protocol, but the short US period cannot establish broad seasonal generalization.\n"
    ab=js('delhi','ablation_summary.json');hy=js('delhi','hybrid_summary.json')
    body+=f"Removing the 24-hour temporal lag worsens Delhi error: the original sheaf gains {ab['no_lag24']['gain_percent']:.2f}\\% over that ablation. However, its gain over identity spatial maps is {ab['identity_spatial_maps']['gain_percent']:.2f}\\%, meaning the identity-map variant performs better under shared temporal maps and settings. This does not establish CCA superiority. Ablations are not independently retuned. The validation-selected convex hybrid improves over the chosen sheaf by {hy['sheaf_causal']['gain_percent']:.2f}\\% on all test cases; it is an exploratory addition, not a replacement for the primary endpoint.\n"
    body+=figure('ablations','Fixed-settings ablations retain countervailing results. Negative gain means the ablation has lower error. Lag removal, identity maps and scalar maps change different aspects of the operator.')
    body+=figure('stress','History, outage length and additional surviving-context loss. Values compare causal sheaf with causal VAR on matched cases within each configuration; configurations are not pooled. Limited eligible long US windows can yield missing entries.')
    gs=js('delhi','grin_summary.json');rows=[]
    for c in gs['choices']:
        z=gs['comparisons'][str(c['seed'])];rows.append([c['seed'],c['width'],c['epoch'],z['cases'],f"{z['gain_percent']:.2f}",f"{z['ci95'][0]:.2f}--{z['ci95'][1]:.2f}"])
    body+=table(['Seed','Width','Epoch','Cases','GRIN gain (\%)','Block interval'],rows,'Every neural seed versus retrospective sheaf on matched Delhi test cases. Positive favors GRIN; these are not causal comparisons.')+figure('neural','All optimization seeds retained. Widths and epochs are chosen on selection validation; the test set is evaluated only after each seed choice.')
    body+=r'''\section{Faults and uncertainty: successes and failures}
Unknown-count support search can return a clean state or one/two faulty station blocks. It fits a training-whitened syndrome and validation classifier, then fixes a five-percent clean calibration cutoff on the separate calibration block. Whole-network-complete 24-hour test episodes do not exist in either recent dataset. That strict criterion is reported as inapplicable; a separate availability extension restricts candidate stations to those fully observed at the hour and injects persistent support only into stations complete across an episode. Missing context is training-mean imputed for the syndrome. This is an empirical accommodation, not exact consistency of the physical network.

Injected families include same/opposite standardized offsets, time-varying random directions, unequal blocks, drift, stuck readings, raw multiplicative gain and one-channel corruption. Random directions are redrawn hourly, so they are not a fixed persistent direction. Support size and alarms are inferred, not supplied from test truth. The model can report at most two blocks and cannot exactly recover three. Hourly exact support, alarm, count and episode denominators are separate; episode-hour rows are dependent.
'''
    rows=[]
    for r in REGIONS:
        f=csv(r,'available_fault_summary.csv');status=js(r,'available_fault_status.json');clean=f[f.kind=='clean'].iloc[0]
        for kind in ['opposite','unequal','drift','stuck','raw_gain']:
            q=f[(f.kind==kind)&(f.true_count==2)].iloc[0];rows.append([NAMES[r],kind.replace('_',' '),int(q.cases),pct(q.alarm_rate),pct(q.exact_support)])
        body+=f"{NAMES[r]} supplies {status['sampled_episodes']} available episodes; clean false alarms occur in {pct(clean.alarm_rate)}\\% of {int(clean.cases)} episode hours. Strong opposite offsets can localize accurately while weak unequal, drift, stuck or gain faults expose poor recovery. These are synthetic injections and do not verify natural field faults.\n"
    body+=table(['Network','Two-fault family','Hours','Alarm (\%)','Exact support (\%)'],rows,'Availability-conditioned persistent episodes, with unknown count. Dependencies prevent treating each hourly row as an independent trial.')+figure('faults','End-to-end exact support across injected families and counts. Maximum reported count two makes three-fault exact recovery impossible. High recovery of large offsets does not generalize to stuck/gain/weak faults.')
    body+=r'''No-feedback PM2.5 intervals use only the calibration block, with pooled residual, station residual and genuinely pattern-varying availability scale controls. These are static snapshot reconstructions with the target station hidden, rather than temporal outage intervals. Positive fixed station scales would cancel and are not credited. Context-loss masks are formed before outcomes. Native-unit full width and abstention are reported alongside coverage. Sequential adaptive intervals are separate revealed-label experiments with correct infinite/full and empty boundaries; they cannot adapt operationally during a dark outage without labels. Time dependence and distribution shift prevent an unconditional exchangeability guarantee.
'''
    rows=[]
    for r in REGIONS:
        f=csv(r,'uncertainty_summary.csv');q=f[(f.group=='all')&(f.context_loss==0)]
        for z in q.itertuples():rows.append([NAMES[r],z.method.replace('_',' '),z.cases,pct(z.coverage),f'{z.finite_mean_width:.1f}'])
    body+=table(['Network','No-feedback interval','Cells','Coverage (\%)','Width'],rows,'Nominal 90\% snapshot PM2.5 intervals at no additional context loss. Width is full interval width in $\mu$g/m$^3$.')+figure('uncertainty','Empirical hourly coverage versus interval width. Day-max calibration improves hourly coverage by widening intervals; weekly-max intervals are entirely infinite and therefore uninformative, omitted from the finite-width plot.')
    ep=csv('epa_la','uncertainty_summary.csv');tail=ep[(ep.group=='extreme')&(ep.context_loss==0)&(ep.method=='station_fixed')].iloc[0];day=csv('delhi','block_interval_day_summary.csv');day=day[(day.context_loss==0)&(day.method=='day_max')].iloc[0]
    body+=f"Delhi has no test PM2.5 values above each station's training 95th percentile, so extreme coverage is undefined there. The US station-fixed interval covers only {pct(tail.coverage)}\\% of {int(tail.cases)} training-threshold extreme cells. Delhi day-max intervals cover {pct(day.simultaneous_coverage)}\\% of {int(day.station_days)} station-days simultaneously, despite 98.7\\% hourly coverage. The denominators include partial observed days and refer only to available truth hours. Weekly calibration has too few groups: 100\\% observed coverage comes from full infinite sets and supplies no decisions. None of these numbers proves calibrated confidence during every real outage.\n"
    body+=r'''\section{Environmental monitoring and decision proxies}
Hourly PM2.5 exceeding 60 $\mu$g/m$^3$ is an illustrative reference, not an official hourly standard, CPCB AQI or GRAP action. Each method chooses an alarm threshold on calibration negatives to target five-percent false alarms. Test sensitivity and false alarms are both reported, so greater sensitivity is not misrepresented as superiority at matched test false-alarm rates.
'''
    rows=[]
    for r in REGIONS:
        for z in csv(r,'alert_summary.csv').itertuples():rows.append([NAMES[r],LABELS[z.method],z.positive_cases,z.negative_cases,pct(z.sensitivity),pct(z.false_alarm_rate)])
    body+=table(['Network','Method','Positive','Negative','Sensitivity (\%)','False alarm (\%)'],rows,'Causal-outage PM2.5 hourly threshold proxy on common truth cells; US sensitivity is undefined with zero positive hours.')+figure('alerts','Actual test trade-offs at calibration-selected cutoffs. The Los Angeles plot shows false alarms only because there are no positive target hours. A target five-percent calibration rate need not transfer to test.')
    pa=js('delhi','paired_alert_summary.json')['var_causal'];body+=f"Delhi sheaf sensitivity exceeds VAR by {pa['sensitivity_difference']['difference_percentage_points']:.2f} percentage points, with paired weekly interval {pa['sensitivity_difference']['ci95'][0]:.2f}--{pa['sensitivity_difference']['ci95'][1]:.2f}. Its false-alarm rate also increases by {pa['false_alarm_difference']['difference_percentage_points']:.2f} points. This is a trade-off, not a demonstrated regulatory decision benefit.\n"
    d=csv('delhi','pm25_daily_summary.csv');q=d[d.method=='sheaf_causal'].iloc[0];c=d[d.method=='city_mean'].iloc[0]
    body+=f"A PM2.5-only daily category proxy requires at least eighteen observed target hours and compares with the actual preceding calendar day. On {int(q.common_station_days)} common station-days the sheaf and city fallback both achieve {pct(q.category_accuracy)}\\% accuracy. Only {int(q.poor_or_worse_truth_days)} truth day is Poor or worse, and every method misses it. These results cannot establish GRAP action recovery or official AQI validity.\n"
    of=js('delhi','official_summary.json');body+=f"Eight date-verified CPCB detail bulletins yield {of['parsed_station_entries']} parsed hotspot entries and only {of['eligible_station_days']} eligible matched station-days. A two-PM concentration proxy agrees in {of['agreements']['sheaf_PM_proxy']}/{of['eligible_station_days']} sheaf and {of['agreements']['city_PM_proxy']}/{of['eligible_station_days']} city cases. Omitted gases, selected hotspots, native reporting windows and the small denominator make this descriptive, not regulatory validation. Original PDFs, rejected alignments and parser corrections are preserved \\citep{{cpcbrecent}}.\n"
    for r in REGIONS:
        z=js(r,'matched_missing_mask_summary.json');body+=f"{NAMES[r]} actual-missing-pattern transplantation hides station pollutant/hour masks on different fully observed test days: {z['cases']} hidden PM2.5 cells and {z['evaluated_patterns']} source patterns. "
        if z['cases']:body+=f"Sheaf PM2.5 error is {z['sheaf_pm25_rmse']:.2f} versus city {z['city_pm25_rmse']:.2f} $\\mu$g/m$^3$ ({z['gain_percent']:.2f}\\% gain). "
        body+='This artificial-mask test does not verify originally missing truth; complete-day selection and repeated recipient days limit generalization.\n'
    body+=r'''Naturally missing-cell estimates and disagreement with the city fallback are retained as exploratory estimates, not measured unobserved exposure. No emissions, weather or health intervention was performed. The supported environmental application is continuity of pollution monitoring, with uncertainty and abstention where the evidence cannot support a decision. Weather-forecast improvement has not been evaluated.

\section{Mathematical and computational checks on recent operators}
'''
    rows=[]
    for r in REGIONS:
        rank=csv(r,'structured_rank_cases.csv');thin=csv(r,'thinning_cases.csv');margin=csv(r,'exhaustive_block_margins.csv');fun=csv(r,'functional_joint_bounded_cases.csv');syn=csv(r,'synthetic_certificate_cases.csv');op=csv(r,'operator_uncertainty_cases.csv');rows.append([NAMES[r],len(rank),int(rank.agree.sum()),len(thin),len(margin),len(fun),len(syn),len(op)])
    body+=table(['Network','Sets','Agree','Thinned','Margins','Real states','Synthetic','Map cases'],rows,'Actual-map analyses. Every nonempty outage subset and every union up to four blocks is enumerated; synthetic feasibility and declared map perturbations are not field uncertainty measurements.')
    jointrows=[]
    for r in REGIONS:
        f=csv(r,'functional_joint_bounded_cases.csv');feasible=f[f.joint_status=='bounded'];jointrows.append([NAMES[r],len(f),len(feasible),int((f.joint_status=='inconsistent').sum()),int((feasible.joint_decision=='abstain').sum()),f'{feasible.joint_extra_lower.median():.3f}'])
    body+=table(['Network','States','Feasible','Inconsistent','Abstained','Median gain'],jointrows,'Joint-score certificate outcomes. Median gain is the dimensionless joint lower bound minus the largest separate lower bound, not a reduction in pollution or prediction error.')
    body+=r'''All 39 feasible Network A and 38 feasible Network B real-state joint cases abstain at the declared residual, fault and state-box budgets. One and two cases, respectively, have inconsistent feasible models and must not be counted as successful threshold certificates. Positive lower-bound tightening demonstrates coupling, but no definite real-data exceedance or safe decision is established by this experiment. The constructive threshold example proves the possible mathematical benefit; it is not a measured regulatory benefit.\par
'''
    body+=figure('margins','Exhaustive singular margins for every one- through four-station union, without connected-set or first-5,000 truncation. Uniform two-fault separation requires the relevant four-block margin and each trial residual condition.')+figure('refinement','Conditional certificate refinement at fixed node budgets on twelve planted feasible cases per network. Computation may tighten outer bounds; nominal rational validity does not estimate physical map error.')
    body+=r'''Forty held-out full states per network evaluate functional residual-ball and bounded-fault envelopes, while twelve planted feasible states check support unions and outer dual containment. Coverage conditioned on the residual/fault/state assumptions is kept distinct from overall empirical coverage. Declared operator entry perturbations at four radii give forty-eight planted cases per network; inflated residual budgets contain the planted truth, but physical CCA-map uncertainty is not estimated. Exact decimal-rational nuisance-span enumeration concerns unrestricted nominal spans; equal spans do not imply equal bounded fault images.

The algebraic finite verification suite passes 37,678 exact rational checks independently of observation age. The code-specific tests check chronology, training normalization, causal invariance, observed preservation, prior cross terms, raw-unit city means and quantile boundaries. Hashes and independently reaggregated case tables are audited separately. Finite examples and internal proof review do not constitute a machine-checked proof of every theorem or external novelty certification.

Artificial one-station repair policies compare trace-based, residual-variance and random selection on held-out outages. Repair gains are exploratory; their Delhi intervals include zero. Simulated software repair is not a demonstrated maintenance or emissions intervention. Per-station singular-margin risk diagnostics have only five Delhi or four US stations and do not establish robust risk prediction.

\section{Discussion and limitations}
The strongest recent empirical result is the pre-specified causal 24-hour reconstruction comparison. It persists across all eligible Delhi days and a secondary US network, yet the US period has only three weekly bins. Separate source networks are useful replication, but not independent proof of universal performance. Seasonal test conditions differ from training; absent extreme Delhi cases and poor extreme US interval coverage restrict environmental claims.

Identity-map superiority under shared settings limits attribution to learned CCA maps. Neural seeds and retrospective gains answer a different information-set question. The exploratory hybrid cannot retrospectively change the primary endpoint. Fault episodes use synthetic injections and availability conditioning; mean-filled natural context changes syndromes, and weak/stuck/gain faults often evade recovery. Calibration-selected alert cutoffs fail to maintain their target test false-alarm rate. Sparse severe daily events and a four-case official comparison cannot demonstrate real GRAP decisions. Naturally dark targets remain unverifiable. Exact nominal coefficient checks do not estimate physical operator uncertainty.

All recent empirical fitting and evaluation use 2026 observations, within twelve months of this study. Historical research is retained separately as supporting evidence, without replacing recent denominators or implying that computational reruns change old observation dates. Public data-age requirements are addressed at the observation level; signed-in portal rules, supervision, personal attestations and organizer acceptance still require the submitter's review.

\section{Conclusion}
The recent-data study supports improved causal monitoring continuity against a stronger tuned filter, while carefully separating normalized error, physical PM2.5 error, reconstruction, detection, uncertainty and environmental decisions. Coupled threshold and exact nominal certificate results provide a mathematical reliability framework under explicit assumptions. The measured outcome is computational monitoring improvement; reduction in pollution or health harm remains a prospective application requiring deployment and validation.

\section{Acknowledgements and assistance}
OpenAQ, the original public monitoring providers, EPA and CPCB supplied public observations or bulletins. Published GRIN source is credited. AI assistance was used in code development, mathematical derivation and checking, experimental implementation, audits and writing. This disclosure does not certify student-only independent authorship. Student contributions, supervising guide, external assistance and required forms must be recorded accurately by the submitter. No names, affiliations, signatures or experimental dates have been invented.
'''
    synopsis={
      'Introduction and objective':'''Air-quality stations can become unavailable while surrounding sensors remain active. A city average may suppress local pollution differences, but sophisticated reconstruction also needs comparison with strong causal controls and explicit uncertainty. This project asks whether vector-valued spatial and temporal consistency relations improve reconstruction of a whole station during a twenty-four-hour outage. A second objective is to identify when a reconstructed pollutant quantity or coupled threshold decision can be bounded despite uncertain station faults. The present empirical core uses only public observations recorded during 2026, with model fitting, selection, calibration and testing separated chronologically. Environmental value is assessed through monitoring continuity and carefully labelled alert proxies; measured pollution reduction is not claimed.''',
      'Innovation':'''The project combines causal sheaf reconstruction with analysis of the actual structured maps, quantity-specific error envelopes and bounded unknown-fault explanations. A joint-threshold result optimizes a coupled pollutant score over support-union feasible sets; an exact nominal dual correction verifies approximate certificates. Established sheaf, sparse recovery and optimization theory is credited. Novelty is the specialized derivation, implementation and evaluated monitoring framework; first-ever mathematical priority is not established. Results distinguish algebraic recoverability, empirical accuracy, uncertainty and regulatory decision validity.''',
      'Methodology':'''Public OpenAQ original measurements supply five eligible Delhi stations and four channels from January to early October 2026. EPA hourly files supply four Los Angeles County stations and three channels from January to March. Native mass and gas-volume units are preserved. Complete calendars are divided into sixty percent training, twenty percent validation and twenty percent testing; validation is split again for model selection and interval calibration. Station eligibility, normalization, graph construction, spatial and temporal maps and covariance fitting use training only. A fixed grid selects sheaf and full-covariance VAR parameters on selection validation. The primary test hides one whole station for twenty-four hours after twenty-four hours of history on every third eligible test day. Matched seasonal, city, Gaussian and spatial controls use identical target cells. All eligible days, longer outages, context losses, partial channels, unknown-count faults, no-feedback intervals, simulated repair, exact certificates and all three retrospective neural seeds are retained. Seven-day paired-bin resampling reports exploratory intervals only when enough bins exist.''',
      'Results and conclusions':f'''Delhi causal reconstruction reduced normalized RMSE by {a['gain_percent']:.1f}% against tuned VAR across {a['cases']} paired station-days and {a['dates']} dates; the weekly-bin interval was {a['ci95'][0]:.1f} to {a['ci95'][1]:.1f}%. The secondary US gain was {us['gain_percent']:.1f}% across {us['cases']} station-days, with uncertainty suppressed for too few bins. Identity spatial maps outperform the selected learned maps under shared settings. Strong injected offsets localize well, but weak, stuck and gain faults expose failures. No-feedback intervals under-cover extreme US observations; conservative weekly sets are entirely infinite. Alert sensitivity improvements incur more false alarms. Actual-map and exact rational mathematical checks support conditional certificates without certifying field assumptions. These findings support better monitoring continuity, rather than demonstrated emissions reduction or verified regulatory decisions.''',
      'Acknowledgements and references':'''Public observations are credited to OpenAQ, original monitoring providers and the United States Environmental Protection Agency. Central Pollution Control Board bulletins and published GRIN code are acknowledged. References include Hansen and Ghrist on sheaf spectral theory, Donoho and Elad on sparse uniqueness, and Gibbs and Candes on adaptive coverage. AI assistance in code, mathematical checking, audits and writing is disclosed. Student and guide contribution records must accurately describe the actual work.''' }
    overlap=js('','PRIOR_RECENT_OVERLAP.json')
    body=body.replace(r'\section{Recent reconstruction results}',f"The preserved earlier supplementary acquisition shares {overlap['shared_hour_channel_rows']:,} station-hour/channel records at three Network A instruments; {overlap['equal_aggregated_values']:,} aggregated values agree. Shared records are from September 12 to October 2; three boundary-hour ozone aggregates differ. Earlier historical cohorts can also overlap this calendar. Thus the present fitting/evaluation blocks are disjoint within the new protocol, but the observations are not a wholly new blinded study. All findings are exploratory; specifying a primary endpoint before these new fits is not external preregistration or proof of no prior test access.\\par\n"+r'\section{Recent reconstruction results}')
    body=body.replace('pre-specified causal 24-hour reconstruction comparison','fixed-protocol causal 24-hour reconstruction comparison')
    body=body.replace('The algebraic finite verification suite passes 37,678 exact rational checks independently of observation age.','A separate incidence-row construction and observation-matrix Kalman implementation reproduce all seven reconstruction methods on all 310 eligible station-days (2,170 case-loss comparisons plus two static-objective checks), using the frozen learned parameters rather than independently refitting them. Every selected neural checkpoint also reproduces all 242 test cases for each of three seeds. Conditional joint-score truth containment is separately checked. The algebraic finite verification suite passes 37,678 exact rational checks independently of observation age.')
    body=body.replace('For noisy size-$t$ support search, sufficient separation uses omitted-block energy','Define the column gain $\\sigma_{\\min}(M)=\\inf_{\\|u\\|_2=1}\\|Mu\\|_2$, which is zero for a column-rank-deficient or underdetermined matrix. For noisy size-$t$ support search, sufficient separation uses omitted-block energy')
    abstract=abstract.replace('Historical experiments remain separate supporting research; novelty priority and competition eligibility are not certified.','Prior observations partly overlap this refitted exploratory study; no independent blinded validation is claimed.')
    synopsis['Results and conclusions']+=' Shared prior observations make the new evaluation exploratory, rather than independently blinded.'
    abstract=anonymous(abstract);body=anonymous(body);synopsis={k:anonymous(v) for k,v in synopsis.items()}
    title=r'''\hypersetup{pageanchor=false}\begin{titlepage}\centering\vspace*{2cm}{\LARGE\bfseries SheafCode\par}\vspace{.5cm}{\Large Recent-Data Causal Air-Quality Reconstruction and Conditional Pollution Certificates\par}\vspace{1cm}{\large Research paper\par}{7 October 2026\par}\vfill\begin{minipage}{.87\linewidth}\small\textbf{Empirical scope.} All empirical fitting, model selection, calibration and testing in this paper use observations recorded during 2026. Public monitoring networks are coded A (India) and B (USA) for anonymous review. Source identifiers, dates, native units and original records are preserved in the reproducibility evidence.\par\medskip\textbf{Assistance.} AI assistance in code, mathematical derivation/checking, audits and writing is disclosed in the acknowledgements.\end{minipage}\end{titlepage}\pagenumbering{roman}\hypersetup{pageanchor=true}\tableofcontents\clearpage\pagenumbering{arabic}
'''
    preamble=b.PREAMBLE.replace(r'\begin{document}',r'\DeclareMathSizes{10.95}{10.95}{8}{8}\newtheorem{corollary}{Corollary}\begin{document}')
    paper=preamble+title+'\n\\begin{abstract}\n'+abstract+'\n\\end{abstract}\n'+body+'\n\\bibliography{references}\\end{document}\n'
    (SRC/'research_paper.tex').write_text(paper,encoding='utf-8');b.compile_pdf('research_paper',PACKAGE/'Upload_Files/SheafCode_Recent_Research_Paper.pdf')
    short=b.PREAMBLE+r'\begin{center}{\Large\bfseries SheafCode}\par Recent-Data Causal Air-Quality Reconstruction and Conditional Pollution Certificates\par Research synopsis\end{center}'+'\n'
    for k,v in synopsis.items():
        if k=='Results and conclusions':short+='\\clearpage\n'
        short+='\\section*{'+k+'}\n'+v.replace('%',r'\%')+'\n'
    short+='\\end{document}\n';(SRC/'synopsis.tex').write_text(short,encoding='utf-8');b.compile_pdf('synopsis',PACKAGE/'Upload_Files/SheafCode_Recent_Research_Synopsis.pdf')
    (PACKAGE/'Upload_Files/Project_Abstract.txt').write_text(abstract.replace(r'\%','%').replace('--','–')+'\n',encoding='utf-8')
    (PACKAGE/'Upload_Files/Portal_Text.txt').write_text('Project title: SheafCode: Recent-Data Causal Air-Quality Reconstruction and Conditional Pollution Certificates\n\nAbstract\n'+abstract.replace(r'\%','%').replace('--','–')+'\n\n'+'\n\n'.join(k+'\n'+v for k,v in synopsis.items()),encoding='utf-8')
    proofs=(ROOT.parent/'Final_Manuscript/Extended_Validation/sections/10_proofs.tex').read_text(encoding='utf-8').replace('The seven fitted operators have zero numerical kernel at their recorded tolerance','The two recent fitted operators have zero numerical kernel at their recorded tolerance').replace(r'$\square$',' QED')
    proofs=proofs.replace('All matrices are fixed before the state being reconstructed.','All matrices are fixed before the state being reconstructed. Residual radii and fault-box bounds are nonnegative; support budgets are integers. Whenever a bounded-fault polyhedron is used for attained extrema, the trusted state bounds are finite and ordered, and the feasible set is nonempty. These model assumptions are distinct from empirical calibration.')
    proofs=proofs.replace('Let $Z$ be the nonempty finite union of bounded-fault feasible state polyhedra and','Let $Z$ be a nonempty finite union of bounded-fault feasible state polyhedra contained in a finite trusted state box $[\\ell,u]$. Let')
    proofs=proofs.replace('For a noisy syndrome $s=D_Fe_F+r$ with exactly $t$ nonzero fault blocks','For a noisy syndrome $s=D_Fe_F+r$ with $1\\leq t\\leq n$ and exactly $t$ nonzero fault blocks')
    proofs=proofs.replace('Define $d_{\\rm block}$ as the minimum number of station blocks in a nonzero vector of $\\ker D$, or infinity if the kernel is zero.','Define $d_{\\rm block}$ as the minimum number of station blocks in a nonzero vector of $\\ker D$, or infinity if the kernel is zero. Here $\\sigma_{\\min}(M)=\\inf_{\\|u\\|_2=1}\\|Mu\\|_2$ is the column gain: it is zero when the matrix has a nontrivial column kernel, including underdetermined matrices.')
    proofs=proofs.replace('The implemented streams predict all simultaneous channels before revealing the hour\'s targets; labels from other channels cannot be silently used earlier.','The recent adaptive stream evaluates one PM2.5 target per station/hour. Multiple simultaneous target channels, if used, must all be predicted before the hour\'s target vector is revealed; earlier target channels cannot silently inform later intervals.')
    (SRC/'complete_theory.tex').write_text(proofs,encoding='utf-8')
    inventory=pd.read_csv(OUT/'RESULT_TABLE_INVENTORY.csv');appendix='\\appendix\n'+proofs+'\n\\clearpage\n\\section{Detailed recent numerical supplement}\n'
    for r in REGIONS:
        appendix+='\\clearpage\n\\subsection{'+NAMES[r]+'}\n'
        f=csv(r,'snapshot_channel_cases.csv');rows=[]
        for (mask,method),g in f.groupby(['mask_type','method']):rows.append([mask.replace('_',' '),LABELS.get(method,method),len(g),f'{np.sqrt(g.mse.mean()):.4f}'])
        appendix+=table(['Mask','Method','Cases','Normalized RMSE'],rows,'Snapshot and partial-channel tests at fixed sampled hours; matching cells are used within each mask.')
        f=csv(r,'history_horizon_context_cases.csv');rows=[]
        for (history,horizon,loss),g in f.groupby(['history','horizon','context_loss']):
            p=g.pivot(index=['day','station'],columns='method',values='mse').dropna();rows.append([history,horizon,f'{loss:.0%}'.replace('%',r'\%'),len(p),f'{100*(1-np.sqrt(p.sheaf_causal.sum()/p.var_causal.sum())):.2f}'])
        appendix+=table(['History h','Outage h','Context loss','Cases',r'VAR gain (\%)'],rows,'All fixed stress configurations; within-configuration normalized station-day RMSE.')
        s=js(r,'repair_summary.json');rows=[]
        for k,z in s.items():rows.append([k.replace('_',' '),z['cases'],f"{z['gain_percent']:.2f}", 'Suppressed' if z['ci95'] is None else f"{z['ci95'][0]:.2f}--{z['ci95'][1]:.2f}"])
        appendix+=table(['Repair policy','Cases',r'Gain (\%)','Block interval'],rows,'Artificial outage repair versus random selection. No physical maintenance or environmental intervention.')
        f=csv(r,'revealed_adaptive_cases.csv');finite=f[np.isfinite(f.radius)];rows=[[len(f),pct(f.covered.mean()),pct(f.full_set.mean()),pct(f.empty_set.mean()),f'{2*finite.radius.mean():.2f}']]
        appendix+=table(['Cells',r'Coverage (\%)',r'Full sets (\%)',r'Empty sets (\%)','Finite width'],rows,'Revealed-label adaptive snapshot intervals. Native PM2.5 full widths exclude full/empty sets; dark-outage feedback is unavailable.')
        f=csv(r,'block_interval_day_summary.csv');rows=[[z.context_loss,z.method.replace('_',' '),z.station_days,pct(z.simultaneous_coverage),pct(z.full_set_fraction)] for z in f.itertuples()]
        appendix+=table(['Context loss','Method','Station-days',r'Simultaneous (\%)',r'Full sets (\%)'],rows,'Simultaneous coverage of available station-day truth, including partial days. Weekly full sets are uninformative.')
        f=csv(r,'functional_joint_bounded_cases.csv');rows=[]
        for method in ['functional','bounded']:
            q=f[f[method+'_assumptions_met']];rows.append([method,len(f),int(f[method+'_covered'].sum()),len(q),int(q[method+'_covered'].sum())])
        appendix+=table(['Envelope','States','Covered','Assumptions met','Conditional covered'],rows,'Forty real held-out states per network. Assumption membership differs from overall empirical coverage.')
        sp=csv(r,'exact_nuisance_span_counts.csv');rows=[[z.station,z.support_count,z.distinct_exact_decimal_spans,f'{z.seconds:.3f}'] for z in sp.itertuples()]
        appendix+=table(['Station','Supports','Distinct exact spans','Seconds'],rows,'Nominal decimal-rational unrestricted nuisance spans. Equal spans do not imply equal bounded images.')
        risk=csv(r,'risk_diagnostic.csv');rows=[[z.station,f'{z.minimum_singular_value:.4f}',f'{z.test_mean_mse:.4f}',z.station_days] for z in risk.itertuples()]
        appendix+=table(['Station','Singular margin','Spatial MSE','Station-days'],rows,'Per-station risk diagnostic with very few station units; not an accuracy certificate.')
    appendix+='\\clearpage\n\\section{Recent evidence-table catalogue}\n'
    appendix+='All case tables, acquisition records, frozen model choices and mathematical witnesses are retained in the accompanying evidence folder. Rows are computational records, not automatically independent sampling units. Machine-readable tables provide the complete numbers without selecting only favourable outcomes.\\par\n'
    for r in REGIONS:
        appendix+='\\subsection{'+NAMES[r]+'}\n'+r'\begin{tabular}{lr}\toprule Table & Rows\\\midrule'+'\n'+'\n'.join(tex(z.table)+f' & {z.rows}'+r'\\' for z in inventory[inventory.region==r].itertuples())+'\n'+r'\bottomrule\end{tabular}'+'\n'
    technical=paper.replace('\\bibliography{references}\\end{document}', '\\clearpage\n'+appendix+'\\bibliography{references}\\end{document}')
                                                                                 
    technical=re.sub(r'\\begin\{tabular\}\{lr\}\\toprule Table & Rows\\\\\\midrule(.*?)\\bottomrule\\end\{tabular\}',lambda q:'\n'.join(line.replace(' & ',': ').replace(r'\\',r'\par') for line in q.group(1).strip().splitlines()),technical,flags=re.S)
    (SRC/'technical_reference.tex').write_text(technical,encoding='utf-8');b.compile_pdf('technical_reference',PACKAGE/'Supporting_Reference/Complete_Recent_Technical_Reference.pdf')
    checks=[]
    def check(k,v):checks.append({'check':k,'passed':bool(v)})
    check('abstract_at_most_250_words',b.words(abstract)<=250)
    counts={k:b.words(v) for k,v in synopsis.items()}
    for (k,v),(lo,hi) in zip(synopsis.items(),[(100,150),(50,100),(150,250),(100,150),(50,100)]):check('synopsis_range/'+k,lo<=b.words(v)<=hi)
    pdfinfo={}
    for sub,fn,cap in [('Upload_Files','SheafCode_Recent_Research_Paper.pdf',20_000_000),('Upload_Files','SheafCode_Recent_Research_Synopsis.pdf',5_000_000),('Supporting_Reference','Complete_Recent_Technical_Reference.pdf',None)]:
        p=PACKAGE/sub/fn;doc=fitz.open(p);full='\n'.join(q.get_text() for q in doc);pdfinfo[fn]={'pages':len(doc),'bytes':p.stat().st_size,'sha256':sha(p)}
        check(fn+'/size_cap',cap is None or p.stat().st_size<=cap);check(fn+'/no_missing_references','??' not in full);check(fn+'/no_iteration_labels',not re.search(r'\bv[1-4]\b',full,re.I));check(fn+'/no_student_paths','OneDrive' not in full and 'Users\\rishi' not in full);check(fn+'/anonymous_network_names',not re.search(r'\bDelhi\b|Los Angeles|California',full,re.I))
        for i,page in enumerate(doc):
            check(fn+f'/page_{i+1}_nonempty',len(page.get_text().strip())>20)
            for block in page.get_text('dict')['blocks']:
                if block['type']==0:
                    x0,y0,x1,y1=block['bbox'];check(fn+f'/page_{i+1}_text_inside_page',min(x0,y0)>=-1 and x1<=page.rect.width+1 and y1<=page.rect.height+1)
        for start in range(0,len(doc),12):
            contact=fitz.open();page=contact.new_page(width=900,height=1200)
            for i in range(start,min(start+12,len(doc))):
                j=i-start;page.show_pdf_page(fitz.Rect(j%3*300,j//3*300,(j%3+1)*300,(j//3+1)*300),doc,i)
            page.get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(REVIEW/f'{p.stem}_contact_{start//12+1}.png')
        doc.close()
    for name in ['research_paper','synopsis','technical_reference']:
        log=(SRC/(name+'.log')).read_text(encoding='utf-8',errors='replace');check(name+'/no_undefined_citations',not re.search(r'(Citation.*undefined|Reference.*undefined|undefined references)',log));check(name+'/no_overfull_boxes','Overfull' not in log)
    docaudit={'status':'passed' if all(c['passed'] for c in checks) else 'failed','abstract_words':b.words(abstract),'synopsis_words':counts,'pdfs':pdfinfo,'checks':checks,'scope':'Document completeness and rendering checks; not unrestricted mathematical originality or organizer eligibility'}
    (REVIEW/'DOCUMENT_CHECKS.json').write_text(json.dumps(docaudit,indent=2),encoding='utf-8')
    (PACKAGE/'README.md').write_text('''# Recent-observation IRIS research package

Use `Upload_Files/SheafCode_Recent_Research_Paper.pdf`, `SheafCode_Recent_Research_Synopsis.pdf`, `Project_Abstract.txt` and `Portal_Text.txt` for the corresponding application fields. No files have been submitted automatically.

The new empirical core uses only 2026 observations for fitting, parameter selection, calibration and testing. Delhi January–early October and the secondary Los Angeles County January–March network are separate. Native gas units, unavailable days, exploratory extensions and failures remain explicit. The older submission documents are superseded for this recent-observation core; historical research is retained separately and is not deleted.

Read `Review_Notes/RECENT_RESULTS_REPORT.md` for findings and limitations, `DOCUMENT_CHECKS.json` for PDF checks, and `WHAT_CHANGED.md` for the changes. `Supporting_Reference/Complete_Recent_Technical_Reference.pdf` contains the paper plus complete conditional mathematical statements/proofs and the table catalogue. The local reproducibility ZIP is supporting evidence, not an upload into the paper slot.

Public IRIS sources checked: https://register.irisnationalfair.org/ and https://iris.exstemplar.com/guidelines.html . Public PDF caps are 20 MB for the paper and 5 MB for synopsis; abstract target 250 words and portal cap 300. The public guideline page carries the prior event edition label, so check the signed-in current application. All recent experiment observations are within twelve months as of 7 October 2026; this does not certify personal eligibility, forms, organizer acceptance or historical supplementary-data acceptance. Supply the required video link (maximum 90 seconds) and accurate student/guide contributions. AI assistance is disclosed. Do not certify independent work that does not match your actual contributions.

Run `Run_Recent_Data_Research.cmd` only to resume/reproduce or regenerate this package. Completed stages are retained; neural epochs resume from matching checkpoints. Numerical truth is never changed to obtain a favourable result.
''',encoding='utf-8')
    (REVIEW/'WHAT_CHANGED.md').write_text('''# Changes for the recent-data submission

- Refitted every empirical model using 2026 observations, rather than adding a short new test to models trained on old data.
- Replaced the primary empirical core with five Delhi and four EPA Los Angeles County sites; removed historical observations from training, selection, calibration and reported recent test denominators.
- Retained full historical research separately. No old raw data, case tables, source code, manuscripts or valid theorem evidence were deleted or relabelled as recent.
- Repeated compatible task families: causal/retrospective outages, strong VAR and simple controls, physical PM2.5, snapshot/partial-channel tests, history/horizon/context stress, fixed-map and lag ablations, three-seed GRIN, convex hybrid, unknown-count persistent faults, no-feedback and revealed adaptive intervals, day/week calibration, actual-map rank/thinning/four-block margins, functional/bounded/joint certificates, exact nominal spans/dual/refinement, declared operator perturbations, simulated repair, alerts/daily proxies/official records, and exploratory natural-missingness estimates plus transplanted-mask validation.
- Do not claim unavailable task cells: strict complete-network fault episodes have no eligible test cases; Delhi extreme-PM coverage has no extreme cases; US alert sensitivity has no positive hours; US bootstrap intervals have too few weekly bins; GRIN is Delhi-only; no direct weather/emissions intervention is available.
- Compact paper retains the main evidence and every consequential failure. Supporting technical reference restores complete theorem proofs and a full evidence-table catalogue; machine-readable cases carry all detailed rows.
- Added new plots and source-linked numbers, preserved native gas units and actual observation dates, and corrected full/infinite interval interpretations. Data-age improvements do not by themselves establish personal eligibility or priority.
''',encoding='utf-8')
    (REVIEW/'ANONYMITY_AND_PRIOR_OVERLAP.md').write_text('''# Anonymous submission and study overlap

The public IRIS anonymity wording prohibits school/city/state disclosure. Upload PDFs, abstract and portal text therefore code the datasets as Network A (India) and Network B (USA). The reproducibility records identify Network A as Delhi and Network B as Los Angeles County. This coding changes presentation only; no observations, parameters or outcomes were changed. Local evidence archives are not mandatory upload documents and must not be uploaded wholesale into the paper field.

The preserved earlier supplementary OpenAQ acquisition shares three location IDs and 4,119 station-hour/channel observations with the new dataset; 4,116 aggregated values agree. Three boundary-hour ozone aggregates differ. Shared September–October records, and possible broader overlap with previous cohorts, mean this is a refitted exploratory temporal holdout, not a wholly new blinded validation. Training/selection/calibration/test remain disjoint within the new protocol. No external preregistration or absence of previous test access is claimed.

Official requirement sources checked 7 October 2026: https://iris.exstemplar.com/guidelines.html and https://register.irisnationalfair.org/ . The former retains a 2025–26 edition label. Personal contributions, guide details, signatures and video cannot be inferred from code. AI assistance remains disclosed; no student-only authorship attestation has been fabricated.
''',encoding='utf-8')
    if docaudit['status']!='passed':raise RuntimeError('Document audit failed: '+', '.join(c['check'] for c in checks if not c['passed']))
    return abstract,pdfinfo

def summary_report(pdfinfo):
    a=js('delhi','reconstruction_summary.json')['primary_every_third_day']['var_causal'];us=js('epa_la','reconstruction_summary.json')['primary_every_third_day']['var_causal']
    text=f'''# Recent-data final research report

Completed 7 October 2026. The empirical core is entirely 2026 observations: all training, selection, calibration and test records are recent. Historical research is preserved separately. Public observation-age requirements are addressed without pretending that old observations were newly collected. Prior supplementary observations overlap three Indian instruments (4,119 shared station-hour/channel records); this is exploratory refitting/evaluation, not an independent blinded study. Mandatory uploads code the networks A (India) and B (USA) for anonymous review.

## Main finding

Delhi causal 24-hour station-outage reconstruction has **{a['gain_percent']:.2f}% lower normalized RMSE than the tuned causal full-covariance VAR**, across **{a['cases']} matched station-days and {a['dates']} dates**; seven-day calendar-bin interval **{a['ci95'][0]:.2f}–{a['ci95'][1]:.2f}%**. All eligible Delhi test days give 16.71% gain across 242 cases. The primary physical PM2.5 gain is 10.45%, a different metric.

The independent US secondary network shows **{us['gain_percent']:.2f}%** gain across **{us['cases']} cases/{us['dates']} dates**. Only three weekly bins exist, so its interval is suppressed. This is useful secondary evidence, not a long seasonal validation.

## Countervailing findings that remain in the paper

- Shared-settings identity spatial maps beat the chosen learned CCA maps in Delhi. Temporal structure helps, but CCA superiority is not demonstrated.
- Strong injected opposite offsets can be identified accurately. Weak unequal, drift, stuck and raw gain faults expose failures; no verified field-fault truth is available.
- Delhi pooled no-feedback hourly coverage is 89.54%. Station-fixed US coverage is 75.82%, and only 17.45% on 212 training-threshold extreme cells. Delhi has no test extremes under that definition.
- Delhi day-max intervals have 98.68% hourly coverage but only 83.08% simultaneous observed-station-day coverage. Weekly-max calibration yields 100% infinite sets and no informative decisions.
- Delhi alert sensitivity improves by 16.40 percentage points over VAR, with an additional 3.56 points of false alarms. The US has no positive hourly reference cases and substantial false alarms.
- Daily PM2.5 category accuracy is tied with city averaging (73.68%, 190 common station-days), with one Poor-or-worse truth day missed by every method. The selected official bulletin comparison has only four eligible cases.
- Simulated repair intervals include zero. Naturally missing target truth remains unknown; transplanted masks are an artificial validation, not field exposure measurement.
- Every feasible real-state joint threshold certificate abstains (39 Indian and 38 US cases); three additional models are inconsistent. Lower-bound tightening is demonstrated, but definite operational threshold certification is not.
- No pollution/emissions reduction, weather forecast improvement, health improvement or real GRAP action recovery has been demonstrated.

## Coverage and verification

Compatible historical task families were rerun on recent operators; `RESULT_TABLE_INVENTORY.csv` indexes every output. Complete structured rank and one- through four-block margin enumerations replace sampled connected-set minima. Nominal support-union, functional/joint envelopes, exact dual correction, anytime refinement and declared map perturbation cases retain their assumptions. The mathematical finite suite passed **37,678 exact rational checks**, supplementing written proofs. Code tests and the final evidence audit validate dates, hashes, information flow and case aggregates. The technical reference includes the complete theorem/proof catalogue.

All three neural seeds and both attempted widths are reported, with validation-selected checkpoints and a matched retrospective comparison. No test-selected best seed is promoted. Exploratory hybrid, block-interval and mask extensions do not replace the fixed primary headline.

## Where to read

- `../Upload_Files/SheafCode_Recent_Research_Paper.pdf`: focused recent-data manuscript.
- `../Supporting_Reference/Complete_Recent_Technical_Reference.pdf`: complete conditional mathematics and case-table inventory.
- `DOCUMENT_CHECKS.json`: PDF/word-count/rendering checks.
- `../../Research_Release/recent_data_results/FINAL_EVIDENCE_AUDIT.json`: reproducibility and numerical audit.
- `WHAT_CHANGED.md`: what changed and what remains separately preserved.

The local code/data evidence supports the reported computational conclusions. It does not prove first-ever novelty, full physical calibration, personal eligibility, or competition selection. The author must understand and accurately acknowledge assisted work.
'''
    (REVIEW/'RECENT_RESULTS_REPORT.md').write_text(text,encoding='utf-8');(OUT/'RECENT_RESULTS_REPORT.md').write_text(text,encoding='utf-8')
                                                                                
    report=r'''\documentclass[11pt,a4paper]{article}\usepackage[margin=1in]{geometry}\usepackage{hyperref}\usepackage{microtype}\setlength{\parindent}{0pt}\setlength{\parskip}{.5em}\begin{document}'''
    for line in anonymous(text).splitlines():
        if line.startswith('# '):report+='\\section*{'+tex(line[2:])+'}\n'
        elif line.startswith('## '):report+='\\subsection*{'+tex(line[3:])+'}\n'
        elif line.startswith('- '):report+='\\par '+tex(line[2:].replace('`',''))+'\n'
        elif line:report+='\\par '+tex(line.replace('**','').replace('`',''))+'\n'
    report+='\\end{document}';(SRC/'results_report.tex').write_text(report.replace('–','--'),encoding='utf-8')
    spec=importlib.util.spec_from_file_location('report_builder',ROOT.parent/'audit_tools/build_iris_submission.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b);b.OUT=PACKAGE;b.SRC=SRC;b.compile_pdf('results_report',REVIEW/'RECENT_RESULTS_REPORT.pdf')

def main():
    if js('', 'neural_progress.json').get('status')!='completed':raise RuntimeError('Do not generate final report before all neural seeds finish')
    for p in [FIG,REVIEW,PACKAGE/'Upload_Files',PACKAGE/'Supporting_Reference']:p.mkdir(parents=True,exist_ok=True)
    plots();evidence_tables();abstract,pdfinfo=documents();summary_report(pdfinfo)
    save('report_status.json',{'status':'completed','package':str(PACKAGE),'figures':len(list(FIG.glob('*.pdf'))),'pdfs':pdfinfo})
    print('Recent research figures, full proofs, submission documents and report completed',flush=True)
if __name__=='__main__':main()
