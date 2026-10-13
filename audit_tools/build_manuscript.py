from pathlib import Path
import json, shutil, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parents[1]
SRC=BASE/'evidence/sheafcode_iris_2026_final/sheafcode_iris_2026_final'
TABLES=BASE/'evidence/sheafcode_iris_2026_final_result_tables/sheafcode_iris_2026_final/results'
OUT=BASE/'Final_Manuscript'
for name in ['sections','figures','data','preview']:(OUT/name).mkdir(exist_ok=True)
for name in ['preamble.tex','main.tex','references.bib','synopsis.tex','portal_fields.tex','video_script.tex']:
    shutil.copy2(BASE/'manuscript'/name,OUT/name)
FR=json.loads((SRC/'results/FINAL_RESULTS.json').read_text())
AU=json.loads((OUT/'audit/numerical_audit.json').read_text())
REGS=list(FR['regions']); LABELS={k:AU['results'][k]['label'] for k in REGS}
SHORT=[LABELS[k].replace('Raw replication ','Raw ') for k in REGS]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220,'font.family':'DejaVu Sans'})
COLORS=['#176b7a','#ce8542','#73798b','#9b536a']

def write(name,text): (OUT/name).write_text(text.strip()+'\n',encoding='utf8')
def pf(x):
    if x is None:return '--'
    if x<.001:return r'$<0.001$'
    return f'{x:.3f}'
def table(headers,rows,caption,label,widths=None):
    cols=widths or ('l'+'r'*(len(headers)-1))
    headers=[h.replace('\\\\','\\') for h in headers]
    caption=caption.replace('\\\\','\\')
    return '\n'.join([r'\begin{table}[H]\centering\small',r'\caption{'+caption+r'}\label{'+label+'}',r'\begin{tabular}{'+cols+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule',*[' & '.join(map(str,r))+r' \\' for r in rows],r'\bottomrule',r'\end{tabular}',r'\end{table}'])
def fig(name,caption,label,width='.98'):
    return r'\begin{figure}[H]\centering'+'\n'+r'\includegraphics[width='+width+r'\linewidth]{figures/'+name+r'.pdf}'+'\n'+r'\caption{'+caption+r'}\label{'+label+'}\n'+r'\end{figure}'
def save(fig_,name):
    fig_.savefig(OUT/f'figures/{name}.pdf',bbox_inches='tight')
    fig_.savefig(OUT/f'figures/{name}.png',bbox_inches='tight')
    plt.close(fig_)
def comparison(name,key,methods,legends):
    fig_,ax=plt.subplots(figsize=(10,5.6));y=np.arange(9)
    rows=[]
    for i,(method,legend) in enumerate(zip(methods,legends)):
        vals=[AU['results'][reg][key][method]['mean'] for reg in REGS]
        ci=np.array([AU['results'][reg][key][method]['ci'] for reg in REGS])
        dy=(i-(len(methods)-1)/2)*.17
        ax.errorbar(vals,y+dy,xerr=np.maximum(0,np.array([np.array(vals)-ci[:,0],ci[:,1]-np.array(vals)])),fmt='o',ms=4.8,capsize=2,color=COLORS[i],label=legend)
        for reg,v,(lo,hi) in zip(REGS,vals,ci):rows.append(dict(network=LABELS[reg],method=legend,mean=v,ci_low=lo,ci_high=hi,n=AU['results'][reg][key][method]['n']))
    ax.set_yticks(y,SHORT);ax.invert_yaxis();ax.set_xlim(left=0);ax.set_xlabel('Mean standardized RMSE (lower is better)');ax.grid(axis='x',alpha=.2)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.13),ncol=2,frameon=False)
    fig_.tight_layout();save(fig_,name);pd.DataFrame(rows).to_csv(OUT/f'data/{name}.csv',index=False)
comparison('snapshot','single_snapshot',['vector_sheaf','gaussian_cond','gnn'],['Spatial sheaf','Gaussian conditional','Masked GCN'])
comparison('channel','channel',['vector_sheaf','gaussian_cond'],['Spatial sheaf','Gaussian conditional'])
comparison('prolonged','prolonged',['st_sheaf','kalman_var','hybrid','gnn'],['Space-time sheaf','Kalman/VAR smoother','Persistence + spatial','Masked GCN'])

certrows=[]
fig_,ax=plt.subplots(figsize=(10,5.2));regs=[r for r in REGS if AU['results'][r]['decile']];y=np.arange(len(regs))
for i,kind in enumerate(['safe','risky']):
    vals=[AU['results'][r]['decile']['rmse_'+kind] for r in regs];ci=np.array([AU['results'][r]['decile'][kind+'_ci'] for r in regs])
    ax.errorbar(vals,y+(i-.5)*.22,xerr=np.maximum(0,np.array([np.array(vals)-ci[:,0],ci[:,1]-np.array(vals)])),fmt='o',capsize=3,color=COLORS[i],label=['Lowest C2* decile','Highest C2* decile'][i])
    for r,v,(lo,hi) in zip(regs,vals,ci):certrows.append(dict(network=LABELS[r],group=kind,mean=v,ci_low=lo,ci_high=hi,n=AU['results'][r]['decile']['n_'+kind]))
ax.set_yticks(y,[LABELS[r].replace('Raw replication ','Raw ') for r in regs]);ax.invert_yaxis();ax.set_xlim(left=0);ax.set_xlabel('Mean standardized RMSE (lower is better)');ax.grid(axis='x',alpha=.2);ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.13),ncol=2);fig_.tight_layout();save(fig_,'certificates');pd.DataFrame(certrows).to_csv(OUT/'data/certificate_deciles.csv',index=False)

fig_,ax=plt.subplots(figsize=(8.5,4.4));thinrows=[]
for i,reg in enumerate(['delhi','kolkata','india']):
    df=pd.read_csv(TABLES/f'v3_{reg}'/'thinning.csv');g=df.groupby('frac_edges').frac_identifiable_all.agg(['mean','min','max'])
    ax.plot(100*g.index,100*g['mean'],marker=['o','s','^'][i],color=COLORS[i],label=LABELS[reg]);ax.fill_between(100*g.index,100*g['min'],100*g['max'],color=COLORS[i],alpha=.10)
    for fraction,row in g.iterrows():thinrows.append(dict(network=LABELS[reg],edge_retention=fraction,identifiable_mean=row['mean'],identifiable_min=row['min'],identifiable_max=row['max']))
ax.set_xlabel('Retained spatial edges (%)');ax.set_ylabel('Artificial outage patterns passing C1 (%)');ax.set_ylim(0,104);ax.set_xlim(10,102);ax.legend(frameon=False);ax.grid(alpha=.2);fig_.tight_layout();save(fig_,'thinning');pd.DataFrame(thinrows).to_csv(OUT/'data/thinning.csv',index=False)

horizons=['1-3','4-6','7-12','13-24','25-48','49-72'];fig_,axes=plt.subplots(3,3,figsize=(11,8.5),sharex=True);hrows=[]
for ax,reg in zip(axes.flat,REGS):
    R=FR['regions'][reg]
    for i,m in enumerate(['st_sheaf','kalman_var','hybrid']):
        vals=[R['prolonged_by_horizon'][m].get(h,np.nan) for h in horizons]
        ax.plot(range(6),vals,marker=['o','s','^'][i],ms=3,color=COLORS[i],label=['Space-time sheaf','Kalman/VAR','Hybrid'][i])
        for h,v in zip(horizons,vals):hrows.append(dict(network=LABELS[reg],method=m,hours=h,pooled_rmse=v))
    ax.set_title(LABELS[reg].replace('Raw replication ','Raw '),fontsize=10);ax.set_ylim(bottom=0);ax.grid(alpha=.15);ax.set_xticks(range(6),horizons,rotation=35,fontsize=8)
for ax in axes[:,0]:ax.set_ylabel('Pooled standardized RMSE')
fig_.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False);fig_.supxlabel('Hours after outage (bins)');fig_.tight_layout(rect=[0,.02,1,.96]);save(fig_,'horizons');pd.DataFrame(hrows).to_csv(OUT/'data/horizons.csv',index=False)

fig_,axes=plt.subplots(1,2,figsize=(10,4.3));faultrows=[]
for i,reg in enumerate(['delhi','kolkata','india']):
    curve=FR['studies'][reg]['faults']['severity_curve_syndrome'];sev=sorted(float(k) for k in curve if float(k)>0)
    vals=[curve[str(s)]['top1'] for s in sev]
    axes[0].plot(sev,vals,marker=['o','s','^'][i],color=COLORS[i],label=LABELS[reg]);
    for s,v in zip(sev,vals):faultrows.append(dict(network=LABELS[reg],severity=s,top1=v))
axes[0].set_xlabel('Synthetic corruption severity (model SD)');axes[0].set_ylabel('Syndrome top-1 localization accuracy');axes[0].set_ylim(0,1);axes[0].legend(frameon=False);axes[0].grid(alpha=.2)
graph=FR['studies']['india']['structure']['A_graph_study'];gt=pd.DataFrame(graph['table']);chosen=gt[gt.variant.isin(['geo_knn_k8','corr_knn_k8','union_geo_corr_k8','geo_knn_k8_dmax25km'])]
axes[1].bar(range(len(chosen)),chosen.test_rmse,color=[COLORS[0],COLORS[2],COLORS[1],COLORS[3]])
axes[1].set_xticks(range(len(chosen)),['Frozen geo','Correlation','Geo + corr\n(validation choice)','25 km cap'],rotation=18,fontsize=8);axes[1].set_ylabel('Mean single-station RMSE');axes[1].set_ylim(0,.85);axes[1].grid(axis='y',alpha=.2);fig_.tight_layout();save(fig_,'secondary');pd.DataFrame(faultrows).to_csv(OUT/'data/fault_severity.csv',index=False);gt.to_csv(OUT/'data/national_graph_study.csv',index=False)

def dataset_table():
    rows=[]
    for reg,R in FR['regions'].items():
        rows.append([LABELS[reg].replace('Raw replication ','Raw '),R['n_stations'],len(R['pollutants']),f"{100*R['obs_fraction']:.1f}",R['n_rows'],R['test_span'][0][:10],R['test_span'][1][:10]])
    return table(['Network','$n$','$p$','Obs. (\\%)','Rows','Test start','Test end'],rows,'Selected networks and chronological test spans. Rows are retained timestamps, not necessarily consecutive hours. All networks use PM$_{2.5}$, PM$_{10}$, NO$_2$, and O$_3$, except Region F, which omits PM$_{10}$.','tab:data')

write('sections/04_data.tex',r'''
\section{Data and experimental protocol}
\subsection{Selected networks and provenance}
Public regulatory air-quality records support six regional networks and a 177-station national network drawn from a historical sub-index archive, together with two raw-concentration datasets. The selected raw datasets contain seven stations in 2025--2026 and four stations in 2019--2025. They provide checks using different collection paths and later test periods; they are not statistically independent replications of the whole study. Regional networks overlap the national network, Region A overlaps Region B, and some raw-data stations and historical periods overlap the main archive.

The historical values are interpreted as pollutant sub-indices and inverted using the documented piecewise-linear breakpoint implementation; values at the 500 cap are removed as censored. The supplied provenance record reports agreement with overlapping raw exports after a one-hour timestamp adjustment. An index rounded to an integer cannot yield an exactly known original concentration. Extreme capped observations are excluded, so the study does not establish performance on those events. Records were accessed through public repository mirrors; direct confirmation against the originating regulator was not supplied for every input. Scientific provenance therefore remains distinct from internal numerical consistency \citep{cpcbresources,rabharmirror}.
'''+'\n'+dataset_table()+r'''
\subsection{Coverage selection and chronological split}
The code selects stations and pollutants using coverage over the available archive, retaining stations with at least 50\% coverage per chosen pollutant over their active span and timestamps with at least 70\% observed selected cells. It maximizes the station--pollutant product, subject to at least four stations and 3,000 retained rows. These rules inspect full-archive availability \emph{before} the split, including future test-period missingness. This is a retrospective cohort-selection limitation even though hidden target values are excluded during reconstruction.

Retained timestamps are split chronologically by row count into 60\% training, 20\% validation, and 20\% test. Means, standard deviations, CCA maps, covariance priors, and model coefficients are fitted on training data. Spatial hyperparameters are selected on validation reconstruction error in the common evaluation space. The temporal weight and hybrid coefficients are selected on validation prolonged windows in the sheaf's own model space. Kalman/VAR ridge and noise parameters are fixed implementation defaults, rather than independently searched validation hyperparameters. Large networks use reduced validation grids and sampled test stations, documented in their manifests.

\subsection{Erasure order and observed-target evaluation}
For every artificial outage, the target is hidden before it can be used to reconstruct natural gaps. Methods receive the remaining measurements and their missingness masks. The sheaf jointly solves all unknown cells. Gaussian conditional prediction also conditions on the surviving coordinates; other baselines use their own masked estimation procedures. Only target cells observed before erasure are scored. Synthetic tests check invariance of several implemented predictions to changes in hidden truth; they do not certify every benchmark path or every preprocessing decision.

Snapshot scores first compute RMSE over observed target cells within a sampled day, then average days within a pattern and patterns within a network. The primary snapshot comparison uses \emph{single-station} patterns. Mixed-size and clustered patterns are secondary summaries. Single-channel scores average days within each station--pollutant target, then targets. Methods with insufficient finite paired units are omitted from inference, not from descriptive reporting.

\subsection{Common metric and statistical scope}
Predictions are inverted to raw concentration units and standardized per station and pollutant using the untransformed training mean and standard deviation. All headline reconstruction scores are evaluated in this common space. A standardized RMSE of one is an error at the scale of training variability, not an established regulatory acceptability threshold.

Prolonged-outage scores take the square root of mean hourly squared error within each run--station failure and average those unit RMSEs. Primary comparisons use one-sided paired Wilcoxon tests for the hypothesis that the competitor has larger error, with Holm correction across up to four competitors \emph{within each network and regime}. This does not correct all comparisons across the study. The reported tests assume suitably independent paired units; overlapping station sets, shared periods, and failures within one run can violate that assumption. A supplementary sensitivity analysis averages paired differences within each run before testing and resamples runs for its interval. It changes both the replicate and weighting and is not a replacement prespecified primary analysis.

Figure error bars are 95\% percentile bootstrap intervals from 10,000 resamples of the displayed pattern, target, or run--station units (fixed seed 927). They describe variation across those sampled units, not full model, temporal, or sensor-calibration uncertainty. Prospective decile tests use one-sided Mann--Whitney comparisons; nominal and supplementary Holm-adjusted values across the seven evaluable networks are distinguished. No significance claims are assigned to exploratory ablations, graph studies, or fault sweeps.
''')

                                                                                          
model=(BASE/'manuscript/sections/05_model.tex').read_text()
model=model.replace('where the coordinates are standardized pollutant concentrations.','where the coordinates are in the selected training-standardized model space: raw concentration or a log/diurnal transform. Headline errors are mapped back to the common raw-concentration standardized space.')
model=model.replace('Each canonical direction receives a positive weight, collected in $W_e$.','The implemented CCA uses covariance ridge regularization $10^{-3}$; canonical correlations are clipped to $[10^{-3},1-10^{-3}]$ when forming weights. An edge with fewer than 200 fully observed training pairs uses fallback identity directions and fixed correlation 0.3. Thus exact population CCA normalizations above describe the ideal construction. Each retained direction receives a positive weight, collected in $W_e$.')
model=model.replace("$P_v$ is the inverse training covariance of station $v$ when that option is selected.",r"$P_v=(\Sigma_v+10^{-3}I)^{-1}$ is the ridge-stabilized training precision of station $v$, estimated from fully observed station rows. A station with insufficient complete rows uses the identity prior.")
model=model.replace('Consider one edge $(u,v)$, retain all $p$ CCA directions,','In the ideal, centered, unregularized-covariance CCA construction, consider one edge $(u,v)$, retain all $p$ CCA directions,')
model=model.replace('the multivariate linear-regression predictor of $x_v$ from $x_u$.','the multivariate linear-regression predictor of $x_v$ from $x_u$. Zero-correlation directions contribute zero; clipping, rank truncation, covariance ridge, identity stalks, and the selected regularization generally break this exact identity in the fitted system.')
model=model.replace('which is the Gaussian conditional expectation of the missing coordinates given the observed coordinates.','which is the zero-mean Gaussian conditional expectation under that precision model. For a nonzero mean the corresponding centered formula must be used. The fitted ridge-stabilized prior gives a regularized analogue.')
model=model.replace(r'\begin{tikzpicture}[',r'\resizebox{.98\linewidth}{!}{\begin{tikzpicture}[').replace(r'\end{tikzpicture}',r'\end{tikzpicture}}')
model=model.replace(r'\begin{keybox}{Interpretability through limiting cases}',r'\paragraph{Interpretability through limiting cases.}').replace(r'\end{keybox}','')
write('sections/05_model.tex',model)

write('sections/06_theory.tex',r'''
\section{Reconstruction and reliability theory}
All statements in this section concern a fixed fitted linear operator in its model coordinates. They do not by themselves guarantee accuracy after nonlinear back-transformation or under distribution shift. Use the unknown set $U$ from \cref{eq:decoder}, including natural gaps when present.

\subsection{C1: structural identifiability}
For $\lambda=0$, the least-squares completion is unique for every observed state precisely when
\begin{equation}
\operatorname{rank}(\delta_U)=|U|\iff\ker\delta_U=\{0\}.
\label{eq:c1}
\end{equation}
Positive-definite regularization ensures a unique numerical solution when C1 fails, but does not establish that the surviving observations contain the missing information. All fitted dense spatial operators have a trivial exact kernel at the recorded numerical tolerance. Every column subset is consequently independent at the exact-algebra level; C1 is nondiscriminative in those operators and is studied empirically after controlled thinning.

\subsection{C2 and \texorpdfstring{C2$^*$}{C2*}: residual sensitivity}
For full-column-rank $\delta_U$, the unregularized residual-to-error amplification is
\begin{equation} C2=1/\sigma_{\min}(\delta_U).\end{equation}
The corresponding norm in the regularized error identity is
\begin{equation}
C2^*=\|G^{-1}\delta_U^\top\|_2,
\qquad G=\delta_U^\top\delta_U+\lambda P_{UU}.
\label{eq:c2star}
\end{equation}
This is a model-coordinate residual sensitivity, not the sensitivity to noise in observed measurements. The latter is the norm of $G^{-1}(\delta_U^\top\delta_O+\lambda P_{UO})$. Regularizer bias and model mismatch are separate contributors to error. Adding constraints need not monotonically improve $C2^*$ or actual reconstruction accuracy.

For the reported prospective ranking, certificates use the \emph{artificial station outage alone}, with other stations treated as structurally available. The actual hourly decoder also includes natural gaps. Its unknown set and sensitivity can therefore differ from the static pattern certificate. The ranking is an empirical diagnostic of artificial-outage geometry, rather than an exact certificate of each incomplete test snapshot.

\subsection{General-\texorpdfstring{$P$}{P} deterministic bound}
With true state $x$, residual $r=\delta x$, and error $e=\hat x_U-x_U$, substitution into the normal equations gives
\begin{equation}
e=-G^{-1}\left[\delta_U^\top r+\lambda(Px)_U\right].
\end{equation}
The triangle inequality and operator norm yield
\begin{equation}
\|e\|_2\le C2^*\|r\|_2+\lambda\|G^{-1}\|_2\|(Px)_U\|_2.
\label{eq:generalbound}
\end{equation}
The bound requires unknown truth to evaluate the residual and prior-bias term. Numerical checks on sampled complete test states are therefore post hoc consistency checks, not operational error bars.

\subsection{C3: typical error under additional assumptions}
For a fixed operator and outage, complete training snapshots give reconstruction errors $e_t$ and the historical estimate
\begin{equation}
C3_{\mathrm{complete}}(U)=\sqrt{\frac{1}{T|U|}\sum_{t=1}^T\|e_t\|_2^2}.
\end{equation}
For a fixed decoder, a stationary and ergodic error process with finite second moment satisfies a law of large numbers for mean squared error. Under train/test distributional stability the estimate then targets the corresponding population RMSE. Fitting the decoder on the same training rows and selecting complete cases introduce additional estimation and selection questions; this fixed-decoder statement is not a proof of consistency for the entire fitted pipeline. Complete-case stability need not extend to naturally incomplete states.

The implementation reports this quantity only with at least 200 complete training rows; otherwise it separately computes $C3_{\mathrm{empirical}}$ from simulated erasures with natural gaps. Both recorded C3 quantities are in the sheaf's model coordinates, whereas the primary RMSE is in the common raw-concentration standardized space. For log/diurnal models they cannot be treated as directly calibrated predictions of the plotted RMSE. The empirical proxy is outside the complete-state statement.

\subsection{Structural resilience and the coding analogy}
For station blocks $S$, define
\begin{equation}
d_{\mathrm{block}}=\min\{|S|:\ker\delta_S\ne\{0\}\},
\end{equation}
with value $+\infty$ when no such subset exists. In the ideal model $\ker\delta$ is a block linear code. Since the fitted dense operators have trivial kernels, their exact-code interpretation contains only the zero state; the useful decoder is the soft regularized inverse problem. Connected-set enumeration and heuristic high-$C2^*$ searches are exploratory conditioning studies, not demonstrated finite code distances or guaranteed worst-case error predictions.
''')

write('sections/07_spacetime.tex',r'''
\section{Space-time reconstruction and observation timing}
The model places a pollutant vector at each station--time vertex. Spatial restrictions repeat at each time step; temporal CCA restrictions connect each station to itself at retained-row offsets 1 and 24. The stacked operator is
\begin{equation}
\delta_{ST}=\begin{bmatrix}I_W\otimes\delta_S\\T_1\\T_{24}\end{bmatrix}.
\end{equation}
Training timestamps have been filtered by coverage, and the temporal fitting code slices the retained training rows rather than checking elapsed time. These offsets are consequently \emph{not always} one hour and twenty-four hours during fitting. In Raw replication 2, 14,188 of 29,017 eligible lag-24 training pairs do not span exactly 24 hours. Test prolonged windows are contiguous hourly runs. The lag-24 ablation establishes the usefulness of the fitted offset relation on these test windows; it does not cleanly isolate a daily-periodicity mechanism. Elapsed-time-matched fitting would require a separate experiment.

For each eligible test run, the failed station has 24 fully observed pre-outage hours and is hidden for an available horizon of 24--72 hours. Other stations retain their natural missingness. Windows are selected only when the target station is fully observed throughout the required history and scoring interval, enabling artificial erasure of known truth. This evaluates recoverable artificial outages, rather than the distribution of real failures. The first 24 outage hours form the primary metric; the available full horizon and later bins are also reported. Short runs contribute fewer late-hour observations.

The space-time sheaf solves the full window jointly, and the Kalman/VAR method uses Rauch--Tung--Striebel smoothing \citep{rauch1965smoothing,lutkepohl2005var}. An early missing reading can therefore use later observations of \emph{surviving stations} inside the window. Hidden values of the failed station remain excluded, but the experiment is retrospective gap reconstruction. It does not establish causal forecasting or real-time performance. Static, persistence, hybrid, and masked-GCN comparators use a smaller temporal information set, so temporal comparisons also reflect different access to history and future surviving observations.
''')

write('sections/08_baselines.tex',r'''
\section{Baselines and comparison fairness}
\textbf{Snapshot reconstruction.} Comparators include a shrinkage-stabilized dense Gaussian conditional model, ordinary kriging, inverse-distance weighting, low-rank completion, scalar per-pollutant sheaves, a scalar graph Laplacian, and a masked graph-convolutional imputer. The low-rank method fits an iterative rank-truncated SVD and projects onto observed coordinates; it is related to completion methods but is not an exact implementation of the nuclear-norm Soft-Impute algorithm \citep{mazumder2010softimpute}. The graph-convolutional architecture is related to established GCN work \citep{kipf2017gcn}.

\textbf{Single-channel reconstruction.} The vector sheaf, Gaussian conditional, low-rank, and masked GCN paths hide the designated channel while allowing surviving same-station channels. The scalar sheaf and kriging code paths hide the whole station before scoring the target pollutant. Those baseline paths therefore do not match the vector model's information set for channel outages. Their results cannot isolate the benefit of pollutant coupling alone. The Gaussian conditional is the central matched-information comparator; the prior/PM$_{10}$/whole-station ablations are mechanism probes with their own restrictions.

\textbf{Prolonged reconstruction.} The state-space baseline fits a dense ridge VAR(1) on low-rank-filled training data, uses diagonal process noise, fixed observation noise, and full-window Kalman smoothing. It has no explicit lag-24 dynamics and its ridge/noise defaults are not separately tuned. Comparators also include last-value persistence, repeated values from the last observed day, a validation-fitted persistence-plus-spatial hybrid, the masked GCN, and the spatial-only sheaf. The GCN is static and is not a recurrent graph imputer such as GRIN \citep{cini2022grin}. The observed advantage is against these implemented baselines, not against every time-series model.

\textbf{Fault localization.} Synthetic corruption severity is varied and the sheaf syndrome is compared with leave-one-out residual, neighbour disagreement, and reading magnitude. Localization accuracy identifies an injected faulty station conditional on a fault; it is distinct from a deployment detector's false-positive rate or calibration.
''')

snaprows=[];prows=[];crows=[];decrows=[]
for reg,R in FR['regions'].items():
    A=AU['results'][reg];sg=A['single_snapshot'];pv=R['P1_singles'].get('gaussian_cond',{})
    imp=100*(1-sg['vector_sheaf']['mean']/sg['gaussian_cond']['mean'])
    snaprows.append([LABELS[reg].replace('Raw replication ','Raw '),sg['vector_sheaf']['n'],f"{sg['vector_sheaf']['mean']:.3f}",f"{sg['gaussian_cond']['mean']:.3f}",f'{imp:.1f}',pf(pv.get('p_holm'))])
    pp=A['prolonged'];pv=R['P3_1_24h']['kalman_var'];cv=A['cluster_sensitivity']['kalman_var']
    prows.append([LABELS[reg].replace('Raw replication ','Raw '),A['n_units'],A['n_runs'],f"{pp['st_sheaf']['mean']:.3f}",f"{pp['kalman_var']['mean']:.3f}",f"{100*pv['rel_improvement']:.1f}",pf(pv['p_holm']),pf(cv['p_holm'])])
    d=A['decile']
    if d:decrows.append([LABELS[reg].replace('Raw replication ','Raw '),f"{d['n_safe']}/{d['n_risky']}",f"{d['rmse_safe']:.3f}",f"{d['rmse_risky']:.3f}",f"{d['ratio']:.2f}",pf(d['p']),pf(d['p_holm_across_networks'])])
    cc=A['channel']; cp=R['P2'].get('gaussian_cond',{})
    crows.append([LABELS[reg].replace('Raw replication ','Raw '),cc['vector_sheaf']['n'],f"{cc['vector_sheaf']['mean']:.3f}",f"{cc['gaussian_cond']['mean']:.3f}",f"{100*(1-cc['vector_sheaf']['mean']/cc['gaussian_cond']['mean']):.1f}",pf(cp.get('p_holm'))])

write('sections/09_results.tex',r'''
\section{Results}
\subsection{Primary single-station snapshots and secondary mixed outages}
For the prespecified single-station snapshot comparison, the spatial sheaf has lower mean error than the dense Gaussian conditional in seven of eight networks with enough paired units for inference. Only Region A and Region D pass the within-network Holm correction. The median relative improvement across those eight comparisons is 3.1\%. Raw replication 2 has four single-station targets, too few for the implementation's five-unit inference threshold; its descriptive means are included. Across all nine descriptive network means, the sheaf is lower in eight. This is evidence of competitiveness, rather than universal or large superiority.
'''+table(['Network','Targets','Sheaf','Gaussian','Gain (\\%)','$p_{\\mathrm{Holm}}$'],snaprows,'Primary single-station snapshots. Gain is $100(1-\\mathrm{RMSE}_{sheaf}/\\mathrm{RMSE}_{Gaussian})$; negative values favor Gaussian. Dashes indicate insufficient paired units for the recorded test. Correction covers the primary competitors within one network.','tab:snapshot')+fig('snapshot','Single-station snapshot reconstruction. Points are means of station-outage pattern scores; bars are pattern-bootstrap 95\% intervals. Raw replication 2 is descriptive. This figure does not combine outage sizes.','fig:snapshot')+r'''
The separate mixed-size summary averages the available single-station, random multi-station, and clustered patterns. That summary yields six improvements among eight testable networks and a median 2.3\% advantage over Gaussian; it is not the primary single-station result. Its sampling weights depend on the enumerated outage sizes and available patterns. National mixed-pattern means are sheaf 1.119 and Gaussian 1.147, with the lowest tested method mean 1.118. These are averages of pattern RMSEs, not a single pooled cell RMSE. They indicate substantial errors under this sampling design but do not alone establish an operational pass/fail threshold.

\subsection{Prospective risk groups and limitations of the certificate}
Before the main test-error loop, the code writes the frozen model's $C2^*$ ordering and top/bottom decile memberships. All seven networks with at least two scored patterns in each group show higher mean error in the high-$C2^*$ group. Six nominal one-sided Mann--Whitney tests have $p<0.05$; five remain below 0.05 after a supplementary Holm correction across the seven network-level decile tests. Small samples and shared station/time information limit inference.
'''+table(['Network','Low/high $n$','Low','High','Ratio','Nominal $p$','Across-network $p_H$'],decrows,'Prospectively assigned artificial-outage risk groups. Low/high refer to $C2^*$ deciles, not groups selected using test error. The ratio is high-risk RMSE divided by low-risk RMSE.','tab:cert')+fig('certificates','Test errors for training-model-designated risk groups. Bars resample scored outage patterns within each group. Networks with fewer than two patterns in either decile are omitted. Group membership may differ in outage size and geometry, and these comparisons do not demonstrate size-independent risk prediction.','fig:cert')+r'''
Spearman correlations between $C2^*$ and mixed-pattern error are heterogeneous: approximately 0.50 in Region A, 0.81 in Region E, and $-0.10$ in Region D. Outage size, station composition, regularizer bias, natural gaps, and distribution shift can all influence the result. Certificates calculated for an artificial outage alone are not the complete missing-set certificate at every hour.

The deterministic general-$P$ bound has zero recorded violations in 1,064 sampled complete-state checks across six networks. The checking code uses at most the first 300 complete test rows, every tenth selected row, and up to 40 target stations. It does not check every available complete state, and zero finite numerical violations does not provide a deployable uncertainty guarantee.

\subsection{Single-channel outages and matched-information comparisons}
Compared with the dense Gaussian conditional, the spatial sheaf improves mean single-channel error in six of nine networks and is worse in three. Gains range from 5.6\% to 11.1\% among the improving networks. This matched-information comparison is more informative about the vector model than scalar/kriging paths that additionally hide other same-station channels.
'''+table(['Network','Targets','Sheaf','Gaussian','Gain (\\%)','$p_{\\mathrm{Holm}}$'],crows,'Single-channel target means. Gaussian and vector sheaf may both use surviving channels at the target station. Within-network corrected values retain the original primary comparison family; station/pollutant dependence remains a limitation.','tab:channel')+fig('channel','Single-channel reconstruction against the matched-information Gaussian comparator. Error bars resample station--pollutant target scores; targets within one station are not fully independent.','fig:channel')+r'''
In Region A, replacing the fitted precision prior with identity increases mean error by 0.005; on eligible non-PM$_{10}$ targets, additionally hiding PM$_{10}$ increases error by 0.016; hiding the whole station increases it by 0.028. Region B gives approximately 0.008, 0.020, and 0.036. These paired ablations do not all share the same target set, so the increases are not additive contributions. The raw replications show heterogeneous effects, and surviving pollutant information is not uniformly beneficial.

\subsection{Prolonged retrospective reconstruction}
With 24 hours of target-station history, first-24-hour mean error is lower for the space-time sheaf than for the implemented Kalman/VAR smoother in all nine networks. The median relative improvement is 34.0\%, computed as the median of nine network-level relative gains. The original run--station analysis passes within-network Holm correction in all nine; treating each run as the replicate instead yields five corrected positive comparisons. The latter sensitivity analysis is exploratory and the number of runs is as low as three. Consequently the manuscript distinguishes the consistent numerical advantage from a broad claim of nine independent statistically confirmed replications.
'''+table(['Network','Units','Runs','ST sheaf','Kalman','Gain (\\%)','Unit $p_H$','Run $p_H$'],prows,'First-24-hour prolonged-outage means and inference. Unit tests use run--station failures; the supplementary run test averages paired differences within each run. Both corrections cover the implemented primary competitors within a network. Shared runs and overlapping networks limit independence.','tab:prolonged')+fig('prolonged','First-24-hour mean unit RMSE. Bars bootstrap run--station units, so they do not account fully for dependence among failures sharing a run. The sheaf and Kalman methods can use later surviving-station observations inside each full window.','fig:prolonged')+r'''
The original unit analysis also favors the space-time sheaf over the static masked GCN in eight of nine networks and the persistence-plus-spatial hybrid in seven of nine after the same within-network correction. Region C does not significantly favor the sheaf over the hybrid, and Region D has a numerically lower hybrid error (0.284 versus 0.300). Baseline information sets and lag structure differ, as described in the methods.

The raw-dataset first-day means are 0.451 versus 0.722 for Kalman in Raw replication 1 and 0.349 versus 0.529 in Raw replication 2. They extend the descriptive result to raw concentration inputs and later periods, but they do not constitute external independent reproduction. The source input for Raw replication 1 was not present in the supplied archives; its saved error tables can be audited, while complete source-data reproduction remains unavailable.
'''+fig('horizons','Errors by elapsed-outage bin through 72 hours. Each point is the square root of pooled hourly squared errors available in that bin, rather than the mean run--station RMSE used in the primary table. Shorter outages do not contribute to later bins; changes across bins therefore mix horizon effects and sample composition.','fig:horizons')+r'''
Exploratory lag ablations use the common subset of run--station units for which every variant exists, over the available full horizon. For Region F, the fitted $\{1,24\}$ model has RMSE 0.526, offset 1 alone 0.643, and offset 24 alone 0.523. Raw replication 2 gives 0.426, 0.580, and 0.428. These differ from the primary first-day values because both horizon and unit set differ. They support use of the fitted offset-24 relation; irregular training timestamps prevent a clean claim that daily periodicity alone explains the gain.

\subsection{National graph construction: limited improvement}
The exploratory national graph study scores 20 selected stations over 12 test days, with a corresponding validation subset. Among connected constructions, mean single-station test error ranges from 0.679 to 0.711. The validation-selected geographic--correlation union at $k=8$ gives 0.679, compared with 0.698 for the frozen geographic graph on this same subset: about 2.8\% lower. This subset is distinct from the main national snapshot experiment and cannot be compared directly with its mixed-pattern mean 1.119.

Aggressive edge-distance caps and correlation filters often fragment the network. Their tested single-station C1 fractions can fall, although fragmentation alone is not equivalent to sheaf rank failure. Diagnostic shifts and unusual channels are plausible contributors to national error, but this study does not causally establish their dominance or rule out every alternative topology. The supported conclusion is that the tested reasonable connected graph alternatives provide only modest improvements on this particular subset.

\subsection{C1 under controlled graph thinning}
Randomly removing spatial edges exposes structural non-identifiability. In Region A, retaining 12\%, 20\%, 30\%, and 45\% of edges yields approximately 43\%, 69\%, 86\%, and 100\% identifiable artificial patterns; Region D gives 30\%, 54\%, 83\%, and 100\%. The national network requires more retention to reach complete identifiability. These are averages across three thinning seeds and the selected pattern sets, not probabilities that every possible network outage is identifiable. Non-identifiable groups often have larger errors, with exceptions.
'''+fig('thinning','Identifiable fraction of artificial patterns after edge thinning. Lines average three seeds; shaded bands give the seed range, not confidence intervals. C1 concerns the thinned artificial-outage operator rather than every hourly natural-gap mask.','fig:thinning')+r'''
\subsection{Secondary fault and network-design studies}
Synthetic fault localization improves with larger corruption in several networks. In Region A, syndrome top-1 accuracy is 0.045 at $0.25\sigma$, 0.555 at $1\sigma$, 0.809 at $1.5\sigma$, 0.881 at $2\sigma$, and 0.939 at $3\sigma$; chance is about $1/33=0.030$. This severity response is conditional localization under the simulated fault distribution, not proof of accurate detection of real faults. National localization remains weaker.
'''+fig('secondary','Exploratory studies. Left: conditional localization of injected faults across positive severity levels; severity is in model-standard-deviation units. Right: selected national graph variants on the same 20-station, 12-day subset; the union is selected using validation, not test error. No confirmatory significance claim is attached.','fig:secondary')+r'''
Connected-set conditioning searches, certificate-guided decommissioning, perturbation tests, and empirical interval checks are secondary. Predicted difficult sets do not consistently exceed random-set errors at every size, and guided decommissioning can beat the random mean without beating the best random sample. The exact fitted kernel is trivial, so finite block-distance lower bounds from subset enumeration are not useful operational coding-distance guarantees. Interval calibration is empirical and carries no conformal or distribution-free coverage guarantee.
''')

abstract='''Air-quality monitoring networks contain redundancy across stations, pollutants, and time. This study asks whether a learned cellular sheaf can reconstruct sensor outages while describing the structural support for each reconstruction.

Stations carry pollutant vectors. Canonical correlation analysis learns edge relationships, and regularized harmonic extension reconstructs missing coordinates using a station covariance prior. The operator provides rank-based identifiability and regularized residual sensitivity, C2*. A space-time extension adds temporal relationships.

Evaluation covers six regional networks, a 177-station national network, and two raw-concentration datasets. Models use chronological splits; artificial targets are erased before natural gaps are reconstructed, and observed target cells are scored in a common training-standardized concentration space.

For primary single-station snapshots, the spatial sheaf has lower mean error than Gaussian conditional prediction in seven of eight testable networks, with median relative improvement 3.1 percent; only two comparisons pass within-network Holm correction. For prolonged retrospective reconstruction, first-24-hour mean error is lower than the implemented Kalman/VAR smoother in all nine networks, with median improvement 34.0 percent. Grouping failures by run supports fewer significance claims than treating each run-station failure separately. Prospectively designated high-C2* groups have higher held-out error in all seven evaluable networks.

These results support competitive reconstruction and empirical risk ranking. Shared datasets, retrospective access to later surviving observations, availability-based cohort selection, irregular temporal training pairs, and incomplete source-data verification limit generalization. The study does not establish real-time forecasting, universally superior imputation, or guaranteed operational error bars.'''
write('sections/01_abstract.tex',r'\begin{abstract}'+'\n'+abstract.replace('C2*',r'C2$^*$')+'\n'+r'\end{abstract}')

write('sections/02_introduction.tex',r'''
\section{Introduction and research questions}
Sensor failure can remove a pollutant channel, a complete station, a clustered station set, or hours of observations. Air-quality reconstruction must estimate missing values from the surviving measurements, but an estimate's plausibility does not establish how strongly the observations constrain it. This study investigates a learned cellular-sheaf operator that supports both reconstruction and algebraic diagnostics \citep{hansen2019spectral}.

A station carries a pollutant vector, and each graph edge compares learned linear combinations of neighboring vectors. The resulting inverse problem combines spatial disagreement with a within-station precision prior. A space-time extension adds pre-outage history and temporal constraints. The research questions are:
\begin{enumerate}[label=\textbf{RQ\arabic*.}]
\item How competitive is reconstruction against dense statistical, spatial, low-rank, and graph-neural baselines when targets are hidden before gap handling?
\item Can structural identifiability and residual sensitivity distinguish difficult artificial outage patterns, including risk groups assigned before their test errors are evaluated?
\item What changes when surviving pollutant channels, spatial relations, or temporal information are removed?
\item Does a space-time operator reduce prolonged retrospective reconstruction error against the implemented temporal baselines?
\item How do graph construction, thinning, and distributional instability constrain the interpretation?
\end{enumerate}
The objective is a transparent connection between reconstruction, outage geometry, and model sensitivity. A sparse inverse problem may remain useful even when a dense statistical baseline matches its accuracy. C1 describes structural uniqueness, C2$^*$ describes one component of sensitivity, and historical error estimates require additional statistical assumptions. None alone certifies the truth of a reconstructed environmental measurement.
''')
write('sections/03_related.tex',r'''
\section{Related work and contribution boundary}
Cellular sheaves and their Laplacians are established tools for vector-valued local-to-global relations \citep{hansen2019spectral}. Connections with network coding also precede this study \citep{ghrist2011network}. CCA is classical multivariate statistics \citep{hotelling1936relations}; sheaf-based low-cost air-quality sensing has been proposed by Pham et al. \citep{pham2022airquality}. The present study consequently does not claim to introduce sheaf theory, CCA, or the first sheaf-based air-quality application.

Kriging and covariance-based conditional prediction provide established spatial and multivariate comparisons \citep{cressie1993statistics}. Low-rank completion \citep{mazumder2010softimpute}, graph convolutions \citep{kipf2017gcn}, and recurrent graph imputation such as GRIN \citep{cini2022grin} offer different representations of redundancy. The implementation tests a static masked GCN, not GRIN itself. Classical VAR and Kalman smoothing provide a temporal comparison \citep{lutkepohl2005var,rauch1965smoothing}.

The contribution is the evaluated combination of learned multi-pollutant spatial and temporal restrictions, structured artificial outages, explicit rank and regularized-sensitivity diagnostics, and prospective risk-group comparisons across several monitoring-network configurations. The targeted literature review supports this contribution boundary but is not proof of exhaustive novelty. The coding analogy applies to an ideal exact kernel; the fitted full-rank operator is used as a soft regularized decoder.
''')
write('sections/10_discussion.tex',r'''
\section{Discussion}
The primary single-station results support a competitive spatial model with modest mean gains. They do not justify treating the stronger mixed-pattern significance counts as primary single-station evidence. Pattern weights, natural-gap filtering, and the selected station set matter, so raw-unit pollutant errors and outage-size summaries are provided in the appendix to aid interpretation.

The largest descriptive advantage occurs in prolonged retrospective reconstruction. The model can use target-station history together with later surviving observations, and the fitted offset-24 relation helps several networks. That advantage is against a VAR(1) smoother with fixed noise/ridge defaults and static or persistence comparators, rather than an exhaustive set of seasonal or recurrent models. Coverage-filtered temporal fitting mixes elapsed durations, preventing a clean attribution to daily recurrence. A strictly online model would require its own masked, elapsed-time-matched evaluation.

Prospective risk grouping is useful because test errors do not determine the ranking. Nonetheless C2$^*$ measures model-coordinate residual sensitivity, not total reconstruction error or uncertainty. High- and low-risk groups can differ in outage size; the actual unknown set also includes natural gaps. Heterogeneous correlations and bias/distributional effects prevent a universal risk-to-error calibration.

The national graph study shows modest improvements for the tested connected alternatives on a selected subset. It does not identify a dominant causal failure mechanism or prove that graph design can never help. Likewise apparent stuck or shifted channels require source-data and calibration investigation before assigning a physical cause.

Negative findings are retained: the hybrid is numerically better in Region D, channel benefits vary, C1 is nondiscriminative for fitted dense operators, clustering by run weakens several temporal significance claims, and exploratory design heuristics are mixed. These findings define the demonstrated scope and motivate further experiments without changing the saved primary observations.
''')
write('sections/11_limitations.tex',r'''
\section{Limitations and unresolved evidence}
\begin{itemize}
\item \textbf{Source verification.} Available data hashes match the recorded prefixes, but one raw replication input is absent. Repository mirrors and overlapping export checks do not verify every measurement against the regulator. Integer-index inversion and exclusion of capped values limit extreme-event interpretation.
\item \textbf{Availability-based selection.} Station/pollutant coverage and retained timestamps are chosen from the full archive before chronological splitting. This uses future availability and describes a retrospectively selected cohort. Prolonged windows further require complete target history and hidden truth.
\item \textbf{Temporal fitting and access.} Offset-1/24 CCA training pairs are drawn from filtered rows and need not span exactly one/24 hours. Full-window smoothing may use later surviving observations, so results do not establish real-time recovery or forecasting.
\item \textbf{Statistical dependence.} Regions overlap and run--station failures share runs. Within-network Holm correction does not make nine independent replications, and unit bootstrap intervals do not capture all temporal or spatial dependence. Run-based sensitivity supports fewer corrected comparisons; it does not resolve dependence from repeated stations or overlapping windows.
\item \textbf{Baseline scope.} Kalman/VAR defaults are fixed and it has no seasonal lag; the masked GCN is static. Some channel baselines erase more information than the vector model. Fairer tuned seasonal/recurrent comparisons could change the conclusions.
\item \textbf{Certificate scope.} Static artificial-outage certificates omit changing natural-gap masks. C3 is in model coordinates and cannot be directly equated to common-space RMSE under nonlinear transforms. The fixed-decoder consistency statement is not a guarantee for a fitted, complete-case-selected pipeline.
\item \textbf{Model and deployment.} Linear restrictions, unusual channels, and distribution shift limit applicability. No regulatory error threshold, causal failure attribution, real-fault false-positive rate, or guaranteed interval coverage is established.
\end{itemize}
''')
write('sections/12_conclusion.tex',r'''
\section{Conclusion}
SheafCode connects missing-measurement reconstruction with rank and residual-sensitivity analysis through one learned operator. Primary single-station snapshots are competitive with dense Gaussian prediction, with median improvement 3.1\% across eight testable networks and corrected positive evidence in two. Prolonged retrospective reconstruction gives lower first-day mean error than the implemented Kalman/VAR smoother in all nine networks, with median relative improvement 34.0\%; the strength of significance depends on the replicate definition.

Prospectively designated high-$C2^*$ groups have higher mean held-out error in seven evaluable networks. This supports empirical structural risk ranking, rather than guaranteed per-reading uncertainty. Graph thinning demonstrates loss of structural uniqueness when constraints are removed; connected national graph alternatives offer modest gains on the tested subset. The study establishes a useful research framework with explicit limits on independence, temporal access, source verification, and generalization.
''')
write('sections/13_repro.tex',r'''
\section{Reproducibility and audit record}
The accompanying audit independently recomputes primary single-station, mixed-pattern, single-channel, and prolonged comparisons from saved per-observation error tables, including paired sample sizes, mean differences, relative gains, one-sided tests, and Holm adjustments. It also verifies decile means/ratios and recorded complete-state bound checks. The audit records 1,635 numerical comparisons with no discrepancies against the supplied machine-readable record. This verifies the stored analysis arithmetic, not the entire origin of each input or correctness of every model implementation.

Nineteen supplied synthetic/theory/leakage tests pass in the audit environment. The source report checker also confirms its render and five recorded assertions. That checker does not audit this rewritten manuscript's interpretive claims, prove novelty, or check every benchmark path. A referenced preprocessing test file and a declared requirements file are absent from the supplied final code archive; their existence or successful execution is not inferred.

All 33 available unique input files match recorded SHA-256 prefixes. The missing raw replication input prevents complete end-to-end reproduction of every network from these archives. Supplementary run-level analysis, plot-data CSVs, evidence checks, and reproducible figure-generation scripts accompany the reference manuscript. Figures are regenerated from unrounded tables or documented result fields; horizon, thinning, and secondary plots identify their distinct aggregation and sampling rules.

The supplied research record documents substantial AI assistance. This reference exemplar also uses AI-assisted auditing and editing. It should be used for understanding and checking the project; authorship and assistance must be represented accurately in any separately prepared competition material.
''')
write('sections/14_acknowledgements.tex',r'''
\section{Acknowledgements and assistance}
The project uses public regulatory air-quality records accessed through documented archive mirrors. No mentor, institutional contribution, or independently collected data is invented in this reference. The supplied record documents substantial AI assistance with implementation, theory checking, experimental design, analysis, and writing support. This reference manuscript and its audit are AI-assisted materials. Actual contributions should be acknowledged accurately.
''')

mixedrows=[];longrows=[];rawrows=[];configrows=[]
for reg,R in FR['regions'].items():
    A=AU['results'][reg];c=R['choice']['vector_sheaf'];label=LABELS[reg].replace('Raw replication ','Raw ')
    mix=A['mixed_snapshot'];mixedrows.append([label,R['n_patterns'],f"{mix['vector_sheaf']:.3f}",f"{mix['gaussian_cond']:.3f}",f"{mix['gnn']:.3f}"])
    h=A['horizon_counts'];longrows.append([label,f"{min(map(int,h))}--{max(map(int,h))}",sum(v for k,v in h.items() if int(k)==72),f"{A['prolonged_long']['st_sheaf']:.3f}",f"{A['prolonged_long']['kalman_var']:.3f}"])
    for pol in R['pollutants']:
        rows=[r for r in R['snapshot_by_size'] if r['size']==1 and r['method']=='vector_sheaf'];grow=[r for r in R['snapshot_by_size'] if r['size']==1 and r['method']=='gaussian_cond']
        if rows:rawrows.append([label,pol.replace('PM2.5',r'PM$_{2.5}$').replace('PM10',r'PM$_{10}$').replace('NO2',r'NO$_2$').replace('O3',r'O$_3$'),f"{rows[0]['rmse_'+pol]:.2f}",f"{grow[0]['rmse_'+pol]:.2f}"])
    configrows.append([label,c['transform'],c['graph'],c['k'],c['r'],c['weighting'],c['stalk'],c['lam'],R['choice']['st_sheaf']['t_scale']])
rawtable='\n'.join([r'\begin{longtable}{llrr}',r'\caption{Single-station snapshot mean RMSE per pollutant in $\mu$g/m$^3$. Each entry averages day-level pollutant RMSEs within a pattern, then patterns; it is not a pooled-cell RMSE. These raw-unit means are not converted using a network-average standard deviation.}\label{tab:raw}\\',r'\toprule Network & Pollutant & Sheaf & Gaussian \\ \midrule',r'\endfirsthead',r'\toprule Network & Pollutant & Sheaf & Gaussian \\ \midrule',r'\endhead',*[' & '.join(r)+r' \\' for r in rawrows],r'\bottomrule\end{longtable}'])
write('sections/15_appendix.tex',r'\appendix'+'\n'+r'\section{Supplementary numeric tables}'+'\n'+table(['Network','Stored patterns','Sheaf','Gaussian','GCN'],mixedrows,'Secondary mixed-size snapshot means. Stored pattern counts include patterns with absent scores; available targets determine each mean.','tab:mixed')+'\n'+table(['Network','Hours (range)','72h units','ST sheaf','Kalman'],longrows,'Mean unit RMSE over each available full outage horizon. A 24--72-hour protocol does not imply every unit lasts 72 hours. Exact duration counts are preserved in the numerical audit.','tab:long')+'\n'+rawtable+'\n'+r'\section{Frozen configuration}'+'\n'+table(['Network','Transform','Graph','$k$','$r$','Weight','Stalk','$\\lambda$','Time'],configrows,'Frozen vector-sheaf settings. reg denotes $\\rho/(1-\\rho)$; invvar denotes $1/[2(1-\\rho)]$. Time is the temporal weight scale; cca+id adds per-pollutant identity restrictions.','tab:config')+r'''
\section{Operational interpretation}
A research implementation can reconstruct unavailable cells, compute diagnostics for the actual missing mask, and compare historical error with held-out calibration data in the same metric space. Static $C2^*$ rankings are useful for studying artificial outage geometry. An online deployment would require a new causal timing protocol, elapsed-time-matched training, verified inputs, and application-specific acceptable-error criteria. Fault localization and network-design findings remain exploratory.

\section{Network-label mapping and study sites}
The anonymized labels preserve the source result ordering. A separate audit mapping records scientific study sites and overlapping networks without introducing school, student-location, or institution claims. Anonymization is a presentation choice in this reference; scientific data provenance is retained.
''')

                                                                                    
portal_fields=[('Project Abstract',abstract),('Introduction \\& Objective',
'''Sensor outages remove channels, stations, or consecutive readings. This study treats reconstruction as a structured inverse problem and asks how strongly the remaining network constrains an estimate. The objectives are to learn multi-pollutant spatial and temporal relationships, reconstruct artificial outages without using hidden truth, derive structural-identifiability and residual-sensitivity diagnostics, and evaluate those diagnostics on held-out errors. Additional experiments examine pollutant information, graph thinning, graph alternatives, and synthetic faults. The study preserves negative results and distinguishes retrospective reconstruction from forecasting. Its conclusions apply to the implemented models and selected public monitoring datasets.'''),('Innovation',
'''The contribution is the evaluated combination of learned multi-pollutant and temporal sheaf restrictions with outage-specific structural diagnostics and prospective risk-group analysis. Sheaf theory, canonical correlation analysis, air-quality sheaf applications, and graph imputation are established. One operator links reconstruction, rank-based identifiability, and regularized residual sensitivity; the study tests where that link is informative and where statistical or data limitations dominate.'''),('Methodology',
'''Stations carry pollutant vectors. Canonical correlation analysis fits edge restrictions, and a regularized inverse problem reconstructs unknown coordinates using a station covariance prior. C1 tests full column rank of the erased operator; C2* measures regularized residual sensitivity. Retained rows are split chronologically 60/20/20. Spatial settings are selected on validation error; temporal scales use validation windows and the Kalman/VAR settings are fixed defaults. Artificial targets are erased before natural gaps are handled, and only previously observed targets are scored in a common training-standardized concentration space. Primary snapshot tests use single-station patterns; channel tests use station-pollutant targets; prolonged tests use run-station failures. One-sided paired comparisons use within-network Holm correction. Supplementary run-based sensitivity checks dependence. Prolonged windows have 24 hours of history and available outage horizons through 72 hours. Full-window methods may use later surviving observations; temporal training offsets need not equal elapsed hours.'''),('Results \\& Conclusions',
'''For primary single-station snapshots, the sheaf has lower mean error than Gaussian conditional prediction in seven of eight testable networks, with median relative improvement 3.1 percent; two comparisons pass within-network Holm correction. Prolonged retrospective reconstruction has lower first-day mean error than the implemented Kalman/VAR smoother in all nine networks, with median improvement 34.0 percent. A run-based sensitivity analysis supports fewer corrected comparisons. Prospectively assigned high-C2* groups have higher error in all seven evaluable networks; six nominal tests and five supplementary across-network Holm tests pass 0.05. Channel benefits vary. Connected national graph alternatives provide modest gains on a selected subset without establishing a dominant failure cause. These results support competitive reconstruction and empirical structural risk ranking. Overlapping datasets, timing assumptions, availability selection, and one missing raw input constrain inference and complete reproduction.'''),('Acknowledgement \\& Reference Links',
r'''Public regulatory air-quality records were accessed through documented archive mirrors. The supplied record documents substantial AI assistance with implementation, theory, experimental design, analysis, and writing support; this reference is also AI-assisted. No mentor or institution is invented. References: Hansen and Ghrist (\url{https://arxiv.org/abs/1808.01513}); Pham et al. (\url{https://arxiv.org/abs/2212.14353}); GRIN (\url{https://arxiv.org/abs/2108.00298}). ISEF judging guidance is available at \url{https://www.societyforscience.org/isef/grand-award/criteria/}.''')]
write('portal_fields.tex',r'\documentclass[11pt]{article}\input{preamble.tex}'+'\n'+r'\title{Short-Form Study Reference}\author{SheafCode}\date{}\begin{document}\maketitle\refwarning'+'\n'+'\n'.join(r'\section*{'+name.replace('\\\\','\\')+'}\n'+text.replace('C2*',r'C2$^*$') for name,text in portal_fields)+'\n'+r'\end{document}')
script='''When an air-quality station fails, a reconstruction can look plausible without being strongly supported by the surviving measurements. SheafCode studies both questions.

Each station carries a vector of pollutants. Learned edge relationships form a cellular sheaf, a sparse linear system that reconstructs missing values and measures structural identifiability and residual sensitivity.

Artificial targets are hidden before natural gaps are reconstructed. The evaluation uses chronological splits and a common concentration metric across nine network configurations.

Primary single-station snapshot improvements are modest. For prolonged retrospective reconstruction, first-day mean error is lower than the implemented Kalman and vector-autoregression smoother in all nine networks, with a median 34 percent improvement. Grouping failures by run weakens several significance claims.

Before test-error evaluation, the model designates high- and low-sensitivity outage groups. High-sensitivity groups later have larger mean errors in all seven evaluable networks.

These results support competitive reconstruction and empirical risk ranking. The full-window models can use later surviving observations, so the findings do not establish real-time forecasting. Dataset overlap, irregular training timestamps, and incomplete input verification remain limitations.'''
write('video_script.tex',r'\documentclass[11pt]{article}\input{preamble.tex}'+'\n'+r'\title{Video Reference Storyboard}\author{SheafCode}\date{}\begin{document}\maketitle\refwarning'+'\n'+r'\section*{Spoken reference script}'+'\n'+script+'\n'+r'''
\section*{Visual storyboard}
\begin{enumerate}
\item 0--15 s: show a station outage and the two research questions.
\item 15--32 s: show station vectors, learned edge rules, and the decoder.
\item 32--60 s: show primary snapshot and first-day temporal figures with their units.
\item 60--76 s: show risk groups assigned before test errors are examined.
\item 76--90 s: show retrospective observation timing and the main limitations.
\end{enumerate}
Timing depends on spoken pace; rehearse the script rather than assuming a word count guarantees a duration.
\end{document}''')
syn=(OUT/'synopsis.tex').read_text()
syn=syn.replace('This study is designed to test whether','This completed study evaluates whether').replace('The proposed contribution','The evaluated contribution').replace('one-hour and twenty-four-hour temporal relations','retained-row offset-1 and offset-24 temporal relations').replace('Temporal edges at lags 1 and 24 hours create the space-time decoder.','Temporal edges at row offsets 1 and 24 create the space-time decoder. Filtered training rows need not be consecutive hours; test prolonged windows are hourly and reconstructed retrospectively.').replace('Statistical comparisons operate at the outage-pattern or run-by-station level, use bootstrap confidence intervals, and apply Holm correction to the pre-specified comparison families.','Primary snapshot comparisons use single-station patterns. Prolonged comparisons use run--station units, with a supplementary run-based dependence check. Within-network Holm correction does not establish independence of overlapping networks.').replace(r'\section{Planned comparisons and falsification criteria}',r'\section{Comparisons and interpretation}')
syn=syn.replace(r'\section{Scope and limitations}',r'\section{Observed results}'+'\n'+portal_fields[4][1].replace('C2*',r'C2$^*$')+'\n'+r'\section{Scope and limitations}')
syn=syn.replace('The study evaluates reconstruction and network resilience rather than causal atmospheric modelling or pollution forecasting.','The study evaluates retrospective reconstruction and network diagnostics. It does not establish causal atmospheric modeling, forecasting, or real-time recovery. Full-archive coverage selection, irregular temporal training pairs, shared networks, and one missing raw input constrain inference and reproduction.')
syn=syn.replace('is a reconstructed blueprint for understanding the final study design','summarizes the completed study')
syn=syn.replace('independent raw-concentration monitoring networks','additional raw-concentration monitoring datasets').replace('independent raw-concentration replication sets','additional raw-concentration datasets')
write('synopsis.tex',syn)

                                                                             
pre=(OUT/'preamble.tex').read_text().replace(r'\usepackage[backend=biber,style=authoryear,natbib=true,maxcitenames=2,maxbibnames=20]{biblatex}',r'\usepackage[authoryear,round]{natbib}')
pre=pre.replace(r'\addbibresource{references.bib}',r'\bibliographystyle{plainnat}')
pre=pre.replace(r'\newcommand{\projecttitle}{SheafCode: Reconstruction and Reliability Analysis of Air-Quality Sensor Failures with Learned Space-Time Cellular Sheaves}',r'\newcommand{\projecttitle}{SheafCode: Reconstruction and Structural Reliability Analysis of Air-Quality Sensor Outages}')
pre=pre.replace(r'\newcommand{\refwarning}{',r'\setlength{\emergencystretch}{2em}'+'\n'+r'\newcommand{\refwarning}{')
pre=pre.replace(r'\usepackage{draftwatermark}',r'\usepackage{draftwatermark}'+'\n'+r'\pdfstringdefDisableCommands{\def\sigma{sigma}\def\lambda{lambda}\def\rho{rho}\def\delta{delta}}')
pre=pre.replace(r'\SetWatermarkText{REFERENCE ONLY - DO NOT SUBMIT}',r'\SetWatermarkText{REFERENCE EXEMPLAR}').replace(r'\SetWatermarkScale{0.42}',r'\SetWatermarkScale{0.32}')
write('preamble.tex',pre)
refs=(OUT/'references.bib').read_text().replace('note         = {Archival access path for the 2017--2020 CPCB-origin data}', 'year         = {n.d.},\n  note         = {Archival access path for the 2017--2020 CPCB-origin data; repository accessed 5 October 2026}')
write('references.bib',refs)
for name in ['main.tex','synopsis.tex']:
    s=(OUT/name).read_text().replace(r'\printbibliography',r'\bibliography{references}')
    s=s.replace(r'\author{[Student Name] \\\small Individual Project \;|\; Grade [XX] \;|\; Category: Software Design}',r'\author{Research reference exemplar}')
    s=s.replace('Competition-facing region names are anonymized.','Network labels are anonymized; this is a completed-study reference.')
    write(name,s)

counts={name:len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b",text)) for name,text in portal_fields}
counts['spoken_script']=len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b",script))
write('audit/word_counts.json',json.dumps(counts,indent=2))
print('Reference word counts',counts)
pd.DataFrame(snaprows,columns=['network','targets','sheaf','gaussian','relative_gain_percent','p_holm']).to_csv(OUT/'data/snapshot_summary.csv',index=False)
pd.DataFrame(prows,columns=['network','units','runs','space_time','kalman','relative_gain_percent','p_holm_unit','p_holm_run']).to_csv(OUT/'data/prolonged_summary.csv',index=False)
pd.DataFrame(rawrows,columns=['network','pollutant','sheaf','gaussian']).to_csv(OUT/'data/raw_pollutant_rmse.csv',index=False)
write('audit/network_mapping.json',json.dumps({LABELS[k]:dict(source_region=k,stations=FR['regions'][k]['n_stations'],test_span=FR['regions'][k]['test_span']) for k in REGS},indent=2))

write('README.md',r'''
# SheafCode completed-study reference

The supplied research is presented as one completed project. These AI-assisted reference documents retain their reference watermark and disclose assistance.

## Read and use

- `main.pdf` and `main.tex`: audited full research reference.
- `synopsis.pdf` / `synopsis.tex`: completed-study synopsis.
- `portal_fields.pdf` / `portal_fields.tex`: consistent short-form reference text; no claim that live portal limits or category requirements were verified.
- `video_script.pdf` / `video_script.tex`: concise reference narration and storyboard; rehearse to verify duration.
- `figures/`: rebuilt vector PDF and high-resolution PNG plots.
- `data/`: exact plotted values, sample counts, uncertainty, and supplementary summaries.
- `FINAL_AUDIT.md`: audit findings, corrections, validation, and unresolved evidence.
- `audit/`: machine-readable numerical and document checks, source mapping, and assistance record.

## Overleaf

Upload the contents of this folder. Select `main.tex`, `synopsis.tex`, `portal_fields.tex`, or `video_script.tex` as the main document. Use pdfLaTeX and BibTeX (bibliography is `plainnat` with `natbib`). Bibliography is included in the full paper and synopsis. Portable local builds use Tectonic and produce the bundled PDFs.

## Scope of confirmation

Saved analysis arithmetic is independently checked against raw error tables. All 34 input hashes match the documented prefixes; one absent input was recovered from its recorded public repository commit and verified against its full documented hash. Sample reconstructions were replayed from both raw datasets. Source authenticity, complete experimental reproduction, novelty, real-time forecasting, and deployment suitability are not certified. The paper describes these limits.
''')

                                                                                      
replacements={
    'and incomplete source-data verification limit generalization.':'and incomplete external source verification limit generalization.',
    'and one missing raw input constrain inference and complete reproduction.':'and limited external source verification constrain generalization.',
    'and incomplete input verification remain limitations.':'and limited external input verification remain limitations.',
    'shared networks, and one missing raw input constrain inference and reproduction.':'shared networks, and limited external source verification constrain inference and generalization.',
    "The source input for Raw replication 1 was not present in the supplied archives; its saved error tables can be audited, while complete source-data reproduction remains unavailable.":"The originally absent Raw replication 1 input was recovered from the documented public repository commit, and its full SHA-256 matches the provenance record. Twenty-two sampled single-station day scores and four full prolonged reconstructions across the two raw datasets reproduce their saved errors within numerical tolerance. This is sampled replay, not a complete independent rerun of all fitting, selection, and experiments.",
    'Available data hashes match the recorded prefixes, but one raw replication input is absent.':'All 34 data hashes match the recorded prefixes; an input absent from the supplied archives was recovered from its documented commit and full hash verified.',
    'All 33 available unique input files match recorded SHA-256 prefixes. The missing raw replication input prevents complete end-to-end reproduction of every network from these archives.':'All 34 unique input files now match recorded SHA-256 prefixes. One absent raw input was recovered from its documented repository commit and its full documented hash verified. Twenty-six sampled reconstruction replay checks across the two raw datasets pass, including four complete prolonged units. Complete training/validation/test reruns of every benchmark and network were not performed.',
    'Gaussian 1.147, with the lowest tested method mean 1.118.':'Gaussian 1.146, with the lowest tested method mean 1.118.',
}
for path in list((OUT/'sections').glob('*.tex'))+[OUT/'synopsis.tex',OUT/'portal_fields.tex',OUT/'video_script.tex']:
    text=path.read_text()
    for old,new in replacements.items():text=text.replace(old,new)
    path.write_text(text,encoding='utf8')

repro=(OUT/'sections/13_repro.tex').read_text().replace('This verifies the stored analysis arithmetic, not the entire origin of each input or correctness of every model implementation.','A further 2,545 supplementary checks verify outage-size/raw-unit summaries, correlations, horizon values, uncertainty ordering, thinning, fault localization, and graph-study values. These checks verify stored analysis arithmetic, not the entire origin of each input or correctness of every model implementation.')
write('sections/13_repro.tex',repro)

data=(OUT/'sections/04_data.tex').read_text()
data=data.replace('The supplied provenance record reports agreement with overlapping raw exports after a one-hour timestamp adjustment.','Rechecking 13 overlapping station--pollutant series after the documented one-hour timestamp adjustment gives correlations 0.9948--0.9997 and median absolute differences 0.14--0.27 $\\mu$g/m$^3$; the 95th-percentile absolute differences reach 2.08 $\\mu$g/m$^3$ in one series. This supports the conversion empirically on those pairs, while retaining rounding and other discrepancies.')
write('sections/04_data.tex',data)
write('audit/input_recovery.json',json.dumps(dict(source_url='https://raw.githubusercontent.com/ajitesh68/Delhi-aqi-forecasting/4a248e45/data/store/cpcb_hourly.parquet',sha256='7096ead5d9619007f663df29b10ff1d582edf45f2a787714f2f9935eee37a3a0',matches_documented_full_hash=True),indent=2))

                                                                                   
from scipy.stats import spearmanr
fig_,axes=plt.subplots(2,2,figsize=(9.5,6.5));scatterrows=[];composition=[]
for ax,reg in zip(axes.flat,['delhi','kolkata','bengaluru','india']):
    s=pd.read_csv(TABLES/f'v3_{reg}'/'test_results.csv')
    s=s[s.method=='vector_sheaf'].groupby('pattern').agg(rmse=('rmse_z','mean'),c2=('c2_star','first'),size=('size','first')).dropna()
    pts=ax.scatter(s.c2,s.rmse,c=s['size'],cmap='viridis',vmin=1,vmax=12,s=19,alpha=.7,edgecolors='none')
    rho=spearmanr(s.c2,s.rmse).statistic
    ax.set_title(f'{LABELS[reg]} (Spearman {rho:.2f})',fontsize=11);ax.set_xlabel('Static artificial-outage C2*');ax.set_ylabel('Pattern mean standardized RMSE');ax.grid(alpha=.15)
    for pattern,row in s.iterrows():scatterrows.append(dict(network=LABELS[reg],pattern=pattern,c2=row.c2,rmse=row.rmse,size=row['size']))
fig_.tight_layout(rect=[0,0,.9,1]);bar=fig_.colorbar(pts,ax=list(axes.flat),fraction=.026,pad=.025);bar.set_label('Erased stations');save(fig_,'risk_association');pd.DataFrame(scatterrows).to_csv(OUT/'data/risk_association.csv',index=False)
for reg in REGS:
    if not AU['results'][reg]['decile']:continue
    s=pd.read_csv(TABLES/f'v3_{reg}'/'test_results.csv');s=s[s.method=='vector_sheaf'].groupby('pattern').agg(size=('size','first'),rmse=('rmse_z','mean'))
    d=json.loads((SRC/'results'/f'v3_{reg}'/'prospective_deciles.json').read_text())
    for kind,field in [('low','safest_decile'),('high','riskiest_decile')]:
        group=s.reindex(d[field]).dropna();composition.append(dict(network=LABELS[reg],group=kind,assigned_n=len(d[field]),scored_n=len(group),median_outage_size=float(group['size'].median()),mean_outage_size=float(group['size'].mean())))
pd.DataFrame(composition).to_csv(OUT/'data/risk_group_composition.csv',index=False)
res=(OUT/'sections/09_results.tex').read_text()
anchor='The deterministic general-$P$ bound has zero recorded violations'
extra=fig('risk_association','Static $C2^*$ versus held-out pattern error in four contrasting networks. Each point is an evaluated artificial outage; color gives erased-station count. These unadjusted mixed-size associations illustrate heterogeneity and possible size confounding.','fig:riskassociation')+'\n'+r'Coverage-based scoring excludes some preassigned patterns, leaving unequal scored groups in several networks. The risk comparisons are conditional on the scored patterns; they do not establish performance on outages without enough observed target truth. Group composition and assigned/scored counts are preserved in the plot-data files.'+'\n\n'
res=res.replace(anchor,extra+anchor)
write('sections/09_results.tex',res)
