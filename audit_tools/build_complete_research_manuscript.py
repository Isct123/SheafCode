                                                                                

                                                                             
                                                      
   
import hashlib,json,re,shutil,subprocess,os,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1]
R=BASE/'Research_Release'
O=BASE/'Final_Manuscript/Complete_Research'
for name in ['sections','figures','data','audit']: (O/name).mkdir(parents=True,exist_ok=True)
sources={};claims=[];figures=[];tables=[]
LABEL={'delhi':'Delhi archive','ncr':'NCR archive','mumbai':'Mumbai archive','kolkata':'Kolkata archive',
       'hyderabad':'Hyderabad archive','bengaluru':'Bengaluru archive','india':'National archive',
       'delhi2025':'Delhi concentration','bengaluru_raw':'Bengaluru concentration','delhi_later':'Later Delhi period'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source(p):
    p=Path(p);sources[str(p.relative_to(BASE))]=sha(p);return p
def csv(p):return pd.read_csv(source(p))
def js(p):return json.loads(source(p).read_text(encoding='utf-8-sig'))
def put(name,text): (O/name).write_text(text.strip()+'\n',encoding='utf-8')
def esc(s):
    s=str(s)
    for a,b in [('\u2014','--'),('\u2013','--'),('\u00b5',r'$\mu$'),('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:s=s.replace(a,b)
    return s
def n(x,d=1):return '--' if pd.isna(x) else f'{float(x):.{d}f}'
def claim(key,value,sourcepath,meaning):
    claims.append({'key':key,'value':value,'source':str(sourcepath.relative_to(BASE)),'meaning':meaning});return value
def table(headers,rows,caption,label):
    tables.append({'label':label,'headers':headers,'rows':rows,'caption':caption})
    width='l'+'r'*(len(headers)-1)
    return '\n'.join([r'\begin{table}[htbp]\centering\small',r'\caption{'+caption+r'}\label{'+label+'}',
        r'\begin{tabular}{'+width+'}',r'\hline',' & '.join(headers)+r' \\ \hline',
        *[' & '.join(map(str,row))+r' \\' for row in rows],r'\hline\end{tabular}\end{table}'])
def figure(p,caption,label,width='.98',page=None):
    p=source(p);target=label.replace(':','_')+'.pdf';shutil.copy2(p,O/'figures'/target)
    figures.append({'label':label,'source':str(p.relative_to(BASE)),'copied_sha256':sha(O/'figures'/target),'caption':caption,'page':page})
    options='width='+width+r'\linewidth'+('' if page is None else ',page='+str(page))
    return r'\begin{figure}[htbp]\centering'+'\n'+r'\includegraphics['+options+']{figures/'+target+'}\n'+r'\caption{'+caption+r'}\label{'+label+'}\n'+r'\end{figure}'


def build():
    recheck=js(O/'audit/RECHECK.json')
    assert recheck['status']=='passed'
                                                                               
    S=csv(R/'joint_decision_results/comparison_sensitivity.csv')
    C=js(R/'extension_results/strong_baselines/summary.json')['comparisons']
    ext=js(R/'validation_results/external_summary.json')
    fs=csv(R/'validation_results/fault_summary.csv')
    alert=csv(R/'validation_results/alert_summary.csv')
    aci=csv(R/'validation_results/alert_paired_week_intervals.csv')
    intervals=csv(R/'followup_results/interval_summary.csv')
    bounded=csv(R/'constrained_results/summary.csv')
    cs=js(R/'constrained_results/manifest.json')
    rank=js(R/'extension_results/structured/summary.json')
    graph=js(R/'validation_results/graph_summary.json')
    joint=csv(R/'joint_decision_results/joint_cases.csv')
    weather=js(R/'results/weather/weather_summary.json')
    d=C['delhi2025']['causal'];gain=claim('causal_delhi_gain',100*d['rmse_relative_gain'],R/'extension_results/strong_baselines/summary.json','Reduction of sqrt(mean matched station-day normalized MSE), relative to full-covariance VAR')
    newer=claim('later_delhi_gain',ext['comparison']['full_covariance_var']['rmse_improvement_percent'],R/'validation_results/external_summary.json','PM2.5 only; PM-only input and four matched instruments')
    fault=fs[(fs.region=='delhi2025')&(fs.fault_count==2)&(fs.direction=='positive')&(fs.magnitude==2)&(fs.method=='calibrated_classifier')].iloc[0]
    fault_gain=claim('unknown_count_two_fault_success',100*fault.exact_end_to_end,R/'validation_results/fault_summary.csv','Synthetic same-sign magnitude-two injection, raw Delhi; unknown count at evaluation')
    abstracts=(f"Air-quality stations can become unavailable while neighbouring sensors remain active, leaving gaps in pollution monitoring. "
        f"SheafCode learns vector-valued spatial and temporal consistency relations and reconstructs missing pollutant channels from available observations. "
        f"The study separates causal reconstruction from retrospective imputation and derives conditional reliability bounds for chosen pollution quantities, unknown station faults and joint pollutant thresholds. "
        f"On the Delhi concentration cohort, causal reconstruction of 24-hour station outages with 24 hours of pre-outage history achieved {gain:.1f}% lower normalized root mean squared error than a validation-tuned full-covariance vector autoregressive filter ({d['n']} paired station-days, {d['clusters']} days). "
        f"The advantage remained positive in calendar-block sensitivity analyses. A non-overlapping later period gave {newer:.1f}% lower PM2.5 error than that filter across {ext['cases']} station-days at four geographically matched stations. "
        f"An official bidirectional graph-imputation architecture performed better in Delhi and worse in Bengaluru in separate retrospective comparisons. "
        f"Fault experiments distinguish alarm, count and localization success; uncertainty studies distinguish revealed-label calibration from intervals that receive no outage labels. "
        f"The mathematical results include sharp functional and joint-threshold envelopes, conditional repair criteria, bounded-fault outer certificates and exact nominal rational verification. "
        f"Several extensions yielded negative results or frequent abstention. Archives overlap, most evaluations are exploratory, and mathematical priority is unestablished. "
        f"The demonstrated application is monitoring continuity; emissions reduction, health benefit and official policy recovery were not measured.")
    assert len(abstracts.split())<=250
    put('abstract.txt',abstracts)
    put('sections/01_abstract.tex',r'\begin{abstract}'+esc(abstracts)+r'\end{abstract}')
    put('sections/02_context.tex',r'''
\section{Research question and contribution}
When an air-quality station loses communication, a model can estimate missing readings from neighbouring stations and the target's pre-outage history. The practical question is whether those estimates outperform realistic alternatives, and whether their uncertainty permits a defensible pollution-threshold decision. Structural recoverability is a separate question: an algebraically identifiable state can still be reconstructed inaccurately when learned relationships have substantial residuals.

The project addresses three linked questions. First, does causal reconstruction improve station-outage error against validation-tuned spatial and temporal baselines? Second, which pollution quantities and threshold decisions are identifiable under declared residual and station-fault assumptions? Third, do interval, fault-alarm and simulated maintenance experiments support a useful monitoring application? Negative answers and failure regions are retained.

The empirical contribution is a reproducible comparison across overlapping Indian monitoring archives, two concentration cohorts and a later same-network period. The methodological contribution is the integration of fitted vector sheaves, causal information restrictions, functional bounds, unknown-support fault envelopes and joint pollutant decisions. The mathematical arguments specialize established structural-rank, sparse-recovery, set-membership and optimization theory. The work does not claim to invent sheaves, Hall matching, Gaussian conditioning, branch-and-bound or conformal calibration.

\section{Related work and originality boundary}
Cellular sheaves encode compatible local measurements through linear restriction maps; their Laplacians and spectral theory are established \citep{hansen2019spectral,ghrist2011network}. Sheaf-based air-quality self-filtering predates this project \citep{pham2022airquality}. Consequently, applying a sheaf to pollution sensors alone is not a first-ever contribution. Canonical correlation analysis supplies the fitted endpoint maps \citep{hotelling1936relations}.

The zero-residual fault uniqueness condition is the block counterpart of classical sparse-representation uniqueness \citep{donoho2003}. Functional observability asks whether a chosen quantity can be determined even when a full state cannot \citep{functional2022}. Bounded-error inverse problems, epigraph optimization, budgeted robust optimization and verified dual corrections motivate the conditional certificates \citep{bounded1990,boyd2004,bertsimas2004,jansson2004}. Sensor-placement theory provides context for the maintenance studies \citep{krause2008}; its guarantees cannot be transferred to a different objective without proof.

GRIN is a published graph recurrent imputation architecture \citep{cini2022grin}. This project uses a pinned official bidirectional implementation with its own outage-training protocol. Its retrospective results are not a comparison against a causal GRIN model or a reproduction of the original published dataset scores. The simpler masked graph controls are separately identified. Adaptive conformal inference can regulate long-run error with revealed labels \citep{gibbs2021}; it does not make unavailable target labels available during an outage.

The claimed project-specific mathematical substance is the complete set of stated assumptions, derivations, verified implementations and application-specific interpretations. A focused independent priority review remains necessary before claiming any theorem is first-ever.
''')
    cohortrows=[]
    for region in ['delhi','ncr','hyderabad','bengaluru','india','delhi2025','bengaluru_raw']:
        j=js(R/f'results_expanded/{region}/cohort.json');manifest=js(R/f'results_expanded/{region}/manifest.json')
        ds=np.load(source(R/f'results_expanded/{region}/dataset.npz'))
        assert ds['X'].shape[1:]==(len(j['stations']),len(j['pollutants']))
        cohortrows.append([esc(LABEL[region]),len(j['stations']),len(j['pollutants']),j['n_train'],j['n_validation'],j['n_test']])
    datatext=r'''
\section{Data, provenance and evaluation populations}
The archive families include Delhi, NCR, Mumbai, Kolkata, Hyderabad, Bengaluru and a national network, with additional concentration data for Delhi and Bengaluru. Some historical archives contain transformed pollutant sub-indices: inversion into concentration-like values is an approximation whose provenance and temporal averaging need care. Direct concentration cohorts are distinguished from these archives throughout. Counts in one cohort cannot be transferred to another by city name alone.

For the hourly cohort protocol, the complete timeline is divided chronologically into 60\% training, 20\% validation and 20\% test before station/channel selection. Selection uses only the entire fixed training window: each selected station/channel must have at least 50\% observed coverage and 720 observed training hours. Candidate pollutant subsets require at least two channels and three stations, and a fixed rule maximizes retained station-channel count. Mumbai and Kolkata do not pass this rule; their supplied snapshot/window evidence is retained separately in Appendix~\ref{app:supplied}, not silently included in the hourly cohort benchmark. The wider supplied cohorts use different eligibility and timestamp protocols.

Missing hours remain explicit. Normalization, transforms, graph construction, CCA and precision fitting use training data. Hyperparameters and calibrated cutoffs use their stated validation sets. Previously examined archives make these internal follow-ups exploratory even when fitting and selection are temporally separated. The comparisons are not a retrospectively declared preregistration.
'''+table(['Cohort','Stations','Channels','Train h','Val h','Test h'],cohortrows,
         'Hourly cohort sizes from the frozen manifests. These are timeline hours, not counts of complete observations or independent samples.','tab:cohorts')+r'''
\subsection{Overlap and the later period}
Delhi archive sites are included in NCR and the national archive, and some concentration-download sites match archive sites geographically. Shared calendar periods do not alone prove duplicate measurements; different download paths do not prove instrument independence. The overlap audit reports shared station identifiers, sites within 200 metres and calendar overlap. Cohorts are partially independent views of monitoring systems, not independent replications.

The later-period study acquires public OpenAQ observations after the model and protocol were fixed. It covers 12 September--2 October 2026 and follows the original Delhi archive endpoint of 11 September. Four exact-coordinate station matches are available: R K Puram, Punjabi Bagh, Jahangirpuri and Mundka. Source records, units, timestamps and download hashes are retained. Duplicate station-hour/channel observations are averaged explicitly. Only PM mass-concentration inputs are used; gas channels are left missing to avoid unverified gas-unit compatibility. This is new-period validation of some of the same instruments, not independent-instrument replication or proof that nobody had previously accessed those observations \citep{openaq2026}.

\subsection{Eligibility and scoring}
Artificial outages hide the target vector before the observation mask is formed. Natural context gaps remain missing. The main prolonged task has 24 hours of target history followed by a 24-hour whole-station outage. Only available target truth is scored, with the recorded eligibility rules; a model estimate at a naturally missing target cell is not verified truth. The later-period study additionally requires at least two surviving PM stations and at least twelve observed PM2.5 target hours in an eligible station-day.

For the main hourly benchmark, let $E_i$ be the mean squared error of the observed target cells in station-day $i$, after division by their station/pollutant training standard deviations. The reported aggregate error is $\sqrt{N^{-1}\sum_i E_i}$. A gain over baseline $B$ is
\[
 100\left(1-\sqrt{\frac{\sum_i E_i^{\rm sheaf}}{\sum_i E_i^B}}\right),
\]
on common eligible station-days. This differs from the supplied snapshot/window protocol's mean of pattern or run-station RMSEs, and from physical-unit PM2.5 error. Raw-unit scores are not obtained by multiplying by a network-average standard deviation.

Calendar-day resampling retains all paired stations on each selected day. Seven- and fourteen-day calendar-bin sensitivities retain longer dependence within each bin but assume resampleable bins; they are not guarantees under arbitrary serial dependence. Networks overlap. The ten-contrast wider percentile sensitivities are exploratory Bonferroni diagnostics, not exact multiple-testing control. Sparse later-period bins are explicitly flagged. All alternative scales and all neural training seeds are retained.
'''
    put('sections/03_data.tex',datatext)
    put('sections/04_method.tex',r'''
\section{Reconstruction and diagnostic methods}
\subsection{Learned spatial and temporal relations}
A station $v$ holds a $p$-channel normalized vector $x_v$. For an edge $e=(u,v)$, weighted CCA restriction maps produce the discrepancy $A_{e,u}x_u-A_{e,v}x_v$. Their stacked matrix is the coboundary $D$; $D^TD$ is the sheaf Laplacian. Graph alternatives use geographic proximity, training correlations or their union. Row weights are clipped functions of fitted canonical correlations. When complete paired training vectors are insufficient, specified identity-direction fallbacks are used. Optional identity rows preserve pollutant-wise relations. Fallback and tied rows have structure; generic entry independence is not assumed for these fitted matrices.

Temporal CCA maps connect station vectors separated by one and twenty-four elapsed hours on a complete hourly timeline. Spatial and temporal rows form the window operator $\mathcal D$. A per-station covariance inverse supplies the positive-definite block prior $P$. With observed coordinates $O$ and unknown coordinates $U$, the fitted decoder solves
\begin{align}
 \widehat z&=\arg\min_z\|D_Uz+D_Ox_O\|_2^2+
       \lambda(x_O,z)^TP(x_O,z),\\
 \widehat z&=-(D_U^TD_U+\lambda P_{UU})^{-1}
                (D_U^TD_O+\lambda P_{UO})x_O.\label{eq:decode}
\end{align}
The ordered pair denotes the full vector in its original coordinate order. The observed--unknown prior term $P_{UO}x_O$ is required for conditioning on surviving channels. In a temporal window, $D$ is replaced by $\mathcal D$ and masks include every naturally missing cell. With $\lambda>0$ and positive-definite $P_{UU}$ the quadratic is strictly convex. A unique regularized solution is not evidence of unregularized identifiability.

\subsection{Causal versus retrospective information}
The causal decoder constructs each estimate from a prefix ending at that hour and the pre-outage target history. It does not use later surviving-station readings. The retrospective decoder may use surviving readings later in the entire outage window. Neither receives the hidden target readings. These information sets are separately evaluated; their gains are not pooled.

\subsection{Baselines and model selection}
Spatial controls include Gaussian conditional reconstruction, scalar relations, inverse-distance weighting and kriging. Temporal controls include seasonal persistence and ridge VAR filtering/smoothing. The stronger VAR control uses training-only iterative rank-truncated PCA completion, correlated innovation covariance and validation selection of transform, ridge, observation noise and covariance shrinkage. That completion routine is not SoftImpute. Filter and smoother results remain separate. Comparisons with unequal available channels or history are identified in the supplied-study appendix.

The official bidirectional GRIN code is pinned to a documented commit. Training uses artificial 24-hour target outages within 48-hour windows, natural gaps remain unlabeled, widths 32/64 are selected on validation, and all three optimization seeds are scored. Adam, gradient clipping, validation early stopping and checkpoint identity hashes are retained. This project-specific adaptation does not reproduce the original GRIN benchmark protocol. A graph-conditioned MLP is an additional single-seed snapshot control, not GRIN.

A convex hybrid selects weights over sheaf, full-covariance VAR and seasonal predictions using a fixed 0.1 grid and validation cases. Outcomes are evaluated without choosing the best test weight or cohort-specific result after scoring. Since the component archives had already been examined, this remains an exploratory extension.

\subsection{Reliability, uncertainty and fault evaluation}
Unregularized identifiability is $\operatorname{rank}(D_U)=|U|$. Singular margins and the regularized residual gain quantify different properties. For the true full state $x$ with residual $r=Dx$, the reconstruction error is
\[
 e_U=-G^{-1}\{D_U^Tr+\lambda(Px)_U\},\qquad G=D_U^TD_U+\lambda P_{UU}.
\]
This identity includes prior bias. An empirical training-error proxy needs representative errors, finite moments and an ergodic law of large numbers; stationarity of $Dx$ alone is insufficient, and self-completed training states can bias the proxy.

Revealed-label adaptive intervals construct every simultaneous pollutant interval before revealing that hour's targets. Full and empty prediction sets are used at the adaptive-level boundaries. The long-run theorem applies to the revealed score stream. Fixed no-feedback intervals calibrate before the outage and receive no hidden-target updates. A constant station scale cancels from an empirical quantile; varying availability scales therefore require direct comparison against unit scale. Width, coverage, full/empty sets, time variation and abstention are reported together.

Fault trials distinguish alarm, estimated count, exact support and localization conditional on knowing the count. The unknown-count classifier uses a training-fitted full-covariance-whitened syndrome, support-search features and separate validation segments for classifier fitting, ridge selection and clean alarm cutoff calibration. All fully observed candidate stations use the same universe. Synthetic faults include positive, negative, random-direction, unequal-size and ramped perturbations in transformed standardized coordinates. They are not verified real sensor faults or necessarily concentration-space two-sigma errors.
''')
    put('sections/05_theory_overview.tex',r'''
\section{Mathematical findings and their scope}
Full statements and proofs appear in Appendix~\ref{app:proofs}. They separate exact model properties from finite-precision computation and environmental validity.

\begin{enumerate}
\item \textbf{Actual structured-map rank.} For invertible selected endpoint blocks, spanning-tree transports and all remaining row constraints determine the exact kernel and outage rank. This resolves the applicability gap left by generic Hall matching. For the seven fitted networks, numerical kernel dimension is zero; exact consistency would then identify only the zero global section, not accurate nonzero pollution.
\item \textbf{Functional and fault robustness.} A chosen pollutant quantity has a sharp residual-ball interval exactly when its coefficient is in the operator row space. Unknown faulty stations produce a support union, so a quantity can sometimes be identified without localizing faults. Noisy exact-support selection requires competitor-specific omitted fault energy; a large first fault cannot protect a tiny second fault.
\item \textbf{Decision and repair bounds.} A finite interval has a minimax midpoint and a conditional projection safeguard. Exact Gaussian trace reductions and sharp worst-case functional repair widths solve specified single-repair model objectives. These facts do not imply submodularity, optimal multi-step greedy repair or better empirical concentration estimates.
\item \textbf{Computational certificates.} Bounded-fault LP envelopes are sharp under the stated model. A polynomial-size support-free dual is generally conservative; a strict-gap example is retained. Support-span compression is exact only for unrestricted nuisance coefficients with exactly equal spans. Anytime branching preserves numerical outer bounds; exact nominal rational residual correction gives a rigorously rounded bound for stored coefficients.
\item \textbf{Learned-map and joint-threshold extensions.} Declared perturbation radii inflate residual budgets through a top-station support function. A joint max-affine pollutant score has a sharp epigraph lower bound that can exceed the maximum of separate pollutant lower bounds. This may certify an exceedance without identifying which pollutant exceeds. Designed examples demonstrate the distinction; the learned-network synthetic experiment did not produce extra exceedance decisions.
\end{enumerate}

Adding equations cannot shrink a truly nested feasible-state envelope, but it can increase a regularized decoder's amplification or actual error. Likewise a mathematical certificate does not estimate whether its physical residual, bias and temporal-box assumptions hold. Floating-point SVD/LP decisions use stated tolerances; only the explicitly rational calculation is an exact certificate of the nominal stored matrix model.
''')
    mainrows=[]
    for region in ['delhi2025','bengaluru_raw']:
        for mode in ['causal','retrospective']:
            q=C[region][mode]
            mainrows.append([esc(LABEL[region]),mode,int(q['n']),int(q['clusters']),n(100*q['rmse_relative_gain']),
                             '['+n(q['difference_ci95'][0],3)+', '+n(q['difference_ci95'][1],3)+']'])
    result=r'\section{Reconstruction results}\subsection{Main concentration-cohort comparison}'+ '\n'
    result+=f"The main Delhi causal result is {gain:.1f}\\% lower normalized RMSE than the stronger full-covariance VAR control, on {d['n']} paired station-days across {d['clusters']} days. This comparison uses the same observed target cells and a prefix-only information set. It is the principal reconstruction finding, rather than the largest percentage in a different protocol.\n"
    result+=table(['Cohort','Mode','Cases','Days',r'Gain (\%)','MSE-difference CI'],mainrows,
        'Sheaf versus validation-tuned full-covariance VAR. Positive gain and positive reference-minus-sheaf MSE difference favor the sheaf. Day-cluster intervals are exploratory and not RMSE-gain intervals.','tab:main')
    result+=figure(R/'Results_Figure_Pack/01_causal_reconstruction.pdf',
        'Causal 24-hour whole-station reconstruction with 24-hour target history. Gains use common station-day mean normalized squared errors; bars are nominal paired-day ratio intervals. Delhi: 399 cases/58 days; Bengaluru: 171/60. The full-covariance VAR is separately tuned. Bengaluru does not beat seasonal persistence in this original component comparison.','fig:causal')
    result+=r'''
Across the seven eligible hourly cohorts, the causal sheaf has favorable point estimates against the tested diagonal-innovation VAR, static Gaussian and spatial sheaf controls. Archive cohorts and concentration cohorts remain separate, and the seven networks are not independent replications. The stronger full-covariance comparison is available for the two concentration cohorts. The static snapshot benchmark does not establish a universal spatial advantage over Gaussian conditioning.
'''
    result+=figure(R/'Results_Figure_Pack/02_all_networks.pdf',
        'Causal reconstruction across the seven hourly cohorts. Point estimates only; the corrected dependence-sensitive values are in the source tables. The archive cohorts contain inverted sub-indices. Excluded Mumbai/Kolkata cohorts and supplied wider-cohort evidence are described separately.','fig:networks')
    result+=r'\subsection{Dependence sensitivity, hybrid and neural controls}'+'\n'
    ss=S[(S.comparison=='Causal sheaf vs VAR')&(S.region=='delhi2025')]
    result+=f"The Delhi gain remains positive in all three calendar-cluster analyses. The wider ten-contrast sensitivity lower endpoints range from {ss.family_sensitivity_low.min():.1f}\\% to {ss.family_sensitivity_low.max():.1f}\\%. These are approximate exploratory bounds, not a finite-sample familywise guarantee. Removing each calendar week in turn preserves the numerical advantage.\n"
    result+=figure(R/'joint_decision_results/Additional_Results_Figures.pdf',
        r'Frozen comparative sensitivity under daily, seven-day and fourteen-day calendar bins. Positive favors the first named method. Dark intervals are nominal 95\%; pale intervals use wider ten-contrast percentile levels. Orange marks fewer than eight occupied bins. GRIN averages MSE across all three trained seeds, not ensemble predictions.','fig:sensitivity',page=2)
    hybrid=js(R/'validation_results/hybrid_summary.json')['comparisons']
    result+=f"The validation-selected hybrid reduced Bengaluru error by {hybrid['bengaluru_raw']['temporal_sheaf']['rmse_improvement_percent']:.1f}\\% relative to its sheaf component, but increased Delhi error by {abs(hybrid['delhi2025']['temporal_sheaf']['rmse_improvement_percent']):.1f}\\%. It is therefore not a universally better replacement. The simpler seasonal baseline is retained.\n"
    rows=[]
    for region in ['delhi2025','bengaluru_raw']:
        for seed in range(3):
            q=graph['comparisons'][region][str(seed)]['temporal_sheaf']
            rows.append([esc(LABEL[region]),seed,int(q['cases']),n(q['rmse_improvement_percent']),
                         '['+n(q['paired_day_ci'][0])+', '+n(q['paired_day_ci'][1])+']'])
    result+=table(['Cohort','Seed','Cases',r'GRIN gain (\%)','Nominal day CI'],rows,
        'Official bidirectional GRIN versus retrospective sheaf. Positive means GRIN has lower RMSE. All seeds are retained; these are project-specific training outcomes, not published benchmark scores.','tab:grin')
    result+=r'''
GRIN outperforms the retrospective sheaf in Delhi and underperforms it in Bengaluru across these seeds. These results rule out a claim of general state-of-the-art superiority. They do not invalidate the separately measured causal sheaf-versus-VAR result, because the information sets differ.
'''
    result+=figure(R/'validation_results/figures/06_official_grin.pdf',
        'Pinned official graph recurrent architecture on matched retrospective station-day outages. Seed-level comparisons retain all three seeds and validation-selected checkpoints. No causal GRIN superiority or reproduction of the original published benchmark is claimed.','fig:grin')
    result+=r'\subsection{Later-period validation and history stress}'+'\n'
    rows=[]
    for ref,label in [('full_covariance_var','Full-covariance VAR'),('seasonal','Seasonal persistence'),('city_mean','City mean')]:
        q=ext['comparison'][ref];rows.append([label,int(q['cases']),int(q['days']),n(q['rmse_improvement_percent']),
             '['+n(q['paired_day_ci'][0])+', '+n(q['paired_day_ci'][1])+']'])
    result+=table(['Comparator','Cases','Days',r'Sheaf gain (\%)','Nominal day CI'],rows,
        'PM2.5 prediction in the later four-station Delhi period. Each station-day is equally weighted using training PM2.5 scales. All methods receive PM-only context and matched pre-outage history.','tab:later')
    result+=r'''
The favorable gain against VAR provides useful temporal validation. The city-mean difference is uncertain. Only three or four longer calendar bins are occupied, so a narrow clustered interval cannot be interpreted as strong external confirmation. This remains a short partial-network study of existing instruments.

History and horizon stress transfer the selected 24-hour parameters unchanged to longer outages and zero-history conditions. In the checkpoint task, Delhi with 24-hour history retains gains against VAR at 72 hours; without target history it can lose to seasonal persistence. Bengaluru likewise can beat VAR while losing to seasonal persistence. Checkpoint-hour metrics and full station-day metrics are not interchangeable.
'''
    result+=figure(R/'validation_results/figures/05_new_period.pdf',
        'Later same-network period: PM-only inputs, four matched instruments, 74 station-days and 19 days. Bars are nominal paired-day ratio intervals. The gain against full-covariance VAR is positive; the city-mean interval crosses zero. This is temporal replication on existing instruments, not an independent sensor network.','fig:later-period')
    result+=figure(R/'Results_Figure_Pack/supplementary/horizon_history_stress.pdf',
        'Fixed-parameter history/horizon stress. Twelve selected starting days; only observed checkpoint cells are scored. This figure uses a different case metric from the main all-outage-hour comparison and retains zero-history and unfavorable seasonal comparisons.','fig:stress')
    put('sections/06_reconstruction.tex',result)
    unc=r'''
\section{Uncertainty, faulty sensors and monitoring thresholds}
\subsection{Intervals with and without target feedback}
Revealed-label adaptive calibration obtains near-nominal average error on the observed score stream, but it cannot update from an unobserved dark-station truth. The corrected boundary rules allow infinite full sets and empty sets. Actual calendar-quarter outcomes are separated from four equal-duration test segments, whose old field names misleadingly suggested calendar quarters. A mean of station worst-quarter values is not the lowest quarter.

The no-feedback comparison below hides test targets throughout prediction and interval construction. Its calibration uses the stated validation labels after model selection, so the empirical coverage is not an independent split-conformal guarantee. Mean widths are full PM2.5 interval widths in concentration units. High accuracy among decided cases must be read together with how often a decision is made.
'''
    rows=[]
    for region in ['delhi2025','bengaluru_raw']:
        for method,short in [('fixed_no_feedback','Fixed'),('availability_scaled_no_feedback','Availability-scaled')]:
            q=intervals[(intervals.region==region)&(intervals.missing_rate==0)&(intervals.method==method)].iloc[0]
            rows.append([esc(LABEL[region]),short,int(q.cases),n(100*q.coverage,2),n(q.mean_width),n(100*q.decided_fraction,2)])
    unc+=table(['Cohort','Interval','Cases',r'Coverage (\%)','Width',r'Decided (\%)'],rows,
        r'No-feedback PM2.5 intervals at zero added context missingness. Width is the mean full interval width in micrograms per cubic metre. Nominal calibration is 90\%; empirical coverage and decision yield differ by cohort.','tab:intervals')
    unc+=figure(R/'Results_Figure_Pack/05_no_feedback_uncertainty.pdf',
        r'No-feedback intervals: coverage and mean full concentration-space width at 0/20/50\% added context-cell missingness. Outcomes are dependent station-hour cases. Availability scaling is not consistently better than a unit-scale control.','fig:intervals')
    unc+=r'\subsection{Fault alarms, count and exact support}'+'\n'
    rows=[]
    for region in ['delhi2025','bengaluru_raw']:
        for count,direction,mag in [(0,'clean',0),(2,'positive',2),(2,'random',2),(2,'unequal',2),(2,'gradual',2)]:
            q=fs[(fs.region==region)&(fs.fault_count==count)&(fs.direction==direction)&(fs.magnitude==mag)&(fs.method=='calibrated_classifier')].iloc[0]
            rows.append([esc(LABEL[region]),direction,int(q.cases),n(100*q.alarm_rate,1),n(100*q.count_accuracy,1),n(100*q.exact_end_to_end,1)])
    unc+=table(['Cohort','Injection','Hours',r'Alarm (\%)',r'Count (\%)',r'Exact (\%)'],rows,
        'Unknown-count classifier at magnitude two, with the clean row shown separately. Exact means correct count/support and, for injected faults, an alarm. A correct no-alarm clean decision also counts as exact. Clean alarm rate is the false-alarm rate.','tab:faults')
    unc+=f"Delhi's positive two-fault injection gives {fault_gain:.1f}\\% exact end-to-end success in {int(fault.cases)} sampled hours. Unequal second faults and gradual perturbations are substantially harder. Bengaluru remains weak in several conditions. The classifier receives no true test fault count, but it is trained on synthetic validation fault families; it is not evidence of arbitrary real-fault robustness. Natural clean recordings may already contain unidentified physical faults.\n"
    unc+=figure(R/'validation_results/figures/02_unknown_count_faults.pdf',
        'Unknown-count injected-fault evaluation with matched candidate universes and clean states. Signed, random and unequal perturbations are retained. Magnitudes refer to transformed standardized coordinates; clean false alarms and exact end-to-end support recovery are distinct quantities.','fig:faults')
    unc+=r'''
The supplied known-count support-search experiment and the simpler penalty decoder are retained in the supplementary evidence. Given-count localization cannot be promoted to end-to-end alarm success. Separate contiguous twelve-hour ramps use fixed supports/directions and no label feedback; the classifier remains stateless and is not a trained change-point detector.

\subsection{Fixed operational cutoffs and episode detection}
The full-stream alert experiment freezes method-specific cutoffs on validation negatives before test scoring. Target outages last 24/72/168 hours; 0/20/50\% additional context channels remain absent throughout an outage, alongside natural gaps. Sheaf and city-mean predictions use the same eligible truth and context masks. PM2.5 above 60 micrograms per cubic metre is an illustrative hourly threshold, not an officially valid daily standard, AQI category or GRAP decision.
'''
    rows=[]
    for region in ['delhi2025','bengaluru_raw']:
        for method in ['sheaf','city_mean']:
            q=alert[(alert.region==region)&(alert.horizon==24)&(alert.missing_rate==0)&(alert.method==method)].iloc[0]
            rows.append([esc(LABEL[region]),esc(method.replace('_',' ')),int(q.cases),int(q.positive_cases),
                         n(100*q.sensitivity),n(100*q.false_positive_rate),n(100*q.episode_detection_rate)])
    unc+=table(['Cohort','Method','Cases','Positive','Sensitivity','FPR','Episodes'],rows,
        'Validation-frozen 24-hour alert cutoffs with no added context loss. Rates are percentages; episode rate is detection among positive episodes. Case denominators are station-hours, not independent environmental events.','tab:alerts')
    q=aci[(aci.region=='bengaluru_raw')&(aci.horizon==24)&(aci.missing_rate==0)].iloc[0]
    unc+=f"For Bengaluru, the sheaf sensitivity difference is {100*q.sensitivity_difference:.1f} percentage points, with a weekly paired interval [{100*q.sensitivity_ci_low:.1f}, {100*q.sensitivity_ci_high:.1f}]. It crosses zero. Thus the full-stream operational study does not establish a reliable sensitivity improvement despite its favorable point estimate. Episode delays are conditional on detection; missing truth breaks episodes, and boundary truncation is reported.\n"
    unc+=figure(R/'validation_results/figures/03_frozen_alerts.pdf',
        'Validation-frozen pollution alerts under persistent target/context outages. Sensitivity must be considered together with false alarms, episode coverage and weekly uncertainty. All tested durations/missingness settings are preserved; this is not actual GRAP invocation.','fig:alerts')
    unc+=r'''
\subsection{Threshold proxies, missing-cell estimates and environmental interpretation}
The descriptive empirical ROC study reports Bengaluru sensitivity 61.25\% versus 40.75\% at an interpolated common 5\% false-positive rate, based on 80 positive cases. This is a test-set operating-point comparison with a nominal paired-day interval, not the validation-frozen full-stream experiment. Its advantage disappears with 50\% added missingness. The two studies therefore answer different questions and are not competing estimates of the same deployed benefit.

Daily threshold and approximate AQI studies retain positive/negative counts, common eligible station-days and false alarms. Official AQI has pollutant-specific averaging and validity rules; transformed sub-indices and daily ozone averages cannot establish official AQI validity \citep{cpcbresources}. Actual GRAP decisions concern Delhi-NCR and invocation records/forecasts, not a station-level threshold alone \citep{caqm2025}. Mumbai threshold examples are illustrative proxies. Where no positive case exists, conditional missed-action rates are undefined rather than zero.

The supplied exploratory Delhi estimate found 151 of 1,980 station-days with at least half of pollutant-hour cells missing and 35 modeled Poor-or-worse estimates. This mask does not mean the entire station was dark for half its hours. These are model estimates at unobserved cells, not independently verified exposure or overlooked health harm. Cohort/protocol differences preclude pooling them with the later-period threshold study. The corrected pipeline preserves natural-gap estimates and repair-priority tables as exploratory diagnostics.
'''
    put('sections/07_monitoring.tex',unc)
    computation=r'''
\section{Conditional certificates, computation and maintenance}
The bounded-fault study evaluates a support union with bounded biases, componentwise residual budgets and optional causal-history boxes. Calibration uses an earlier part of the examined test archive; evaluation uses its later part. Representative exchangeability is a theorem condition, not a demonstrated fact of the pollution process. Comparing temporal constraints with a no-history model is nested only when both use the same joint residual budget.
'''
    cases=csv(R/'constrained_results/cases.csv')
    computation+=f"There are {len(cases):,} scored method/case rows and {cs['conditional_containment_checks']:,} checks for cases satisfying the declared assumptions. Rows from methods/scenarios sharing truth are not independent observations. Temporal constraints can narrow bounds but useful threshold decisions remain rare.\n"
    computation+=figure(R/'Results_Figure_Pack/07_temporal_bounds_and_abstention.pdf',
        'Conditional bounded-fault quantities with and without causal temporal constraints under the same joint residual budget. Coverage, width, inconsistency and decision yield are distinct. Calibration and evaluation are disjoint archive periods that had previously been examined.','fig:bounds')
    computation+=r'''
The support-free dual is computationally smaller but may give wider intervals than exhaustive support optimization. In a three-repeat synthetic benchmark with eighteen fault candidates and budget three, the reported median speedup is about 408-fold, with wider bounds. This is a runtime/tightness tradeoff on synthetic systems, not a measured field acceleration. The exact nominal rational checker corrects approximate dual equality using trusted state bounds and outward rounding; it checks stored coefficients rather than physical model error.
'''
    computation+=figure(R/'Results_Figure_Pack/08_computation_tightness_tradeoff.pdf',
        'Synthetic bounded-fault runtime and interval tightness. Three timing repeats per configuration; the support-free outer certificate is faster but generally wider than the sharp sparse union. Exact rational correction verifies a nominal bound, not optimizer optimality or physical validity.','fig:compute')
    computation+=r'''
Support-span compression reduces redundant synthetic nuisance families, but the fitted Delhi and Bengaluru examples provide no support-span reduction and can run slower. Equal-span equivalence applies to unrestricted nuisance coefficients; adding a componentwise bias box destroys that equivalence in general. Anytime branching narrows the bounded-fault convex relaxation toward exact support enumeration. The sixteen planted-state examples per network attain their exhaustive endpoints by the tested seven-node budget, an outcome of these examples rather than a universal seven-node theorem.
'''
    computation+=figure(R/'validation_results/figures/04_certificate_refinement.pdf',
        'Anytime LP refinement and declared map-uncertainty scenarios on frozen learned maps with synthetic feasible states. Numerical outer bounds and exact-support endpoints are separated. Perturbation radii are simulation assumptions, not estimated physical uncertainty.','fig:refinement')
    additional=((joint.joint_lower>1+1e-6)&(joint.independent_lower<=1+1e-6))
    assert int(additional[joint.region=='coupled_example'].sum())==50
    assert int(additional[joint.region!='coupled_example'].sum())==0
    computation+=r'''
\subsection{Joint pollutant decisions}
The joint-threshold certificate minimizes the maximum of affine pollutant scores over each allowed support. Fifty designed coupled examples produce an exceedance certificate that separate pollutant minima cannot establish. Neither of the twelve-case learned-map synthetic groups produces an additional exceedance decision. The theoretical distinction is therefore proved and demonstrated, but not established as a monitoring improvement. The scores are shifted standardized coordinates; they do not implement official raw-concentration AQI.
'''
    computation+=figure(R/'joint_decision_results/Additional_Results_Figures.pdf',
        'Joint versus maximum separate lower score. The diagonal means no extra lower-bound information. The designed coupled family demonstrates a strict improvement; Delhi and Bengaluru use planted synthetic states on actual learned maps. No additional exceedance certificates were obtained in those learned-map groups.','fig:joint',page=1)
    computation+=r'''
\subsection{Simulated repair selection}
Gaussian trace-reduction selection is model-optimal among the specified single-station candidates and improves empirical reconstruction against expected random repair in six of seven tested cohorts. It is essentially indistinguishable from a simpler variance-only policy and is worse than random in the Bengaluru archive cohort. The separate worst-case functional-width policy does not establish a mean-error advantage over random expected repair. Four inconsistent Delhi pairs are explicitly excluded from the feasible-pair comparison.

These outcomes show why a proof of optimality for one model objective does not prove empirical superiority in another metric. Transformed-coordinate averages are not physical concentration means after nonlinear transformation. No physical sensor repair or maintenance-cost reduction was measured.
'''
    computation+=figure(R/'Results_Figure_Pack/10_repair_policy_tradeoffs.pdf',
        'Simulated two-station outages followed by one repair. Worst-case functional-width selection is compared with Gaussian trace, variance and expected random repair. The objective and outcome metric differ; no consistent advantage of the advanced policy or physical intervention is established.','fig:repair')
    put('sections/08_certificates.tex',computation)
    put('sections/09_discussion.tex',r'''
\section{Discussion, environmental relevance and limitations}
The most useful empirical result is causal monitoring continuity during a Delhi station outage, supported by realistic classical comparators and temporal sensitivity. The later period provides partial same-instrument temporal validation. Modern retrospective comparison is mixed: a bidirectional neural model outperforms the sheaf in Delhi, so a general superiority claim would be incorrect. Spatial snapshot, hybrid, repair and meteorology studies also reveal failure regions.

The mathematical work explains what can be guaranteed in a declared inverse-problem model, which pollution quantities can remain identifiable without a full state or fault support, and how to compute conditional bounds. The empirical work tests whether that framework is useful under archive-derived residuals. Broad intervals and frequent abstention are scientifically relevant outcomes: exact optimization cannot make an inaccurate feasible-state assumption true or force useful decisions from weak information.

The environmental pathway is improved availability and reliability of monitoring information, potentially informing maintenance and pollution alerts. This study does not establish emission reductions, reduced personal exposure, improved health, successful official policy invocation or a field-ready alarm system. Such claims require independent instruments, validated intervention records and prospective outcomes. An estimated high concentration at an unobserved cell is not verified exposure.

Important limitations include archive reuse, source overlap, artificial missingness and fault directions, incomplete station-day eligibility, validation reuse after model selection, dependence beyond the chosen calendar blocks, small later-period coverage, approximate sub-index inversion and potential unit incompatibility. Different studies use different state transforms, residual norms, case units and target populations. Numerical rank depends on tolerance. SVD/LP theorem tests are not independent proof review, and no exhaustive check of every network outage or independent retraining of every historical model is claimed.

\subsection{Meteorology and broader exploratory studies}
Adding neighbouring temperature, humidity, wind, rainfall and solar-radiation channels in Bengaluru does not improve the matched reconstruction endpoint: its error is approximately 0.1\% worse. The one-hour PM2.5 forecast with meteorology is about 2.0\% worse than the matched weather-free ridge model. This is not a weather-prediction advance. The archive repair, geographic-correlation graph, thinning, risk-ranking and concentration-space ablations are retained in the appendices with their own units and eligibility.

\section{Conclusion}
SheafCode combines learned vector consistency relations with missing-data reconstruction and conditional pollutant-specific reliability analysis. The strongest evidence is lower causal outage error against a tuned full-covariance VAR filter in Delhi, with supportive but limited later-period validation. Results against stronger neural and simpler seasonal comparators are mixed. The conditional proofs and numerical implementations clarify identifiability, unknown-fault quantities, repair objectives and joint-threshold information. They support a monitoring research framework whose usefulness and limitations are explicitly measured; they do not establish first-ever mathematical priority or cleaner air.

\section{Reproducibility and assistance}
The companion package records source, data and result SHA-256 values, saved case tables, selected settings, complete neural seed outcomes, proof sources and vector figures. The final audit traces publication tables to their generating CSV/JSON files, verifies common-case arithmetic and checks the rendered document. A full independent rerun of every historical fitting procedure is outside this audit. Existing negative results and unsupported-claim corrections remain available in the evidence index.

This integrated document was drafted with AI assistance from the supplied research materials and completed local analyses. It is a research reference; this audit does not certify a student's authorship, independence or competition eligibility. Any use should reflect the researcher's own understanding and an accurate account of contributions and assistance. No identity, mentorship or institution is invented.
''')
                                                                            
    legacy=source(BASE/'Final_Manuscript/sections/09_results.tex').read_text()
    legacy=legacy.replace(r'\section{Results}',r'\section{Supplied snapshot and window evidence}\label{app:supplied}')
    legacy=legacy.replace('prespecified','designated').replace('preregistered','pre-ranked')
    for name in ['snapshot','channel','prolonged','horizons','certificates','risk_association','thinning','secondary']:
        p=source(BASE/f'Final_Manuscript/figures/{name}.pdf');shutil.copy2(p,O/f'figures/{name}.pdf')
        figures.append({'label':'supplied-'+name,'source':str(p.relative_to(BASE)),
                        'copied_sha256':sha(O/f'figures/{name}.pdf'),'caption':'Supplied audited protocol; original detailed caption retained','page':None})
    intro=r'''
This appendix preserves the supplied study's quantitative evidence and statistical units. It is supporting evidence, not the main causal hourly-cohort benchmark. Region A/B/C/D/E/F denote Delhi/NCR/Mumbai/Kolkata/Hyderabad/Bengaluru; National denotes the national archive; Raw 1/Raw 2 denote the supplied Delhi/Bengaluru concentration-source configurations. Station/channel membership differs from the strict hourly cohorts in Table~\ref{tab:cohorts}. In particular, the supplied four-channel Bengaluru configuration is not the later three-channel cohort. The supplied timeline/selection has acknowledged limitations, including adjacent retained rows not necessarily representing elapsed hours. Raw pollutant values in inverted-sub-index cohorts remain approximations. No original 34\% median retrospective result is pooled with the 23.2\% causal full-covariance comparison.

'''
    section=r'\section{Supplied snapshot and window evidence}\label{app:supplied}'
    legacy=legacy.replace(section,section+'\n'+intro,1)
    put('sections/11_supplied_results.tex',legacy)
    oldappendix=source(BASE/'Final_Manuscript/sections/15_appendix.tex').read_text()
    oldappendix=oldappendix.replace(r'\appendix','')
    oldappendix=oldappendix.replace('An online deployment would require','For this supplied window protocol, an online deployment would require')
    put('sections/12_supplied_tables.tex',oldappendix)
                                                                                  
    index=[
      ('Supplied snapshot, channel, prolonged, risk and thinning','Final_Manuscript/audit/numerical_audit.json','Appendix supplied results; nine cohort configurations','Saved tables audited; sampled raw replay documented'),
      ('Supplied graph topology, ablations, fault severity and decommissioning','Final_Manuscript/sections/09_results.tex','Appendix supplied results','Exploratory, subset/candidate-specific'),
      ('Corrected hourly cohort, snapshots and first prolonged study','Research_Release/docs/OVERNIGHT_REPORT.md','Data section and supporting tables','Seven eligible networks; two exclusions'),
      ('Expanded causal/retrospective archive comparison','Research_Release/expanded_docs/reviewed_comparisons.csv','Main reconstruction and full supplement','Point estimates independently recomputed'),
      ('Full-covariance VAR','Research_Release/extension_results/strong_baselines/summary.json','Main reconstruction','Source checks, prefix tests and arithmetic'),
      ('Horizon/history stress','Research_Release/docs/AMBITIOUS_TECHNICAL_VALIDATION.md','Main figure and appendix','Different checkpoint metric'),
      ('Generic/Hall and support-pattern rank','Research_Release/docs/CORRECTED_THEORY.md','Proof appendix and corrections','Generic idealization; not CCA general-position proof'),
      ('Actual-map holonomy and outage rank','Research_Release/extension_results/structured/summary.json','Theory and proof appendix','5,394 sampled/exhaustive small-family agreements'),
      ('Functional bounds and sharp repair','Research_Release/extension_results/functional/summary.json','Theory, repair and appendix','Conditional on residual model'),
      ('Prior-metric sensitivity and monotonicity counterexample','Research_Release/extensions/structured/THEORY_AND_NOVELTY.md','Proof appendix','Actual error need not improve with more rows'),
      ('Revealed-label intervals and actual calendar-quarter replay','Research_Release/docs/interval_replay_audit.json','Interval section and appendix','212 station/setting replay checks recorded'),
      ('No-feedback fixed/availability intervals','Research_Release/followup_results/interval_summary.csv','Interval table and figure','No hidden-target update; empirical coverage'),
      ('Early corrected unknown-count support-search faults','Research_Release/docs/fault_summary.csv','Fault section and appendix','Localization separated from detection'),
      ('Unknown-count whitened classifier and continuous ramps','Research_Release/validation_results/fault_summary.csv','Main fault table and full supplement','New summary independently aggregated'),
      ('Daily threshold, preceding-calendar-day and common cases','Research_Release/docs/environment_comparisons.csv','Monitoring section and appendix','Proxy, not official AQI or GRAP'),
      ('Supplied missing-cell exposure estimate','Final_Manuscript/NEW_MATERIAL_REVIEW.md','Monitoring limitations','151/1,980 and 35 estimates; not verified exposure'),
      ('Frozen hourly alert cutoffs','Research_Release/validation_results/alert_summary.csv','Main monitoring table and full supplement','24/72/168-hour persistent masks'),
      ('Matched empirical ROC operating points','Research_Release/constrained_results/matched_false_alarm_comparisons.csv','Monitoring section and appendix','Different endpoint, unadjusted exploratory intervals'),
      ('Gaussian model-based repair','Research_Release/docs/sensor_repair_summary.json','Repair and appendix','5,582 model-optimal cases; no advantage over variance'),
      ('Worst-case functional repair policy','Research_Release/docs/sensor_repair_trials.csv','Repair and appendix','No demonstrated mean-error advantage'),
      ('Meteorology reconstruction and one-hour forecast','Research_Release/results/weather/weather_summary.json','Discussion and appendix','Negative matched-comparator findings'),
      ('Graph-conditioned MLP','Research_Release/followup_results/neural_control/summary.json','Methods and appendix','Single-seed snapshot control; not GRIN'),
      ('Official bidirectional GRIN','Research_Release/validation_results/graph_summary.json','Main neural table and figure','Three seeds, retrospective only'),
      ('Validation-selected convex hybrid','Research_Release/validation_results/hybrid_summary.json','Main reconstruction and appendix','Bengaluru gain, Delhi degradation'),
      ('Unknown-support unrestricted quantity envelopes','Research_Release/extensions/followup/ROBUST_QUANTITY_THEOREM.md','Proof appendix and numerical supplement','Unboundedness/abstention retained'),
      ('Support-span compression','Research_Release/extensions/followup/SUBSPACE_COMPRESSION_THEOREM.md','Proof and computation section','Unrestricted faults; actual-map no acceleration'),
      ('Bounded-fault causal-box calibration','Research_Release/constrained_results/manifest.json','Conditional certificates and appendix','11,360 method rows; 10,512 conditional checks'),
      ('Support-free budget dual and strict gap','Research_Release/extensions/followup/CONSTRAINED_QUANTITY_THEORY.md','Proof and computation section','Conservative polynomial-size LP'),
      ('Exact rational correction','Research_Release/extensions/followup/VERIFIED_CERTIFICATE_THEOREM.md','Proof and computation section','Stored nominal coefficients; no physical guarantee'),
      ('Anytime support refinement','Research_Release/validation_results/refinement_cases.csv','Proof and computation section','Synthetic planted states on actual maps'),
      ('Uncertain-map inflation','Research_Release/extensions/validation/OPERATOR_UNCERTAINTY_THEOREM.md','Proof and computation section','Declared perturbation radii only'),
      ('Overlap and geographic membership','Research_Release/docs/overlap_with_coordinates.json','Data section and supplement','Shared sites not independent instruments'),
      ('Later-period public observations','Research_Release/validation_results/external_summary.json','Main later-period table','Short four-site PM-only period'),
      ('Joint max-affine thresholds','Research_Release/joint_decision_results/joint_cases.csv','Proof and computation section','Designed benefit; zero extra learned-map decisions'),
      ('Calendar sensitivity and neural seed mean loss','Research_Release/joint_decision_results/comparison_sensitivity.csv','Main sensitivity figure','Ten fixed contrasts, all block scales and seeds'),
      ('Earlier reports and supplied archives','Final_Manuscript/Complete_Research/audit/SOURCE_INVENTORY.csv','Companion evidence index','Unsupported/stale claims are not promoted')]
    pd.DataFrame(index,columns=['family','source','manuscript_location','verification_scope']).to_csv(O/'audit/EVIDENCE_COVERAGE.csv',index=False)
    maptext=r'\section{Evidence coverage and verification boundaries}\label{app:coverage}'+'\n'
    maptext+=r'The companion source inventory includes supplied archives, papers, audit reports, experiment scripts and saved tables. This evidence map covers every identified substantive study family. Coverage means representation and a stated verification scope; it does not mean every archived trial was independently replayed.'+'\n'
    maptext+=r'\begin{longtable}{p{.30\linewidth}p{.29\linewidth}p{.31\linewidth}}\hline Study family & Location & Scope \\ \hline\endfirsthead\hline Study family & Location & Scope \\ \hline\endhead'+'\n'
    maptext+='\n'.join(esc(a)+' & '+esc(c)+' & '+esc(d)+r' \\' for a,b,c,d in index)
    maptext+=r'\hline\end{longtable}'+'\n'
    maptext+=r'''
\subsection{Corrections that constrain the final claims}
The supplied thinning summary's claimed 294/294 prediction is not supported: the detailed review aggregates 293 predicted and 294 observed non-identifiable single stations, with 203 rank disagreements across all thinning configurations. Main unthinned Hall/rank agreement alone does not prove learned-map general position. The abstract's worst-quarter figure was a nested average rather than the minimum individual quarter. Both claims are replaced by accurately described quantities.

The earlier finite boundary substitutes do not inherit the arbitrary-sequence adaptive guarantee; the corrected full/empty convention is required. A fixed station scale cancels exactly. Two-station margins are not four-station margins for two-fault decoding, sampled connected-set minima are not global guarantees, and knowing the true fault count is not an operational alarm evaluation. The noiseless theorem assumes an exact clean model, which is restrictive for full-rank fitted sheaves.

The supplied AQI/GRAP and missing-exposure results are retained as approximate threshold/model-estimate evidence, with common eligible days, positive-case denominators and source limitations. Period overlap is not proof of duplicate records or independence. Model-source changes cannot be combined with old result numbers. Old pending-status prose is superseded by actual completed artifact status; it is not copied as a current limitation.

\subsection{Local audit scope}
The current recheck reran 89 core/extension tests and one pinned published-architecture integration test in separate invocations. The read-only saved-result audit checks frozen sources, data and outputs, comparison arithmetic, fault aggregation, conditional containment, syntax and archive integrity. Written proofs were internally checked for assumptions and counterexamples. Full independent model retraining, every historical trial replay, external peer review and comprehensive priority certification are not claimed. These boundaries remain explicit even after manuscript reconciliation.
'''
    put('sections/13_coverage.tex',maptext)
                                                                             
                                                                        
    supplement=r'\section{Complete numerical supplement by protocol}\label{app:numerics}'+'\n'
    fulltables=[
      (R/'expanded_docs/reviewed_comparisons.csv','Expanded cohort comparisons',['region','mode','reference','cases','days','rmse_gain']),
      (R/'docs/prolonged_comparisons.csv','Smaller sampled-day prolonged comparisons',None),
      (R/'followup_results/interval_summary.csv','No-feedback interval outcomes',['region','missing_rate','method','cases','coverage','mean_width','decided_fraction']),
      (R/'validation_results/fault_summary.csv','Unknown-count fault outcomes',['region','fault_count','direction','magnitude','method','cases','alarm_rate','exact_end_to_end']),
      (R/'validation_results/alert_summary.csv','Persistent-outage frozen alerts',['region','horizon','missing_rate','method','cases','positive_cases','sensitivity','false_positive_rate']),
      (R/'validation_results/alert_paired_week_intervals.csv','Paired weekly alert uncertainty',['region','horizon','missing_rate','weeks','sensitivity_difference','sensitivity_ci_low','sensitivity_ci_high']),
      (R/'constrained_results/summary.csv','Bounded-fault conditional outcomes',['region','horizon','scenario','method','cases','coverage','assumption_frequency','decided_fraction']),
      (R/'docs/environment_comparisons.csv','Daily threshold paired comparisons',None),
      (R/'docs/sensor_repair_trials.csv','Repair trial diagnostics',None),
      (R/'joint_decision_results/comparison_sensitivity.csv','Calendar-block comparison sensitivity',['comparison','region','cluster_days','clusters','gain_percent','nominal_95_low','nominal_95_high']),
      (R/'validation_results/refinement_summary.csv','Anytime refinement',['region','node_budget','cases','median_width','median_exact_width','completed_fraction']),
    ]
                                                                                 
    for path,title,cols in fulltables:
        if not path.exists():
            raise FileNotFoundError(path)
        frame=csv(path)
        shutil.copy2(path,O/'data'/(path.parent.name+'_'+path.name))
        supplement+=r'\subsection{'+esc(title)+'}'+'\n'
        if cols is None or len(frame)>220:
            supplement+=f"The complete table contains {len(frame):,} rows and {len(frame.columns)} columns and is included in the companion data directory. Its units and eligibility are those of the generating study; rows are not automatically independent replicates.\n"
            continue
        cols=[c for c in cols if c in frame]
                                                                          
        short={'full_covariance_penalized_ls':'LS','calibrated_classifier':'Classifier',
               'availability_scaled_no_feedback':'Scaled','fixed_no_feedback':'Fixed',
               'joint_budget_no_history':'Joint no-history','joint_temporal':'Joint temporal',
               'unrestricted_fault':'Unrestricted','bounded_fault':'Bounded',
               'temporal_sheaf':'Sheaf','full_covariance_var':'Full VAR',
               'seasonal_persistence':'Seasonal','spatial_sheaf':'Spatial',
               'stable_var':'Diagonal VAR','city_mean':'City mean',
               'delhi2025':'Delhi conc.','bengaluru_raw':'Bengaluru conc.',
               'hyderabad':'Hyderabad','bengaluru':'Bengaluru arch.','delhi':'Delhi arch.','ncr':'NCR','india':'National'}
        widths=[.94/len(cols)]*len(cols)
        supplement+=r'\begingroup\small\setlength{\tabcolsep}{2pt}\begin{longtable}{'+''.join('p{'+f'{w:.3f}'+r'\linewidth}' for w in widths)+r'}\hline'+'\n'
        supplement+=' & '.join(esc(c.replace('_',' ')) for c in cols)+r' \\ \hline\endfirsthead'+'\n'
        supplement+=' & '.join(esc(c.replace('_',' ')) for c in cols)+r' \\ \hline\endhead'+'\n'
        for _,row in frame.iterrows():
            vals=[]
            for col in cols:
                x=row[col]
                if isinstance(x,(int,np.integer)):s=str(x)
                elif isinstance(x,(float,np.floating)):s=n(x,3)
                else:s=short.get(str(x),str(x).replace('_',' '))
                vals.append(esc(s))
            supplement+=' & '.join(vals)+r' \\'+'\n'
        supplement+=r'\hline\end{longtable}\endgroup'+'\n'
    put('sections/14_numeric_supplement.tex',supplement)
                                                                                 
    proof=source(BASE/'audit_tools/complete_research_proofs.tex').read_text(encoding='utf-8')
    put('sections/10_proofs.tex',proof)
    pre=source(BASE/'Final_Manuscript/preamble.tex').read_text()
    pre=pre.replace(r'\usepackage{draftwatermark}','')
    pre='\n'.join(line for line in pre.splitlines() if not line.startswith(r'\SetWatermark'))
    pre=pre.replace('Reference exemplar - not for submission','Integrated research reference')
    pre+=r'''\DeclareMathSizes{10.95}{10.95}{8}{8}
\newtheorem{proposition}{Proposition}
\newtheorem{corollary}{Corollary}
'''
    put('preamble.tex',pre)
    bibliography=source(BASE/'Final_Manuscript/references.bib').read_text()
    extra=source(BASE/'audit_tools/complete_research_references.bib').read_text()
    put('references.bib',bibliography+'\n'+extra)
    main=r'''\documentclass[11pt]{article}
\input{preamble.tex}
\title{SheafCode: Causal Air-Quality Reconstruction and Conditional Pollution-Decision Certificates}
\author{Integrated research reference}
\date{6 October 2026}
\begin{document}\maketitle
\begin{center}\small This AI-assisted research reference integrates the supplied evidence and completed local analyses. Authorship, first-ever priority and competition eligibility are not certified.\end{center}
\input{sections/01_abstract.tex}
\tableofcontents\clearpage
\input{sections/02_context.tex}
\input{sections/03_data.tex}
\input{sections/04_method.tex}
\input{sections/05_theory_overview.tex}
\input{sections/06_reconstruction.tex}
\input{sections/07_monitoring.tex}
\input{sections/08_certificates.tex}
\input{sections/09_discussion.tex}
\clearpage\bibliography{references}
\clearpage\appendix
\input{sections/10_proofs.tex}
\clearpage\input{sections/11_supplied_results.tex}
\input{sections/12_supplied_tables.tex}
\clearpage\input{sections/13_coverage.tex}
\clearpage\input{sections/14_numeric_supplement.tex}
\end{document}
'''
    put('main.tex',main)
    (O/'audit/CLAIMS.json').write_text(json.dumps(claims,indent=2))
    (O/'audit/FIGURES.json').write_text(json.dumps(figures,indent=2))
    (O/'audit/TABLES.json').write_text(json.dumps(tables,indent=2))
    (O/'audit/BUILD_INPUTS.json').write_text(json.dumps(sources,indent=2))
    print('Manuscript sources built:',len(figures),'figures;',len(sources),'source files;',len(abstracts.split()),'abstract words',flush=True)


def compile_pdf():
    env=os.environ.copy();env['TECTONIC_CACHE_DIR']=str(BASE/'audit_tools/tex_cache')
    compiler=BASE/'audit_tools/tectonic/tectonic.exe'
    result=subprocess.run([str(compiler),'--only-cached','--keep-logs','--outdir',str(O),str(O/'main.tex')],cwd=O,env=env,capture_output=True,text=True)
    (O/'audit/COMPILE_LOG.txt').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError('TeX compilation failed; see audit/COMPILE_LOG.txt')
    shutil.copy2(O/'main.pdf',BASE/'Final_Manuscript/SheafCode_Complete_Manuscript.pdf')
    print('Compiled integrated manuscript:',O/'main.pdf',flush=True)


if __name__=='__main__':build();compile_pdf()
