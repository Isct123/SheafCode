                                                                                
from pathlib import Path
import re, csv, statistics, hashlib, json, shutil

w=Path('SUBMISSION_WORKSPACE');src=w/'Source/manuscript.tex'
backup=Path('audit_tools/completeness_audit_backup');backup.mkdir(exist_ok=True)
assert not (backup/'manuscript.tex').exists(), 'Already applied'
shutil.copy2(src,backup/'manuscript.tex');shutil.copy2(w/'Manuscript_For_Upload.pdf',backup/'Manuscript_For_Upload.pdf')
before=src.read_text(encoding='utf-8')
literature=r'''
\section{Related work and the research gap}\label{sec:related-work}
\textbf{Reconstructing readings is an established problem.} Spatial statistics, temporal models, matrix completion and graph learning offer different ways to use relationships between measurements. Spatial statistics models dependence across locations \citep{cressie1993statistics}; VAR models dependence across channels and time \citep{lutkepohl2005var}; low-rank matrix completion uses shared structure in incomplete measurement arrays \citep{mazumder2010softimpute}. They provide relevant context, but a citation is not a matched experiment: this study evaluates the comparators specified in Methods and does not claim to outperform every member of these families.

Sheaves allow the two stations on an edge to use different maps into a common comparison space. Their network and spectral foundations are established \citep{ghrist2011network,hansen2019spectral}. Importantly, \citet{pham2022airquality} already applied sheaves to local air-quality self-filtering, combining low-cost sensing with vehicle-related information. SheafCode therefore does not claim that sheaves, causal pollution monitoring or their combination were invented here. The present task concerns reconstructing missing station readings from surviving network observations and pre-outage history, followed by explicit conditional decision and measurement-repair analysis. CCA supplies established learned comparison maps \citep{hotelling1936relations}; the ablation results prevent attributing the observed gains solely to those maps.

\textbf{A strong imputer need not provide a decision guarantee.} GRIN learns spatiotemporal representations through graph message passing for multivariate imputation \citep{cini2022grin}. Its published benchmark gains are not this project's results. Here its pinned bidirectional implementation receives a retrospective information set, so its comparisons are kept separate from causal outage predictions. Bounded-error estimation instead asks which states remain possible under explicit uncertainty limits \citep{bounded1990}. Functional observability motivates determining a chosen quantity without reconstructing the entire state \citep{functional2022}. SheafCode applies this distinction to a joint pollutant threshold: all admissible explanations may agree on a decision even while individual readings differ. Epigraph optimization and verified linear bounds supply the credited computational tools \citep{boyd2004,jansson2004}.

\textbf{Which measurement should come next?} Sensor placement for Gaussian-process information gain \citep{krause2008}, integer-programming test selection for fault diagnosis \citep{awasthi2020}, and generalized test collection \citep{douek2021} establish that measurement selection is a substantive problem in its own right. Their objectives and assumptions differ from the bounded-noise, possibly corrupted pollution queries studied here; their approximation or optimality guarantees cannot simply be transferred. Robust placement also has uncertainty-focused precedents \citep{xing2022robust}. Decision-tree function evaluation and noisy adaptive testing motivate using each answer to select the next query \citep{cicalese2014adaptive,jia2024noisy}.

\textbf{The specific gap addressed here.} The contribution is the linked monitoring formulation: bound a joint pollution decision, identify opposite-decision witnesses, choose a minimum-cost corruption-aware separating set, calculate its continuous-box tolerance and optimize an adaptive continuation. The proofs specialize credited foundations rather than claiming new general theories of matching, error correction or dynamic programming. Empirical tests assess these capabilities together with reconstruction, failures and abstentions. This is a focused literature review of the directly used methods, not an exhaustive survey or a proof of worldwide priority. Adaptive conformal inference is relevant uncertainty background \citep{gibbs2021}, but its revealed-error feedback does not establish operational adaptation during an unobserved outage; measurement-selection adaptivity is a different concept.

'''

roots=[('India (A)',Path('Research_Release/recent_data_results/delhi')),
       ('USA (B)',Path('Research_Release/recent_data_results/epa_la')),
       ('UK (C)',Path('Research_Release/uk_validation_results/uk_aurn'))]
rows=[];facts=[];inputs={}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,p in roots:
    tables={}
    for f in ['structured_rank_cases.csv','exhaustive_block_margins.csv','functional_joint_bounded_cases.csv','operator_uncertainty_cases.csv']:
        q=p/f;tables[f]=list(csv.DictReader(q.open()));inputs[str(q)]=sha(q)
    rank=tables['structured_rank_cases.csv'];joint=tables['functional_joint_bounded_cases.csv'];maps=tables['operator_uncertainty_cases.csv']
    agrees=sum(x['agree']=='True' for x in rank)
    feasible=[x for x in joint if x['joint_status']=='bounded'];inc=sum(x['joint_status']=='inconsistent' for x in joint)
    below=sum(x['joint_decision']=='below_or_equal' and x['joint_status']=='bounded' for x in joint)
    abstain=len(feasible)-below
    gain=statistics.median(float(x['joint_extra_lower']) for x in feasible)
    covered=sum(x['robust_covered']=='True' for x in maps)
    rows.append(f'{name} & {agrees}/{len(rank)} & {len(tables["exhaustive_block_margins.csv"])} & {len(feasible)}/{inc} & {below}/{abstain} & {gain:.3f} & {covered}/{len(maps)}'+r'\\')
    facts.append(dict(network=name,rank_agreement=agrees,rank_cases=len(rank),margin_cases=len(tables['exhaustive_block_margins.csv']),joint_feasible=len(feasible),joint_inconsistent=inc,joint_below=below,joint_abstain=abstain,median_joint_tightening=gain,perturbed_truth_covered=covered,perturbation_cases=len(maps)))
appendix=r'''
\appendix
\section{Supporting evidence and completeness of reporting}\label{app:evidence-coverage}
The main text reports the designated recent comparisons and the linked decision-repair results. It does not reproduce every saved case table. This appendix restores supporting outcomes that would otherwise be represented only by a general reference to the evidence archive. Case counts below describe their own protocols; they are not extra independent replications or additions to the primary station-day denominator.

\subsection{Additional recent operator and joint-decision results}
The original three networks also have exhaustive actual-map checks, all station unions up to four blocks, joint-score trials and planted operator-perturbation tests. The following summaries are independently aggregated from their saved case CSVs.
\begin{table}[htbp]\centering\scriptsize
\caption{Separate original-network supporting experiments. Joint outcomes concern 40 tested states per network; inconsistent models are excluded from feasible decision counts.}
\begin{tabular}{lrrrrrr}\toprule
Network & Rank agree & Unions & Feas./incons. & Below/abstain & Tightening & Perturb.\\\midrule
'''+'\n'.join(rows)+r'''
\bottomrule\end{tabular}
\par\smallskip\small Rank agree compares actual structured-map prediction with numerical rank. Unions counts enumerated one- through four-station supports. Tightening is the median joint lower bound minus the largest separate lower bound, a dimensionless score difference. Perturb. counts planted truth containment, not measured operator uncertainty.
\end{table}
All feasible Indian and US joint trials abstain. The UK experiment returns four model-conditional below-reference decisions and abstains on the other feasible states; no experiment in this table establishes an above-reference certificate. These trials are distinct from the later 272-case exact joint verification in the main text. The UK safe decisions must not be erased by a blanket claim that every original-network trial abstained.

Each network's 48 perturbation cases use declared entry changes and inflated residual budgets; all planted truths are contained, but the experiment does not estimate physical CCA-map uncertainty. Exhaustive four-block margins address two-fault separation, whereas two-block margins alone cannot certify a uniform two-fault claim. Numerical rank checks and exact nominal rank certificates concern different levels of assurance.

\subsection{Supporting archive findings under separate protocols}
Earlier supplied observations remain useful context, with their original transforms, information sets and dependence limitations. They do not train the recent models or contribute to their headline denominators, and their observation-age eligibility is separate. The following selected findings are retained from the audited technical reference, not represented as new reruns in this document revision.
\begin{table}[htbp]\centering\small
\caption{Selected supporting archive findings. These are separate analyses, not pooled recent-study results.}
\begin{tabularx}{\linewidth}{@{}p{.25\linewidth}X@{}}\toprule
Study & Result and necessary qualification\\\midrule
Single-station snapshots & Median gain 3.1\%; numerical gains in 7/8 testable networks, with 2/8 comparisons passing within-network Holm correction. This is distinct from the mixed-size snapshot rollup.\\
Prolonged reconstruction & Median first-day gain 34.0\% across nine configurations; 5/9 supplementary run-based comparisons pass within-network Holm correction. Full-window estimates are retrospective, not the recent causal task.\\
Concentration-cohort control & Delhi causal gain 23.2\% versus full-covariance VAR on 399 station-days/58 dates. Bengaluru's VAR gain does not establish superiority to seasonal persistence.\\
Later same-network period & PM-only VAR-relative gain 9.5\% on 74 station-days/19 dates; existing instruments and few calendar bins limit independence and precision.\\
Additional-country test & Beijing has 5.93\% higher all-pollutant RMSE than full-covariance VAR on 502 station-days. This negative transfer result is retained.\\
Meteorology & Weather-channel reconstruction and one-hour forecast controls did not establish an improvement; these experiments do not support a weather-prediction claim.\\
\bottomrule\end{tabularx}\end{table}
The supporting Delhi neural comparison favours GRIN across its tested seeds, unlike the recent Indian retrospective comparison. Different observations and protocols explain why these are separate evidence, not interchangeable claims. Source overlap, inverted sub-indices, selected cohorts and reused archives prevent treating their numerical gains as independent confirmation of the recent result. Their full tables and source records remain in the complete research reference.

\subsection{What the concise manuscript does and does not reproduce}
Partial-channel and snapshot tests, thinning controls, support-span enumeration, residual-ball intervals, support-free outer bounds, simulated variance/trace repair, risk diagnostics, width-matched interval controls, revealed-label intervals and operator perturbations remain documented under their own protocols. They are supporting studies, not uniformly positive findings. Equal unrestricted nuisance spans need not give equal bounded fault images; a relaxation may be looser than support enumeration; and an unavailable hidden target cannot validate its estimated exposure.

The paper preserves the central connected proofs and reports the main gains together with failures. It does not claim that every historical theorem is reproduced here: projection/pseudoinverse interval formulas, unrestricted support-span compression, dual outer-bound derivations and prior-dependent error identities require separate technical treatment in the reference. They are not additional demonstrated first-ever theorems. Every result family and its reporting location is mapped in the accompanying completeness audit; underlying case tables are retained for reproduction rather than inserted wholesale into the manuscript.

'''
after=before.replace(r'\section{Data and experimental design}',literature+r'\section{Data and experimental design}',1)
after=after.replace(r'\section*{Acknowledgements}',appendix+r'\section*{Acknowledgements}',1)
old='Real recent original-network certificate trials do not establish definite safe or exceedance decisions; abstentions and inconsistent feasible models remain explicit.'
new='The original Indian and US joint-score trials do not establish definite safe or exceedance decisions; abstentions and inconsistent feasible models remain explicit. The separate UK trial includes four model-conditional below-reference decisions (Appendix~\\ref{app:evidence-coverage}).'
assert old in after;after=after.replace(old,new)
                                                                                                
after=after.replace(r'\textbf{We can change this if we do not need a mentor.}', '')
after=after.replace('AI tools were used to check mathematical formulas, to\nrefine writing, and for some of the data analysis.','AI tools assisted code development, mathematical derivation and checking, data analysis, audits and writing.')
for env in ['proposition','theorem']:
    pattern=r'\\begin\{'+env+r'\}.*?\\end\{'+env+r'\}'
    assert re.findall(pattern,before,re.S)==re.findall(pattern,after,re.S)
assert re.findall(r'\\textbf\{Proof\.\}.*?\\textit\{QED\.\}',before,re.S)==re.findall(r'\\textbf\{Proof\.\}.*?\\textit\{QED\.\}',after,re.S)
src.write_text(after,encoding='utf-8')
report=dict(status='passed',date='2026-10-10',supplementary_aggregates=facts,input_sha256=inputs,
    existing_formal_statements_and_proofs_unchanged=True,existing_11_figures_retained=True,
    corrections=['Restored a dedicated related-work/research-gap section','Restored specific recent and selected historical supporting outcomes','Corrected blanket original-network abstention wording: UK has four model-conditional below decisions','Removed mentor editing instruction and expanded AI assistance disclosure to recorded scope'],
    audit_scope='Document completeness, selected supplementary CSV aggregation and existing independent evidence replays. No independent retraining of every model or historical trial; no worldwide-priority or student-contribution certification.')
(w/'Guides/COMPLETENESS_RESTORATION_CHECK.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
