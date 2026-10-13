                                                                                      
from pathlib import Path
import re
import json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'SUBMISSION_WORKSPACE/Final_IRIS/Natural_Readability_Final'
OUT = ROOT / 'SUBMISSION_WORKSPACE/Final_IRIS/Concise_Readable_Final'
s = (BASE / 'SheafCode_IRIS_Final.tex').read_text(encoding='utf-8')
original = s

def between(start, end, replacement):
    global s
    assert s.count(start) == 1, start
    a = s.index(start)
    b = s.index(end, a)
    s = s[:a] + replacement.strip() + '\n\n' + s[b:]

def replace(old, new):
    global s
    assert s.count(old) == 1, old[:100]
    s = s.replace(old, new)

between('Air-quality stations measure pollutants', r'\section{Related work', r'''
Air-quality stations measure particulate matter, nitrogen dioxide (NO2) and ozone; their readings inform reporting and emergency responses \citep{caqm2025}. PM2.5 and PM10 denote particles with diameters up to 2.5 and 10 micrometres. When a station fails, a city average may miss local differences. We ask three questions:
\begin{enumerate}
\setlength{\itemsep}{.15em}\setlength{\parskip}{0pt}
\item \textbf{Reconstruction:} can surviving readings and the station's earlier history estimate its missing concentrations during an outage?
\item \textbf{Certification:} does every state allowed by the observations and error limits give the same threshold decision?
\item \textbf{Measurement planning:} which extra readings distinguish opposite decisions despite noise or corrupted responses?
\end{enumerate}

For hour $t$, a \emph{causal} estimate uses only observations available through $t$. This describes its information set, not causal inference about pollution sources. A \emph{retrospective} estimate also uses later observations.

We represent stations as vertices of a graph and link pairs whose readings we compare. A pollutant vector lists one station's concentrations. A \emph{sheaf} adds rules for comparing transformed vectors, allowing related stations to report different raw concentrations.

An estimate alone cannot certify a decision. Allowed PM2.5 readings of 30 and 40 micrograms per cubic metre ($\mu$g/m$^3$) give opposite answers at an illustrative reference of 35. A \emph{certificate} holds for every state allowed by the declared assumptions; otherwise we report uncertainty. Here ``safe'' means non-exceedance of that reference, not absence of health risk.

Our contributions are a causal reconstruction implementation, joint threshold certificates, and fixed and adaptive measurement plans with a tolerance for changes in archived scenarios. States giving the same decision need not be distinguished. The Indian benchmark is the primary reconstruction comparison; external cohorts test transfer (Section~\ref{sec:results}). These stages use separate experimental protocols and have not been tested together as a deployed system.
''')

between('Spatial statistics describes relationships', r'\section{Methodology}', r'''
Spatial statistics models relationships between locations. Vector autoregression (VAR) predicts several channels from earlier readings; matrix completion fills gaps using a small number of shared patterns, called low-rank structure \citep{cressie1993statistics,lutkepohl2005var,mazumder2010softimpute}. GRIN learns relationships by passing information along graph edges \citep{cini2022grin}. Its tested bidirectional implementation uses observations before and after the gap, so we compare it with retrospective sheaf. These controls represent specified implementations, not every method in their families.

Sheaf network and spectral methods are established \citep{ghrist2011network,hansen2019spectral}; previous air-quality work used sheaves for local self-filtering with low-cost sensors and vehicle-related information \citep{pham2022airquality}. Canonical correlation analysis (CCA) finds correlated linear combinations of two sets of readings \citep{hotelling1936relations}. Our \emph{ablation}, which changes one method component at a time, checks whether reconstruction gains depend on CCA rather than direct comparisons using identity maps.

Bounded-error estimation retains all states allowed by error limits; functional observability asks whether a chosen quantity can be determined without recovering the whole state \citep{bounded1990,functional2022}. Sensor placement and test selection distinguish possibilities, while adaptive testing chooses later queries from earlier answers \citep{krause2008,awasthi2020,douek2021,xing2022robust,cicalese2014adaptive,jia2024noisy}. We specialize these ideas to joint pollution decisions, corrupted measurements and continuous variation around archived scenarios.

The analysis uses established tools: epigraph optimization bounds the largest score with an extra variable, verified inequalities check numerical bounds, and branch and bound divides fault possibilities into smaller searches \citep{boyd2004,jansson2004,land1960}. Adaptive conformal inference adjusts intervals using revealed prediction errors \citep{gibbs2021}; a silent station supplies no such labels during its outage. This focused review makes no exhaustive priority claim.
''')

between('Training data fit the comparison maps.', 'In sheaf terminology', r'''
Training fits the graph and maps; a later selection period chooses settings such as the temporal weight (Section~\ref{sec:experiments}). We build comparisons in three steps:
\begin{enumerate}
\setlength{\itemsep}{.15em}\setlength{\parskip}{0pt}
\item \textbf{Choose neighbours.} Average each pair's absolute training correlations across channels. Each station selects its three strongest available neighbours, or all if fewer; either endpoint's choice creates an edge.
\item \textbf{Fit spatial maps.} CCA gives correlated directions greater weight. With fewer than 200 complete paired vectors, identity maps compare normalized readings directly.
\item \textbf{Add temporal comparisons.} Link each station's readings one and twenty-four hours apart, with a selection-chosen weight. Missing entries remain unknown variables.
\end{enumerate}
''')
replace('This is why we call the model a sheaf. The graph expresses relationships between measurements, not physical routes along which pollution must travel.', 'The graph describes measurement relationships, not physical pollution routes.')

between('The decoder selects the state with the smallest objective value.', r'\section{Mathematical foundations', r'''
Reconstruction minimizes an objective; certification checks agreement across all allowed states. Low average error alone cannot prove a particular alert.

A \emph{query} obtains a genuinely new reading, such as an independently checked measurement. Reusing a recorded number adds no information, and a failed station may not answer. Our simulations reveal archived readings; query savings count measurements, not field expenditure.
''')
between('The methodology gives a best estimate', r'\subsection{What does one observed station anchor?}', r'''
We first examine exact recoverability, then certify decisions under uncertainty and select measurements to resolve disagreement. Each guarantee depends on its stated assumptions.
''')
between('Imagine three stations connected in a chain', 'For each edge, suppose', r'''
In a chain A--B--C, exact invertible comparison rules let an observation at A determine B and C. The observed station \emph{anchors} its connected component. If the whole component is silent, ambiguity may remain.
''')
replace('A cycle can restrict those directions, so knowing that the graph is connected is not enough to determine $q_C$. Here homogeneous comparisons mean zero discrepancy on every edge.', 'Cycles can restrict $q_C$. Homogeneous comparisons mean zero discrepancy on every edge.')
between('With identity comparison maps, the rule simply says', r'\subsection{Can a decision survive', r'''
Identity maps require equal vectors: one observation fixes a component, while a wholly silent component leaves $p$ directions free. Other cycle constraints may force $q_C=0$. Singular maps and partly observed vectors require different analysis. The result follows spanning-tree elimination \citep{ghrist2011network,hansen2019spectral}; graph matching alone cannot establish rank---the number of independent columns---when coefficients have algebraic relationships \citep{hall1935}. Exact uniqueness does not imply accuracy in noisy data.
''')
between('The previous proposition describes a decision that holds', r'\par\noindent\begin{minipage}', r'''
Numerical optimization may return only approximate coefficients. Known variable ranges let us bound the leftover terms instead of ignoring them. Here $x$ has $d$ optimization variables; $a_{ij},b_i$ describe $m$ inequalities and $\ell_j,u_j$ the variable limits. The inequality weights $\lambda_i$ are separate from the decoder weight $\lambda$.
''')
between('A certificate may also require checking many possible groups', 'To bound a linear score', r'''
Searching many fault supports can take time. At any interruption, we retain a verified feasible score and an upper bound for unexplored cases.
''')
between('The earlier results tell us when the available information', 'Let $Z_-$ and $Z_+$', r'''
When both decisions remain possible, a query plan must separate opposite states. It need not identify every distinct pollution vector.
''')
between("Our query plan so far distinguishes states", 'A small example shows why the radius matters.', r'''
Future readings may differ from archived examples. For a scalar scenario $a_i$, allow independent changes of at most $\rho$ at every station:
\[
\|x-a_i\|_\infty\leq\rho.
\]
The \emph{stability radius} limits this scenario change, whereas $\eta_v$ limits measurement error.
''')
between('The stability theorem assumes we choose our full set', 'For a finite scalar scenario bank', r'''
A fixed plan chooses every query in advance. An adaptive plan uses each answer to choose the next station and stops when only one decision remains.
''')
replace('This illustrates the mechanism, not a measured network. The implemented recursion queries each station once and uses a declared search limit. Its search can grow exponentially: adding stations can multiply the number of states it must examine', 'This is illustrative. The implementation queries each station once and has a declared search limit; its search can grow exponentially as stations are added')

between('The mathematics above says what is possible', r'\subsection{Data, units and chronological roles}', r'''
We test reconstruction accuracy and the decision procedures separately, using chronological periods to keep future observations out of causal inputs.
''')
between('A \\emph{cohort} is a group of stations', 'India uses local time', r'''
A \emph{cohort} is a group of stations evaluated together; a \emph{role} adds a fixed chronological split. Each calendar has four successive periods: training fits the model, selection chooses hyperparameters (settings such as regularization weights), calibration sets interval and alert cutoffs, and testing measures performance.

Table~\ref{tab:units} lists native units: ppb and ppm mean parts per billion and million. Normalization rescales channels for fitting without converting their physical units.
\begin{table}[htbp]\centering\small
\caption{Native pollutant units in the original networks. A dash marks an unused channel.}\label{tab:units}
\begin{tabular}{lllll}\toprule
Network & PM2.5 & PM10 & Ozone & NO2\\\midrule
India & $\mu$g/m$^3$ & $\mu$g/m$^3$ & $\mu$g/m$^3$ & ppb\\
US & $\mu$g/m$^3$ & --- & ppm & ppb\\
UK & $\mu$g/m$^3$ & $\mu$g/m$^3$ & $\mu$g/m$^3$ & ---\\\bottomrule
\end{tabular}\end{table}
''')

between('Controls are causal VAR', 'Root mean squared error (RMSE)', r'''
The main control is causal VAR with cross-channel covariance. Simpler controls use the city mean, previous daily readings (seasonal persistence), learned covariance (Gaussian conditioning), or spatial sheaf alone. Table~\ref{tab:model-settings} gives selection settings. Ridge penalizes large fitted coefficients; covariance shrinkage reduces estimated cross-channel dependence.

A random forest (RF) averages decision-tree predictions. Its causal inputs are current surviving values, one- and twenty-four-hour lags, the last actual target history, missing-data indicators and calendar/station indicators. GRIN uses later context and is tested retrospectively. For each provider, its external comparison uses the first alphabetically sorted eligible cohort, requiring ten training and three selection cases. Finite tuning budgets do not establish globally optimal performance.
\begin{table}[htbp]\centering\small
\caption{Model settings. Selection data choose among alternatives; an epoch is one pass through training data.}\label{tab:model-settings}
\begin{tabularx}{\linewidth}{@{}p{.16\linewidth}X@{}}\toprule
Method & Settings\\\midrule
Sheaf & Decoder weight $\lambda$: 0.1 or 1; temporal scale: 0.5 or 2.\\\addlinespace
VAR \citep{lutkepohl2005var} & Ridge: 1 or 100; noise scale: 0.05 or 0.5; covariance shrinkage: 0, 0.2 or 1.\\\addlinespace
RF \citep{breiman2001} & 64 trees; depth at most 12; minimum leaf size 5 or 20 training examples; at most 80 sampled training days.\\\addlinespace
GRIN \citep{cini2022grin} & Three random seeds; hidden-layer width 32 or 64; at most 30 epochs; stop after seven epochs without selection improvement.\\\bottomrule
\end{tabularx}\end{table}
''')
replace('It gives larger errors more weight, and lower values mean better reconstruction.', 'Lower RMSE means better reconstruction; squaring gives larger errors more weight.')

between('We report reconstruction performance first', r'\subsection{Does reconstruction improve, and where?}', '')
replace('A method that performs best during selection may therefore lose that advantage later.', 'Selection preference does not always transfer.')
between('Retrospective sheaf gains against GRIN in India are', r'\begin{table}', r'''
Table~\ref{tab:grin} retains every GRIN seed and interval. Retrospective sheaf wins in India, US D1 and UK E1; GRIN wins in Illawarra F1 on all three seeds. Both methods receive later context, so this does not establish performance against a causal neural method during a live outage.
''')
replace('Thus CCA superiority is not established. A selection-chosen hybrid gains 1.66\\% over the primary sheaf setting, without replacing it. Temporal context helps here; learned spatial maps are not the sole explanation.', 'A selection-chosen hybrid gains 1.66\\% over the primary sheaf setting. Temporal context helps here, but CCA superiority is not established; neither alternative replaces the primary result.')
replace('The definitions of extreme readings and predicted-risk groups are those in Section~\\ref{sec:experiments}.', '')
replace('The intervals remain least reliable during unusually high pollution.', '')
replace('The guard issued no wrong certificates in the tested trials and saved 17.24\\% of measurements without corruption. With one corruption, however, only 20.53\\% of trials receive certificates. Savings include trials ending in abstention.', 'With one corruption, only 20.53\\% of trials receive certificates. Table~\\ref{tab:chronological-guard} includes measurements spent on abstentions in its savings.')
between('The theorems rely on exact conditions', r'\begin{table}', r'''
Table~\ref{tab:verification-summary} summarizes checks against independent computations. A matrix's kernel contains vectors it maps to zero; modular checks use arithmetic modulo primes to certify rank. These checks test implementation correctness, not physical sensor assumptions.
''')
replace('These are different endpoints. The other checks support the algorithms\' internal correctness. They do not establish physical error bounds, efficient scaling to large networks, or independently reproduced performance.', 'These are separate endpoints; neither check establishes physical error bounds, efficient scaling or independently reproduced performance.')

between('Earlier experiments used different datasets', r'\subsection{What environmental claim', r'''
Table~\ref{tab:historical-results} retains earlier findings under different data and information rules. These studies may overlap accessed archives and are not pooled with the recent benchmark. Holm correction accounts for testing several comparisons.
\begin{table}[htbp]\centering\small
\caption{Earlier studies under their own protocols, including negative findings.}\label{tab:historical-results}
\begin{tabularx}{\linewidth}{@{}p{.24\linewidth}X@{}}\toprule
Study & Result and qualification\\\midrule
Historical snapshots & Median gain 3.1\% over dense Gaussian conditioning; 7/8 favour sheaf, 2/8 pass within-network Holm correction.\\\addlinespace
First-day retrospective & Median gain 34.0\% over the implemented Kalman/VAR smoother across nine configurations; 5/9 supplementary run-based comparisons pass Holm.\\\addlinespace
Delhi causal & Gain 23.2\% over full-covariance VAR on 399 station-days and 58 dates.\\\addlinespace
Later same network & PM-only gain 9.5\% on 74 station-days and 19 dates; not an independent replication.\\\addlinespace
Beijing & All-pollutant RMSE 5.93\% higher than VAR on 502 station-days.\\\addlinespace
Neural and weather & Historical Delhi favours GRIN on every seed; weather controls establish no improvement.\\\bottomrule
\end{tabularx}\end{table}
''')
replace('The alert results show a tradeoff between finding high-pollution hours and raising false alarms. ', '')
replace('This is a tradeoff, not a win at equal false alarms.', 'Higher sensitivity comes with more false alarms.')
between('The supported benefit is computational monitoring continuity', r'\section{Conclusion}', r'''
\section{Discussion and limitations}

The Indian benchmark supports improved reconstruction, while external losses and the identity-map ablation limit claims about transfer and CCA. Three further limitations matter for monitoring:
\begin{itemize}
\setlength{\itemsep}{.2em}\setlength{\parskip}{0pt}
\item \textbf{Uncertainty and faults.} Tail intervals and some fault types perform poorly. All positive joint truth cases abstain, so verified computation has not demonstrated successful positive-alert certification in these cases.
\item \textbf{Query access and assumptions.} Later states may leave archive boxes. Direct readings avoid this membership assumption but still need justified error bounds and genuinely new measurements. Frequent one-corruption abstention and competitive simple orders limit operational claims.
\item \textbf{Validation and scale.} Cohorts, dates, seeds and stress trials are dependent; later additions are exploratory. Small-network and synthetic checks do not establish efficient scaling or natural-fault performance.
\end{itemize}

A prospective study needs new outage periods, independent instrument-error limits and practical access to query readings. We measured computational reconstruction and conditional query planning, not pollution or emissions reductions, health benefits or deployed regulatory decision recovery. Supporting research files remain in the authors' archive; the principal protocols, comparisons, statements and proofs are included here.
''')
between('SheafCode treats an outage as a problem', r'\section*{Data and code availability}', r'''
SheafCode reconstructs missing readings, checks joint threshold decisions across allowed states, and selects measurements to resolve disagreement despite noise or corruption. The fixed Indian causal benchmark improves on VAR, while external losses remain. Finite-bank decisions can need fewer queries than full-pattern identification; the stability theorem gives a sharp conditional tolerance and a failure witness. Later-date membership failures show why that guarantee cannot be assumed in the field. Monitoring continuity during real outages remains the intended application.
''')

for pattern in [r'\\begin\{figure\}.*?\\end\{figure\}',
                r'\\begin\{(?:proposition|theorem)\}.*?\\end\{(?:proposition|theorem)\}',
                r'\\textbf\{Proof\.\}.*?\\textit\{QED\.\}']:
    assert re.findall(pattern,s,re.S) == re.findall(pattern,original,re.S), pattern

def prose_words(text):
    text = text.split(r'\begin{document}',1)[1].split(r'\begin{thebibliography}',1)[0]
    text = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}',' ',text,flags=re.S)
    text = re.sub(r'\\[A-Za-z]+(?:\[[^\]]*\])?', ' ', text)
    text = re.sub(r'\$.*?\$|\\\[.*?\\\]', ' ', text, flags=re.S)
    return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", text))

s = re.sub(r'\n{4,}', '\n\n\n', s)

between('When an air-quality station stops reporting', r'\end{abstract}', r'''
When an air-quality station stops reporting, SheafCode estimates its missing readings, checks whether all allowed states agree on an alert, and selects extra measurements to resolve disagreement. Learned spatial and temporal comparisons use only information available at prediction time. On a fixed Indian 24-hour outage benchmark, normalized root mean squared error (RMSE) is 16.48\% lower than a tuned causal vector autoregressive (VAR) model across 77 matched station-days; physical PM2.5 error is 10.45\% lower. In 20 exploratory US, UK and Australian cohorts, normalized error is lower than VAR in 13 and a causal random forest in 18. Losses, fault-localization failures and weak extreme-pollution interval coverage are retained. The mathematical analysis gives model-conditional joint threshold certificates, measurement plans tolerant of noise and corruption, a sharp tolerance for scenario change, and adaptive queries. In ten finite-bank settings where both tasks are feasible, deciding the alert needs 25\% fewer queries than identifying the full pattern. A later-date simulation without corrupted queries saves 17.24\% of measurements against querying every station, with no wrong issued certificates under the tested error budget. Future observations sometimes leave the declared uncertainty model. The demonstrated benefits concern computational reconstruction under simulated outages and conditional query planning; pollution, emissions and health-risk reductions were not measured.
''')
replace('For each edge, suppose a selected $p$-row relation $M_{uv}x_u=N_{uv}x_v$ has invertible endpoint maps, so the rule works in both directions. Keep any extra comparison rows as constraints. A connected component $C$ is a group of stations linked by paths. Choose a root station and a spanning tree, which connects the component without cycles. A root vector $a$ then determines each station along the tree, written $x_v=T_va$, where $T_v$ is the product of comparison transports along that tree path. However, edges outside the tree can impose extra conditions. Let $K_C$ be the set of root vectors satisfying \\emph{all} these conditions, and let $q_C=\\dim K_C$ count the independent directions still free. Cycles can restrict $q_C$. Homogeneous comparisons mean zero discrepancy on every edge. We write $|C|$ for the number of stations in component $C$.', r'''Assume each edge has an invertible $p$-row relation $M_{uv}x_u=N_{uv}x_v$; retain extra rows as constraints. A connected component $C$ contains stations linked by paths. Choose a root and a spanning tree connecting them without cycles. Its root vector $a$ determines $x_v=T_va$, where $T_v$ combines the transports along the tree path. Let $K_C$ contain roots satisfying all remaining constraints, and $q_C=\dim K_C$ count their free directions. Homogeneous comparisons have zero discrepancy on every edge; $|C|$ counts the component's stations.''')
replace('A surviving reading may itself be faulty. Let $e$ be a correction to the observed coordinates, so their possible clean values are $x_O+e$; $e_v$ denotes the correction block for station $v$. For any vector $w$, write $\\|w\\|_\\infty=\\max_j|w_j|$, its largest absolute entry.', r'''Let $e$ correct observed readings to possible clean values $x_O+e$, with station block $e_v$. The norm $\|w\|_\infty=\max_j|w_j|$ is a vector's largest absolute entry.''')
replace('For a fixed fault support, we compute the bounds with linear programs (LPs): optimization problems with a linear objective and linear constraints. The extra variable introduced below lets a joint maximum score fit this form. A feasible witness is an allowed state that demonstrates one of the possible decisions.', 'Each fault support gives linear programs (LPs), with linear objectives and constraints. An extra variable represents the maximum score. A feasible witness is an allowed state demonstrating a decision.')
replace('The factor two appears because each state allows an error of $\\eta_v$: their reading boxes overlap unless the centres differ by more than $2\\eta_v$ in some coordinate. The stations in $E(a,b)$ have disjoint boxes. Each separating set is an edge of a conflict hypergraph; unlike a graph edge, it can contain any number of stations. A multicover selects stations to hit every such set the required number of times. An exact oracle checks whether a proposed plan leaves any opposite pair unresolved. The optimization over conflicts discovered so far is called the master problem.', r'''Each state permits error $\eta_v$, so reading boxes are disjoint only when their centres differ by more than $2\eta_v$ in some coordinate. Each separating set $E(a,b)$ is a conflict hypergraph edge, which can contain many stations. A multicover hits every edge the required number of times. An exact oracle finds unresolved opposite pairs; the master problem optimizes over conflicts found so far.''')
replace("A small example shows why the radius matters. Suppose the archived states are $(0,0)$ (non-exceedance) and $(4,10)$ (exceedance), the threshold for the larger station reading is five, and each queried reading has error at most one. With no corrupted queries, measuring either station distinguishes these two centres. But they do not tolerate the same amount of change: measuring the first station works only for $\\rho<1$, while measuring the second works for $\\rho<4$. Thus two plans with the same query cost can have different robustness.", r'''Consider archived states $(0,0)$ and $(4,10)$, with opposite decisions at a threshold of five for the maximum reading. With clean error at most one and no corruption, either station distinguishes the centres. But the first tolerates only $\rho<1$, while the second tolerates $\rho<4$: equal-cost plans can differ in robustness.''')
replace("There is one more limit: an archived state's decision label must not change inside its allowed box.", "The decision must also stay unchanged inside each scenario box.")
replace('As both the scenario radius $\\rho$ and the measurement error $\\eta_v$ grow, the possible responses from opposite decisions may start to overlap. The theorem finds the exact largest tolerance for a chosen query set, then asks how it changes with the query budget.', r'As $\rho$ and $\eta_v$ grow, opposite scenarios may permit overlapping responses. The theorem gives the exact separation tolerance for a query set and budget.')
replace("The example now also shows why the inequality in the theorem is strict. At $\\rho=4$, states $(0,4)$ and $(4,6)$ still have opposite labels, yet both can produce a second-station reading of five within the allowed measurement error. Exactly at that boundary, the query cannot distinguish them. The radius is therefore a \\emph{conditional tolerance}, not a promise that tomorrow's pollution lies in one of the boxes. These independent boxes also need not satisfy the original coupled sheaf constraints", r'''At $\rho=4$, opposite states $(0,4)$ and $(4,6)$ can both report five at the second station, showing why the boundary is excluded. The radius is conditional on scenario membership; independent boxes need not satisfy the original coupled sheaf constraints''')
replace('To test prediction intervals, snapshot trials hide only the current target reading. A prediction interval gives a lower and upper concentration bound around an estimate. Calibration uses absolute PM2.5 errors, pooled or station-wise, to aim for 90\\% coverage: the fraction of hidden truths lying inside the intervals. A percentile marks a position in the training distribution: the 95th percentile has about 95\\% of training readings below it. Risk grouping uses predicted risk against a training 75th-percentile threshold; extremes for evaluation use truth above the training 95th percentile. No hidden labels update intervals. Their empirical coverage does not justify the separate bounded set $Z$.', r'''Snapshot trials hide the current target reading and place a prediction interval around its estimate. Calibration uses pooled or station-wise absolute PM2.5 errors to target 90\% coverage, the fraction of hidden truths inside the intervals. The training 95th percentile has about 95\% of training values below it. Evaluation extremes exceed that level; risk grouping instead uses predictions against the training 75th percentile. Hidden labels never update intervals, and empirical coverage does not justify the separate bounded set $Z$.''')
replace('Fault localization depends strongly on the failure type. Exact support recovery means identifying precisely the faulty stations, with no extras or omissions. Injected equal opposite two-fault offsets yield exact support recovery in 99.6\\% of Indian and 95.8\\% of US hours; unequal offsets yield only 0.7\\% and 15.0\\%. Drift, stuck, gain and one-channel faults also expose failures. Detection, estimated count and exact localization are different endpoints. The two-fault tests use hours with the required readings available: 456 Indian hours and 432 US hours. They restrict the candidate faulty stations and fill missing context with means, which limits interpretation. The fault procedure reports at most two faulty stations, so it cannot recover a three-fault support exactly.', r'''Exact support recovery identifies every faulty station with no extras. Equal opposite two-fault offsets yield 99.6\% recovery in India and 95.8\% in the US, versus 0.7\% and 15.0\% for unequal offsets. Drift, stuck, gain and one-channel faults also expose weaknesses (Figure~\ref{fig:fault-outage}). Detection, count and localization are separate endpoints. Tests use 456 Indian and 432 US hours with required readings available; restricted candidates and mean-filled context limit interpretation. Reporting at most two faults makes three-fault exact recovery impossible.''')

(OUT/'SheafCode_IRIS_Final.tex').write_text(s,encoding='utf-8')
(ROOT/'audit_tools/final_iris_review/concise_compile/SheafCode_IRIS_Final.tex').write_text(s,encoding='utf-8')
report = json.loads((BASE/'final_check.json').read_text(encoding='utf-8'))
report.update(fresh_directory_single_file_compile_passed=False, visual_layout_review_completed=False,
              clean_compile_pending=True, baseline_prose_words=prose_words(original),
              revised_prose_words=prose_words(s), scope='Shorter readable version; all figures, statements and proofs preserved exactly; no experiments rerun.')
(OUT/'final_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('Approximate prose words, including tables but excluding diagrams and math:',prose_words(original),'->',prose_words(s))
