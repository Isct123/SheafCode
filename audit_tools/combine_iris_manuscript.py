                                                                                  
from pathlib import Path
import hashlib, json, re, shutil, zipfile

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'SUBMISSION_WORKSPACE' / 'Combined_IRIS'
SRC = OUT / 'Source'
INPUT = OUT / 'Inputs'
SRC.mkdir(parents=True, exist_ok=True)
INPUT.mkdir(exist_ok=True)
base_path = Path(r'C:\Users\rishi\Downloads\SheafCode_IRIS_Submission_Corrected_Final.tex')
base = base_path.read_text(encoding='utf-8')
audited = (ROOT/'SUBMISSION_WORKSPACE/Source/manuscript.tex').read_text(encoding='utf-8')
shutil.copy2(base_path, INPUT/base_path.name)
shutil.copy2(Path(r'C:\Users\rishi\Downloads\main.pdf'), INPUT/'comparison_manuscript.pdf')

def result(label):
    for block in re.findall(r'\\begin\{(?:proposition|theorem)\}.*?\\end\{(?:proposition|theorem)\}', audited, re.S):
        if r'\label{'+label+'}' in block:
            return block
    raise ValueError(label)

def figure(file, label, caption, width='.90'):
    return r'\begin{figure}[htbp]\centering\includegraphics[width='+width+r'\linewidth]{figures/'+file+'.pdf}\n'+r'\caption{'+caption+r'}\label{'+label+r'}\end{figure}'+'\n'

preamble = base.split(r'\begin{document}')[0]
parts = [preamble+r'\begin{document}\maketitle\vspace{-1.5em}'+'\n']
parts.append(r'''
\begin{abstract}
Air-quality stations can stop reporting while nearby stations keep working. We ask three questions: how can we estimate the missing readings using information available during the outage, when do all plausible explanations agree on a pollution decision, and which extra readings could resolve a disagreement? SheafCode learns spatial and temporal comparisons between pollutant vectors. On a fixed Indian 24-hour outage benchmark, its normalized root mean squared error is 16.48\% lower than a tuned causal vector autoregressive (VAR) model across 77 matched station-days; physical PM2.5 error is 10.45\% lower. Across 20 exploratory US, UK and Australian cohorts, it has lower normalized error than VAR in 13 and a causal random forest in 18. Losses, faulty-sensor failures and poor extreme-event interval coverage are retained. Separate mathematical results test a joint threshold over every state in a declared uncertainty model. They characterize corruption-aware measurement repair, give a sharp continuous-box stability radius, and select queries adaptively. In ten finite-bank settings where both tasks are feasible, deciding the threshold uses 25\% fewer queries than identifying the complete archived scenario. A later-date direct-reading simulation without corrupted queries saves 17.24\% of measurements compared with querying every station, with no wrong issued certificates under the tested error budget. Archive membership fails on some later observations, so model-based guarantees remain conditional. The supported application is computational monitoring continuity; no reduction in pollution, emissions or health risk was measured.
\end{abstract}

\section{Research question and contribution}
Air-quality stations measure pollutants such as particulate matter, nitrogen dioxide and ozone. Their readings help inform air-quality reporting and emergency responses \citep{caqm2025}. If one station stops reporting, we lose information about that location even while its neighbours keep working. A city average can fill the gap, but it may miss local differences. We ask whether nearby readings and the station's earlier history can give a better estimate during the outage.

The timing matters. A method that fills a historical gap may use readings recorded after the gap. During an ongoing outage, those readings do not yet exist. Our estimate for hour $t$ therefore uses only observations available through $t$. We use \emph{causal} in this information-set sense, not to claim causal inference about pollution sources.

We represent the network with a \emph{sheaf}: each station carries a pollutant vector, and each edge compares translated versions of two station vectors. This lets stations follow a learned relationship without requiring their raw concentrations to be equal. Spatial comparisons and earlier readings together constrain the missing vector.

A good estimate still leaves a second question. Suppose it falls just below an alert reference. Another state allowed by the observations and error limits might lie above it. Some surviving readings might also be faulty. We therefore ask whether \emph{every} state in a declared uncertainty model gives the same threshold decision. If so, we can certify that decision within the model; otherwise we report uncertainty. Here ``safe'' means only non-exceedance of the stated mathematical reference, not absence of health risk.

Finally, when two decisions remain possible, what should we measure? We need to distinguish states with opposite decisions; states with the same decision need not be told apart. This motivates selecting measurements for the decision rather than reconstructing every concentration. We allow noisy and corrupted responses, then ask how much archived scenarios can change before a plan fails and whether each answer can guide the next query.

Our contributions follow these questions: a causal reconstruction implementation; joint threshold certificates under explicit fault and residual limits; and decision-specific fixed, stable and adaptive measurement plans. The fixed Indian comparison finds 16.48\% lower normalized RMSE than tuned VAR (Section~\ref{sec:results}). We also retain external losses and operational limitations. These are applications and specialized results built on established sheaf, optimization and test-selection theory, not inventions of those foundations.

\section{Related work and the research gap}
Spatial statistics models dependence between locations, VAR models relationships across time and channels, and matrix completion uses shared low-rank structure in incomplete arrays \citep{cressie1993statistics,lutkepohl2005var,mazumder2010softimpute}. GRIN learns spatiotemporal relationships through graph message passing \citep{cini2022grin}. We evaluate specified controls, not every method in these families. GRIN's tested bidirectional implementation is compared with retrospective sheaf, separately from live causal reconstruction.

Sheaf network and spectral methods are established \citep{ghrist2011network,hansen2019spectral}. Earlier air-quality work used sheaves for local self-filtering with low-cost sensing and vehicle-related information \citep{pham2022airquality}. Our task is to reconstruct a missing station from surviving observations and pre-outage history, then examine what decisions and extra measurements those observations justify. Canonical correlation analysis (CCA), used to learn comparison maps, is also established \citep{hotelling1936relations}; our identity-map ablation prevents attributing the reconstruction gains specifically to CCA.

Bounded-error estimation asks which states remain possible under declared limits; functional observability asks whether a chosen quantity can be determined without recovering the entire state \citep{bounded1990,functional2022}. Sensor placement, fault-diagnosis test selection and generalized test collection study which measurements distinguish possibilities \citep{krause2008,awasthi2020,douek2021,xing2022robust}. Adaptive testing uses earlier answers to choose later queries \citep{cicalese2014adaptive,jia2024noisy}. Their objectives and assumptions differ, so their guarantees cannot simply be transferred to our problem.

The specific contribution is the connection between joint pollution decisions, opposite-decision witnesses, corruption-aware separating sets, a continuous-box tolerance and adaptive query policies. Epigraph optimization, verified bounds and branch and bound supply credited tools \citep{boyd2004,jansson2004,land1960}. This is a focused review of directly relevant work, not an exhaustive priority claim. Adaptive conformal inference concerns revealed prediction errors \citep{gibbs2021}; a silent station does not supply those labels during its outage.

\section{Methodology}\label{sec:method}
\subsection{Why compare translated station readings?}
Station $v$ carries a normalized vector $x_v\in\R^p$ of $p$ pollutant channels. An edge $uv$ has maps $M_{uv}$ and $N_{uv}$ into a shared comparison space. Its discrepancy is $M_{uv}x_u-N_{uv}x_v$. For example, two stations may rise and fall together while one usually reports higher concentrations. Comparing translated readings lets us use that relationship without forcing equality.

Training means and standard deviations set each station/channel's scale. Each station selects its three strongest available neighbours by mean absolute within-channel training correlation (or all other stations if fewer), and we take the union of these edges. CCA learns spatial comparison maps; correlated directions receive greater weight. Fewer than 200 complete paired vectors trigger identity-map fallbacks. Learned temporal comparisons also link the same station one and twenty-four hours apart, with a selection-chosen temporal weight. Missing entries remain unknown variables; they are not turned into observed truth.

In sheaf terminology, the station and edge vector spaces are \emph{stalks}, and the endpoint maps are \emph{restriction maps}. Stacking the spatial discrepancies gives a coboundary $\delta$. Exact agreement means $\delta x=0$; its squared disagreement is $\|\delta x\|_2^2=x^T\delta^T\delta x$. The matrix $\delta^T\delta$ is the sheaf Laplacian. These definitions explain the method's name without assuming that air moves along graph edges.
''')
parts.append(figure('network_structure','fig:network','The fitted five-station Indian graph and one comparison edge. Positions are schematic. Orange marks an illustrative hidden station, not a recorded outage. Endpoint maps translate vectors before comparing them.','.88'))
parts.append(r'''
\subsection{How do the comparisons give an estimate?}
Let $D$ stack spatial and temporal discrepancies over the prediction window. Split its columns into missing coordinates $U$ and observed coordinates $O$. Write $z=x_U$, with $x_O$ fixed, and let $x(z)$ be the filled full state. We minimize
\begin{equation}\label{eq:decoder}
J(z)=\|D_Uz+D_Ox_O\|_2^2+\lambda x(z)^TPx(z),\qquad\lambda>0.
\end{equation}
The first term penalizes disagreement. The second uses a positive-definite precision $P$, built from inverse regularized within-station training covariances and repeated over the window, to discourage implausible combinations. Differentiating gives
\begin{equation}\label{eq:normal-equations}
(D_U^TD_U+\lambda P_{UU})\hat z
=-(D_U^TD_O+\lambda P_{UO})x_O.
\end{equation}
The matrix on the left is positive definite, so the estimate is unique. The $P_{UO}x_O$ term lets observed channels inform missing channels at the same station. With no prior term, this is a harmonic extension: choose the missing entries to minimize the network's disagreement energy. Regularization makes the solve unique even when comparisons alone leave ambiguity.

For hour $t$, the causal solve uses only the available prefix through $t$. The hidden station's actual pre-outage history is allowed; its hidden outage readings and future surviving observations are excluded. Later predictions never become new observations. A retrospective solve may use later context and is reported separately (Figure~\ref{fig:causal-workflow}).
''')
parts.append(figure('causal_workflow','fig:causal-workflow','The information boundary at the first hidden hour. Available history and current surviving readings can enter the estimate; hidden target truth and future readings cannot. Naturally missing context is not assumed available.','.90'))
parts.append(r'''
\subsection{Why check a decision separately?}
Equation~\ref{eq:decoder} chooses one estimate. Certification asks whether every state satisfying declared error limits agrees on a threshold decision. These are different uses of the model: small average error does not imply a valid certificate.

A \emph{query} obtains a new reading not already available to the procedure, such as an independently checked measurement. Reading the same recorded number again adds no information, and a failed station cannot be assumed to answer. Our tests simulate revealing archived station readings. Query savings count those readings, not money saved in a deployed network.

\section{Mathematical foundations and guarantees}\label{sec:theory}
The results below answer increasingly demanding questions. They start with exact comparisons, then allow ambiguous states and faults, then select new readings. Each guarantee requires its stated assumptions.

\subsection{What does one observed station anchor?}
On an exact edge, suppose a selected $p$-row relation $M_{uv}x_u=N_{uv}x_v$ has invertible endpoint maps. Keep any extra rows as constraints. A root value $a$ can then be transported along a spanning tree to $x_v=T_va$. Let $K_C$ be the roots satisfying all remaining edge and extra-row constraints in component $C$, and $q_C=\dim K_C$. A cycle can restrict the root, so connectivity alone does not determine $q_C$.
''')
parts.append(result('prop:anchors'))
parts.append(r'''
\textbf{Proof.} Subtract two solutions agreeing on observed stations. Their difference satisfies the homogeneous relations. Zero at one observed vertex propagates through invertible tree maps to its entire component. On a wholly hidden component, differences correspond exactly to roots in $K_C$, independently of other components. Invertible transport takes a nonzero root to a nonzero value at every vertex, proving the support claim. \textit{QED.}

For identity maps, one observed station fixes its component; hiding the entire component leaves $p$ free directions. Other cycle constraints can instead give $q_C=0$. Singular maps or partial-channel observations need different coordinate-level analysis. This is specialized spanning-tree elimination, not a new connectivity theorem \citep{ghrist2011network,hansen2019spectral}. Generic matching \citep{hall1935} does not prove rank for tied learned maps. Exact uniqueness also says nothing by itself about error on noisy pollution data.

\subsection{Can a decision survive uncertainty in individual readings?}
To state the uncertainty model explicitly, let $e$ correct the observed coordinates: their possible clean values are $x_O+e$. Let $B_U$ be a finite box for the unknown state, $r\geq0$ the componentwise comparison-residual budgets, $\beta$ a fault-magnitude limit and $t$ the maximum number of faulty surviving stations. For each station support $F$ with $|F|\leq t$, define
\[
Z_F=\{z\in B_U:\ \exists e,\ e_v=0\ (v\notin F),\ \|e\|_\infty\leq\beta,
\ |D_Uz+D_O(x_O+e)|\leq r\},\qquad Z=\bigcup_F Z_F.
\]
The residual inequality holds row by row. Each $Z_F$ is a bounded polyhedron; the finite union is compact but may not be convex. Extra trusted linear constraints may be included. If $Z$ is empty, the model is inconsistent and we abstain. Truth belongs to $Z$ only when the declared box, fault and residual assumptions hold \citep{bounded1990}.

Let $s_k(z)=h_k^Tz+c_k$ be pollutant $k$'s affine score relative to its reference, including normalization offsets. Set $S(z)=\max_k s_k(z)$; exceedance means $S>1$, and non-exceedance means $S\leq1$. These references do not automatically implement official AQI or GRAP.
''')
parts.append(result('prop:joint-short').replace('sharp enclosing range','sharp enclosing interval'))
parts.append(r'''
\textbf{Proof.} Compactness makes both extrema attainable. For fixed $z$, the smallest $y$ satisfying every $s_k(z)\leq y$ is their maximum, so its minimum on each support gives the lower endpoint. Finite support and pollutant maxima can be interchanged for the upper endpoint. States attaining $L$ and $U$ witness the stated decisions or disagreement. The score range may have gaps inside its enclosing interval. \textit{QED.}

If $a,b\in[0,2]$ and $a+b=2$, either concentration may be zero, but $\max(a,b)\geq1$. With both references 0.75, every state exceeds some reference although neither concentration is identified (Figure~\ref{fig:joint-example}). Equal weights give $(a+b)/(2\cdot0.75)=4/3$, an easily checked certificate. More generally, any nonnegative pollutant weights summing to one give a weighted score no larger than $S$; a lower bound above one therefore suffices. Weights may need to differ by fault case: the two feasible score vectors $(2,0)$ and $(0,2)$ each exceed, but a single common weighting has worst value at most one. This uses the established epigraph principle \citep{boyd2004}.
''')
parts.append(figure('joint_example','fig:joint-example','Illustrative concentrations, not measurements. The segment $a+b=2$ avoids the box where both are at most 0.75, even though either individual value can be zero.','.48'))
parts.append(r'''
\subsection{How can we trust an approximate calculation?}
A solver may give almost the right weights for combining inequalities. An unmatched coefficient is still an error unless we bound its effect. Finite variable ranges let us do that explicitly.
''')
parts.append(result('prop:weighted-safe'))
parts.append(r'''
\textbf{Proof.} Multiply each valid inequality by its nonnegative weight and add. Its left side is $q(x)-\sum_j\delta_jx_j$. Each leftover term is at most its larger endpoint value on $[\ell_j,u_j]$. Add those bounds; applying the result to $-q$ gives a lower bound. \textit{QED.}

For $2x+y\leq5$, $0\leq x\leq2$, $0\leq y\leq3$, the weight 1.5 bounds $3x+2y$ by nine. Weight 1.49 instead leaves $0.02x+0.51y$ and gives $7.45+0.04+1.53=9.02$. Ignoring the leftovers would incorrectly claim 7.45. Evaluate the certificate using exact arithmetic or controlled outward rounding; ordinary floating-point evaluation alone is not a proof. Combine only constraints valid in the same fault case. This verifies declared coefficients, not physical accuracy \citep{jansson2004}.

\subsection{What if the fault search stops early?}
We may have to search many fault supports. Introduce a binary activation $a_v$ for each candidate faulty station, impose $|e_{vk}|\leq\beta a_v$ and $\sum_v a_v\leq t$, and relax unfixed activations to $[0,1]$ at a search node. The relaxation contains every genuine explanation in that node. A fractional activation is a bounding device, not a physical partial fault.
''')
parts.append(result('prop:search-bounds'))
parts.append(r'''
\textbf{Proof.} Every explanation is resolved, covered by an unresolved node, or pruned below the incumbent. These cases give the upper bound; a verified feasible state gives the lower bound. Branching on zero or one preserves coverage. A child lies inside its parent, so the smaller of their valid bounds remains valid. A depth-$n$ binary tree has at most $2^{n+1}-1$ nodes; exact leaf optimization resolves the finite cases. \textit{QED.}

A feasible state above a threshold proves that exceedance is possible, not that every state exceeds. Conservative bounds remain valid but need not close the gap. To minimize, maximize the negative score and reverse endpoints. This applies classical branch and bound \citep{land1960}.

\subsection{Which readings would resolve opposite decisions?}
Write $Z_-$ and $Z_+$ for declared non-exceedance and exceedance states. There are $n$ candidate query stations; station $v$ would report a vector $m_v(z)$ at positive integer cost $c_v$. A clean reading has componentwise error at most $\eta_v$; up to $f$ queried stations may return arbitrary vectors. The fixed plan is chosen before its responses.

For each opposite pair $a\in Z_-$, $b\in Z_+$, define
\[
E(a,b)=\{v:\|m_v(a)-m_v(b)\|_\infty>2\eta_v\}.
\]
At these stations the two clean-reading boxes are disjoint. The sets $E(a,b)$ form a conflict hypergraph: each set says which stations could distinguish one pair. To withstand corruption, we need enough separators that the two explanations cannot both blame their disagreement on faults.
''')
parts.append(result('prop:repair'))
parts.append(r'''
\textbf{Proof.} A common response compatible with both states must be corrupted under at least one explanation at every queried separator. Their two corruption sets cover at most $2f$ stations, so $2f+1$ separators rule this out. Conversely, split at most $2f$ separators into two sets of size at most $f$; report the other state's vector on each respective set. Elsewhere, report a coordinatewise midpoint inside both clean boxes. Both decisions then fit one response. Discovered conflicts are necessary constraints, so the master optimum is a lower bound. An accepted candidate is feasible for all pairs and hence globally optimal. A rejected candidate satisfies every old conflict, so its new conflict is distinct; only $2^n$ sets exist. An infeasible master proves impossibility. \textit{QED.}

With conflicts $\{A,B\}$ and $\{B,C\}$, $B$ alone resolves both when $f=0$. With one corrupted query, each pair needs three separators, making this example impossible. This is the familiar two-explanation error-correction argument \citep{donoho2003}, applied to decision-specific multicover \citep{awasthi2020,douek2021}. The implemented oracle exhaustively checks a finite bank, not an arbitrary continuous sheaf polyhedron. Near-threshold states may be inseparable at positive noise, so a margin or abstention region is necessary.
''')
parts.append(figure('repair_intuition','fig:repair-intuition','Two mathematical examples. Left: $B$ hits both conflicts when $f=0$; with $f=1$ each pair needs three separators. Right: expanded reading intervals meet at the strict stability boundary. These are not measured pollution outcomes.','.90'))
parts.append(r'''
\subsection{How far may an archived scenario change?}
A future state need not equal an archived vector. Around each scalar scenario $a_i$, allow independent coordinate changes of at most $\rho$: $\|x-a_i\|_\infty\leq\rho$. For the network-maximum reference $\tau$, let
\[
G=\min_i|\max_v a_{iv}-\tau|>0,\qquad0\leq\rho<G.
\]
This keeps every box's decision label unchanged. State mismatch $\rho$ differs from reading error $\eta_v$: together they widen possible clean responses by $\rho+\eta_v$.
''')
parts.append(result('thm:repair-radius'))
parts.append(r'''
\textbf{Proof.} Under scenario $i$, every clean reading in $[a_{iv}-\rho-\eta_v,a_{iv}+\rho+\eta_v]$ is attainable. Two such intervals are disjoint exactly when $h_v(i,j)>\rho$. The repair proposition requires at least $q$ disjoint queried intervals for every opposite pair; sorting makes this equivalent to $\rho<r(R)$. At a positive boundary, a weakest pair has at most $2f$ separators. Split their corruptions as above; elsewhere choose shared responses and project them onto the two true-state boxes, giving clean errors at most $\eta_v$. The condition $\rho<G$ preserves opposite labels. Larger budgets retain smaller plans. Finite minima, maxima and order statistics select original margins, proving the frontier claim. \textit{QED.}

For centres $(0,0)$ and $(4,10)$, reference five, reading error one and $f=0$, either station separates the centres. Station one tolerates $\rho<1$; station two tolerates $\rho<4$. At $\rho=4$, states $(0,4)$ and $(4,6)$ keep opposite labels and both give second-station reading five. The strict boundary describes a real ambiguity in the declared model. The radius is a conditional tolerance, not a future-coverage guarantee, and the independent boxes need not satisfy coupled sheaf constraints \citep{xing2022robust}.

\subsection{Can each answer choose the next query?}
A fixed plan pays for every selected reading. An adaptive plan can stop or choose a different station after an answer. For a finite scalar bank, let $U$ be the unqueried stations and initialize each scenario's remaining corruption budget to $b_i=f$. After querying $v$ and receiving $y$, $T_{v,y}(b)$ subtracts one from $b_i$ exactly when $y$ lies outside scenario $i$'s closed clean interval. Discard negative-budget candidates. A response is compatible when at least one candidate remains.
\begin{proposition}[Exact adaptive repair]\label{prop:adaptive}
If remaining candidates have one label, $V(U,b)=0$. If labels differ and $U$ is empty, $V(U,b)=\infty$. Otherwise the optimal worst-case additional cost is
\[
V(U,b)=\min_{v\in U}\left(c_v+\max_{y\,\mathrm{compatible}}
V(U\setminus\{v\},T_{v,y}(b))\right).
\]
Singleton interval endpoints, open gaps and the two exterior intervals give finitely many updates representing every real response. Adaptive cost is at most optimal fixed-plan cost, and adaptivity cannot make an impossible instance feasible.
\end{proposition}
\textbf{Proof.} A history fits scenario $i$ exactly when at most $f$ readings lie outside its clean intervals. Thus $(U,b)$ retains everything relevant to the continuation. Induct on $|U|$: any policy first selects a station, and an adversary may choose any compatible response; the recurrence chooses the best worst-case continuation. Interval membership is constant on each response class. A fixed plan is an available policy. An ambiguous full transcript defeats every decision tree, so adaptivity cannot restore feasibility. \textit{QED.}

For three possible zones, imagine one station identifies the zone and one station per zone distinguishes its two decisions. A fixed plan needs all three zone stations; an adaptive plan queries the zone identifier and then its zone station, needing two. This illustrates the mechanism, not a measured network. The implemented recursion is exponential, queries each station once and uses a declared search limit \citep{cicalese2014adaptive,jia2024noisy}.

\section{Experimental design}\label{sec:experiments}
\subsection{Data, units and chronological roles}
The original networks use five Indian stations with four channels (A), four US stations with three (B), and five UK stations with three (C), from OpenAQ, EPA AirData and DEFRA AURN \citep{openaqrecent,eparecent,aurn2026,openair2026}. Indian PM2.5, PM10 and ozone are in $\mu$g/m$^3$, and NO2 in ppb. US PM2.5 is in $\mu$g/m$^3$, ozone in ppm and NO2 in ppb; UK PM2.5, PM10 and ozone are in $\mu$g/m$^3$. Normalization does not physically interchange these units. India uses local time and the US/UK use UTC. US monitor occurrence codes are selected by site/channel training eligibility, and negative concentrations are excluded. Preliminary or unratified records remain a limitation.

Observations begin no earlier than 14 October 2025. Original intended calendars end on 5 October, 31 March and 30 June 2026 for A, B and C; acquired Indian records stop on 2 October, leaving final days missing. Intended training/selection/calibration/test timeline positions are 4003/667/667/1335, 1296/216/216/432 and 2606/434/435/869 hours respectively. These positions are not all observed measurements.

A fixed exploratory expansion attempts 22 regional cohorts: ten evaluable US counties, four UK regions, six NSW regions and two US counties without usable test observations. Its 20 evaluable cohorts contain 73 station memberships, 549,302 observed pollutant-hour cells and 2,114 paired test station-days. NSW hourly data and timestamps follow official source conventions \citep{nswapi2026,nswtime2026}. EPA expansion observations run through March 2026, UK/NSW through June. A longer Indian resplit reuses observations and supplies 579 station-days.

Training sets normalization, graph and maps; selection chooses hyperparameters; calibration sets interval and alert cutoffs; testing scores performance. External splits are chronological 60/10/10/20. Eligibility normally needs 50\% training coverage, 720 observed training hours, three stations and two channels including PM2.5. Missing source hours remain missing. Cohorts are selected deterministically, share sources and sometimes stations, and are not independent replications. Later additions are exploratory, not preregistered or blinded. Observation-age eligibility depends on the actual submission date.

\subsection{Matched outage tests and controls}
We hide a station's complete vector for 24 hours after 24 hours of available history, using every third eligible test day and requiring at least $12p$ observed target cells. Methods share the same truth cells. Tuned full-covariance causal VAR is the main control \citep{lutkepohl2005var}; others are concentration-space city mean, seasonal persistence, static Gaussian conditioning and spatial-only sheaf. Sheaf searches $\lambda\in\{0.1,1\}$ and temporal scale $\{0.5,2\}$; VAR searches ridge $\{1,100\}$, noise scale $\{0.05,0.5\}$ and covariance shrinkage $\{0,0.2,1\}$.

The causal random forest uses surviving current readings, one-/twenty-four-hour lags, last actual pre-outage target, masks and calendar/station indicators. Channel-specific forests use 64 trees, depth at most 12, selected leaf size 5 or 20 and at most 80 sampled training days \citep{breiman2001}. Pinned bidirectional GRIN uses three seeds, widths 32/64, at most 30 epochs and seven-epoch early stopping; it is compared with retrospective sheaf \citep{cini2022grin}. Finite tuning budgets are not globally optimized performance bounds. External GRIN roles are selected from training masks: the first lexicographic eligible cohort per provider with ten training and three selection cases.

Each station-day contributes mean normalized squared error $E$. On matched cases, gain against baseline $B$ is
\[
100\left(1-\sqrt{\frac{\sum E_{\rm sheaf}}{\sum E_B}}\right).
\]
Positive gain means lower RMSE. This averages squared error within station-days before comparing roots, not daily percentage gains or pooled-cell error. Physical PM2.5 error is separate. Paired seven-day calendar-bin bootstrap intervals use 2,000 draws and are suppressed below five bins. Fourteen-day sensitivities are retained in the reference. These fixed bins differ from the stationary bootstrap \citep{politis1994}; intervals do not establish independent cases, simultaneous significance or protection from shift.

\subsection{Decision and query protocols}
For fixed repair, at most 24 evenly spaced complete PM2.5 \emph{test} hours per role form a declared offline bank. All 23 roles are attempted; 18 supply complete vectors. References 15/35/60 $\mu$g/m$^3$, noise limits 1/3/5 and corruption budgets zero/one produce 414 settings. Near-reference states are excluded and counted under a declared margin rule. The bank uses test truth as possible states; it is not a prospective prediction or independently validated field model. Pattern identification distinguishes entire archived scenarios, not arbitrary continuous states.

Chronological transfer instead uses at most 24 equally spaced \emph{training} vectors. Separate calibration gives the 90\%-rank nearest-centre distance in maximum-coordinate norm. Each radius is capped at 95\% of its centre-to-threshold semantic margin. The illustrative network-maximum reference is 35 $\mu$g/m$^3$ and clean-reading error at most one. Test outcomes choose neither bank nor radius. Clean trials use negative/zero/positive error endpoints; corruption trials vary the faulty station, high/low corrupt readings and clean-error signs. Trials reuse hours. The companion reference describes additional protocols; case tables and audit outputs remain in the research archive.

\section{Results and environmental relevance}\label{sec:results}
\subsection{Does reconstruction improve, and where?}
The fixed Indian benchmark has 77 matched station-days on 16 dates. Sheaf lowers normalized RMSE by \textbf{16.48\%} against tuned VAR (seven-day interval 12.62--20.16\%). Physical PM2.5 gain is 10.45\% (2.75--14.31\%). These measure different quantities. Original US gain is 8.68\% on 24 station-days/six dates (interval suppressed), and UK gain 3.79\% on 60 station-days/12 dates (1.78--7.83\%). The US physical PM2.5 comparison has 23 eligible cases.
''')
parts.append(figure('primary_reconstruction','fig:headline','The fixed comparison against five matched controls. Positive gain favours causal sheaf normalized RMSE. India uses 77 station-days and seven-day intervals; the US secondary comparison uses 24 and insufficient bins for intervals. Networks are not pooled.','.88'))
parts.append(r'''
Original all-day gains against VAR are 16.71\% (India, 242 station-days), 7.62\% (USA, 68; interval suppressed) and 1.71\% (UK, 174; interval $[-1.16,2.96]$). The longer Indian resplit gives 4.65\% on 579 station-days (1.89--7.37\%), reusing observations rather than replacing the headline.

Sheaf has lower normalized error than VAR in \textbf{13/20} external cohorts and than the tested forest in \textbf{18/20} (Figure~\ref{fig:external}). Lower Hunter loses to both; several UK cohorts lose to VAR. These are numerical counts, not adjusted significance claims. Selection-only choice among sheaf/VAR/forest beats VAR in just 9/20 test cohorts, showing that selection preference need not transfer.
''')
parts.append(figure('external_comparisons','fig:external','Every evaluable external cohort, including losses. Positive gain favours causal sheaf. Lines are paired seven-day intervals; points alone have insufficient bins. The controls can disagree. Complete region/code mappings, physical PM2.5 outcomes and split counts are in the companion reference.','.87'))
parts.append(r'''
\subsection{What do neural controls and ablations explain?}
Retrospective sheaf gains against GRIN in India are 11.37\%, 13.95\% and 15.49\% on 242 shared cases per seed. Selected US D1 and UK E1 seeds favour sheaf; Illawarra F1 favours GRIN in all three (sheaf gains $-0.85\%,-0.41\%,-0.68\%$). All seeds and intervals are retained in the reference. These comparisons do not establish a causal neural advantage.

Removing the 24-hour lag worsens Indian error, but identity spatial maps beat the chosen CCA maps: the chosen CCA setting has 5.71\% higher RMSE than the identity setting (the reported gain is $-5.71\%$). Thus CCA superiority is not established. A selection-chosen hybrid gains 1.66\% over the primary sheaf setting, without replacing it. Temporal context helps here; learned spatial maps are not the sole explanation.

\subsection{Where do faults and uncertainty estimates fail?}
Injected equal opposite two-fault offsets yield exact support recovery in 99.6\% of Indian and 95.8\% of US hours; unequal offsets yield only 0.7\% and 15.0\%. Drift, stuck, gain and one-channel faults also expose failures. Detection, estimated count and exact localization are different endpoints. Availability-conditioned two-fault families have 456 Indian/432 US hours; restricted candidates and mean-filled missing context limit interpretation. The decoder reports at most two faults, making three-fault exact recovery impossible.

The uncertainty results are \emph{no-feedback snapshots}, not intervals validated through uninterrupted outages. Extremes are truth above each training 95th percentile; predicted-risk grouping uses a training 75th-percentile threshold and current predictions, not hidden truth. Across 49,996 external cells, fixed/risk-grouped pooled coverage is 87.5\%/88.9\%, versus nominal 90\%; on 2,309 extreme cells it is 25.2\%/45.7\%. Risk grouping widens intervals. The reference retains cohort-macro coverage, widths, sparse-calibration infinite intervals and exclusions. Original Indian fixed coverage is 89.5\% on 5,467 cells, width 62.5 $\mu$g/m$^3$; no Indian test extremes meet this definition. US station-fixed coverage is 75.8\% on 1,613 cells and 17.45\% on 212 extreme cells. Tail uncertainty remains weak.

\subsection{Can the decision need fewer readings than the full pattern?}
Without query corruption, all 61 two-label finite-bank settings admit a decision plan; only ten also admit full-pattern identification. On those ten, decision repair uses \textbf{18 rather than 24 queries} (25\% fewer, six strictly cheaper). In the other 51, decisions are identifiable but full patterns are not. With one corruption, only 15/61 decision plans are feasible and 46 impossible.
\begin{table}[htbp]\centering\small
\caption{Finite archive repair. Counts are dependent settings. Full-pattern impossibility counts only feasible decision settings; query totals compare the common feasible set.}
\begin{tabular}{lrrrrr}\toprule
Faults & Two labels & Decision feasible & Pattern impossible & Both feasible & Queries (D/F)\\\midrule
0 & 61 & 61 & 51 & 10 & 18/24\\
1 & 61 & 15 & 11 & 4 & 18/19\\\bottomrule
\end{tabular}\label{tab:repair-summary}\end{table}

Adaptive repair lowers summed worst-case costs from 84 to 83 across 61 clean feasible settings (one improved), and from 58 to 52 across 15 one-corruption settings (10.34\%, six improved). The same 46 remain impossible. At minimum fixed cost, maximizing interior stability radius improves 12/61 clean and 4/15 one-corruption settings; extra budget improves a further 21 and three. These sums reuse observations and are not expected field expenditure.

\subsection{Do archive guarantees survive later dates?}
Across 18 chronological roles, only \textbf{10,292/13,187} eligible later hour-role vectors lie inside the boxes (78.05\%; role coverage 49.02--95.47\%). Assuming membership anyway gives 984 wrong among 37,044 issued decisions without corruption and 5,528 among 152,362 with one corrupt query. All errors lie outside the boxes. Calibration did not establish future coverage.

We therefore check an elementary \emph{direct-reading guard}. Exceedance is certified after more than $f$ queried lower bounds exceed the reference: at least one of those readings must be clean. With $f=0$, non-exceedance needs every station queried and every upper bound at or below the reference. With one arbitrary corruption and no physical concentration bounds, non-exceedance is generally unidentifiable, so the guard abstains. This counting rule adds no new theorem.
\begin{table}[htbp]\centering\small
\caption{Later-date direct-reading stress simulation. Sequential and full-network guards give identical decisions/abstentions on matched responses. Savings include abstentions and count queries, not field expenditure.}
\begin{tabular}{lrrrr}\toprule
Fault budget & Trials & Issued / wrong & Query saving & Abstentions\\\midrule
0 & 39,561 & 39,309 / 0 & 17.24\% & 252\\
1 & 207,936 & 42,699 / 0 & 11.02\% & 165,237\\\bottomrule
\end{tabular}\label{tab:chronological-guard}\end{table}
With one corruption, only 20.53\% of trials receive certificates. Box-plan prioritization costs 41/15 more queries than simple training-frequency order. Validation-selected ordering improves on that simple order by only 0.219\%/0.080\%, with losses in 7/4 of eighteen roles. We establish fewer queries than full measurement under simulated budgets, not superiority to simple orders.
''')
parts.append(figure('operational_validation','fig:operational-validation','Archive membership and direct-reading outcomes across later observations. Membership is measured, not guaranteed by the calibration rank. Certificate issuance and query savings include dependent stress trials; one-corruption abstentions are frequent.','.90'))
parts.append(r'''
\subsection{Do the implementations follow the mathematics?}
Across 23 fitted roles, 1,032 graph subset/tolerance checks agree with direct rank. All nominal full operators have zero kernel; separate nonzero-kernel and singular controls test the other cases. Exact modular checks certify full column rank for all 23 operators and 246 selected endpoint blocks. These are claims about stored coefficients, not physical consistency.

The 272 hidden-target joint cases have 260 feasible models and 12 inconsistent ones. Verification gives 155 below-reference certificates and 117 abstentions \emph{including those 12}; all 40 positive truth cases abstain. There is no measured above-reference joint certificate. Separate scalar searches and planted joint-positive controls remain in the reference.

Independent checks confirm all 414 fixed configurations, 122 stability frontiers (594 budget points) and 76 adaptive optima (72,950 subproblems). Exact tests include 239 boundary counterexamples, and direct replay reconstructs all 247,497 stress trials. Sixty synthetic multicover cases reach 128 stations; greedy matches 19/24 small optima and otherwise needs at most one extra query. These support implementation correctness, not physical error limits, polynomial scaling or independent refitting.

\subsection{What did earlier studies find?}
Separate historical protocols give a median 3.1\% snapshot gain versus dense Gaussian conditioning (7/8 numerical gains, 2/8 passing within-network Holm correction), and a median 34.0\% first-day retrospective gain versus an implemented Kalman/VAR smoother across nine configurations (5/9 supplementary run-based comparisons pass Holm). A Delhi causal cohort gains 23.2\% over full-covariance VAR on 399 station-days/58 dates; a later same-network PM-only period gains 9.5\% on 74 station-days/19 dates. Beijing instead has 5.93\% higher all-pollutant RMSE than VAR on 502 station-days, and a historical Delhi neural comparison favours GRIN on every seed. Weather controls do not establish improvement. These observations and information sets differ from the recent benchmark, may overlap earlier accessed archives, and are not pooled or independent headline replications.

\subsection{What environmental claim does the evidence support?}
At an illustrative hourly PM2.5 reference of 60 $\mu$g/m$^3$, calibrated Indian sheaf alerts have 69.1\% sensitivity and 9.6\% false positives, versus VAR's 52.7\% and 6.1\%, on 1,177 positive/4,012 negative hours. This is a tradeoff, not a win at equal false alarms. References and test-label ROC curves do not implement official regulatory alerts.

Transplanting natural missing-data patterns onto observed Indian values gives PM2.5 RMSE 19.65 versus city mean 20.46 $\mu$g/m$^3$, a 3.94\% gain on 719 hidden cells/155 patterns; the US counterpart has eight cells. On 190 Indian daily proxy cases, sheaf and city mean tie at 73.7\%, and all methods miss the sole Poor-or-worse day. Four official bulletin matches are descriptive \citep{cpcbrecent,caqm2025}. Original Indian/US joint-score trials give no definite decision; the separate UK trial has four model-conditional below-reference decisions. These are distinct from the later 272-case verification.

The supported benefit is computational monitoring continuity and conditional query planning. No physical intervention, pollution reduction, emissions reduction, health benefit or deployed regulatory decision recovery was measured.

\section{Discussion and limitations}
The questions led to different kinds of evidence. Hidden readings let us measure reconstruction error. Declared feasible states let us prove conditional decisions. Archived scenarios let us test exact query plans. A gain in one task does not establish a gain in the others.

The Indian reconstruction result suggests that surviving readings and temporal context can help, but transfer is mixed. Identity maps beating CCA caution against attributing improvement to a more elaborate spatial fit. Poor tail intervals and weak-fault localization show where average performance hides important failures. The joint verifier's abstention on every positive truth case is numerical assurance, not successful alert detection.

For repair, the main operational problem is model membership: a later state can fall outside the archive boxes. Direct readings avoid that particular assumption but still require justified reading-error bounds and access to genuinely new measurements. High one-corruption abstention and simple-order controls limit claims of operational advantage. Cohorts, dates, seeds and stress trials are dependent, and additions after earlier outcomes are exploratory. Exact small-network and synthetic checks do not resolve exponential scaling or validate natural faults.

A prospective study needs new outage periods, independently justified instrument-error limits, and a practical way to obtain query readings. The research archive and companion reference retain supporting protocols, failures and detailed implementation checks.

\section{Conclusion}
SheafCode treats an outage as a problem of both estimation and information. Learned spatial and temporal comparisons reconstruct missing readings from available observations. Separate mathematics checks whether every admissible state agrees on a joint threshold and, if not, selects readings that can distinguish opposite decisions despite noise or corruption.

The fixed Indian causal benchmark improves on tuned VAR, while other networks retain losses. Finite-bank decision repair can need fewer queries than full-pattern identification, and the stability theorem gives a sharp conditional tolerance with a failure witness. Later-date membership failures explain why those guarantees cannot be assumed in the field. The strongest supported application is computational monitoring continuity.

\section*{Data, code and supplementary material}
Public providers and access resources are cited above. The supplied research workspace retains acquisition requests, station IDs, native units, source hashes, model records, case tables and code in \path{Research_Release/}, with independent replay tools in \path{audit_tools/}. No public archive URL is asserted. The companion \texttt{technical\_reference.pdf} contains the longer audited report, complete external/seed tables, supporting figures and proof detail; case-level files remain in the research archive. The LaTeX handoff includes bibliography and figure assets. Supplementary upload availability should be checked in the submission portal.

\section*{Acknowledgements}
We thank OpenAQ, the US EPA, DEFRA AURN and the NSW Government for public monitoring data, the Central Pollution Control Board for its bulletins, and the authors of GRIN for code. We are grateful to our school computer science teacher for guidance. AI tools assisted code development, mathematical derivation and checking, data analysis, audits and writing. The authors take responsibility for the content and conclusions.

\begingroup\small\bibliography{references}\endgroup
\end{document}
''')

text='\n'.join(parts)
                                                                           
text=text.replace('.pdf}\\n', '.pdf}\n')
(SRC/'manuscript.tex').write_text(text,encoding='utf-8')
shutil.copy2(ROOT/'SUBMISSION_WORKSPACE/Source/references.bib',SRC/'references.bib')
shutil.copytree(ROOT/'SUBMISSION_WORKSPACE/Source/figures',SRC/'figures',dirs_exist_ok=True)
with zipfile.ZipFile(r'C:\Users\rishi\Downloads\SheafCode_IRIS_Submission_Standalone_and_Supplement.zip') as archive:
    supplemental=archive.read('SheafCode_Supplementary_Technical_Reference.tex').decode('utf-8-sig')
(SRC/'technical_reference.tex').write_text(supplemental,encoding='utf-8')
manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(INPUT.iterdir()) if p.is_file()}
manifest['canonical_source_sha256']=hashlib.sha256((ROOT/'SUBMISSION_WORKSPACE/Source/manuscript.tex').read_bytes()).hexdigest()
(OUT/'INPUT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'source':str(SRC/'manuscript.tex'),'source_words':len(text.split()),'figures':text.count(r'\begin{figure}'),'propositions':text.count(r'\begin{proposition}'),'theorems':text.count(r'\begin{theorem}')},indent=2))
