                                                                                          
from pathlib import Path
import os, re, json, shutil, subprocess, hashlib, csv
import fitz

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'IRIS_Submission'
SRC = OUT / 'Editable_Source'
ORIGINAL = ROOT / 'Final_Manuscript/Extended_Validation'

ABSTRACT = r'''Air-quality stations can become unavailable while neighbouring sensors remain active, leaving gaps in pollution monitoring. SheafCode learns vector-valued spatial and temporal consistency relations and reconstructs missing pollutant channels from available observations. The study separates causal reconstruction from retrospective imputation and derives conditional reliability bounds for chosen pollution quantities, unknown station faults and joint pollutant thresholds. On the Delhi concentration cohort, causal reconstruction of 24-hour station outages with 24 hours of pre-outage history achieved 23.2\% lower normalized root mean squared error than a validation-tuned full-covariance vector autoregressive filter (399 paired station-days, 58 days). An independent implementation reproduced every case, with a seven-day calendar-block gain interval of 19.1--27.4\%. A later non-overlapping period gave 9.5\% lower PM2.5 error against the filter across 74 station-days at four matched stations. A fixed-protocol Beijing test found 5.9\% higher error than that filter across 502 station-days, despite gains over simpler baselines. Mathematical results characterize actual structured-map rank, quantity-specific uncertainty, bounded-fault outer certificates and coupled pollutant thresholds. Exact finite rational checks supplement written proofs. Persistent fault and no-feedback interval experiments reveal important operational limitations: stuck and gain faults often evade recovery, and Bengaluru extreme-pollution coverage is only 3.35\% despite 92.15\% overall coverage. A small official-hotspot comparison is descriptive rather than regulatory validation. Historical benchmarks are distinguished from recent observations; archives overlap and most analyses are exploratory. The demonstrated application is monitoring continuity, not measured emissions reduction or health benefit.'''

BODY = r'''
\section{Introduction and objective}
Missing sensor readings can interrupt pollution surveillance precisely when reliable local information is needed. Replacing a dark station by a city average is simple, but suppresses spatial differences. A model that exploits neighbouring channels and pre-outage history may provide a better estimate. Its algebraic recoverability, empirical accuracy and ability to support a pollution decision must nevertheless be tested separately.

The primary research question is whether a vector-valued sheaf model improves \emph{causal} whole-station reconstruction against a validation-tuned full-covariance vector autoregressive (VAR) filter under matched information. Secondary questions concern the reliability of selected pollutant quantities, unknown station faults, joint thresholds and simulated repair. The hypothesis is evaluated through artificial outages with held-out truth, realistic controls and explicit failure tests. Naturally missing target readings cannot verify a reconstruction.

\section{Literature review and innovation}
Cellular sheaves encode compatible local measurements by linear restriction maps and have an established spectral theory \citep{hansen2019spectral,ghrist2011network}. Sheaf-based air-quality filtering predates this project \citep{pham2022airquality}; canonical correlation analysis (CCA) is also established \citep{hotelling1936relations}. Thus a sheaf application alone is not a first-ever contribution.

The project combines training-fitted spatial/temporal maps, causal reconstruction, actual-map rank analysis and quantity-specific decision certificates. Rather than treating accurate reconstruction as guaranteed by rank, it separates prior bias, residual sensitivity, calibration and physical validity. The central mathematical decision result evaluates a \emph{joint} pollutant score over unknown-fault explanations: a coupled feasible set can certify that some pollutant exceeds a threshold even when no individual pollutant is guaranteed to exceed. A second result corrects approximate dual vectors by exact nominal rational arithmetic.

These results specialize functional observability, sparse uniqueness, set-membership estimation and convex optimization \citep{functional2022,donoho2003,bounded1990,boyd2004,jansson2004}. The innovation claim is the project's integrated derivation, implementation and evaluated monitoring application. Unrestricted mathematical priority has not been established. GRIN supplies an additional published graph-imputation control \citep{cini2022grin}; retrospective neural results are kept separate from the causal comparison.

\section{Data and methodology}
\subsection{Sources, chronology and populations}
Historical Indian archives include Delhi, NCR, Hyderabad, Bengaluru and a national network. Direct concentration cohorts include Delhi and Bengaluru; some other archives contain inverted pollutant sub-indices and are not treated as verified instantaneous concentrations. A strict hourly protocol divides the complete timeline chronologically into 60\% training, 20\% validation and 20\% test before station/channel selection. Each retained station/channel needs at least 50\% observed training coverage and 720 observed training hours. A fixed rule maximizes eligible station-channel count, with at least three stations and two channels. Mumbai and Kolkata do not meet this hourly rule; their separate supplied comparisons are supplementary.

\begin{table}[htbp]\centering\small
\caption{Strict hourly cohorts. Hours count timeline positions, not independent observations. Historical archives and direct concentrations are distinct.}
\begin{tabular}{lrrrrr}\toprule
Cohort & Stations & Channels & Training h & Validation h & Test h\\\midrule
Delhi archive &12&2&17784&5928&5928\\
NCR archive &17&2&17784&5928&5928\\
Hyderabad archive &4&4&17021&5674&5674\\
Bengaluru archive &3&2&17784&5928&5928\\
National archive &60&2&17784&5928&5928\\
Delhi concentration &7&4&8202&2734&2735\\
Bengaluru concentration &3&3&36820&12274&12274\\\bottomrule
\end{tabular}\end{table}

Missing hours remain explicit. Normalization, graph selection, CCA and precision fitting use training data; hyperparameters use validation. Related archives share instruments and periods, so seven cohorts are not seven independent replications. Public-source overlap is assessed using identifiers, coordinate matches within 200 metres and dates; calendar overlap alone is not proof of duplicated observations.

An additional UCI Beijing network contains twelve stations and four concentration channels (PM2.5, PM10, NO2, O3), hourly from March 2013 to February 2017 \citep{chen2017beijing}. A fixed training-only correlation graph is used because the archive lacks station coordinates. Delhi-selected sheaf settings are held fixed while Beijing maps are fitted on Beijing training data; this is algorithm transfer, not zero-shot map transfer. A separate VAR is validation-tuned. Every seventh complete test day is selected with a fixed anchor.

Recent-period validation uses public OpenAQ observations from 12 September--2 October 2026 at four geographically matched Delhi instruments \citep{openaq2026}. Only PM mass-concentration channels are used; unverified gas-unit compatibility is avoided. Duplicate station-hour/channel records are averaged explicitly. The later period does not overlap the original Delhi archive endpoint, but uses some of the same instruments. Source dates, units, hashes, fitting settings and split memberships are retained in the technical evidence.

\subsection{Reconstruction model and causal information}
Station $v$ holds a normalized pollutant vector $x_v\in\mathbb R^p$. On edge $e=(u,v)$, weighted CCA maps create discrepancy $A_{e,u}x_u-A_{e,v}x_v$. Stacking discrepancies defines the operator $D$ and Laplacian $D^TD$. Graph alternatives use geography, training correlations or their union. Specified identity fallbacks handle insufficient paired training samples; their tied structure is retained. Temporal relations connect observations one and twenty-four \emph{elapsed} hours apart.

Let $O$ index available coordinates and $U$ unknown coordinates. With a positive-definite block prior $P$ and $\lambda>0$, the decoder minimizes
\begin{align}
\widehat z&=\arg\min_z\|D_Uz+D_Ox_O\|_2^2+\lambda x(z)^TPx(z),\label{eq:objective}\\
\widehat z&=-(D_U^TD_U+\lambda P_{UU})^{-1}
(D_U^TD_O+\lambda P_{UO})x_O.\label{eq:decode}
\end{align}
Here $x(z)$ is the full vector with observed entries $x_O$ and unknown entries $z$ in their original order. The $P_{UO}x_O$ term conditions on surviving channels. In a temporal window $D$ is replaced by the spatial/temporal operator. Natural gaps remain unknown rather than zero observations. Strict convexity gives a unique regularized estimate; it does not establish unregularized identifiability or accuracy.

For a true state $x$ and residual $r=Dx$, writing $G=D_U^TD_U+\lambda P_{UU}$,
\[
\widehat z-x_U=-G^{-1}\{D_U^Tr+\lambda(Px)_U\}.
\]
This separates consistency residual from prior bias. The causal decoder uses a prefix ending at the prediction hour and pre-outage target history. The retrospective decoder may use later surviving-station readings; neither receives hidden target readings. Their information sets and reported gains are not pooled.

\subsection{Controls, selection and statistics}
The main task hides an entire station for 24 hours after 24 hours of history. Methods share target truth cells and surviving context. Controls include seasonal persistence, raw city mean, static Gaussian conditioning and spatial sheaf reconstruction. The stronger VAR uses training-only iterative rank-truncated PCA completion, correlated innovation covariance and validation-selected transform, ridge, noise and covariance shrinkage. Filtering and smoothing are distinguished.

For station-day $i$, $E_i$ is mean squared error over available target cells after division by station/channel training standard deviations. Aggregate RMSE is $\sqrt{N^{-1}\sum_iE_i}$. Relative gain over baseline $B$ is
\[
100\left(1-\sqrt{\frac{\sum_iE_i^{\rm sheaf}}{\sum_iE_i^B}}\right).
\]
Positive values favor the sheaf. This metric differs from physical PM2.5 error and mean run-station RMSE in supporting protocols. Paired calendar-day and seven-/fourteen-day bin resampling preserve within-bin dependence but require resampleable bins. Intervals are exploratory percentile estimates, not guarantees under arbitrary temporal dependence. Sparse-bin results and overlapping cohorts are explicitly identified.

The official bidirectional GRIN implementation is pinned. Artificial target outages, natural-gap masking, validation-selected widths/checkpoints and all three optimization seeds are retained. It uses a retrospective information set. A validation-grid convex hybrid combines sheaf, VAR and seasonal estimates. Examined archives make these analyses exploratory; no retrospective preregistration is claimed.

\subsection{Reliability and fault experiments}
Revealed-label adaptive intervals construct all simultaneous pollutant intervals before revealing that hour's targets, with full/empty sets at the adaptive-level boundaries. The long-run adaptation argument concerns a revealed score stream \citep{gibbs2021}; labels are unavailable during a truly dark outage. Separate fixed intervals receive no hidden-test-target feedback. Model selection had already used validation, so their empirical coverage is not an independent split-conformal guarantee. A fixed positive station scale cancels from an empirical quantile and is not credited with improving coverage.

Unknown-count fault classification uses a training-fitted whitened syndrome and separately calibrated validation cutoffs. Alarm, count, conditional localization and exact end-to-end support are different outcomes. Persistent episodes evaluate signed offsets, unequal blocks, drift, stuck values, gains and single-channel errors. Candidate universes are matched, and true test fault counts are not supplied to the detector. Raw multiplicative gains are applied before the frozen transform; standardized-coordinate gains are separately labeled. A detector with maximum output count two cannot exactly identify three injected faults.

\section{Mathematical results}
\subsection{Structured recoverability and quantity-specific uncertainty}
Generic Hall/matching rank is valid for independent generic entries with the declared support, not automatically for tied CCA/identity maps. For actual maps, assume every edge has a selected $p$-row block invertible at both endpoints. Root a spanning tree in each connected component $C$ and transport $x_v=T_va$. Stack these transports as $T_C$, and set $H_C=D_CT_C$, $q_C=p-\operatorname{rank}H_C$. Then
\[
\ker D_C\simeq\ker H_C,\qquad
\operatorname{rank}D_S=p|S|-\sum_{C\subseteq S}q_C.
\]
\textit{Derivation.} Tree equations determine every block from the root; all remaining rows require $H_Ca=0$. If any vertex is observed as zero in a kernel vector, invertible transport forces the whole component to zero. Only fully erased components contribute null directions; rank-nullity gives the formula. Thin/singular endpoint blocks lie outside this assumption. The fitted seven-network operators have zero numerical kernel at their stated tolerance; this means only the zero state is exactly consistent, not that real pollution is accurately reconstructed.

For $\|Az+b\|_2\leq\varepsilon$, a target $h^Tz$ has a finite residual-ball envelope precisely when $h$ lies in the row space of $A$. With $z_0=-A^+b$ and $\varepsilon\geq\|b+Az_0\|$, its sharp endpoints are
\[
h^Tz_0\ \pm\ \sqrt{\varepsilon^2-\|b+Az_0\|^2}\,\|A^{+T}h\|.
\]
Null directions explain why a quantity may remain identifiable when the full state is not. Sparse exact-syndrome uniqueness for every at-most-$t$ fault requires $d_{\rm block}>2t$, with distance infinity for a zero kernel. For noisy size-$t$ support search, sufficient separation uses omitted-block energy $\sigma_{\min}(D_{F\cup S})\|e_{F\setminus S}\|>2\|r\|$, not the entire fault norm. Classical sparse uniqueness is credited \citep{donoho2003}; full proofs and counterexamples are preserved in the technical reference.

\subsection{Joint pollutant-threshold theorem}
Let $Z$ be a nonempty finite union of bounded-fault feasible polyhedra, restricted to a finite trusted state box. Each support allows at most $t$ faulty stations, componentwise residual budget $\varepsilon$, and bounded station biases. Let $s_k(z)=h_k^Tz+c_k$, and $S(z)=\max_ks_k(z)$.

\begin{proposition}[Sharp joint threshold envelope]
The sharp enclosing interval of $S$ is $[L,U]$. For each support, minimize $y$ subject to its state constraints and $s_k(z)\leq y$ for every $k$; $L$ is the smallest of these LP values. $U$ is the largest support/pollutant affine maximum. All feasible explanations exceed threshold one iff $L>1$; all are at or below it iff $U\leq1$. If $L\leq1<U$, feasible explanations give both decisions.
\end{proposition}
\textit{Proof.} Each nonempty support polyhedron is compact, and the finite union is compact. Continuous $S$ therefore attains its extrema. Epigraph minimization equals $\min S$ on each support; finite support/pollutant maxima commute for $U$. Attained endpoints prove both threshold equivalences and ambiguity. Empty unions report inconsistency. The interval encloses the score range, which need not be continuous. \hfill$\square$

Writing $l_k=\min_Zs_k$, $L\geq\max_kl_k$, possibly strictly. For $z_1+z_2=2$, $0\leq z_k\leq2$, and $s_k=z_k/0.75$, both separate minima are zero while $L=4/3$. Thus every feasible state has an exceedance although neither pollutant individually must exceed. This is a monitoring-specific use of established epigraph optimization \citep{boyd2004}, not an assertion of new general convex theory.

\subsection{Exact nominal correction under unknown faults}
The model is $|Az+b+B_Fe_F|\leq\varepsilon$, $|F|\leq t$, $|e_F|\leq\beta$, $\ell\leq z\leq u$. Let $B_j$ be station $j$'s fault block. For any approximate dual vector $w$, set $\delta=h-A^Tw$ and $\gamma_j=\|B_j^Tw\|_1$. Interpret stored finite coefficients as exact rational numbers.
\begin{proposition}[Verified nominal upper bound]
Every feasible nominal state satisfies
\[
h^Tz\leq-b^Tw+\varepsilon^T|w|
+\beta\operatorname{top}_t(\gamma)
+\sum_k\max(\delta_k\ell_k,\delta_ku_k),
\]
where $\operatorname{top}_t$ sums the $t$ largest nonnegative station leverages. Applying the bound to $-h$ gives a lower bound.
\end{proposition}
\textit{Proof.} Substitute the residual equation in $h^Tz=w^TAz+\delta^Tz$. Its residual term is at most $\varepsilon^T|w|$, at most $t$ bounded station faults contribute no more than the top-$t$ leverages, and maximizing $\delta^Tz$ over the finite box gives the last term. Exact rational evaluation with outward rounding verifies the stated bound. \hfill$\square$

This corrects approximate dual equality but certifies the stored model, not the accuracy of physical maps or LP optimality \citep{jansson2004}. A support-free optimized dual can be wider than exhaustive sparse-support envelopes. Branching/refinement is exact only with exhaustive exact solves; equal-span compression requires unrestricted nuisance coefficients. Declared map-perturbation radii inflate budgets but are simulation assumptions, not measured field uncertainties. Further derivations, calibration conditions, repair objectives and strict-gap examples remain in the full technical reference.

\section{Results and numerical validation}
\subsection{Primary causal reconstruction and independent replay}
Delhi's main causal comparison gives \textbf{23.18\% lower normalized RMSE} than the full-covariance VAR across 399 common station-days and 58 dates. A second implementation independently loads the supplied raw parquet, assembles the fitted quadratic and implements the causal solver and Kalman control without importing project reconstruction/filter code. All individual losses reproduce, with maximum MSE discrepancy $8.88\times10^{-16}$. Its seven-day calendar-bin percentile gain interval is [19.14, 27.44]\%, with sixteen occupied bins. This strengthens implementation reproducibility, not independence of instruments or laboratories.

\begin{table}[htbp]\centering\small
\caption{Selected matched reconstruction results. Positive gain means lower sheaf RMSE. Causal and retrospective information sets and endpoint metrics differ.}
\begin{tabular}{lrrr}\toprule
Comparison & Cases & Days & Gain (\%)\\\midrule
Delhi concentration, causal vs full VAR &399&58&23.18\\
Delhi concentration, retrospective vs full VAR &399&58&26.9\\
Bengaluru concentration, causal vs full VAR &171&60&19.0\\
Later Delhi PM2.5, causal vs full VAR &74&19&9.5\\
Beijing four-pollutant, causal vs full VAR &502&42&-5.93\\
Beijing four-pollutant, causal vs seasonal &502&42&61.73\\
Beijing four-pollutant, causal vs raw city mean &502&42&28.83\\\bottomrule
\end{tabular}\end{table}
\begin{figure}[htbp]\centering\includegraphics[width=.96\linewidth]{figures/fig_causal.pdf}
\caption{Matched causal 24-hour whole-station outages after 24 hours of history. Delhi: 399 station-days/58 dates; Bengaluru: 171/60. Bars are nominal paired-day ratio intervals. Bengaluru does not beat seasonal persistence in this component comparison.}\end{figure}

Across eligible hourly cohorts the sheaf favors the tested diagonal-innovation VAR, static Gaussian and spatial sheaf controls numerically, but the stronger full-covariance comparison is available only for the two concentration cohorts. Delhi calendar-bin sensitivity preserves the gain; the wider ten-contrast sensitivity lower endpoints range from 15.7\% to 16.4\%. These are exploratory diagnostics rather than exact familywise guarantees. The hybrid improves Bengaluru relative to its sheaf component by 5.6\%, but worsens Delhi by 3.5\%.

\subsection{Later period, independent network and neural controls}
The four-station later-period PM2.5 comparison contains 74 station-days and nineteen dates. Gains are 9.5\% against full VAR (nominal day interval [5.0,13.1]\%), 33.6\% against seasonal persistence, and 2.2\% against city mean ([-4.8,8.2]\%). Only three/four longer bins are occupied, limiting dependence-aware inference. This is temporal validation on existing instruments.

\begin{figure}[htbp]\centering\includegraphics[width=.96\linewidth]{figures/fig_later-period.pdf}
\caption{Recent non-overlapping period at four matched instruments, with PM-only context and matched history. Nominal paired-day intervals do not establish general external replication; the city-mean interval includes zero.}\end{figure}

Beijing loses to full VAR: four-pollutant error is 5.93\% higher, with gain interval [-8.16,-3.71]\%; PM2.5 error is 8.54\% higher across 501 cases. The sheaf nevertheless beats raw city mean and seasonal persistence. The raw-city comparator averages concentrations before target normalization; the initial standardized-mean diagnostic is preserved separately, not used as the publication comparator. This negative network outcome is retained rather than followed by test-based tuning.

\begin{figure}[htbp]\centering\includegraphics[width=.96\linewidth]{figures/online_02_network_transfer.pdf}
\caption{Beijing: twelve stations, 502 four-pollutant station-days/42 dates, and 501 PM2.5 cases. Seven-day calendar-group intervals are exploratory. Negative reduction denotes higher sheaf error.}\end{figure}

Official bidirectional GRIN lowers retrospective Delhi RMSE relative to the sheaf by 4.4\%, 2.6\% and 4.7\% across three seeds. In Bengaluru, its gains are -9.7\%, -8.3\% and -11.0\%. All seeds/checkpoints are retained. This rejects universal state-of-the-art superiority; it does not erase the separately measured causal sheaf-versus-VAR advantage. History/horizon stress also retains unfavorable zero-history and seasonal comparisons. Checkpoint-hour and full station-day metrics are distinct.

\subsection{Persistent faults and uncertainty failure boundaries}
The frozen unknown-count detector is evaluated in 24-hour episodes. Of forty-eight planned starts, 29 Delhi and 33 Bengaluru starts have at least two fully observed stations throughout the episode. Most two-fault scenarios contain 696 and 792 dependent hours respectively (Bengaluru stuck: 768). Delhi exact alarm-and-support success is 96.98\% for same-sign shifts, 97.27\% for opposite signs, but only 10.06\% for unequal faults, 36.64\% for drift and 1.44\% for stuck readings. Corresponding Bengaluru success is 10.10\%, 44.32\%, 8.21\%, 2.40\% and 0.26\%. Clean alarm rates are 3.30\% and 6.06\%. Raw concentration gain 1.5 gives only 6.61\% and 0.76\% exact support success.

\begin{figure}[htbp]\centering\includegraphics[width=.96\linewidth]{figures/online_03_persistent_faults.pdf}
\caption{Persistent two-fault end-to-end support recovery. Count is unknown to the predictor. Standardized gains and raw-concentration multiplication are different perturbations. Strong signed-shift recovery does not imply general sensor-fault robustness.}\end{figure}

The expanded no-feedback interval study evaluates every available target under snapshot station hiding and 0/20/50\% added context losses, with three fixed interval methods. It contains 459,216 method/case rows, not independent observations. At zero extra loss, station-fixed PM2.5 coverage is 93.69\% over 16,692 Delhi cases, with mean full width 86.0 micrograms/m$^3$; Bengaluru coverage is 92.15\% over 34,332 cases, width 51.5. Above station-specific training 95th-percentile pollution, Bengaluru coverage falls to \textbf{3.35\% across 837 cases}, width 61.1. Delhi has only one such extreme case, insufficient for a tail conclusion. Calendar-group intervals with fewer than five clusters are suppressed.

\begin{figure}[htbp]\centering\includegraphics[width=.96\linewidth]{figures/online_04_uncertainty_shift.pdf}
\caption{No-feedback uncertainty: coverage, width and decision yield must be considered together. The 90\% line is a calibration target, not a shifted-distribution guarantee. High overall coverage hides severe Bengaluru extreme-pollution failure.}\end{figure}

\subsection{Pollution alerts and official-record comparison}
Validation-frozen PM2.5 alert thresholds use an illustrative hourly 60 micrograms/m$^3$ reference. At 24-hour outages with no added context losses, Delhi sensitivity is 31.0\% for sheaf and 22.9\% for city mean, with false-positive rates 2.5\% and 1.3\% over 16,530 common station-hours (3,645 positive). Bengaluru sensitivity is 42.4\% versus 33.2\%, false-positive rates 3.3\% versus 3.8\%, over 34,173 hours (2,026 positive). The weekly paired Bengaluru sensitivity-difference interval [-4.9,24.0] percentage points crosses zero. This does not establish a reliable deployed alert benefit.

Official CPCB documents are acquired at protocol-fixed weekly dates, retaining source PDFs, dates and hashes. Of 34 requested documents, 33 are acquired. Detailed bulletins list selected hotspots rather than a representative station sample. Fourteen eligible common station-days across ten dates remain after coverage/window alignment. A selected four-pollutant proxy uses the preceding 24-hour window ending at 16:00 and eight-hour ozone averaging. Category agreement with official records is 13/14 for observed proxy, 12/14 for sheaf and 6/14 for city fallback. Omitted channels, units and official rolling/validity rules prevent treating this as official AQI recovery \citep{cpcbresources}. It is descriptive, not an actual GRAP invocation study \citep{caqm2025}.

\subsection{Certificates, repair and supporting research}
Bounded-fault experiments preserve 11,360 method/case rows and 10,512 checks for assumption-satisfying cases. Joint residual/history budgets can narrow bounds, but useful decisions remain rare. A synthetic eighteen-candidate, three-fault runtime test reports about 408-fold median speedup for a support-free dual across three timing repeats, at the cost of wider bounds. This is a synthetic tradeoff, not field acceleration. Equal-span compression gives no benefit on the fitted Delhi/Bengaluru examples and may run slower.

The joint-threshold theorem's strict improvement is realized in fifty designed coupled examples. Neither twelve-case learned-map synthetic group produces additional exceedance decisions. Gaussian trace repair selection improves error against expected random repair in six of seven cohorts, but is essentially indistinguishable from a variance-only policy and worse in one archive. Worst-case functional-width selection has no established mean-error advantage; four inconsistent Delhi pairs are excluded explicitly. Model-optimal repair objectives do not imply measured maintenance benefit.

Separate supporting snapshot/window studies remain in the technical reference: primary single-station spatial gains in seven of eight comparisons, median 3.1\%, with two adjusted significant comparisons; retrospective prolonged gains in nine of nine configurations, median 34.0\%, with five of nine supplementary run-based adjusted comparisons; risk-ranking directions in seven of seven evaluable networks, five adjusted comparisons. These use different populations, information sets and units and do not replace the causal headline. Historical thinning reports 293 predicted versus 294 observed non-identifiable singles and 203 total rank disagreements, rather than perfect agreement.

Exact rational verification passes 37,678 finite checks, including exhaustive 19,683 ternary three-station operators, sparse collisions, noisy-support boundaries, joint-score examples, equal spans, dual corrections and adaptive telescoping. Seven additional unit tests and 161,411 evidence aggregation/source checks pass. Counts are verification workload, not environmental samples or unrestricted formal proof certification.

\section{Discussion, environmental relevance and limitations}
The strongest demonstrated empirical result is reproducible causal reconstruction in the Delhi concentration cohort, supported by shorter recent-period validation. The central theoretical contribution is conditional decision analysis that distinguishes recovering a vector, identifying a quantity and certifying a coupled exceedance. The method is useful only when its information set, residual assumptions and uncertainty behavior are appropriate.

Environmental relevance is \emph{monitoring continuity}: better estimates during communication gaps could support local pollution assessment and prioritization. Reconstruction does not remove pollutants. No emissions intervention, field repair, exposure reduction or health benefit was measured. Model estimates at naturally missing cells remain exploratory; the historical 35 modeled Poor-or-worse days among 151 heavily missing station-days are not verified unmeasured exposure.

Beijing losses, Delhi neural-control losses, realistic fault failures, sparse alert evidence and extreme-pollution undercoverage are substantive performance limits. Regularization ensures a unique answer, not a correct one. More equations can increase actual decoder error. Generic matching assumptions cannot be transferred to structured fitted maps. Compact-set certificates are conditional on finite bounds and model validity; calibration does not establish exchangeability during distribution shift. Exact rational evaluation certifies stored coefficients rather than true physical relations.

Dataset provenance matters for both scientific interpretation and competition eligibility. Historical observations are dated explicitly; recent computations do not make them recently collected environmental measurements. Recent-period validation is identified separately. Historical public benchmarks are supporting evidence whose acceptance under any data-age rule requires organizer clarification. Dataset geography throughout refers to monitoring locations, not the student's school or residence.

\section{Conclusion}
SheafCode reduces matched causal Delhi outage RMSE by 23.18\% against a tuned full-covariance VAR, reproduces this result independently at implementation level, and obtains a favorable shorter later-period PM2.5 comparison. Actual-map rank, functional envelopes, joint-threshold LPs and exact nominal dual correction make reliability assumptions explicit. Additional-network, neural, fault and uncertainty tests limit universal and operational claims. The supported impact is computational monitoring continuity with conditional decision analysis; cleaner air and stronger deployment guarantees remain unmeasured.

\section*{Acknowledgements and assistance disclosure}
\addcontentsline{toc}{section}{Acknowledgements and assistance disclosure}
Public data providers and the cited software/research authors are acknowledged. AI assistance was used in code development, mathematical derivation and checking, experimental auditing, and manuscript drafting. Assistance is disclosed rather than presented as independent student-only work. No mentor identity or institutional affiliation is inferred here. The student must review the text, explain the research and accurately document personal contributions and supervision separately.

The accompanying technical reference preserves full derivations, study tables and audit trails. Frozen code/data identities and prediction CSVs remain in the validated research package. No underlying evidence was refitted or discarded for this paper.
'''

SYNOPSIS = {
'Introduction and objective': '''Air-quality monitoring loses local information when stations stop transmitting. City-average substitution can conceal spatial differences, while complex reconstruction may appear reliable despite large model error. This project asks whether learned vector-valued spatial and temporal consistency relations improve causal reconstruction during whole-station outages, and when reconstructed pollutant quantities support defensible threshold decisions. The primary objective is a matched comparison against a validation-tuned full-covariance vector autoregressive filter, using only surviving observations and pre-outage history. Secondary objectives are to derive conditional quantity and joint-threshold bounds, evaluate unknown-count fault detection, examine uncertainty under pollution shift, and assess monitoring relevance against archived observations. The hypothesis concerns improved monitoring estimates under specified conditions, rather than measured emissions reduction. Historical public benchmarks and recent-period validation are explicitly distinguished.''',
'Innovation': '''The contribution combines causal vector-sheaf reconstruction with actual structured-map rank analysis and pollution-specific reliability certificates. A coupled feasible-state model can guarantee that at least one pollutant exceeds a threshold even when no individual pollutant has a guaranteed exceedance. An exact nominal correction makes approximate dual bounds valid for stored coefficients. These application-specific derivations and implementations extend an integrated monitoring framework; they do not claim invention of sheaves, sparse uniqueness or convex optimization. Empirical failure tests distinguish algebraic recoverability from operational reliability.''',
'Methodology': '''Complete hourly timelines are split chronologically into training, validation and test before cohort selection. Training determines eligible station/channels, normalization, graphs, CCA restriction maps and covariance priors. Validation selects hyperparameters and baseline settings. Artificial 24-hour whole-station outages follow 24 hours of history; causal estimates use only prefixes ending at the prediction hour. Matched baselines include seasonal persistence, raw city mean and a tuned full-covariance VAR filter. Paired station-day errors use training-standardized target channels, with calendar-block sensitivity analyses. Retrospective GRIN comparisons use a separate information set. A second independent implementation replays every Delhi headline case from the supplied raw data. Further tests use a protocol-fixed twelve-station Beijing benchmark and a recent four-station Delhi period. Mathematical proofs cover structured rank, quantity envelopes, bounded unknown faults and joint thresholds; exact finite rational checks supplement them. Persistent synthetic faults retain clean alarms, count errors and failed localization. Fixed uncertainty intervals receive no hidden test-target feedback and are evaluated overall and in training-defined pollution extremes. Official selected-hotspot records provide a small common-eligibility proxy comparison, with regulatory limitations retained. No test-based tuning or pooling of unlike protocols is used to manufacture the headline.''',
'Results and conclusions': '''Delhi causal reconstruction reduces normalized RMSE by 23.18% against full VAR across 399 station-days and 58 dates; independent replay reproduces every case. The exploratory seven-day gain interval is 19.14-27.44%. A recent non-overlapping period gives 9.5% lower PM2.5 error across 74 station-days. Beijing instead has 5.93% higher four-pollutant error than VAR, although simpler baselines are beaten. Persistent stuck/gain faults often fail. Bengaluru fixed intervals achieve 92.15% overall coverage but only 3.35% on 837 extreme-pollution cases. Conditional joint-threshold proofs and 37,678 exact finite checks support the mathematics, without certifying first-ever priority or physical validity. The result supports monitoring continuity under specified conditions, not universal superiority, official policy recovery or measured environmental improvement.''',
'Acknowledgements and references': '''Public data providers and cited software authors are acknowledged. AI assisted code development, derivation/checking, auditing and drafting; personal contributions and supervision must be documented accurately. Key sources: Hansen and Ghrist (2019), cellular sheaf spectral theory; Pham et al. (2022), prior sheaf air-quality monitoring; Cini et al. (2022), GRIN; Donoho and Elad (2003), sparse uniqueness; Boyd and Vandenberghe (2004), convex optimization; Gibbs and Candes (2021), adaptive conformal inference; UCI Beijing Multi-Site Air Quality, DOI 10.24432/C5RK5G. Full bibliographic records appear in the research paper.'''
}

PREAMBLE = r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx,microtype,tabularx}
\usepackage[authoryear,round]{natbib}\bibliographystyle{plainnat}
\usepackage{hyperref}\hypersetup{colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue,pdfauthor={},pdftitle={SheafCode: Causal Reconstruction and Conditional Pollution-Decision Certificates}}
\usepackage{fancyhdr}\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small SheafCode}\fancyfoot[C]{\thepage}\setlength{\headheight}{14pt}
\setlength{\parindent}{0pt}\setlength{\parskip}{.42em}\setlength{\emergencystretch}{2em}
\newtheorem{proposition}{Proposition}
\begin{document}
'''

def words(s):
    s = re.sub(r'\\[a-zA-Z]+(?:\[[^]]*\])?', ' ', s)
    return len(re.findall(r"\b[\w]+(?:[.'%-][\w]+)*\b", s))

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def compile_pdf(name, target):
    env = os.environ.copy(); env['TECTONIC_CACHE_DIR'] = str(ROOT/'audit_tools/tex_cache')
    r = subprocess.run([str(ROOT/'audit_tools/tectonic/tectonic.exe'), '--only-cached', '--keep-logs', '--outdir', str(SRC), str(SRC/(name+'.tex'))],cwd=SRC,env=env,capture_output=True,text=True)
    (OUT/'Review_Notes'/f'{name}_compile.txt').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
    if r.returncode: raise RuntimeError(r.stdout+r.stderr)
    shutil.copy2(SRC/(name+'.pdf'), target)

def main():
    for d in [SRC, SRC/'figures', OUT/'Upload_Files', OUT/'Supporting_Reference', OUT/'Review_Notes']: d.mkdir(parents=True,exist_ok=True)
    original_pdf=ROOT/'Final_Manuscript/SheafCode_Validated_Manuscript.pdf'
    original_hash=sha(original_pdf)
    for fn in ['fig_causal.pdf','fig_later-period.pdf','online_02_network_transfer.pdf','online_03_persistent_faults.pdf','online_04_uncertainty_shift.pdf']:
        shutil.copy2(ORIGINAL/'figures'/fn,SRC/'figures'/fn)
    shutil.copy2(ORIGINAL/'references.bib',SRC/'references.bib')
    title=r'''\begin{titlepage}\centering\vspace*{2cm}
{\LARGE\bfseries SheafCode\par}\vspace{.5cm}
{\Large Causal Air-Quality Reconstruction and Conditional Pollution-Decision Certificates\par}
\vspace{1.5cm}{\large Research paper\par}\vspace{.4cm}{6 October 2026\par}
\vfill\begin{minipage}{.88\linewidth}\small
\textbf{Research scope.} Computational reconstruction of public monitoring observations, mathematical reliability analysis, and held-out artificial-outage evaluation. Historical benchmark observations and recent-period validation are identified separately. Dataset place names identify monitoring locations only.\par\medskip
\textbf{Assistance.} AI assistance in development, mathematical checking, auditing and writing is disclosed in the acknowledgements. This paper does not certify independent student-only authorship or first-ever priority.
\end{minipage}\end{titlepage}
\tableofcontents\clearpage
'''
    title=title.replace(r'\begin{titlepage}',r'\hypersetup{pageanchor=false}\begin{titlepage}').replace(r'\end{titlepage}',r'\end{titlepage}\pagenumbering{roman}\hypersetup{pageanchor=true}').replace(r'\tableofcontents\clearpage',r'\tableofcontents\clearpage\pagenumbering{arabic}')
    paper=PREAMBLE+title+'\\begin{abstract}\n'+ABSTRACT+'\n\\end{abstract}\n'+BODY.replace(r'$\square$', ' QED')+'\n\\bibliography{references}\\end{document}\n'
    (SRC/'research_paper.tex').write_text(paper,encoding='utf-8')
    synopsis=PREAMBLE+r'\begin{center}{\Large\bfseries SheafCode}\par Causal Air-Quality Reconstruction and Conditional Pollution-Decision Certificates\par\small Research synopsis\end{center}'+'\n'
    synopsis+=r'\section*{Project abstract}'+'\n'+ABSTRACT+'\n'
    for k,v in SYNOPSIS.items():
        synopsis+=r'\section*{'+k+'}\n'+v.replace('%',r'\%')+'\n'
    synopsis+=r'\end{document}'
    (SRC/'synopsis.tex').write_text(synopsis,encoding='utf-8')
    text=ABSTRACT.replace(r'\%', '%').replace('--','–')
    (OUT/'Upload_Files/Project_Abstract.txt').write_text(text+'\n',encoding='utf-8')
    fields='Project title: SheafCode: Causal Air-Quality Reconstruction and Conditional Pollution-Decision Certificates\n\nProject abstract\n'+text+'\n\n'
    for k,v in SYNOPSIS.items(): fields+=k+'\n'+v+'\n\n'
    (OUT/'Upload_Files/Portal_Text.txt').write_text(fields,encoding='utf-8')
    compile_pdf('research_paper',OUT/'Upload_Files/SheafCode_Research_Paper.pdf')
    compile_pdf('synopsis',OUT/'Upload_Files/SheafCode_Research_Synopsis.pdf')
    shutil.copy2(original_pdf,OUT/'Supporting_Reference/Complete_Technical_Reference.pdf')
    rules='''# Published IRIS format checked 6 October 2026

Sources: https://iris.exstemplar.com/guidelines.html and https://register.irisnationalfair.org/ ; examples: https://register.irisnationalfair.org/examples/research-paper and https://register.irisnationalfair.org/examples/synopsis . The guidelines page carries a 2025–26 edition label; the registration page explicitly addresses the current event. Requirements visible in a signed-in application have not been inspected.

The public pages ask for synopsis and full-paper PDFs, an abstract, project details and a YouTube video link. PDF caps are 5 MB and 20 MB respectively; the video maximum is 90 seconds. A paper includes a title page, contents, methods, discussion, conclusion, acknowledgements and references. No hard paper page limit or compulsory font was found. The shorter paper length is an editorial choice.

Published section word ranges: introduction/objective 100–150; innovation 50–100; methodology 150–250; results/conclusions 100–150; acknowledgements/references 50–100. Abstract target 250 words, portal hard cap 300. These apply to synopsis/portal text, not each full-paper section.

Anonymity excludes student school/location identifiers; none are added. Geographic dataset names remain to make provenance scientifically intelligible. Experimental-data age is limited to twelve months by the public guidelines. Historical public benchmark acceptance needs organizer clarification; recent analysis dates do not change observation dates. Personal eligibility, supervision, contribution records, and any signed forms cannot be certified from the repository.
'''
    (OUT/'Review_Notes/REQUIREMENTS.md').write_text(rules,encoding='utf-8')
    changes='''# What was shortened or omitted from the upload paper

No underlying research results, code, data, or the 66-page manuscript were deleted. The complete manuscript is copied to Supporting_Reference/Complete_Technical_Reference.pdf; it is a technical evidence record, not an additional mandatory upload.

## Retained in the focused paper
- Main causal reconstruction comparison, matched information, correct metric and statistical units, independently replayed results and interval.
- Recent-period validation, Beijing loss to the stronger VAR, neural comparisons across every seed, seasonal/hybrid caveats and overlap.
- Structured-map derivation, functional envelope, sparse/noisy fault boundary, and full short proofs of the joint-threshold and exact nominal dual-correction results.
- Persistent fault failures, clean false alarms, severe tail undercoverage, alert uncertainty, selected official-record denominators, and environmental limitations.
- Summaries of all other major study families, historical benchmark distinctions, assistance disclosure and full cited bibliography.

## Moved to the preserved technical reference
- Complete catalogue of fifteen numbered mathematical propositions/corollaries and their long derivations; the focused paper gives the central proofs and summarizes the remaining results.
- Exhaustive supplied snapshot/window, channel, thinning, outage ranking and cohort-specific tables; numerical summaries and caveats remain in the paper.
- Detailed per-network/per-seed tables, alternative statistical scales, all history/horizon and missingness settings, and run-level sensitivity tables.
- Full interval feedback/boundary diagnostics, given-count decoding experiments, equal-span/computation/refinement/repair benchmarks and their counterexamples.
- Acquisition/replay details, data/config hashes, every test/audit record and the 36-study-family inventory. These remain in the original validated research package as well.
- Twenty-two of the twenty-seven figure placements. The five retained plots show causal reconstruction, recent-period validation, Beijing generalization, persistent fault failures and uncertainty shift. Neural, operational-alert, computation, repair and joint-threshold outcomes remain in text; their plots remain in the technical reference.

## Editorial changes
- Replaced the sprawling reference layout with a title page, contents, introduction/objective, literature/innovation, reproducible methods, focused mathematics, results, discussion, conclusion and references.
- Removed reader-navigation and audit-process repetition from the upload narrative; preserved assistance disclosure and all scientifically consequential limitations.
- Added a separate short synopsis and portal text within published word ranges. It describes completed work accurately rather than fabricating a prospectively written plan.
- Added an explicit observation-age distinction. Historical benchmarks are not relabeled as newly collected data.
- Added no student names, school, residence, signatures, mentor identities or unsupported novelty/impact claims. Dataset locations remain essential provenance.
'''
    (OUT/'Review_Notes/WHAT_WAS_SHORTENED.md').write_text(changes,encoding='utf-8')
    readme='''# IRIS submission documents

## Intended upload files
1. Upload_Files/SheafCode_Research_Paper.pdf — focused full research paper.
2. Upload_Files/SheafCode_Research_Synopsis.pdf — short synopsis using published section ranges.
3. Upload_Files/Project_Abstract.txt — abstract for the portal.
4. Upload_Files/Portal_Text.txt — copy/paste short sections if the application asks for them. Do not upload this text as an extra scientific PDF unless requested.

Editable_Source contains reproducible LaTeX and the five selected existing plot PDFs. Supporting_Reference/Complete_Technical_Reference.pdf preserves the complete research record; upload it only if supplementary material is explicitly allowed. Review_Notes explains format sources, shortening decisions and checks. Nothing has been submitted to IRIS by this script.

## Before clicking Submit
- Resolve whether historical public benchmarks satisfy IRIS's data-age restriction. The current public guidelines say experimental data must be no more than twelve months old. These documents preserve actual dates; they cannot certify organizer acceptance. The recent-period study is genuine separate evidence, not a relabeling of old observations.
- Review and understand the manuscript and assistance disclosure, and accurately record your own contributions, supervising guide and any required forms in the application. Names/affiliations were not invented or added to anonymous PDFs.
- Supply the required YouTube link (maximum 90 seconds), choose the category matching your actual contribution, and complete student/team details. Those are not present in the repository package.
- Check any additional signed-in portal constraints. Public requirements were checked; the private application was not inspected. Do not upload the whole folder or the large technical ZIP into the paper slot.

The scientific package is condensed and checked. Eligibility, personal attestations and video completion remain the submitter's actions. The original full manuscript and validated research package are unchanged.
'''
    (OUT/'README.md').write_text(readme,encoding='utf-8')
    checks=[]
    def check(name,ok):
        checks.append({'check':name,'passed':bool(ok)})
    check('original_manuscript_unchanged',sha(original_pdf)==original_hash)
    check('abstract_at_most_250_words',words(ABSTRACT)<=250)
    data=ROOT/'Research_Release/online_validation_results'
    replay=json.loads((data/'replay_summary.json').read_text(encoding='utf-8'))
    check('headline_gain_matches_replayed_results',round(replay['comparison']['gain_percent'],2)==23.18)
    check('headline_cases_and_dates_match',replay['comparison']['cases']==399 and replay['comparison']['days']==58)
    check('headline_calendar_interval_matches', [round(v,2) for v in replay['comparison']['ci95']]==[19.14,27.44])
    interval_rows=list(csv.DictReader((data/'uncertainty_shift_corrected_summary.csv').open(encoding='utf-8',newline='')))
    for region,group,expected_cases,expected_coverage in [('delhi2025','all',16692,93.69),('bengaluru_raw','all',34332,92.15),('bengaluru_raw','True',837,3.35)]:
        row=next(r for r in interval_rows if r['region']==region and float(r['extra_missing_rate'])==0 and r['method']=='station_fixed' and r['group']==group and r['group_type']==('all' if group=='all' else 'train_extreme'))
        check('coverage_source/'+region+'/'+group,int(row['cases'])==expected_cases and round(100*float(row['coverage']),2)==expected_coverage)
    official=json.loads((data/'official_summary.json').read_text(encoding='utf-8'))
    check('official_record_denominator_matches',official['eligible_test_station_days']==14 and official['downloaded_files']==33)
    exact=json.loads((data/'exact_math_results.json').read_text(encoding='utf-8'))
    check('exact_finite_check_count_matches',len(exact['checks'])==37678 and all(c['passed'] for c in exact['checks']))
    limits=[(100,150),(50,100),(150,250),(100,150),(50,100)]
    counts={}
    for (k,v),(lo,hi) in zip(SYNOPSIS.items(),limits):
        counts[k]=words(v); check('section_word_range/'+k,lo<=words(v)<=hi)
    pdfs={}
    for fn,cap in [('SheafCode_Research_Paper.pdf',20_000_000),('SheafCode_Research_Synopsis.pdf',5_000_000)]:
        p=OUT/'Upload_Files'/fn
        doc=fitz.open(p); full='\n'.join(page.get_text() for page in doc)
        pdfs[fn]={'pages':len(doc),'bytes':p.stat().st_size,'sha256':sha(p)}
        check(fn+'/size_cap',p.stat().st_size<=cap)
        check(fn+'/no_missing_references','??' not in full)
        check(fn+'/no_iteration_labels',not re.search(r'\bv[1-4]\b',full,re.I))
        check(fn+'/no_student_path','Users\\rishi' not in full and 'OneDrive' not in full)
        check(fn+'/not_false_first_ever_claim','first-ever contribution' not in full or 'not a first-ever contribution' in full)
        for n,page in enumerate(doc):
            check(fn+f'/page_{n+1}_nonempty',len(page.get_text().strip())>15)
            for block in page.get_text('dict')['blocks']:
                if block['type']!=0: continue
                x0,y0,x1,y1=block['bbox']; check(fn+f'/page_{n+1}_text_inside_page',x0>=-1 and y0>=-1 and x1<=page.rect.width+1 and y1<=page.rect.height+1)
                                           
        for start in range(0,len(doc),12):
            sheet=fitz.open(); page=sheet.new_page(width=900,height=1200)
            for i in range(start,min(start+12,len(doc))):
                idx=i-start; rect=fitz.Rect((idx%3)*300,(idx//3)*300,(idx%3+1)*300,(idx//3+1)*300)
                page.show_pdf_page(rect,doc,i)
            page.get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(OUT/'Review_Notes'/f'{p.stem}_contact_{start//12+1}.png')
        doc.close()
    for fn in ['research_paper','synopsis']:
        log=(SRC/(fn+'.log')).read_text(encoding='utf-8',errors='replace')
        check(fn+'/no_undefined_citations',not re.search(r'(Citation.*undefined|Reference.*undefined|undefined references)',log))
        check(fn+'/no_overfull_boxes','Overfull' not in log)
    audit={'status':'passed' if all(c['passed'] for c in checks) else 'failed','abstract_words':words(ABSTRACT),'synopsis_section_words':counts,'pdfs':pdfs,'checks':checks,'source_full_manuscript_sha256':original_hash,'scope':'Document/content checks only. Data-age eligibility and private portal requirements unresolved.'}
    (OUT/'Review_Notes/DOCUMENT_CHECKS.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k!='checks'},indent=2))
    failed=[c['check'] for c in checks if not c['passed']]
    if failed: raise RuntimeError('Document checks failed: '+', '.join(failed))

if __name__=='__main__': main()
