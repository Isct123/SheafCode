                                                                                       
from pathlib import Path
import difflib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'audit_tools/final_iris_review'
OUT = ROOT / 'SUBMISSION_WORKSPACE/Final_IRIS/Compact_Readable_Final'
SRC = OUT / 'SheafCode_IRIS_Final.tex'
BACKUP = REVIEW / 'before_standard_definition_edit'
BACKUP.mkdir(exist_ok=True)
for name in [SRC.name, 'SheafCode_IRIS_Final_Tightened.pdf', 'SheafCode_Final_Overleaf.zip', 'README.md', 'final_check.json']:
    if not (BACKUP / name).exists():
        shutil.copy2(OUT / name, BACKUP / name)
old = (BACKUP / SRC.name).read_text(encoding='utf-8')
inventory = (REVIEW / 'repetition_replacements.txt').read_text(encoding='utf-8')
paragraphs = {int(m[1]): m[2].strip() for m in re.finditer(r'^\[(\d+)\][ \t]*(.*?)\s*(?=^\[\d+\]|\Z)', inventory, re.M | re.S)}
replacements = {
5: '',
6: r'''SheafCode represents stations as a graph and compares transformed pollutant vectors along its edges. The sheaf maps allow related stations to have different raw concentrations.''',
11: r'''Existing reconstruction approaches include spatial statistics, VAR and low-rank matrix completion \citep{cressie1993statistics,lutkepohl2005var,mazumder2010softimpute}. GRIN uses a graph neural network \citep{cini2022grin}. Its tested bidirectional implementation uses observations before and after the gap, so its comparison is with retrospective sheaf reconstruction. The results concern these implementations, rather than every method in each family.''',
12: r'''Sheaf network and spectral methods are established \citep{ghrist2011network,hansen2019spectral}. Earlier air quality work used sheaves to filter local readings from low-cost sensors with vehicle-related information \citep{pham2022airquality}. Our spatial maps use canonical correlation analysis (CCA) \citep{hotelling1936relations}. An ablation replaces these maps with identity maps to test whether CCA helps beyond comparing normalized readings directly.''',
13: r'''Recovering a decision under bounded errors without identifying the whole state connects this work to bounded-error estimation and functional observability \citep{bounded1990,functional2022}. Sensor placement, test selection and adaptive testing address related measurement choices \citep{krause2008,awasthi2020,douek2021,xing2022robust,cicalese2014adaptive,jia2024noisy}. Our analysis applies these ideas to joint pollution decisions, corrupted responses and continuous changes around archived scenarios.''',
14: r'''The analysis draws on epigraph optimization, verified inequalities and branch and bound \citep{boyd2004,jansson2004,land1960}. Adaptive conformal inference relies on revealed prediction errors \citep{gibbs2021}. A silent station provides no such feedback during its outage, limiting how its intervals can be updated.''',
17: r'''Station $v$ has $p$ pollutant channels, written as a vector $x_v\in\R^p$. Each channel is normalized by subtracting its training mean and dividing by its training standard deviation.''',
18: r'''On edge $uv$, matrices $M_{uv}$ and $N_{uv}$ transform the two stations' readings into a shared space, giving the discrepancy $M_{uv}x_u-N_{uv}x_v$. Two stations may rise and fall together even when one usually records higher concentrations. These transformations use that relationship without forcing their raw readings to be equal.''',
20: r'''Stacking station vectors gives $x$, and stacking spatial discrepancies gives the coboundary $\delta$. Exact agreement means $\delta x=0$. The squared disagreement is $\|\delta x\|_2^2=x^T\delta^T\delta x$, with sheaf Laplacian $\delta^T\delta$. The graph describes measurement relationships rather than physical pollution routes.''',
28: r'''The left-hand matrix is positive definite, so the estimate is unique even when the comparisons alone leave ambiguity. The term $P_{UO}x_O$ lets observed channels inform missing channels at the same station. This procedure is the \emph{decoder}.''',
33: r'''Queries require new readings, such as independently checked measurements. Reusing a recorded value adds no information, and a failed station may not answer. Our simulations reveal archived readings, so savings count measurements rather than field expenditure.''',
45: r'''The correction $e$ changes observed readings $x_O$ to possible clean values $x_O+e$. Its station block is $e_v$.''',
48: r'''The residual inequality applies row by row. Each $Z_F$ is a bounded polyhedron, and their finite union $Z$ is compact, so continuous scores attain their extrema. The union need not be convex. Additional trusted linear constraints may be included. An empty $Z$ indicates inconsistency and leads to abstention. The true state belongs to $Z$ only if the declared assumptions hold \citep{bounded1990}.''',
75: r'''Both states allow error $\eta_v$, so their reading boxes are disjoint only when centres differ by more than $2\eta_v$ in some coordinate. The sets $E(a,b)$ form a conflict hypergraph for the multicover problem below. An exact oracle finds opposite pairs that the current plan cannot distinguish, and a master problem chooses the cheapest plan for the conflicts found so far.
\par\noindent\begin{minipage}{\linewidth}''',
88: r'''Larger scenario changes and measurement errors can allow opposite scenarios to share a response. The theorem gives the exact separation tolerance within the label-preserving range and its dependence on query budget.''',
95: r'''An early answer may rule out enough scenarios to avoid some remaining measurements. The adaptive policy chooses its next station from the candidates still allowed by the responses and stops when they share one decision.''',
106: r'''Each experimental \emph{role} pairs a station cohort with a fixed chronological split. Each timeline has four successive periods. Training fits the model, selection chooses hyperparameters, calibration sets interval and alert cutoffs, and testing measures performance.''',
112: r'''The exploratory expansion attempts 22 cohorts, comprising ten evaluable US counties, four UK regions, six New South Wales (NSW) regions in Australia, and two US counties without test data. The 20 evaluable cohorts contain 73 station memberships, 549,302 observed pollutant-hour cells and 2,114 paired station-days. Memberships can count a station in more than one cohort.''',
116: r'''The main comparison is causal VAR with cross-channel covariance. Other controls use the city mean, seasonal persistence, Gaussian conditioning based on learned covariance, or spatial sheaf alone. Selection settings appear in Table~\ref{tab:model-settings}.''',
119: r'''For each matched station-day, $E$ is mean squared normalized error over scored cells. Against baseline $B$, the percentage gain is''',
143: r'''Equal opposite two-fault offsets give 99.6\% exact support recovery in India and 95.8\% in the US, compared with 0.7\% and 15.0\% for unequal offsets. Drift, stuck readings, gain and one-channel faults also expose weaknesses (Figure~\ref{fig:fault-outage}). The tests use 456 Indian and 432 US hours with required readings available. Restricted candidates and mean-filled context limit interpretation. Reporting at most two faults makes three-fault exact recovery impossible.''',
151: r'''Without query corruption, all 61 settings containing both decision labels allow decision recovery, but only ten allow full-pattern identification. Across those ten, decision repair uses \textbf{18 rather than 24 queries}, saving 25\%, with six strictly cheaper settings. In the other 51, only the decision is identifiable. With one corruption, 15/61 decision plans remain feasible and 46 become impossible.''',
170: r'''Earlier studies examined snapshots, retrospective reconstruction, and other places and periods (Table~\ref{tab:historical-results}). Their results show how conclusions depend on the task and available observations, including cases where sheaf loses. Potentially overlapping archives are not pooled with the recent fixed benchmark. Multiple comparisons use Holm correction.''',
174: r'''At an illustrative hourly PM2.5 reference of 60 $\mu$g/m$^3$, calibrated Indian sheaf alerts detect 69.1\% of 1,177 above-reference hours, compared with VAR's 52.7\%. False positives among 4,012 below-reference hours are 9.6\% and 6.1\%, respectively. Higher sensitivity therefore comes with more false alarms. These illustrative references and receiver operating characteristic (ROC) analyses do not implement official regulatory alerts.''',
}
new = old
for index, replacement in replacements.items():
    before = paragraphs[index]
    assert new.count(before) == 1, index
    new = new.replace(before, replacement, 1)
new = re.sub(r'\n{4,}', '\n\n\n', new)
SRC.write_text(new, encoding='utf-8', newline='\n')
(OUT / 'standard_definition_edit.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile='before_standard_definition_edit/SheafCode_IRIS_Final.tex', tofile='SheafCode_IRIS_Final.tex')), encoding='utf-8')
(REVIEW / 'standard_definition_edit_stats.json').write_text(json.dumps({'edited_paragraphs': len(replacements), 'removed_paragraphs': [5]}, indent=2)+'\n', encoding='utf-8')
print(f'Edited {len(replacements)} paragraphs; saved the previous source, PDF and package.')
