                                                                          

                                                                          
                                                                          
   
from pathlib import Path
import csv
import difflib
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'SUBMISSION_WORKSPACE/Final_IRIS'
BASE = ROOT / 'SUBMISSION_WORKSPACE/Standalone_IRIS'
INPUT = Path(r'C:\Users\rishi\.codex\attachments\a775a17b-76c1-4c98-9a10-395c3b5ff76a\Pasted text.txt')
source = INPUT.read_text(encoding='utf-8-sig')
baseline = (BASE / 'SheafCode_IRIS_Standalone.tex').read_text(encoding='utf-8')
figures = re.findall(r'\\begin\{figure\}.*?\\end\{figure\}', source, re.S)
assert len(figures) == 11
by_label = {re.search(r'\\label\{([^}]+)\}', f).group(1): f for f in figures}

def read_csv(relative):
    with (ROOT / relative).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

radius_rows = read_csv('Research_Release/decision_repair_results/RADIUS_CONFIGURATIONS.csv')
runtime = json.loads((ROOT / 'Research_Release/math_integration_results/REPORT_FACTS.json').read_text())
transfer = read_csv('Research_Release/operational_validation_results/COHORT_TRANSFER.csv')
cohorts = read_csv('Research_Release/expanded_validation_results/COHORT_RESULTS.csv')
cohort_order = [r['cohort'] for r in cohorts if r['code'][0] in 'DEF' and r['cases'] != '0']
interval_rows = read_csv('Research_Release/weakness_resolution_results/uncertainty_decisions/INTERVAL_SUMMARY.csv')
widths = {
    'network': '.98', 'workflow': '.98', 'joint': '.43', 'repair': '.96',
    'primary': '.86', 'external': '.76', 'robust': '.96', 'risk': '.96',
    'radius': '.90', 'operational': '.98', 'math': '.98',
}
labels = {
    'network': 'network', 'workflow': 'causal-workflow', 'joint': 'joint-example',
    'repair': 'repair-intuition', 'primary': 'headline', 'external': 'external',
    'robust': 'fault-outage', 'risk': 'risk', 'radius': 'repair-radius',
    'operational': 'operational-validation', 'math': 'math-integration',
}
notes = []
diagram_sources = {}
for key, destination in labels.items():
    figure = by_label['fig:' + key]
    start = figure.index(r'\begin{tikzpicture}')
    end = figure.index(r'\end{tikzpicture}') + len(r'\end{tikzpicture}')
    diagram = figure[start:end]
    if key == 'repair':
        diagram = diagram.replace('(axis cs:-4.9,1) -- (axis cs:5.0,1)', '(axis cs:-4.9,1) -- (axis cs:4.9,1)')
        diagram = diagram.replace('(axis cs:5.1,1) -- (axis cs:15,1)', '(axis cs:5.1,1) -- (axis cs:14.9,1)')
        notes.append('rho=3.9 intervals corrected to [-4.9,4.9] and [5.1,14.9].')
    if key == 'math':
        diagram = diagram.replace('(0,0.0255)', f"(0,{runtime['median_exhaustive_seconds']:.8f})")
        diagram = diagram.replace('(1,0.0786)', f"(1,{runtime['median_search_seconds']:.8f})")
        notes.append('Runtime bars use saved medians: 0.02499815 and 0.07701090 seconds.')
    if key == 'radius':
        plots = list(re.finditer(r'\\addplot coordinates \{.*?\};', diagram, re.S))
        assert len(plots) == 2
        for match, faults in reversed(list(zip(plots, [0, 1]))):
            rows = [r for r in radius_rows if r['status'] == 'certified_inside_radius' and int(r['query_faults']) == faults]
            assert len(rows) == (61 if faults == 0 else 15)
            assert all(float(r['same_cost_radius']) >= float(r['baseline_radius']) for r in rows)
            coords = ' '.join(f"({float(r['baseline_radius']):.8g},{float(r['same_cost_radius']):.8g})" for r in rows)
            diagram = diagram[:match.start()] + r'\addplot coordinates {' + coords + '};' + diagram[match.end():]
        notes.append('Radius scatter coordinates regenerated from saved CSV with consistent precision; no optimization rerun.')
    if key == 'risk':
        plots = list(re.finditer(r'\\addplot\[c(?:blue|orange)\] coordinates \{.*?\};', diagram, re.S))
        assert len(plots) == 4
        settings = [('all', 'station_fixed'), ('all', 'risk_station'), ('extreme', 'station_fixed'), ('extreme', 'risk_station')]
        point_sets = []
        for group, method in settings:
            rows = {r['cohort']: r for r in interval_rows if r['group'] == group and r['method'] == method and float(r['context_loss']) == 0}
            point_sets.append({20-i: 100*float(rows[c]['coverage']) for i, c in enumerate(cohort_order) if c in rows and int(rows[c]['cells']) > 0})
        for match, points in reversed(list(zip(plots, point_sets))):
            coords = ' '.join(f'({v:.8g},{y})' for y, v in points.items())
            color = 'cblue' if '[cblue]' in match.group() else 'corange'
            diagram = diagram[:match.start()] + r'\addplot[' + color + r'] coordinates {' + coords + '};' + diagram[match.end():]
        connectors = list(re.finditer(r'\\draw\[black!45, line width=1.0pt\] \(axis cs:([-\d.]+),(\d+)\) -- \(axis cs:([-\d.]+),(\d+)\);', diagram))
        assert len(connectors) == 39
        for index, match in reversed(list(enumerate(connectors))):
            first = 0 if index < 20 else 2
            y = int(match.group(2))
            a, b = point_sets[first][y], point_sets[first+1][y]
            replacement = rf'\draw[black!45, line width=1.0pt] (axis cs:{a:.8g},{y}) -- (axis cs:{b:.8g},{y});'
            diagram = diagram[:match.start()] + replacement + diagram[match.end():]
        notes.append('All 78 coverage points and 39 connecting lines use the saved interval summary, eliminating intermediate rounding differences.')
    if key == 'operational':
        plot = re.search(r'\\addplot\[fill=cblue, draw=none\] coordinates \{.*?\};', diagram, re.S)
        assert len(transfer) == 18
        coords = ' '.join(f"({100*float(r['box_coverage']):.8g},{18-i})" for i, r in enumerate(transfer))
        diagram = diagram[:plot.start()] + r'\addplot[fill=cblue, draw=none] coordinates {' + coords + '};' + diagram[plot.end():]
        diagram = diagram.replace('xticklabels={No corruption, One corruption}', r'xticklabels={{No\\corruption},{One\\corruption}}')
        diagram = diagram.replace(r'x tick label style={font=\scriptsize}', r'x tick label style={font=\scriptsize,align=center}')
        notes.append('Membership bars use the saved 18-role coverage values in the matching label order.')
    if key == 'network':
        diagram = diagram.replace('{Cell poset}', '{Station-to-edge maps}')
    if key == 'workflow':
        diagram = diagram.replace(r'Decision bounds and queries\\when ambiguity remains', r'Decision check and extra readings\\when uncertainty remains')
    diagram_sources[destination] = diagram

                                                                                 
style_start = source.index(r'\usepackage{tikz,pgfplots}')
style_end = source.index(r'\renewcommand{\topfraction}')
style = source[style_start:style_end]
text = baseline.replace(r'\usepackage{amsmath,amssymb,booktabs,graphicx,microtype,tabularx}',
                        r'\usepackage{amsmath,amssymb,booktabs,graphicx,microtype,tabularx}' + '\n' + style)

old_figures = re.findall(r'\\begin\{figure\}.*?\\end\{figure\}', baseline, re.S)
assert len(old_figures) == 11
for old in old_figures:
    destination = re.search(r'\\label\{fig:([^}]+)\}', old).group(1)
    key = next(k for k, v in labels.items() if v == destination)
    caption = old[old.index(r'\caption'):old.index(r'\end{figure}')]
    if key == 'network':
        caption = r'\caption{The Indian station graph (left), its station-to-edge restriction maps (middle), and the shared comparison space $C_a=\R^4$ on edge $a$ (right). Orange marks an illustrative hidden station and its three comparisons. Positions are schematic.}\label{fig:network}'
    if key == 'external':
        caption = r'\caption{Every evaluable external cohort, including losses; codes and physical outcomes are in Table~\ref{tab:external}. Error bars are paired seven-day intervals; bars without them have insufficient bins.}\label{fig:external}'
    replacement = r'\begin{figure}[htbp]\centering' + '\n' + r'\resizebox{' + widths[key] + r'\linewidth}{!}{%' + '\n' + diagram_sources[destination] + '\n}\n' + caption + '\n' + r'\end{figure}'
    text = text.replace(old, replacement, 1)

                                                                           
edits = [
    ('normalized root mean squared error is', 'normalized root mean squared error (RMSE) is'),
    ('The demonstrated benefit concerns computational monitoring continuity, not measured reductions in pollution, emissions or health risk.',
     'The demonstrated benefit is keeping estimates available during outages; we did not measure reductions in pollution, emissions or health risk.'),
    ('Their environmental motivation is keeping computational monitoring useful during outages.',
     'The aim is to help monitoring continue during outages.'),
    ('Its strongest supported application is computational monitoring continuity.',
     'It can help keep estimates and decision checks available during outages.'),
    (r'Station $v$ carries a normalized vector $x_v\in\R^p$.',
     r'Station $v$ carries a normalized vector $x_v\in\R^p$, where $p$ is the number of pollutant channels.'),
    ('The station and edge spaces are sheaf',
     'The full vector $x$ stacks the station vectors. The station and edge spaces are sheaf'),
    ('Let $e$ correct observed coordinates, giving possible clean values $x_O+e$.',
     r'Let $e$ correct observed coordinates, giving possible clean values $x_O+e$. We use $\|w\|_\infty=\max_j|w_j|$ for the largest absolute entry of a vector.'),
    ('Exact uniqueness does not imply accurate prediction in noise.',
     'Exact uniqueness does not imply accurate prediction in noise. We next replace exact agreement with error limits and ask whether the decision still holds.'),
    (r'For supports $F$ with $|F|\leq t$, define',
     r'A support $F$ lists the surviving stations that may be faulty. For $|F|\leq t$, define'),
    ('Each $Z_F$ is a bounded polyhedron and their finite union is compact, though possibly nonconvex.',
     'Each $Z_F$ is a bounded polyhedron, a set described by finitely many linear inequalities. Their finite union is compact, though possibly nonconvex.'),
    (r'Let $s_k(z)=h_k^Tz+c_k$ be pollutant $k$\'s affine reference score, including normalization offsets.',
     r'Let $s_k(z)=h_k^Tz+c_k$ be pollutant $k$\'s affine reference score, with fixed coefficients $h_k,c_k$ that include normalization offsets.'),
    ('Exact decision conditions still need reliable computation.',
     r'The bounds in Proposition~\ref{prop:joint-short} still need reliable computation.'),
    ('These are mathematical references, not automatically official AQI or GRAP.',
     'These are mathematical references, not automatically the official Air Quality Index (AQI) or Graded Response Action Plan (GRAP).'),
    ('Fault-support searches may stop early.',
     'Even a reliable bound for one fault case does not cover the others. There may be many possible faulty-station groups, and checking them all may take too long. We therefore keep bounds that remain valid if the search stops early.'),
    ('For a finite scalar bank, let $U$ be unqueried stations and $b_i=f$ initially for scenario $i$.',
     'For a finite scalar bank, let $U$ be unqueried stations and $b=(b_i)$ list the remaining corruption budgets, initially $b_i=f$ for each scenario $i$. Let $V(U,b)$ be the least possible worst-case cost of the remaining queries.'),
    ('These stations have disjoint clean-reading boxes. Their sets form a conflict hypergraph: we must cover each opposite pair with enough separators to withstand faults.',
     'These stations have disjoint clean-reading boxes. Each separating set is an edge of a conflict hypergraph; unlike a graph edge, it can contain any number of stations. A multicover selects stations to hit every such set the required number of times. An exact oracle checks whether a proposed plan leaves any opposite pair unresolved.'),
    (r"Tomorrow's readings may differ from archived centres $a_i$.",
     r"For the next two results, we use a \emph{scenario bank}: a finite collection of archived station vectors $a_i$. Each station now has one pollutant reading. This makes exact query planning manageable, but tomorrow's readings may differ from the archived centres."),
    ('one-/twenty-four-hour lags, last actual target history, masks and calendar/station indicators',
     'one-/twenty-four-hour lags, last actual target history, missing-data indicators and calendar/station indicators'),
    ('The causal forest uses', 'The causal random forest (RF) uses'),
    ('clean calibration sets the alarm cutoff.',
     'clean calibration sets the alarm cutoff. Offset tests use same or opposite signs, random directions, or unequal magnitudes; a gain fault rescales a raw reading.'),
    ('Unknown states lie in standardized $[-6,6]$, one bounded surviving-station fault is permitted, and residual budgets come from calibration.',
     r'These snapshot checks use only the spatial rows of $D$. Unknown states lie in standardized $[-6,6]$, with $t=1$ and $\beta=1$ in normalized units. If $n_{\rm cal}$ complete calibration vectors are available, every row receives the same residual budget: the $\lceil0.9(n_{\rm cal}+1)\rceil$-th smallest calibration value of $\|Dx\|_\infty$. Native-channel references are PM2.5 60, PM10 and ozone 100 $\mu$g/m$^3$, ozone 0.05 ppm, and NO2 40 ppb.'),
    ('each box radius is capped at 95\\% of its semantic margin.',
     "each box radius is capped at 95\\% of its centre's distance from the decision threshold."),
]
for old, new in edits:
    if old == new:
        continue
    if old not in text:
                                                                               
        old = old.replace("\\'", "'")
        new = new.replace("\\'", "'")
    assert old in text, old
    text = text.replace(old, new, 1)
text = text.replace(r'Positive noise can make near-threshold pairs inseparable.',
                    r'Positive noise can make near-threshold pairs inseparable. The right panel of Figure~\ref{fig:repair-intuition} previews the next question: $\rho$ measures how far a true reading may differ from an archived one.')
text = text.replace('NSW illawarra', 'NSW Illawarra').replace('NSW lower hunter', 'NSW Lower Hunter')
text = text.replace('NSW newcastle local', 'NSW Newcastle local').replace('NSW north west slopes', 'NSW North West Slopes')
text = text.replace('NSW sydney east', 'NSW Sydney east').replace('NSW sydney north west', 'NSW Sydney north west')
text = text.replace('[11pt,a4paper]', '[10pt,a4paper]').replace('[margin=1in]', '[margin=.85in]')
text = text.replace(r'\DeclareMathSizes{10.95}{10.95}{8}{8}', r'\DeclareMathSizes{10}{10}{7}{5}')
text = text.replace(r'\usepackage[authoryear,round]{natbib}', r'\usepackage[numbers,sort&compress]{natbib}')
text = text.replace('\\textit{QED.}\n', '\\textit{QED.}\n\n')
text = text.replace(r'\begingroup\footnotesize\setlength{\bibsep}{1pt}', r'\begingroup\footnotesize\raggedright\setlength{\bibsep}{1pt}')
text = text.replace(r'\setlength{\bibsep}{1pt}', r'\setlength{\bibsep}{0pt}')

                                                                                  
import sys
if '--embed-bibliography' in sys.argv:
    bbl_path = OUT / 'SheafCode_IRIS_Final.bbl'
    if not bbl_path.exists():
        bbl_path = ROOT / 'audit_tools/final_iris_review/SheafCode_IRIS_Final.bbl'
    bbl = bbl_path.read_text(encoding='utf-8')
    expected = {k.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}', text) for k in group.split(',')}
    actual = set(re.findall(r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}', bbl))
    assert expected == actual, (expected - actual, actual - expected)
    text = text.replace(r'\bibliographystyle{plainnat}', '')
    text = text.replace(r'\bibliography{references}', '\n' + bbl + '\n')

OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'SheafCode_IRIS_Final.tex').write_text(text, encoding='utf-8')
shutil.copy2(BASE / 'references.bib', OUT / 'references.bib')
formal = lambda s: [re.sub(r'\s+', ' ', m.group()).strip() for m in re.finditer(r'\\begin\{(proposition|theorem)\}.*?\\end\{\1\}', s, re.S)]
proofs = lambda s: [re.sub(r'\s+', ' ', m.group()).strip() for m in re.finditer(r'\\textbf\{Proof\.\}.*?\\textit\{QED\.\}', s, re.S)]
assert formal(baseline) == formal(text)
assert proofs(baseline) == proofs(text)
assert len(re.findall(r'\\begin\{figure\}', text)) == 11
assert r'\includegraphics' not in text
all_labels = re.findall(r'\\label\{([^}]+)\}', text)
assert len(all_labels) == len(set(all_labels))
assert set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', text)) <= set(all_labels)
assert not re.search(r'(?:technical|companion) reference', text, re.I)
report = {
    'user_diagrams_included': 11,
    'figures_editable_in_source': True,
    'mathematical_statements_unchanged': True,
    'core_proofs_unchanged': True,
    'figure_transcription_corrections': notes,
    'input_sha256': hashlib.sha256(INPUT.read_bytes()).hexdigest(),
    'baseline_sha256': hashlib.sha256((BASE / 'SheafCode_IRIS_Standalone.tex').read_bytes()).hexdigest(),
    'scope': 'Editorial and layout revision; plotted corrections use saved results. No experiments rerun.',
}
(OUT / 'final_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
(OUT / 'final_changes.diff').write_text(''.join(difflib.unified_diff(baseline.splitlines(keepends=True), text.splitlines(keepends=True), fromfile='SheafCode_IRIS_Standalone.tex', tofile='SheafCode_IRIS_Final.tex')), encoding='utf-8')
print(json.dumps(report, indent=2))
