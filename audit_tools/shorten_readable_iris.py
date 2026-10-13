                                                                                   
from pathlib import Path
import csv
import json
import re
import difflib

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'SUBMISSION_WORKSPACE/Final_IRIS/Appendix_Readable_Final'
OUT = ROOT / 'SUBMISSION_WORKSPACE/Final_IRIS/Shorter_Readable_Final'
OUT.mkdir(exist_ok=True)
original = (BASE / 'SheafCode_IRIS_Final.tex').read_text(encoding='utf-8')
source = original.replace(r'\usepackage[margin=1in]{geometry}', r'\usepackage[margin=0.8in]{geometry}')

def csv_rows(relative):
    with (ROOT / 'Research_Release' / relative).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

cohorts = {r['code']: r for r in csv_rows('expanded_validation_results/COHORT_RESULTS.csv')}
forests = {r['code']: r for r in csv_rows('weakness_resolution_results/NONLINEAR_SUMMARY.csv')}

                                                                             
table = next(t for t in re.findall(r'\\begin\{table\}.*?\\end\{table\}', source, re.S)
             if r'\label{tab:external}' in t)
rows = []
for line in table.splitlines():
    if re.match(r'^[DEF]\d+ &', line):
        code, region, cases, gain, rf_gain, pm = [x.strip() for x in line.removesuffix(r'\\').split('&')]
        record = cohorts[code]
        assert int(cases) == int(record['cases'])
        assert abs(float(gain) - float(record['gain'])) < .0051
        assert abs(float(rf_gain) - float(forests[code]['gain'])) < .0051
        assert abs(float(pm) - float(record['pm25_gain'])) < .0051
        ci = (f"[{float(record['ci_low']):.2f}, {float(record['ci_high']):.2f}]"
              if record['ci_low'] else r'---')
        rf_ci_values = json.loads(forests[code]['ci7'] or '[]')
        rf_ci = f'[{rf_ci_values[0]:.2f}, {rf_ci_values[1]:.2f}]' if rf_ci_values else r'---'
        rows.append(f'{code} & {region} & {cases} & {gain} {ci} & {rf_gain} {rf_ci} & {pm}' + r'\\')
new_table = r'''\begin{table}[htbp]\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{Every evaluable external cohort: gain (\%) and paired seven-day interval on matched station-days. Negative values are losses.}
\label{tab:external}\begin{tabular}{@{}llrrrr@{}}\toprule
Code & Region & Cases & vs VAR [interval] & vs RF [interval] & PM2.5 vs VAR\\\midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabular}

\par\smallskip\footnotesize VAR/RF gains use normalized multichannel error; PM2.5 gains use physical concentration error. Brackets give paired seven-day bootstrap intervals; a dash means insufficient calendar bins. EPA codes identify state/county; UK/NSW names identify regions. EPA 06-073 and 06-111 lack usable test observations and are excluded, not successes.
\end{table}'''
source = source.replace(table, new_table)
source = source.replace(r'Figure~\ref{fig:external}', r'Table~\ref{tab:external}')

                                                                             
removed = ['external', 'risk', 'operational-validation', 'math-integration']
def keep_figure(match):
    label = re.search(r'\\label\{fig:([^}]+)\}', match.group()).group(1)
    return '' if label in removed else match.group()
source = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', keep_figure, source, flags=re.S)
source = source.replace(r'Figure~\ref{fig:math-integration}', r'Table~\ref{tab:verification-summary}')

                                                                        
needle = r'Stress and scaling & Direct replay reconstructs all 247,497 stress trials. Sixty synthetic multicover cases reach 128 stations; greedy matches 19/24 small optima and otherwise needs at most one extra query. \\ \bottomrule'
replacement = r'''Stress and scaling & Direct replay reconstructs all 247,497 stress trials. Sixty synthetic multicover cases reach 128 stations; greedy matches 19/24 small optima and otherwise needs at most one extra query. \\ \addlinespace
Search and runtime & For the search with the largest initial gap, the upper bound falls from 3.58 to 2.86 after seven nodes; a feasible lower bound of 2.86 appears after eight (linear objective, offset excluded). Median times are 0.025 seconds for support LPs and 0.077 for verified search; verification adds overhead. \\ \bottomrule'''
assert needle in source
source = source.replace(needle, replacement)
source = source.replace('there is no measured above-reference joint certificate. \\\\ \\addlinespace',
    'there is no measured above-reference joint certificate. The separate scalar check gives 216 below-reference decisions, 44 uncertain cases and 12 inconsistent cases. \\\\ \\addlinespace')

                                                                           
source = source.replace(r'\begingroup\footnotesize\raggedright\setlength{\bibsep}{0pt}',
                        r'\begingroup\footnotesize\raggedright\setlength{\bibsep}{0pt}\setlength{\parskip}{0pt}')

                                                                             
appendix_start = source.index(r'\clearpage' + '\n' + r'\appendix')
appendix = source[appendix_start:]
appendix = appendix.replace('\\clearpage\n', '')
appendix = appendix.replace(r'\par\noindent\begin{minipage}{\linewidth}' + '\n', '')
appendix = appendix.replace(r'\end{minipage}\par', '')
source = source[:appendix_start] + appendix

                                                                               
for pattern in [r'\\begin\{(?:proposition|theorem)\}.*?\\end\{(?:proposition|theorem)\}',
                r'\\textbf\{Proof\.\}.*?\\textit\{QED\.\}',
                r'\\\[.*?\\\]', r'\\begin\{equation\}.*?\\end\{equation\}']:
    assert re.findall(pattern, original, re.S) == re.findall(pattern, source, re.S)
old_figures = re.findall(r'\\begin\{figure\}.*?\\end\{figure\}', original, re.S)
new_figures = re.findall(r'\\begin\{figure\}.*?\\end\{figure\}', source, re.S)
assert all(f in new_figures for f in old_figures[:4])

def protected_prose(text):
    text = re.sub(r'\\begin\{(?:figure|table)\}.*?\\end\{(?:figure|table)\}', '', text, flags=re.S)
    text = text.replace(r'Figure~\ref{fig:external}', r'Table~\ref{tab:external}')
    text = text.replace(r'Figure~\ref{fig:math-integration}', r'Table~\ref{tab:verification-summary}')
    text = text.replace(r'\usepackage[margin=1in]{geometry}', r'\usepackage[margin=0.8in]{geometry}')
    text = text.replace(r'\setlength{\parskip}{0pt}', '')
    text = text.replace(r'\clearpage', '')
    start = text.index(r'\appendix')
    appendix = text[start:].replace(r'\par\noindent\begin{minipage}{\linewidth}', '').replace(r'\end{minipage}\par', '')
    return re.sub(r'\s+', '', text[:start] + appendix)
assert protected_prose(original) == protected_prose(source), 'An explanation changed'

(OUT / 'SheafCode_IRIS_Final.tex').write_text(source, encoding='utf-8')
(OUT / 'display_changes.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True), source.splitlines(True),
    fromfile='27-page manuscript', tofile='Shorter manuscript')), encoding='utf-8')
print(json.dumps({'removed_duplicate_figures': removed, 'retained_figures': len(new_figures),
    'all_explanatory_prose_preserved': True, 'all_statements_and_proofs_preserved': True}))
