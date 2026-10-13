                                                            
from pathlib import Path
import csv
import hashlib
import json
import re
import zipfile
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'SUBMISSION_WORKSPACE/Final_IRIS/Natural_Readability_Final'
source = (OUT / 'SheafCode_IRIS_Final.tex').read_text(encoding='utf-8')
log_path = OUT / 'SheafCode_IRIS_Final.log'
if not log_path.exists():
    log_path = ROOT / 'audit_tools/final_iris_review/readable_compile/SheafCode_IRIS_Final.log'
log = log_path.read_text(encoding='utf-8', errors='replace')
assert source == (ROOT / 'audit_tools/final_iris_review/readable_compile/SheafCode_IRIS_Final.tex').read_text(encoding='utf-8')
assert (OUT / 'SheafCode_IRIS_Final.pdf').read_bytes() == (ROOT / 'audit_tools/final_iris_review/readable_compile/SheafCode_IRIS_Final.pdf').read_bytes()
assert not re.search(r'undefined|Overfull|Missing character|LaTeX Error', log, re.I)
assert not any(x in source for x in [r'\includegraphics', r'\input{', r'\bibliography{'])
labels = re.findall(r'\\label\{([^}]+)\}', source)
assert len(labels) == len(set(labels))
assert set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', source)) <= set(labels)
citations = {k.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}', source) for k in group.split(',')}
entries = set(re.findall(r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}', source))
assert citations == entries
figures = {re.search(r'\\label\{fig:([^}]+)\}', f).group(1): f for f in re.findall(r'\\begin\{figure\}.*?\\end\{figure\}', source, re.S)}
assert len(figures) == 11
previous = (ROOT / 'audit_tools/final_iris_review/previous_16_page_version/SheafCode_IRIS_Final.tex').read_text(encoding='utf-8')
normalize = lambda text: re.sub(r'\s+', '', re.sub(r'\biff\b', 'if and only if', text))
for pattern in [r'\\begin\{(?:proposition|theorem)\}.*?\\end\{(?:proposition|theorem)\}',
                r'\\textbf\{Proof\.\}.*?\\textit\{QED\.\}']:
    old_blocks = re.findall(pattern, previous, re.S)
    new_blocks = re.findall(pattern, source, re.S)
    assert len(old_blocks) == len(new_blocks) == 7
    assert list(map(normalize, old_blocks)) == list(map(normalize, new_blocks))

def csv_rows(path):
    with (ROOT / 'Research_Release' / path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

                                                                           
primary = re.findall(r'\(([-.\d]+),(\d+)\) -= \(([-.\d]+),0\) \+= \(([-.\d]+),0\)', figures['headline'])
assert len(primary) == 10
methods = ['var_causal', 'seasonal', 'city_mean', 'gaussian', 'spatial_sheaf']
for region, points in zip(['delhi', 'epa_la'], [primary[:5], primary[5:]]):
    data = json.loads((ROOT / f'Research_Release/recent_data_results/{region}/reconstruction_summary.json').read_text())['primary_every_third_day']
    for (value, y, low_error, high_error), method in zip(points, methods):
        record = data[method]
        assert abs(float(value) - record['gain_percent']) < .0051
        if record['ci95']:
            assert abs(float(value) - float(low_error) - record['ci95'][0]) < .012
            assert abs(float(value) + float(high_error) - record['ci95'][1]) < .012
        else:
            assert float(low_error) == float(high_error) == 0

cohorts = [r for r in csv_rows('expanded_validation_results/COHORT_RESULTS.csv') if r['code'][0] in 'DEF' and r['cases'] != '0']
forests = {r['code']: r for r in csv_rows('weakness_resolution_results/NONLINEAR_SUMMARY.csv')}
series = re.findall(r'\\addplot\[fill=(?:cblue|corange), draw=none\] coordinates \{(.*?)\};', figures['external'], re.S)
assert len(series) == 2
for index, string in enumerate(series):
    points = re.findall(r'\(([-.\d]+),(\d+)\) -= \(([-.\d]+),0\) \+= \(([-.\d]+),0\)', string)
    assert len(points) == 20
    for (value, y, low_error, high_error), record in zip(points, cohorts):
        other = record if index == 0 else forests[record['code']]
        assert abs(float(value) - float(other['gain'])) < .0051
        interval = [record['ci_low'], record['ci_high']] if index == 0 else json.loads(other['ci7'] or '[]')
        if interval and interval[0] != '':
            assert abs(float(value) - float(low_error) - float(interval[0])) < .012
            assert abs(float(value) + float(high_error) - float(interval[1])) < .012

intervals = csv_rows('weakness_resolution_results/uncertainty_decisions/INTERVAL_SUMMARY.csv')
series = re.findall(r'\\addplot\[c(?:blue|orange)\] coordinates \{(.*?)\};', figures['risk'], re.S)
settings = [('all', 'station_fixed'), ('all', 'risk_station'), ('extreme', 'station_fixed'), ('extreme', 'risk_station')]
coverage_checks = 0
for string, (group, method) in zip(series, settings):
    for value, y in re.findall(r'\(([-.\d]+),(\d+)\)', string):
        cohort = cohorts[20 - int(y)]['cohort']
        record = next(r for r in intervals if r['cohort'] == cohort and r['group'] == group and r['method'] == method and float(r['context_loss']) == 0)
        assert abs(float(value) - 100*float(record['coverage'])) < 1e-5
        coverage_checks += 1
assert coverage_checks == 78

                                                                                
transfer = csv_rows('operational_validation_results/COHORT_TRANSFER.csv')
membership = re.search(r'\\addplot\[fill=cblue, draw=none\] coordinates \{(.*?)\};', figures['operational-validation'], re.S).group(1)
membership_points = re.findall(r'\(([-.\d]+),(\d+)\)', membership)
membership_labels = re.search(r'yticklabels=\{([^}]+)\}', figures['operational-validation']).group(1).split(',')
role_labels = {'original_usa': 'Original USA', 'original_india': 'Original India', 'india_long_test': 'India long test'}
nsw_labels = {'nsw_illawarra': 'NSW Illawarra', 'nsw_lower_hunter': 'NSW Lower Hunter',
              'nsw_newcastle_local': 'NSW Newcastle local', 'nsw_north_west_slopes': 'NSW North West Slopes',
              'nsw_sydney_east': 'NSW Sydney east', 'nsw_sydney_north_west': 'NSW Sydney north west'}
for row in transfer:
    role = row['cohort']
    if role.startswith('epa_'):
        role_labels[role] = 'EPA ' + role[4:].replace('_', ' ')
    elif role.startswith('uk_'):
        role_labels[role] = 'UK ' + role[3:].replace('_', ' ')
    elif role.startswith('nsw_'):
        role_labels[role] = nsw_labels[role]
assert len(membership_points) == len(transfer) == 18
for value, y in membership_points:
    label = membership_labels[int(y)-1]
    row = next(r for r in transfer if role_labels[r['cohort']] == label)
    assert abs(float(value) - 100*float(row['box_coverage'])) < 1e-5

                                                                   
old_plot = pymupdf.open(ROOT / 'SUBMISSION_WORKSPACE/Standalone_IRIS/figures/fault_outage_robustness.pdf')
old_text = old_plot[0].get_text()
old_values = re.findall(r'^(?:n/a|[-−]?\d+\.\d+)$', old_text, re.M)
new_cells = re.findall(r'\\node\[font=\\scriptsize, text=(?:black|white|black!55)\] at \(([-\d.]+),([-\d.]+)\) \{(.*?)\};', figures['fault-outage'])
new_values = [v.replace('$-$', '-').replace('−', '-') for x, y, v in new_cells]
assert len(old_values) == len(new_values) == 100
assert [x.replace('−', '-') for x in old_values] == new_values

doc = pymupdf.open(OUT / 'SheafCode_IRIS_Final.pdf')
page_text = [page.get_text() for page in doc]
assert sorted(map(int, re.findall(r'Figure (\d+):', '\n'.join(page_text)))) == list(range(1, 12))
for i, page in enumerate(doc):
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines', []):
            for span in line['spans']:
                x0, y0, x1, y1 = span['bbox']
                assert x0 >= -1 and y0 >= -1 and x1 <= page.rect.width+1 and y1 <= page.rect.height+1, (i, span)
abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', source, re.S).group(1)
abstract = re.sub(r'\\[a-zA-Z]+(?:\[[^\]]*\])?', '', abstract).replace('{', '').replace('}', '')
abstract_words = len(abstract.split())
assert abstract_words <= 300
references_page = next(i+1 for i, t in enumerate(page_text) if re.search(r'^References$', t, re.M))
report = json.loads((OUT / 'final_check.json').read_text())
report.update(compilation_passed=True, pages_including_references=len(doc), references_begin_on_page=references_page,
              abstract_words=abstract_words, bibliography_entries=len(entries), citations_and_cross_references_resolve=True,
              all_eleven_figure_captions_rendered=True, no_overfull_boxes=True, pdf_text_within_page_bounds=True,
              single_file_with_embedded_bibliography_and_figures=True,
              main_text_font_points=int(re.search(r'\\documentclass\[(\d+)pt', source).group(1)),
              mathematical_statement_count=len(re.findall(r'\\begin\{(?:proposition|theorem)\}', source)),
              proof_count=len(re.findall(r'\\textbf\{Proof\.\}', source)),
              membership_bar_and_role_label_checks=18, clean_compile_pending=False,
              primary_gain_checks=10, external_gain_checks=40, coverage_point_checks=coverage_checks, heatmap_cell_checks=100,
              source_sha256=hashlib.sha256((OUT / 'SheafCode_IRIS_Final.tex').read_bytes()).hexdigest(),
              pdf_sha256=hashlib.sha256((OUT / 'SheafCode_IRIS_Final.pdf').read_bytes()).hexdigest())
with zipfile.ZipFile(OUT / 'SheafCode_Final_Overleaf.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    archive.write(OUT / 'SheafCode_IRIS_Final.tex', 'SheafCode_IRIS_Final.tex')
with zipfile.ZipFile(OUT / 'SheafCode_Final_Overleaf.zip') as archive:
    assert archive.testzip() is None and archive.namelist() == ['SheafCode_IRIS_Final.tex']
    archive.extractall(ROOT / 'audit_tools/final_iris_review/natural_readability_package_check')
(OUT / 'final_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
(OUT / 'README.md').write_text(
    f'SheafCode_IRIS_Final.pdf is the readable standalone version. It has {len(doc)} pages including references; references begin on page {references_page}. It exceeds the recommended 10–15 pages because the explanations, proofs, 11 figures and supporting tables are retained. No technical reference is required.\n\n'
    'For Overleaf, upload SheafCode_Final_Overleaf.zip or the TEX file itself and compile SheafCode_IRIS_Final.tex. All 11 editable TikZ/PGFPlots diagrams and the bibliography are embedded. No separate images or bibliography files are needed.\n\n'
    'The mathematical statements and proofs are preserved from the corrected standalone paper. Plotted corrections use saved results; no new experiments were run.\n', encoding='utf-8')
print(json.dumps(report, indent=2))
