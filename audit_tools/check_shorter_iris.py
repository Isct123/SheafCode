                                                                         
from pathlib import Path
import csv
import hashlib
import json
import re
import sys
import zipfile
import pymupdf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'SUBMISSION_WORKSPACE/Final_IRIS/Shorter_Readable_Final'
BASE = ROOT / 'SUBMISSION_WORKSPACE/Final_IRIS/Appendix_Readable_Final'
REVIEW = ROOT / 'audit_tools/final_iris_review/shorter_final'
FRESH = ROOT / 'audit_tools/final_iris_review/shorter_package_compile'
REVIEW.mkdir(exist_ok=True)
FRESH.mkdir(exist_ok=True)
source_path = OUT / 'SheafCode_IRIS_Final.tex'
source = source_path.read_text(encoding='utf-8')
original = (BASE / source_path.name).read_text(encoding='utf-8')
log = (OUT / 'SheafCode_IRIS_Final.log').read_text(encoding='utf-8', errors='replace')
assert not re.search(r'undefined|Overfull|Missing character|LaTeX Error', log, re.I)
assert not any(x in source for x in [r'\includegraphics', r'\input{', r'\bibliography{'])
assert r'\documentclass[11pt,a4paper]{article}' in source
assert r'\usepackage[margin=0.8in]{geometry}' in source

labels = re.findall(r'\\label\{([^}]+)\}', source)
assert len(labels) == len(set(labels))
assert set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', source)) <= set(labels)
citations = {k.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}', source) for k in group.split(',')}
entries = set(re.findall(r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}', source))
assert citations == entries and len(entries) == 32

patterns = {
    'formal_statements': r'\\begin\{(?:proposition|theorem)\}.*?\\end\{(?:proposition|theorem)\}',
    'complete_proofs': r'\\textbf\{Proof\.\}.*?\\textit\{QED\.\}',
    'displayed_math': r'\\\[.*?\\\]|\\begin\{equation\}.*?\\end\{equation\}',
    'bibliography': r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}',
}
counts = {}
for name, pattern in patterns.items():
    old = re.findall(pattern, original, re.S)
    new = re.findall(pattern, source, re.S)
    assert old == new, name
    counts[name] = len(new)
assert counts['formal_statements'] == counts['complete_proofs'] == 7

def protected_prose(text):
    text = re.sub(r'\\begin\{(?:figure|table)\}.*?\\end\{(?:figure|table)\}', '', text, flags=re.S)
    for old, new in [('fig:external', 'tab:external'), ('fig:math-integration', 'tab:verification-summary')]:
        text = text.replace('Figure~\\ref{' + old + '}', 'Table~\\ref{' + new + '}')
    text = text.replace(r'\usepackage[margin=1in]{geometry}', r'\usepackage[margin=0.8in]{geometry}')
    text = text.replace(r'\setlength{\parskip}{0pt}', '').replace(r'\clearpage', '')
    start = text.index(r'\appendix')
    appendix = text[start:].replace(r'\par\noindent\begin{minipage}{\linewidth}', '').replace(r'\end{minipage}\par', '')
    return re.sub(r'\s+', '', text[:start] + appendix)
assert protected_prose(original) == protected_prose(source)

def blocks(text, kind):
    return {re.search(r'\\label\{([^}]+)\}', b).group(1): b
            for b in re.findall(r'\\begin\{' + kind + r'\}.*?\\end\{' + kind + r'\}', text, re.S)}
old_figures = blocks(original, 'figure')
figures = blocks(source, 'figure')
assert len(figures) == 7
for label, figure in figures.items():
    assert figure == old_figures[label], label
assert list(figures)[:4] == list(old_figures)[:4]
old_tables = blocks(original, 'table')
tables = blocks(source, 'table')
assert len(tables) == 10
for label, table in tables.items():
    if label not in ['tab:external', 'tab:verification-summary']:
        assert table == old_tables[label], label

def csv_rows(path):
    with (ROOT / 'Research_Release' / path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))
cohorts = {r['code']: r for r in csv_rows('expanded_validation_results/COHORT_RESULTS.csv')}
forests = {r['code']: r for r in csv_rows('weakness_resolution_results/NONLINEAR_SUMMARY.csv')}
external_checks = 0
for row in tables['tab:external'].splitlines():
    if not re.match(r'^[DEF]\d+ &', row): continue
    code, region, cases, var, rf, pm = [x.strip() for x in row.removesuffix(r'\\').split('&')]
    c, f = cohorts[code], forests[code]
    assert int(cases) == int(c['cases'])
    assert abs(float(pm) - float(c['pm25_gain'])) < .0051
    for value, result, interval in [(var, c['gain'], [c['ci_low'], c['ci_high']]),
                                    (rf, f['gain'], json.loads(f['ci7'] or '[]'))]:
        nums = re.findall(r'-?\d+(?:\.\d+)?', value)
        assert abs(float(nums[0]) - float(result)) < .0051
        if interval and interval[0] != '':
            assert len(nums) == 3
            assert all(abs(float(a) - float(b)) < .0051 for a, b in zip(nums[1:], interval))
        else:
            assert len(nums) == 1 and '---' in value
        external_checks += 1
assert external_checks == 40
assert '216 below-reference decisions, 44 uncertain cases and 12 inconsistent cases' in tables['tab:verification-summary']

doc = pymupdf.open(OUT / 'SheafCode_IRIS_Final.pdf')
assert len(doc) == 22
texts = [page.get_text() for page in doc]
assert not any('??' in txt for txt in texts)
reference_page = next(i + 1 for i, txt in enumerate(texts) if '\nReferences\n' in txt)
appendix_page = next(i + 1 for i, txt in enumerate(texts) if '\nProofs\n' in txt)
proof_links = []
for page in doc:
    for link in page.get_links():
        if link.get('nameddest', '').startswith(('appendix.A', 'subsection.A.')):
            assert link.get('page', -1) >= appendix_page - 1
            proof_links.append(link)
assert len(proof_links) == 8
captions = '\n'.join(texts)
assert set(re.findall(r'Figure (\d+):', captions)) == {str(i) for i in range(1, 8)}
assert set(re.findall(r'Table (\d+):', captions)) == {str(i) for i in range(1, 11)}
for i, page in enumerate(doc):
    for block in page.get_text('blocks'):
        assert block[0] >= -1 and block[2] <= page.rect.width + 1
        assert block[1] >= -1 and block[3] <= page.rect.height + 1
    page.get_pixmap(matrix=pymupdf.Matrix(.62, .62)).save(REVIEW / f'page_{i+1:02}.png')
for start in range(0, len(doc), 8):
    canvas = Image.new('RGB', (1000, 1460), '#cccccc')
    draw = ImageDraw.Draw(canvas)
    for k in range(8):
        i = start + k
        if i >= len(doc): break
        im = Image.open(REVIEW / f'page_{i+1:02}.png')
        im.thumbnail((470, 337))
        x, y = 15 + k % 2 * 500, 25 + k // 2 * 365
        canvas.paste(im, (x, y))
        draw.text((x, y - 15), f'Page {i+1}', fill='black')
    canvas.save(REVIEW / f'contact_{start+1:02}.png')
for i in [13, 17, 20, 21]:
    doc[i].get_pixmap(matrix=pymupdf.Matrix(1.35, 1.35)).save(REVIEW / f'full_page_{i+1}.png')

package = OUT / 'SheafCode_Final_Overleaf.zip'
if '--final' not in sys.argv:
    with zipfile.ZipFile(package, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(source_path, source_path.name)
with zipfile.ZipFile(package) as archive:
    assert archive.namelist() == [source_path.name]
    assert archive.read(source_path.name) == source_path.read_bytes()
    archive.extractall(FRESH)

fresh_ok = False
if '--final' in sys.argv:
    fresh_doc = pymupdf.open(FRESH / 'SheafCode_IRIS_Final.pdf')
    assert [p.get_text() for p in fresh_doc] == texts
    fresh_log = (FRESH / 'SheafCode_IRIS_Final.log').read_text(encoding='utf-8', errors='replace')
    assert not re.search(r'undefined|Overfull|Missing character|LaTeX Error', fresh_log, re.I)
    fresh_ok = True

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = {
    'pages': len(doc), 'previous_pages': 27, 'pages_saved': 5,
    'references_begin_page': reference_page, 'proofs_begin_page': appendix_page,
    'body_font_pt': 11, 'margins_inches': .8,
    'explanatory_prose_preserved': True, 'mathematical_statements_preserved': True,
    'complete_proofs_preserved': True, 'displayed_math_preserved': True,
    'bibliography_entries_preserved': 32, 'counts': counts,
    'figures': len(figures), 'tables': len(tables), 'remaining_figures_unchanged': True,
    'methodology_figures_1_to_4_preserved': True,
    'external_gain_and_interval_checks': external_checks, 'proof_links_resolve': len(proof_links),
    'overfull_boxes': 0, 'undefined_references': 0, 'fresh_zip_compilation_verified': fresh_ok,
    'experiments_rerun': False,
    'source_sha256': sha(source_path), 'pdf_sha256': sha(OUT / 'SheafCode_IRIS_Final.pdf'),
    'zip_sha256': sha(package), 'previous_source_sha256': sha(BASE / source_path.name),
}
(OUT / 'final_check.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
(OUT / 'README.md').write_text(
    f'SheafCode_IRIS_Final.pdf contains {len(doc)} pages including references and complete proofs. '
    f'References begin on page {reference_page}; the proof appendix begins on page {appendix_page}. '
    'The previous 27-page Appendix_Readable_Final bundle is unchanged.\n\n'
    'All explanatory prose, motivations, examples, six propositions, the stability theorem, '
    'seven complete proofs and displayed mathematics are preserved. Figures 1–4 are unchanged. '
    'The external-cohort chart is consolidated into its table, retaining every cohort outcome and '
    'paired interval. Repeated coverage, archive-membership and verification charts are removed; '
    'their principal findings remain in the prose and tables, with the unique verification '
    'bounds and runtimes added to the verification table. Seven figures and ten tables remain. '
    'Body text is 11pt with 0.8-inch margins; reference spacing is tightened and the proof appendix '
    'flows after references to avoid an almost-empty final page.\n\n'
    'Upload SheafCode_Final_Overleaf.zip to Overleaf and compile SheafCode_IRIS_Final.tex. '
    'The ZIP contains one standalone TEX file with all figures and references embedded. '
    + ('A fresh compilation of the extracted ZIP matches the delivered PDF text. ' if fresh_ok else '')
    + 'Cross-references resolve and there are no overfull boxes. No new experiments were run. '
    'display_changes.diff records the editorial changes; final_check.json records preservation '
    'checks and artifact hashes.\n', encoding='utf-8')
print(json.dumps(manifest, indent=2))
