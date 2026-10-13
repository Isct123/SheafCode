                                                                            
from pathlib import Path
import difflib
import hashlib
import json
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'SUBMISSION_WORKSPACE/Clarity_Fixes'
OUT = ROOT / 'SUBMISSION_WORKSPACE/Standalone_IRIS'
baseline = (BASE / 'SheafCode_IRIS_Clarity_Final_Fixed.tex').read_text(encoding='utf-8')
reference = (ROOT / 'SUBMISSION_WORKSPACE/Combined_IRIS/Source/technical_reference.tex').read_text(encoding='utf-8')
body = (ROOT / 'audit_tools/standalone_iris_body.tex').read_text(encoding='utf-8')
blocks = {}
for match in re.finditer(r'\\begin\{(proposition|theorem)\}.*?\\end\{\1\}', baseline, re.S):
    label = re.search(r'\\label\{(?:prop|thm):([^}]+)\}', match.group()).group(1)
    blocks[label] = match.group()
    after = baseline[match.end():]
    proof = re.match(r'\s*(\\textbf\{Proof\.\}.*?\\textit\{QED\.\})', after, re.S).group(1)
    body = body.replace('@@FORMAL_' + label + '@@', match.group())
    body = body.replace('@@PROOF_' + label + '@@', proof)

def table_containing(text, needle):
    index = text.index(needle)
    start = text.rfind(r'\begin{table}', 0, index)
    end = text.index(r'\end{table}', index) + len(r'\end{table}')
    return text[start:end]

external = table_containing(reference, r'\label{tab:external}')
external = external[:external.index(r'\par\smallskip')] + r'''
\par\smallskip\small VAR/RF gains use normalized multichannel error; PM2.5 gains use physical concentration error. EPA codes identify state/county; UK/NSW names identify regions. EPA 06-073 and 06-111 lack usable test observations and are excluded, not successes.
\end{table}'''
grin = table_containing(reference, 'Every tested GRIN seed:')
grin = grin.replace(r'\begin{tabular}', r'\label{tab:grin}\begin{tabular}', 1)
intervals = table_containing(reference, 'No-feedback PM2.5 snapshot intervals, nominal 90')
intervals = intervals.replace(r'\begin{tabular}', r'\label{tab:intervals}\begin{tabular}', 1)
intervals = intervals[:intervals.index(r'\par\smallskip')] + r'''
\par\smallskip\small Macro coverage averages cohorts (20 overall, 19 extreme); pooled coverage weights cells. Width is the cohort mean finite full width in $\mu$g/m$^3$; full intervals are infinite and excluded from finite widths. Full frequency is cell pooled. Worst station-quarter coverage is 57.8\% fixed/74.6\% risk grouped.
\end{table}'''
for key, value in {'external': external, 'grin': grin, 'intervals': intervals,
                   'repair-summary': table_containing(baseline, r'\label{tab:repair-summary}'),
                   'chronological-guard': table_containing(baseline, r'\label{tab:chronological-guard}')}.items():
    body = body.replace('@@TABLE_' + key + '@@', value)
assert '@@' not in body

preamble = baseline[:baseline.index(r'\begin{abstract}')]
preamble = re.sub(r'\n{3,}', '\n\n', preamble)
preamble = preamble.replace(r'\setlength{\parskip}{.28em}', r'\setlength{\parskip}{.20em}')
preamble = preamble.replace(r'\begin{document}', r'''\setlength{\textfloatsep}{12pt plus 2pt minus 2pt}
\setlength{\floatsep}{10pt plus 2pt minus 2pt}
\AtBeginDocument{\setlength{\abovedisplayskip}{8pt plus 2pt minus 2pt}\setlength{\belowdisplayskip}{8pt plus 2pt minus 2pt}}
\begin{document}''', 1)
body = body.replace(r'\begingroup\small\bibliography{references}\endgroup', r'\begingroup\footnotesize\setlength{\bibsep}{1pt}\bibliography{references}\endgroup')
text = preamble + body
OUT.mkdir(parents=True, exist_ok=True)
target = OUT / 'SheafCode_IRIS_Standalone.tex'
if not target.exists() or target.read_text(encoding='utf-8') != text:
    target.write_text(text, encoding='utf-8')
shutil.copy2(BASE / 'references.bib', OUT / 'references.bib')
shutil.copytree(BASE / 'figures', OUT / 'figures', dirs_exist_ok=True)
(OUT / 'README.md').write_text(
    'Submit SheafCode_IRIS_Standalone.pdf as the manuscript. No technical reference is required.\n\n'
    'For Overleaf, upload Standalone_Overleaf.zip and select SheafCode_IRIS_Standalone.tex as the main document.\n'
    'The ZIP contains the source, bibliography and figures; it contains no companion manuscript.\n', encoding='utf-8')
(OUT / 'standalone_changes.diff').write_text(''.join(difflib.unified_diff(
    baseline.splitlines(keepends=True), text.splitlines(keepends=True),
    fromfile='SheafCode_IRIS_Clarity_Final_Fixed.tex', tofile=target.name)), encoding='utf-8')

formal = lambda s: [re.sub(r'\s+', ' ', m.group()).strip() for m in re.finditer(r'\\begin\{(proposition|theorem)\}.*?\\end\{\1\}', s, re.S)]
assert formal(baseline) == formal(text), 'A mathematical statement changed'
proofs = lambda s: [re.sub(r'\s+', ' ', m.group()).strip() for m in re.finditer(r'\\textbf\{Proof\.\}.*?\\textit\{QED\.\}', s, re.S)]
assert proofs(baseline) == proofs(text), 'A core proof changed'
figures = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', text)
assert len(figures) == 11 and len(set(figures)) == 11
assert not re.search(r'(?:technical|companion) reference|supplementary material|retained in the reference', text, re.I)
labels = re.findall(r'\\label\{([^}]+)\}', text)
assert len(labels) == len(set(labels))
assert set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', text)) <= set(labels)
bib = (OUT / 'references.bib').read_text(encoding='utf-8')
keys = set(re.findall(r'@\w+\s*\{\s*([^,\s]+)', bib))
citations = {k.strip() for g in re.findall(r'\\cite\w*(?:\[[^\]]*\])*\{([^}]+)\}', text) for k in g.split(',')}
assert citations <= keys
assert all((OUT / f).exists() for f in figures)
assert (BASE / 'SheafCode_IRIS_Clarity_Final_Fixed.tex').read_text(encoding='utf-8') == baseline
numbers = lambda s: {n.replace(',', '') for n in re.findall(r'(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?![\w.])', s)}
missing = sorted(numbers(baseline) - numbers(text))
report = {'formal_statements_unchanged': True, 'core_proofs_unchanged': True, 'original_unchanged': True,
          'original_pdf_figures_included': len(figures), 'citations_and_references_resolve': True,
          'baseline_numeric_literals_not_repeated': missing,
          'scope': 'Standalone editorial revision; no new fitting or experiments.'}
(OUT / 'standalone_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({'file': str(target), 'source_words': len(text.split()), 'missing_numeric_literals': missing}))

pdf_path = target.with_suffix('.pdf')
if pdf_path.exists() and pdf_path.stat().st_mtime >= target.stat().st_mtime:
    import pymupdf
    log = target.with_suffix('.log').read_text(encoding='utf-8', errors='replace')
    assert 'Overfull' not in log and 'undefined' not in log.lower()
    pdf = pymupdf.open(pdf_path)
    assert all(p.get_text().strip() for p in pdf)
    pdf_text = '\n'.join(p.get_text() for p in pdf)
    assert re.findall(r'Figure\s+(\d+)\s*:', pdf_text) == [str(i) for i in range(1, 12)]
    reference_page = next((i + 1 for i, p in enumerate(pdf) if re.search(r'(?m)^References\s*$', p.get_text())), None)
    for page in pdf:
        for word in page.get_text('words'):
            assert word[0] >= -1 and word[1] >= -1 and word[2] <= page.rect.width + 1 and word[3] <= page.rect.height + 1
    report.update({'compilation_passed': True, 'pages_including_references': len(pdf),
                   'references_begin_on_page': reference_page, 'eleven_figure_captions_rendered': True,
                   'no_overfull_boxes': True, 'pdf_text_within_page_bounds': True,
                   'source_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                   'pdf_sha256': hashlib.sha256(pdf_path.read_bytes()).hexdigest()})
    (OUT / 'standalone_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with zipfile.ZipFile(OUT / 'Standalone_Overleaf.zip', 'w', zipfile.ZIP_DEFLATED) as package:
        for p in [target, OUT / 'references.bib', *[OUT / f for f in figures]]:
            package.write(p, p.relative_to(OUT))
    with zipfile.ZipFile(OUT / 'Standalone_Overleaf.zip') as package:
        assert package.testzip() is None
        assert not any('technical_reference' in name for name in package.namelist())
    print(json.dumps({'verified_pages': len(pdf), 'source_figures': len(figures)}))
