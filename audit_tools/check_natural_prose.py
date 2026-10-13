                                                                                  
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import shutil
import sys
import zipfile
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'audit_tools/final_iris_review'
OUT = ROOT / 'SUBMISSION_WORKSPACE/Final_IRIS/Compact_Readable_Final'
STAGE = REVIEW / 'natural_build'
FRESH = REVIEW / 'natural_package_compile'
SRC = OUT / 'SheafCode_IRIS_Final.tex'
BACKUP = REVIEW / 'before_natural_prose_rewrite'
PDF = OUT / 'SheafCode_IRIS_Final_Headings.pdf'
ZIP = OUT / 'SheafCode_Final_Overleaf.zip'

old = (BACKUP / SRC.name).read_text(encoding='utf-8')
new = SRC.read_text(encoding='utf-8')
assert new[new.index(r'\appendix'):] == old[old.index(r'\appendix'):]
for env in ['proposition', 'theorem', 'equation', 'thebibliography', 'tikzpicture']:
    pat = rf'\\begin\{{{env}\}}.*?\\end\{{{env}\}}'
    assert re.findall(pat, new, re.S) == re.findall(pat, old, re.S), env
assert re.findall(r'\\\[.*?\\\]', new, re.S) == re.findall(r'\\\[.*?\\\]', old, re.S)
heading_changes = json.loads((REVIEW / 'descriptive_heading_changes.json').read_text(encoding='utf-8'))
old_headings = re.findall(r'\\(?:sub)?section\*?\{.*?\}', old)
expected_headings = [next((h.replace(before, after) for before, after in heading_changes.items()
                          if '{'+before+'}' in h), h) for h in old_headings]
assert re.findall(r'\\(?:sub)?section\*?\{.*?\}', new) == expected_headings
before_headings = (REVIEW / 'before_descriptive_headings' / SRC.name).read_text(encoding='utf-8')
strip_headings = lambda s: re.sub(r'\\(?:sub)?section\{[^\n]*?\}', lambda m: m[0].split('{')[0]+'{}', s)
assert strip_headings(new) == strip_headings(before_headings)
for macro in ['label', 'citep', 'citet', 'ref', 'eqref']:
    pat = rf'\\{macro}\{{(.*?)\}}'
    assert Counter(re.findall(pat, new)) == Counter(re.findall(pat, old)), macro
tables = lambda s: re.findall(r'\\begin\{table\}.*?\\end\{table\}', s, re.S)
assert len(tables(old)) == len(tables(new)) == 10
for before, after in zip(tables(old), tables(new)):
    body = lambda s: re.search(r'\\begin\{tabularx?\}.*?\\end\{tabularx?\}', s, re.S)[0]
    assert Counter(re.findall(r'\d+(?:[.,]\d+)*', body(before))) == Counter(re.findall(r'\d+(?:[.,]\d+)*', body(after)))
assert len(re.findall(r'\\begin\{figure\}', new)) == 7
assert new.count(r'\begin{proof}') == 7
assert len(re.findall(r'\\begin\{proposition\}', new)) == 6
assert len(re.findall(r'\\begin\{theorem\}', new)) == 1
labels = re.findall(r'\\label\{([^}]+)\}', new)
assert len(labels) == len(set(labels))
assert set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', new)) <= set(labels)
keys = set(re.findall(r'\\bibitem(?:\[.*?\])?\{([^}]+)\}', new, re.S))
assert {k for group in re.findall(r'\\cite[pt]?\{([^}]+)\}', new) for k in group.split(',')} <= keys

def check_log(path):
    s = path.read_text(encoding='utf-8', errors='replace')
    assert not re.search(r'undefined|Overfull|Missing character|LaTeX Error', s, re.I), path

check_log(STAGE / 'SheafCode_IRIS_Final.log')
if '--package' in sys.argv:
    shutil.copy2(STAGE / 'SheafCode_IRIS_Final.pdf', PDF)
    for suffix in ['log', 'aux', 'out']:
        shutil.copy2(STAGE / f'SheafCode_IRIS_Final.{suffix}', OUT / f'SheafCode_IRIS_Final.{suffix}')
    with zipfile.ZipFile(ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(SRC, SRC.name)
    FRESH.mkdir(exist_ok=True)
    with zipfile.ZipFile(ZIP) as archive:
        assert archive.namelist() == [SRC.name]
        assert archive.read(SRC.name) == SRC.read_bytes()
        archive.extractall(FRESH)
    print('Protected content verified. Standalone ZIP extracted for a fresh compilation.')
    sys.exit(0)

assert '--final' in sys.argv
check_log(FRESH / 'SheafCode_IRIS_Final.log')
with zipfile.ZipFile(ZIP) as archive:
    assert archive.namelist() == [SRC.name]
    assert archive.read(SRC.name) == SRC.read_bytes()
with pymupdf.open(PDF) as doc, pymupdf.open(FRESH / 'SheafCode_IRIS_Final.pdf') as fresh:
    texts = [page.get_text() for page in doc]
    assert texts == [page.get_text() for page in fresh]
    assert PDF.read_bytes() == (STAGE / 'SheafCode_IRIS_Final.pdf').read_bytes()
    proof_page = next(i+1 for i,t in enumerate(texts) if '\nProofs\n' in t)
    ref_page = next(i+1 for i,t in enumerate(texts) if '\nReferences\n' in t)
    assert '\n'.join(texts[proof_page-1:]).count('Proof.') == 7
    proof_links = [link for page in doc for link in page.get_links()
                   if link.get('nameddest', '').startswith(('appendix.A', 'subsection.A.'))]
    assert len(proof_links) == 8
    assert all(proof_page-1 <= link['page'] < len(doc) for link in proof_links)
    for page in doc:
        for b in page.get_text('blocks'):
            assert -1 <= b[0] < b[2] <= page.rect.width+1
            assert -1 <= b[1] < b[3] <= page.rect.height+1
    pages = len(doc)
hashfile = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
with pymupdf.open(BACKUP / 'SheafCode_IRIS_Final_Updated.pdf') as previous:
    previous_pages = len(previous)
manifest = {
    'revision': 'natural-prose manuscript with descriptive research headings',
    'pages': pages, 'previous_pages': previous_pages, 'additional_pages': pages-previous_pages,
    'body_font_pt': 11, 'margins_inches': 0.8,
    'references_begin_page': ref_page, 'proofs_begin_page': proof_page, 'proofs_end_page': pages,
    'rewritten_body_paragraphs': 106, 'captions_rewritten': 17,
    'descriptive_headings_updated': 19, 'paragraphs_preserved_from_natural_revision': True,
    'section_structure_preserved': True, 'formal_statements_preserved_exactly': True,
    'display_equations_preserved_exactly': True, 'complete_proofs_preserved_exactly': 7,
    'propositions': 6, 'theorems': 1, 'figure_contents_preserved_exactly': 7,
    'table_count': 10, 'table_numeric_values_preserved': True,
    'citation_and_reference_targets_preserved': True, 'bibliography_preserved_exactly': True,
    'overfull_boxes': 0, 'undefined_references': 0, 'proof_links_resolve': len(proof_links),
    'fresh_zip_compilation_verified': True, 'pdf_text_within_page_bounds': True,
    'new_experiments_run': False, 'source_sha256': hashfile(SRC),
    'pdf_filename': PDF.name, 'pdf_sha256': hashfile(PDF), 'zip_sha256': hashfile(ZIP),
    'previous_source_sha256': hashfile(BACKUP / SRC.name),
}
(OUT / 'final_check.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
(OUT / 'README.md').write_text(
    f'The current compact manuscript is SheafCode_IRIS_Final.tex. Its compiled PDF is '
    f'{PDF.name} ({pages} pages, including references and all seven proofs). '
    f'References begin on page {ref_page}, and Appendix A occupies pages {proof_page}-{pages}.\n\n'
    'This revision rewrites the prose throughout the abstract, introduction, related work, '
    'methods, theory explanations, experiments, results, discussion and conclusion. '
    'Captions and descriptive table text also use complete sentences. Basic pollutant '
    'definitions and unnecessary textbook glosses have been removed. The structure, '
    'research questions, motivations, findings and qualifications are retained.\n\n'
    'Question-style headings have been replaced with descriptive research headings. '
    'The motivating paragraphs remain unchanged. The preceding natural-prose version '
    'is backed up in audit_tools/final_iris_review/before_descriptive_headings/. '
    'descriptive_headings.diff records these heading-only changes.\n\n'
    'The formal statements, displayed equations, all seven expanded proofs, figure contents, '
    'table values and bibliography are preserved. No new experiments were run. '
    'The previous prose revision is backed up in '
    'audit_tools/final_iris_review/before_natural_prose_rewrite/. '
    'natural_prose_rewrite.diff records this revision. Older PDF filenames in this directory '
    'belong to previous revisions; use the Headings PDF listed above.\n\n'
    'Upload SheafCode_Final_Overleaf.zip to Overleaf and compile SheafCode_IRIS_Final.tex. '
    'The ZIP contains a single TEX file with embedded TikZ figures and bibliography. '
    'A fresh extraction compiles and reproduces the delivered PDF text. References resolve '
    'and there are no overfull boxes. final_check.json records the checks and artifact hashes.\n',
    encoding='utf-8')
root_readme = OUT.parent / 'README.md'
s = root_readme.read_text(encoding='utf-8')
first = ('The current compact manuscript is [Compact_Readable_Final](Compact_Readable_Final/README.md) '
         f'({pages} pages), with the full prose rewritten in natural complete sentences and all seven '
         'expanded formal proofs retained. The separate '
         '[Appendix_Readable_Final](Appendix_Readable_Final/README.md) version has 30 pages '
         'and has not received this prose rewrite.\n\n')
root_readme.write_text(first + s[s.index('\n\n')+2:], encoding='utf-8')
print(json.dumps(manifest, indent=2))
