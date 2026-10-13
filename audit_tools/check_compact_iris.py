                                                                         
from pathlib import Path
import hashlib
import json
import re
import sys
import zipfile
import pymupdf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'SUBMISSION_WORKSPACE/Final_IRIS/Compact_Readable_Final'
BASE = ROOT / 'audit_tools/final_iris_review/before_layout_tighten'
REVIEW = ROOT / 'audit_tools/final_iris_review/compact_layout_review'
FRESH = ROOT / 'audit_tools/final_iris_review/compact_package_compile'
FRESH.mkdir(exist_ok=True)
source_path = OUT / 'SheafCode_IRIS_Final.tex'
source = source_path.read_text(encoding='utf-8')
old = (BASE / source_path.name).read_text(encoding='utf-8')
restored = re.sub(r'\n\n% Compact float spacing.*?\\setlength\{\\belowcaptionskip\}\{0pt\}\n', '', source, flags=re.S)
restored = re.sub(r'\n% Compact appendix headings.*?\\makeatother\n', '', restored, flags=re.S)
restored = restored.replace(r'\resizebox{.46\linewidth}{!}', r'\resizebox{.52\linewidth}{!}')
restored = re.sub(r'\\subsection\{((?:Proposition|Theorem)~\\ref\{[^}]+\})\.\}',
    r'\\subsection{Proof of \1}', restored)
assert restored == old, 'Change outside the requested layout/heading edits'
for pattern in [r'\\begin\{(?:proposition|theorem)\}.*?\\end\{(?:proposition|theorem)\}',
                r'\\textbf\{Proof\.\}.*?\\textit\{QED\.\}']:
    before, after = re.findall(pattern, old, re.S), re.findall(pattern, source, re.S)
    assert before == after and len(after) == 7
assert len(re.findall(r'\\begin\{figure\}', source)) == 7
assert len(re.findall(r'\\begin\{table\}', source)) == 10
log = (OUT / 'SheafCode_IRIS_Final.log').read_text(encoding='utf-8', errors='replace')
assert not re.search(r'undefined|Overfull|Missing character|LaTeX Error', log, re.I)
labels = re.findall(r'\\label\{([^}]+)\}', source)
assert len(labels) == len(set(labels))
assert set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', source)) <= set(labels)

doc = pymupdf.open(OUT / 'SheafCode_IRIS_Final.pdf')
assert len(doc) == 21
texts = [p.get_text() for p in doc]
proofs_page = next(i+1 for i, txt in enumerate(texts) if '\nProofs\n' in txt)
assert proofs_page == 21
assert all(f'A.{i}' in texts[-1] for i in range(1,8))
assert texts[-1].count('QED.') == 7
reference_page = next(i+1 for i, txt in enumerate(texts) if '\nReferences\n' in txt)
links = [link for page in doc for link in page.get_links()
         if link.get('nameddest', '').startswith(('appendix.A', 'subsection.A.'))]
assert len(links) == 8 and all(link.get('page') == 20 for link in links)
for i, page in enumerate(doc):
    for b in page.get_text('blocks'):
        assert b[0] >= -1 and b[2] <= page.rect.width + 1
        assert b[1] >= -1 and b[3] <= page.rect.height + 1
    page.get_pixmap(matrix=pymupdf.Matrix(.62,.62)).save(REVIEW / f'page_{i+1:02}.png')
for start in range(0,len(doc),8):
    canvas = Image.new('RGB',(1000,1460),'#cccccc')
    draw = ImageDraw.Draw(canvas)
    for k in range(8):
        i = start+k
        if i >= len(doc): break
        im = Image.open(REVIEW / f'page_{i+1:02}.png')
        im.thumbnail((470,337))
        x,y = 15+k%2*500,25+k//2*365
        canvas.paste(im,(x,y));draw.text((x,y-15),f'Page {i+1}',fill='black')
    canvas.save(REVIEW / f'contact_{start+1:02}.png')
doc[-1].get_pixmap(matrix=pymupdf.Matrix(1.35,1.35)).save(REVIEW / 'final_proofs.png')

package = OUT / 'SheafCode_Final_Overleaf.zip'
if '--final' not in sys.argv:
    with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.write(source_path,source_path.name)
with zipfile.ZipFile(package) as z:
    assert z.namelist() == [source_path.name]
    assert z.read(source_path.name) == source_path.read_bytes()
    z.extractall(FRESH)
fresh_ok = False
if '--final' in sys.argv:
    fresh = pymupdf.open(FRESH / 'SheafCode_IRIS_Final.pdf')
    assert [p.get_text() for p in fresh] == texts
    fresh_log = (FRESH / 'SheafCode_IRIS_Final.log').read_text(encoding='utf-8',errors='replace')
    assert not re.search(r'undefined|Overfull|Missing character|LaTeX Error',fresh_log,re.I)
    fresh_ok = True
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = {
    'pages':21,'previous_pages':22,'original_pages':27,'pages_saved_this_revision':1,
    'body_font_pt':11,'margins_inches':.8,'references_begin_page':reference_page,
    'proofs_begin_page':21,'all_seven_complete_proofs_on_final_page':True,
    'explanations_and_motivations_preserved':True,'proof_bodies_preserved_exactly':True,
    'formal_statements_preserved_exactly':True,'figures':7,'tables':10,
    'figure_3_width_fraction':.46,'previous_figure_3_width_fraction':.52,
    'overfull_boxes':0,'undefined_references':0,'proof_links_resolve':8,
    'fresh_zip_compilation_verified':fresh_ok,
    'source_sha256':sha(source_path),'pdf_sha256':sha(OUT/'SheafCode_IRIS_Final.pdf'),
    'zip_sha256':sha(package),'previous_source_sha256':sha(BASE/source_path.name),
}
(OUT / 'final_check.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
(OUT / 'README.md').write_text(
    'This standalone manuscript has 21 pages, including references and all seven complete proofs. '
    'The earlier 22-page Shorter_Readable_Final and 27-page Appendix_Readable_Final bundles are preserved.\n\n'
    'Only the requested layout was changed: reduced space around floats and captions, '
    'Figure 3 scaled from 52% to 46% of the text width, and compact appendix headings that run '
    'into the proof paragraphs. All explanations, motivations, examples, formulas, formal '
    'statements and proof bodies remain unchanged. The body and proofs remain 11pt with '
    '0.8-inch margins. All proofs fit on page 21.\n\n'
    'Upload SheafCode_Final_Overleaf.zip to Overleaf and compile SheafCode_IRIS_Final.tex. '
    'The ZIP contains a single TEX file with embedded figures and references. '
    + ('The freshly extracted ZIP compiles and matches the delivered PDF text. ' if fresh_ok else '')
    + 'Cross-references resolve and there are no overfull boxes. layout_changes.diff records '
    'the exact changes; final_check.json records preservation checks and artifact hashes.\n',encoding='utf-8')
print(json.dumps(manifest,indent=2))
