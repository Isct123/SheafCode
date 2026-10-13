                                                                     
from pathlib import Path
import hashlib
import json
import re
import shutil
import sys
import zipfile

import pymupdf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'audit_tools/final_iris_review'
APPENDIX = (REVIEW / 'expanded_proofs.tex').read_text(encoding='utf-8')
NAMES = ['Appendix_Readable_Final', 'Compact_Readable_Final']
if '--compact' in sys.argv:
    NAMES = ['Compact_Readable_Final']
CHANGES = [
    (r'\usepackage{amsmath,amssymb,booktabs,graphicx,microtype,tabularx}',
     r'\usepackage{amsmath,amssymb,amsthm,booktabs,graphicx,microtype,tabularx}'),
    ('The nonnegative integer $f$ bounds queried stations that may instead return arbitrary vectors.',
     'Every vector in each clean-error box is permitted, independently across stations and coordinates. '
     'The nonnegative integer $f$ bounds queried stations that may instead return arbitrary vectors.'),
    ('For a finite scalar scenario bank, let $U$ be the stations not yet queried.',
     'For a finite scalar scenario bank with independent closed clean-response intervals, '
     'query each station at most once. Let $U$ be the stations not yet queried.'),
    (r'whose supremum is $\min\{G,r_B\}$; neither boundary is included.',
     r'whose supremum is $\min\{G,r_B\}$. The separation boundary is excluded; '
     r'$\rho<G$ is the chosen sufficient strict label-preservation range, '
     r'and failure at $\rho=G$ is not asserted.'),
]


def body(text):
    result = text[:text.index(r'\appendix')].rstrip()
    if result.endswith(r'\clearpage'):
        result = result[:-len(r'\clearpage')].rstrip()
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_log(path):
    log = path.read_text(encoding='utf-8', errors='replace')
    assert not re.search(r'undefined|Overfull|Missing character|LaTeX Error', log, re.I), path


for name in NAMES:
    out = ROOT / 'SUBMISSION_WORKSPACE/Final_IRIS' / name
    previous = REVIEW / 'before_proof_expansion' / name
    stage = REVIEW / 'expanded_build' / name
    fresh = REVIEW / 'expanded_package_compile' / name
    fresh.mkdir(parents=True, exist_ok=True)
    src = out / 'SheafCode_IRIS_Final.tex'
    text = src.read_text(encoding='utf-8')
    old = (previous / src.name).read_text(encoding='utf-8')
    expected = body(old)
    for before, after in CHANGES:
        assert expected.count(before) == 1
        expected = expected.replace(before, after)
    if name == 'Compact_Readable_Final':
        expected = expected.replace(
            r'For hour $t$, a \emph{causal} estimate uses only observations available through $t$. This describes its information set, not causal inference about pollution sources. A \emph{retrospective} estimate also uses later observations.',
            r'During an outage, we estimate missing readings using only observations available at the time. We call this a \emph{causal} estimate; the term refers to the information used, without implying causal inference about pollution sources. A \emph{retrospective} estimate also uses later observations.')
    assert text == expected + '\n\n' + APPENDIX, 'Unapproved body change'
    assert len(re.findall(r'\\begin\{proof\}', text)) == 7
    assert len(re.findall(r'\\end\{proof\}', text)) == 7
    assert len(re.findall(r'\\begin\{proposition\}', text)) == 6
    assert len(re.findall(r'\\begin\{theorem\}', text)) == 1
    figures = len(re.findall(r'\\begin\{figure\}', text))
    tables = len(re.findall(r'\\begin\{table\}', text))
    assert figures == len(re.findall(r'\\begin\{figure\}', old))
    assert tables == len(re.findall(r'\\begin\{table\}', old))
    for env in ['figure', 'table', 'thebibliography']:
        pattern = rf'\\begin\{{{env}\}}.*?\\end\{{{env}\}}'
        assert re.findall(pattern, text, re.S) == re.findall(pattern, old, re.S)
    labels = re.findall(r'\\label\{([^}]+)\}', text)
    assert len(labels) == len(set(labels))
    assert set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', text)) <= set(labels)
    bibkeys = set(re.findall(r'\\bibitem(?:\[.*?\])?\{([^}]+)\}', text, re.S))
    citekeys = {key for group in re.findall(r'\\cite[pt]?\{([^}]+)\}', text)
                for key in group.split(',')}
    assert citekeys <= bibkeys
    check_log(stage / 'SheafCode_IRIS_Final.log')
    doc = pymupdf.open(stage / 'SheafCode_IRIS_Final.pdf')
    texts = [p.get_text() for p in doc]
    proof_page = next(i+1 for i, t in enumerate(texts) if '\nProofs\n' in t)
    reference_page = next(i+1 for i, t in enumerate(texts) if '\nReferences\n' in t)
    appendix_text = '\n'.join(texts[proof_page-1:])
    assert all(f'A.{i}' in appendix_text for i in range(1, 8))
    assert appendix_text.count('Proof.') == 7
    proof_links = [link for page in doc for link in page.get_links()
                   if link.get('nameddest', '').startswith(('appendix.A', 'subsection.A.'))]
                                                                       
    assert len(proof_links) == 8
    assert all(proof_page-1 <= link['page'] < len(doc) for link in proof_links)
    for page in doc:
        for block in page.get_text('blocks'):
            assert -1 <= block[0] < block[2] <= page.rect.width+1
            assert -1 <= block[1] < block[3] <= page.rect.height+1
    images = []
    for i in range(proof_page-1, len(doc)):
        path = stage / f'proof_page_{i+1}.png'
        doc[i].get_pixmap(matrix=pymupdf.Matrix(1.2, 1.2)).save(path)
        images.append(path)
    canvas = Image.new('RGB', (1000, 750*((len(images)+1)//2)), '#cccccc')
    draw = ImageDraw.Draw(canvas)
    for k, path in enumerate(images):
        im = Image.open(path)
        im.thumbnail((480, 690))
        x, y = 10+k%2*500, 30+k//2*750
        canvas.paste(im, (x, y))
        draw.text((x, y-20), f'Page {proof_page+k}', fill='black')
    canvas.save(stage / 'proof_contact.png')
    package = out / 'SheafCode_Final_Overleaf.zip'
    pdf_path = out / 'SheafCode_IRIS_Final.pdf'
    if '--final' in sys.argv:
        saved_manifest = json.loads((out / 'final_check.json').read_text(encoding='utf-8'))
        pdf_path = out / saved_manifest.get('pdf_filename', pdf_path.name)
    if '--final' not in sys.argv:
        for suffix in ['pdf', 'log', 'aux', 'out']:
            artifact = stage / f'SheafCode_IRIS_Final.{suffix}'
            try:
                shutil.copy2(artifact, out / artifact.name)
            except PermissionError:
                if suffix != 'pdf':
                    raise
                pdf_path = out / 'SheafCode_IRIS_Final_Updated.pdf'
                shutil.copy2(artifact, pdf_path)
        with zipfile.ZipFile(package, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            z.write(src, src.name)
    with zipfile.ZipFile(package) as z:
        assert z.namelist() == [src.name]
        assert z.read(src.name) == src.read_bytes()
        z.extractall(fresh)
    fresh_ok = '--final' in sys.argv
    if fresh_ok:
        check_log(fresh / 'SheafCode_IRIS_Final.log')
        with pymupdf.open(fresh / 'SheafCode_IRIS_Final.pdf') as fresh_doc:
            assert [p.get_text() for p in fresh_doc] == texts
        assert pdf_path.read_bytes() == (stage / 'SheafCode_IRIS_Final.pdf').read_bytes()
    with pymupdf.open(previous / 'SheafCode_IRIS_Final.pdf') as old_doc:
        old_pages = len(old_doc)
    manifest = {
        'revision': 'expanded formal proof appendix',
        'pages': len(doc), 'previous_pages': old_pages,
        'additional_pages': len(doc)-old_pages,
        'body_font_pt': 11,
        'margins_inches': 1 if name.startswith('Appendix') else .8,
        'references_begin_page': reference_page,
        'proofs_begin_page': proof_page, 'proofs_end_page': len(doc),
        'complete_proofs': 7, 'propositions': 6, 'theorems': 1,
        'body_changes': ['amsthm proof environments', 'independent clean-error boxes',
                         'one query per station for adaptive model',
                         'sufficient label-preservation boundary clarified']
                        + (['causal and retrospective introduction reworded']
                           if name == 'Compact_Readable_Final' else []),
        'remaining_body_preserved_exactly': True,
        'figures_preserved_exactly': figures, 'tables_preserved_exactly': tables,
        'bibliography_preserved_exactly': True,
        'overfull_boxes': 0, 'undefined_references': 0,
        'proof_links_resolve': len(proof_links),
        'fresh_zip_compilation_verified': fresh_ok,
        'source_sha256': digest(src), 'pdf_filename': pdf_path.name, 'pdf_sha256': digest(pdf_path),
        'zip_sha256': digest(package), 'previous_source_sha256': digest(previous / src.name),
    }
    (out / 'final_check.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    (out / 'README.md').write_text(
        f'This standalone manuscript has {len(doc)} pages, including references and all seven '
        f'expanded proofs. Appendix A occupies pages {proof_page}–{len(doc)}. '
        f'The current compiled PDF is {pdf_path.name}. '
        f'The previous {old_pages}-page bundle is preserved in '
        f'audit_tools/final_iris_review/before_proof_expansion/{name}/.\n\n'
        'The appendix now gives explicit tree transports and ambiguity spaces, compactness and '
        'LP arguments, weighted-inequality derivations, the search coverage invariant, both '
        'directions of the query-repair condition, stability-boundary witnesses, and the '
        'adaptive-state and recurrence proof. Each proof uses a separate heading and a standard '
        'proof environment. The text remains 11pt.\n\n'
        'The body clarifies independent clean-response boxes, one query per station in the '
        'adaptive model, and the sufficient strict label-preservation range. '
        + ('The causal/retrospective introduction paragraph was subsequently reworded for flow. '
           if name == 'Compact_Readable_Final' else '')
        + 'All other body '
        f'content, {figures} figures, {tables} tables and the bibliography are preserved. '
        'No new experiments were run for this proof revision.\n\n'
        'Upload SheafCode_Final_Overleaf.zip to Overleaf and compile SheafCode_IRIS_Final.tex. '
        'The ZIP contains one TEX file with embedded figures and references, using the standard '
        'amsthm package. '
        + ('A fresh extraction compiles and matches the delivered PDF text. ' if fresh_ok else '')
        + 'Cross-references resolve and there are no overfull boxes. proof_expansion.diff '
        'records the source changes; final_check.json records validation and artifact hashes.\n',
        encoding='utf-8')
    print(json.dumps({'version': name, **manifest}, indent=2))
