                                                                                    
from pathlib import Path
import hashlib
import json
import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'IRIS_Submission_Recent_Data'
DEST = PACKAGE / 'Supporting_Reference'
RECENT = PACKAGE / 'Upload_Files/SheafCode_Full_Empirical_Manuscript.pdf'
if not RECENT.exists():
    RECENT = PACKAGE / 'Upload_Files/SheafCode_Three_Network_Full_Manuscript.pdf'
if not RECENT.exists():
    RECENT = DEST / 'Complete_Recent_Technical_Reference.pdf'
HISTORICAL = ROOT / 'Final_Manuscript/SheafCode_Validated_Manuscript.pdf'
TARGET = DEST / 'SheafCode_Complete_Research_Reference.pdf'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_page(doc, title, paragraphs):
    page = doc.new_page(width=595, height=842)
    page.insert_text((48, 64), title, fontsize=18)
    y = 98
    for paragraph in paragraphs:
        rect = fitz.Rect(48, y, 547, y + 120)
        result = page.insert_textbox(rect, paragraph, fontsize=11, lineheight=1.35)
        if result < 0:
            raise RuntimeError('Text exceeds the bridge-page allocation')
        y += 120 - result + 22
    if y > 794:
        raise RuntimeError('Bridge page exceeds printable area')


def main():
    global TARGET
    sources = [RECENT, HISTORICAL]
    before = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    doc = fitz.open()
    text_page(doc, 'SheafCode: Complete Research Reference', [
        'Recent empirical core, complete conditional mathematics, and historical supporting research. Compiled 7 October 2026. This reading volume preserves both reviewed manuscripts in full, with original figures, tables, references and page numbering.',
        'Part I is the current recent-observation study and complete mathematical treatment. It retains the original Indian, US and UK benchmarks and adds the broad external validation, including every favorable, negative and unavailable cohort. The longer Indian resplit reuses observations and remains a separate sensitivity. Dataset-specific percentages are not pooled into one accuracy claim.',
        'Part II is historical supporting evidence. It preserves the earlier broad project, supplied snapshot and prolonged-outage analyses, mathematical extensions and additional computational validation. Its own abstract and headline describe that historical protocol; they do not replace the current primary endpoint.',
        'This is a complete research reference, not the designated IRIS upload. Organizer acceptance of older observations as supporting evidence is unconfirmed, and the historical document retains geographic source identities. Use the anonymous recent paper and synopsis for the currently prepared upload fields.'
    ])
    recent_start = len(doc)
    with fitz.open(RECENT) as recent:
        doc.insert_pdf(recent)
        recent_pages = len(recent)
        recent_toc = recent.get_toc()
    text_page(doc, 'Historical Supporting Evidence', [
        'The following manuscript is retained in its entirety. It contains the earlier reconstruction results, uncertainty and fault experiments, decision proxies, supplied-study numerical tables, conditional proofs, evidence coverage map, and six additional computational validation tracks. No original page or figure has been removed from this preserved manuscript.',
        'Read each result with its own dataset, cohort, information set, statistical unit and denominator. In particular, retrospective prolonged-outage gains, causal hourly-cohort gains and single-station snapshot gains measure different tasks. They must not be pooled or compared as one evolving accuracy percentage.',
        'Historical source data and partially overlapping networks are not independent recent replications. Shared recent observation records are disclosed in Part I. Negative transfer results, weak-fault failures, interval undercoverage and inconclusive environmental proxies remain evidence, rather than being dropped in favor of larger numerical gains.',
        'Current clarified mathematical assumptions in Part I govern the current study. Previously rejected propositions and unsupported operational guarantees are not reinstated by preserving historical evidence. More pages or theorem labels do not establish first-ever novelty, stronger statistical evidence or demonstrated pollution reduction.'
    ])
    historical_start = len(doc)
    with fitz.open(HISTORICAL) as historical:
        doc.insert_pdf(historical)
        historical_pages = len(historical)
        historical_toc = historical.get_toc()
    toc = [[1, 'Reading guide and evidence boundaries', 1],
           [1, 'Part I: Recent study and complete technical supplement', recent_start + 1]]
    toc += [[level + 1, title, page + recent_start] for level, title, page in recent_toc if page > 0]
    toc += [[1, 'Historical evidence: interpretation guide', historical_start],
            [1, 'Part II: Entire historical research manuscript', historical_start + 1]]
    toc += [[level + 1, title, page + historical_start] for level, title, page in historical_toc if page > 0]
    doc.set_toc(toc)
    doc.set_metadata({'title': 'SheafCode: Complete Research Reference',
                      'subject': 'Recent primary study and historical supporting evidence',
                      'author': 'Research reference; assisted work disclosed in sources'})
    temp = TARGET.with_suffix('.tmp.pdf')
    doc.save(temp, garbage=4, deflate=True)
    doc.close()
    try:
        temp.replace(TARGET)
    except PermissionError:
                                                                              
        TARGET = DEST / 'SheafCode_Complete_Research_Reference_With_Expanded_Evidence.pdf'
        temp.replace(TARGET)
    checks = []
    with fitz.open(TARGET) as combined:
        for source, offset in [(RECENT, recent_start), (HISTORICAL, historical_start)]:
            with fitz.open(source) as original:
                for i, page in enumerate(original):
                    retained = combined[offset + i]
                    text_equal = page.get_text() == retained.get_text()
                    a = page.get_pixmap(matrix=fitz.Matrix(.5, .5), alpha=False)
                    b = retained.get_pixmap(matrix=fitz.Matrix(.5, .5), alpha=False)
                    pixels_equal = a.samples == b.samples and a.width == b.width and a.height == b.height
                    if not text_equal or not pixels_equal:
                        raise RuntimeError(f'Page preservation failed: {source.name}, {i + 1}')
                    checks.append({'source': source.name, 'source_page': i + 1,
                                   'combined_page': offset + i + 1,
                                   'text_identical': text_equal, 'render_identical': pixels_equal})
        pages = len(combined)
        combined[0].get_pixmap(matrix=fitz.Matrix(1, 1)).save(DEST / 'Complete_Reference_Cover.png')
    after = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    if before != after:
        raise RuntimeError('Frozen source manuscript changed')
    manifest = {'status': 'passed', 'date': '2026-10-07', 'output': str(TARGET.relative_to(ROOT)),
                'sha256': sha(TARGET), 'bytes': TARGET.stat().st_size, 'pages': pages,
                'recent_pages': recent_pages, 'historical_pages': historical_pages,
                'recent_start_page': recent_start + 1, 'historical_start_page': historical_start + 1,
                'sources_unchanged': True, 'source_sha256': before, 'page_checks': checks,
                'scope': 'Document preservation and assembly; no additional training or independent proof review'}
    (PACKAGE / 'Review_Notes/COMPLETE_REFERENCE_CHECKS.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (DEST / 'COMPLETE_REFERENCE_GUIDE.md').write_text(f'''# Complete project reading copy

Open [{TARGET.name}]({TARGET.name}): **{pages} pages**.

- Page 1: reading guide, primary result and submission boundaries.
- Pages {recent_start + 1}-{recent_start + recent_pages}: the entire recent study and technical reference, including complete proofs and detailed numerical supplement.
- Page {historical_start}: historical-evidence interpretation guide.
- Pages {historical_start + 1}-{historical_start + historical_pages}: all {historical_pages} pages of the preserved integrated historical manuscript, including its figures, references, supplied-study tables and additional validation tracks.

Both source manuscripts remain unchanged. Every retained page passed exact extracted-text and half-resolution rendered-pixel comparisons. The checks establish faithful assembly, not a new scientific audit or retraining. Earlier inaccurate draft claims are not restored as valid results; use the reviewed evidence and the current explicit assumptions.

The recent primary endpoint remains the headline. Historical results provide additional context and evidence under their own protocols; larger gains from different information sets must not be substituted for that endpoint. Shared observations are not independent replication. All negative results and limitations remain visible.

This combined reference is separate from the recent reproducibility ZIP, which remains the already verified frozen release. Historical raw data and code are available in `Final_Manuscript/SheafCode_Validated_Research_Package.zip` and the original project folders. Rebuild this reading copy with `audit_tools/python/python.exe audit_tools/build_complete_research_reference.py` after an intentionally reviewed source change; no experiments run in this command.

The designated anonymous IRIS paper and synopsis remain in `Upload_Files`. This combined reference contains older observations and historical geography, and is not a replacement mandatory upload without organizer acceptance. No result, abstract or submitted-data date has been silently changed.
''', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ['status', 'pages', 'recent_pages', 'historical_pages', 'bytes', 'sources_unchanged']}))


if __name__ == '__main__':
    main()
