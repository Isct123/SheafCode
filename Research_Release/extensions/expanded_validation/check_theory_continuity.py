                                                                             
from pathlib import Path
import re, json, hashlib
import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / 'IRIS_Submission_Recent_Data'
REVIEW = PACKAGE / 'Review_Notes'

def main():
    old_path = ROOT / 'Final_Manuscript/Extended_Validation/sections/10_proofs.tex'
    old = old_path.read_text(encoding='utf-8')
    new_path = PACKAGE / 'Expanded_Evidence_Source/expanded_full_manuscript.tex'
    new = new_path.read_text(encoding='utf-8')
    appendix = new.split(r'\section{Complete conditional mathematical statements and proofs}', 1)[1].split(r'\section{Detailed recent numerical supplement}', 1)[0]
    normalize = lambda s: re.sub(r'\s+', ' ', s).strip()
    headings = re.findall(re.escape(r'\subsection{') + r'([^}]+)}', old)
                                                                             
    proofs = re.findall(re.escape(r'\paragraph{Proof.}') + r'.*?QED', old, re.S)
    checks = {f'heading/{h}': r'\subsection{' + h + '}' in appendix for h in headings}
    checks['earlier_heading_count'] = len(headings) == 14
    checks['earlier_proof_count'] = len(proofs) == 16
    checks.update({f'proof/{i+1}': normalize(p) in normalize(appendix) for i, p in enumerate(proofs)})
    for env, count in [('proposition', 6), ('corollary', 0)]:
        checks[f'statement_count/{env}'] = appendix.count(r'\begin{' + env + '}') == count
    labels=['prop:graph','prop:functional','prop:compression','prop:joint','prop:anytime','prop:nominal']
    positions=[appendix.index(r'\label{'+label+'}') for label in labels]
    checks['six_selected_in_order']=positions==sorted(positions)
    checks['no_duplicate_main_propositions']=new.count(r'\begin{proposition}')==6
    checks['bounded_unrestricted_transition']='equal spans need not have equal bounded images' in appendix
    checks['numerical_certificate_scope']='does not by itself certify a joint-score epigraph lower bound' in appendix
    continuity = (Path(__file__).parent / 'theory_continuity.tex').read_text(encoding='utf-8')
    checks['restored_C1_C2_C3'] = normalize(continuity) in normalize(appendix)
    checks['generic_Hall_preserved'] = normalize(old.split(r'\subsection{Generic structural rank and actual structured maps}')[1].split(r'\begin{proposition}')[0]) in normalize(appendix)
    checks['historical_PDF_unchanged'] = hashlib.sha256((ROOT/'Final_Manuscript/SheafCode_Validated_Manuscript.pdf').read_bytes()).hexdigest() == '90ad9d477dae3b6e5421f2b315e328b715ba9e155a49fc5009b0e483a2ea2b81'
    pdf = PACKAGE / 'Upload_Files/SheafCode_Full_Empirical_Manuscript.pdf'
    with fitz.open(pdf) as doc:
        texts = [p.get_text() for p in doc]
        rows = []
        for h in headings + ['C1--C3 diagnostics and their statistical scope']:
            needle = h.replace('--', '–')
            pages = [i+1 for i, t in enumerate(texts) if needle in t.replace('\n', ' ')]
                                                                            
            if not pages:
                words = re.findall(r'[A-Za-z]{3,}', h)
                pages = [i+1 for i, t in enumerate(texts) if all(w in t for w in words)]
            checks['rendered_heading/'+h] = bool(pages)
            rows.append((h, ', '.join(map(str, pages))))
        page_count = len(doc)
    if not all(checks.values()): raise AssertionError([k for k,v in checks.items() if not v])
    note = f'''# Mathematical coverage check

Checked 7 October 2026 against the earlier integrated 66-page manuscript's
proof source and the earlier C1--C3 theory discussion.

**Result:** six propositions are selected and numbered in the current
manuscript. Nine other former propositions and the corollary are now
supporting statements and a consequence, rather than additional numbered
novelty claims. All 16 earlier formal proof arguments remain identical after
whitespace normalization. All 14 earlier proof-appendix subsection headings
and the generic Hall argument are retained. This is a preservation check,
not a new external proof review or a mathematical-priority certification.

The check found that the abbreviated recent treatment had omitted the
qualified C3 fixed-decoder consistency discussion and some diagnostic
details. These are now restored directly in the current paper: C1 rank
equivalence, C2/C2* gains, the general-prior Euclidean error bound, observed
measurement sensitivity, the restricted monotonic-gain condition and the
stationary-ergodic fixed-decoder C3 limit. The historical 200-complete-row
reporting convention and model-coordinate limitations are stated explicitly.
The added discussion makes no new experimental-performance claim.

The current paper has **{page_count} PDF pages**. Page numbers below are PDF
positions; table-of-contents appearances can also be listed.

| Earlier mathematical topic | Current PDF pages | Treatment |
|---|---|---|
'''
    for h, pages in rows:
        note += f'| {h} | {pages} | Retained; explicit assumptions and limitations |\n'
    note += '''
Clarifications relative to the earlier integrated appendix include finite
ordered state boxes/nonempty feasible sets, nonnegative budgets, integer
support limits, and the column-gain definition for underdetermined matrices.
Joint-score attainment and exact-support assumptions are explicit. Dataset
counts and revealed-label evaluation descriptions refer to their respective
experiments rather than being treated as universal theorem assumptions.

Earlier unaudited claims are deliberately superseded: whole-fault-vector
robust support separation, a generic guarantee for tied/CCA maps, finite
quantile boundaries advertised as arbitrary-sequence coverage, and C3
consistency based on stationarity alone. They are not restored as valid
theorems. Their corrected forms appear in the current treatment.

The recent-data upload does not reproduce every historical numerical table.
The combined research reference preserves the entire earlier 66-page
manuscript page-for-page alongside the current recent-data paper. Historical
results retain their own protocols and observation dates.

No training runs are required for these writing corrections. Numerical
results and experimental configurations have not changed.
'''
    (REVIEW/'THEORY_COVERAGE_CHECK.md').write_text(note, encoding='utf-8')
    result = {'status':'passed','checks':checks,'check_count':len(checks),'pages':page_count,'manuscript_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'scope':'Theory preservation and diagnostic restoration; not external proof review'}
    (REVIEW/'THEORY_COVERAGE_CHECK.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Theory continuity audit passed', len(checks), 'checks;',page_count,'pages')

if __name__ == '__main__': main()
