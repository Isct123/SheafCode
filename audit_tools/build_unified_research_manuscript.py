                                                                            
from pathlib import Path
import importlib.util
import json
import re
import shutil
import hashlib
import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'IRIS_Submission_Recent_Data'
SRC = PACKAGE / 'Full_Manuscript_Source'
OLD = ROOT / 'Final_Manuscript/Extended_Validation'
RECENT = PACKAGE / 'Editable_Source'


def main():
    SRC.mkdir(exist_ok=True)
    shutil.copytree(RECENT / 'figures', SRC / 'figures', dirs_exist_ok=True)
    shutil.copytree(OLD / 'figures', SRC / 'historical_figures', dirs_exist_ok=True)
    for name in ['complete_theory.tex', 'references.bib']:
        shutil.copy2(RECENT / name, SRC / name)
    selected = sorted(p for p in (OLD / 'sections').glob('*.tex')
                      if not p.name.startswith(('01_', '10_')))
    chapters = [(p, p.read_text(encoding='utf-8')) for p in selected]
    labels = set(re.findall(r'\\label\{([^}]+)\}', '\n'.join(s for _, s in chapters)))
    def historical_refs(match):
        names = match.group(2).split(',')
        return '\\' + match.group(1) + '{' + ','.join('hist:' + n if n in labels else n for n in names) + '}'
    archive = []
    for path, text in chapters:
        text = re.sub(r'\\(label|ref|eqref|cref|Cref|pageref)\{([^}]+)\}', historical_refs, text)
        text = text.replace('{figures/', '{historical_figures/')
                                                                                    
        name = 'historical_' + path.name
        (SRC / name).write_text(text, encoding='utf-8')
        archive.append('\\clearpage\\input{' + name + '}')
    recent = (RECENT / 'technical_reference.tex').read_text(encoding='utf-8')
    body = recent.split('\\begin{document}', 1)[1].rsplit('\\bibliography{references}', 1)[0]
    body = body.replace('All empirical fitting, model selection, calibration and testing in this paper use observations recorded during 2026.',
                        'The primary empirical study uses only 2026 observations for fitting, model selection, calibration and testing. Historical supporting analyses are separately identified in the appendices and are not included in recent-study denominators.')
    body = body.replace('Recent-Data Causal Air-Quality Reconstruction and Conditional Pollution Certificates',
                        'Causal Air-Quality Reconstruction and Conditional Pollution Certificates')
    body = body.replace('\\end{abstract}', '\nHistorical supporting analyses are retained in separate appendices under their original protocols; they do not contribute to the recent primary comparison.\n\\end{abstract}', 1)
    intro = r'''
\section{Scope of the integrated evidence}
The recent-observation study is the main empirical argument. The supporting historical chapters preserve the broader project: supplied snapshot/channel comparisons, prolonged retrospective outages, earlier causal hourly cohorts, diagnostic rankings, faults, intervals, conditional decision certificates, simulated repair and additional computational validation. Their dataset ages, observation windows, information sets and statistical units differ. They are not pooled with the recent primary endpoint and do not constitute additional independent recent replications. Shared observations are disclosed in the recent study.

Historical support includes both favorable and unfavorable outcomes. Its original abstracts and title pages are omitted to give this manuscript one title, one abstract, one contents page and one consolidated bibliography. The current complete mathematical appendix supplies the clarified assumptions and proofs; an older duplicate proof chapter is not repeated. Previously rejected propositions are not restored as valid claims. More observations or larger retrospective gains do not establish causal superiority under a different protocol.

This integrated manuscript is a complete scientific reading document. IRIS acceptance of historical observation results as supplementary evidence has not been confirmed. The published rule limits experimental data to twelve months at submission. Historical appendix data must not be represented as recent; the anonymous recent-only manuscript remains available if that distinction is not accepted. Accurate student/guide contributions and assisted-work disclosure remain necessary.
'''
    body = body.replace('\\section{Introduction and objective}', intro + '\n\\section{Introduction and objective}', 1)
    boundary = r'''
\clearpage\section{Historical supporting research: reading boundary}
The remaining appendices contain earlier reviewed analyses, not results from the current recent-data fits. Numerical headlines and cohort counts in these sections apply only to their stated historical protocols. Any statement describing a previously ``main'' or ``primary'' result denotes the endpoint of that historical experiment; the current manuscript headline remains the recent Network A causal comparison. Native units, source limitations, failures and statistical-unit caveats are retained.

The following sections reuse historical source identities for reproducibility. This complete reading document is therefore not certified as an anonymous or data-age-compliant IRIS upload. No claim is made that adding an appendix exempts experimental observations from organizer rules. The recent-only paper contains the current validated empirical core and can be uploaded without this unconfirmed historical supplement.
'''
    preamble = (OLD / 'preamble.tex').read_text(encoding='utf-8')
    preamble = preamble.replace('margin=0.78in', 'margin=1in').replace('Integrated research reference', 'Recent study and supporting evidence')
    source = '\\documentclass[11pt,a4paper]{article}\n' + preamble + '\n\\begin{document}\n' + body + boundary + '\n'.join(archive) + '\n\\clearpage\\bibliography{references}\n\\end{document}\n'
    (SRC / 'full_manuscript.tex').write_text(source, encoding='utf-8')
    spec = importlib.util.spec_from_file_location('builder', ROOT / 'audit_tools/build_iris_submission.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    builder.SRC = SRC
    target = PACKAGE / 'Supporting_Reference/SheafCode_Full_Manuscript.pdf'
    builder.compile_pdf('full_manuscript', target)
    log = (SRC / 'full_manuscript.log').read_text(encoding='utf-8', errors='replace')
    if 'undefined' in log.lower() or 'multiply defined' in log.lower():
        raise RuntimeError('Resolve undefined or repeated LaTeX references before completion')
    with fitz.open(target) as doc:
        pages = len(doc)
        text = '\n'.join(page.get_text() for page in doc)
        checks = {'one_abstract': text.count('Abstract') == 1,
                  'historical_boundary': 'Historical supporting research: reading boundary' in text,
                  'recent_headline': '16.5%' in text,
                  'historical_prolonged': '34.0%' in text,
                  'historical_transfer_failure': '5.93' in text,
                  'size_below_20mb': target.stat().st_size < 20_000_000,
                  'all_selected_chapters': all((SRC / ('historical_' + p.name)).exists() for p in selected)}
        doc[0].get_pixmap().save(PACKAGE / 'Supporting_Reference/Full_Manuscript_Cover.png')
    if not all(checks.values()):
        raise RuntimeError(str(checks))
    manifest = {'status': 'passed', 'pages': pages, 'bytes': target.stat().st_size,
                'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'checks': checks,
                'historical_chapters': [p.name for p in selected],
                'scope': 'Unified typesetting and cross-reference checks; historical submission eligibility unconfirmed'}
    (PACKAGE / 'Review_Notes/UNIFIED_MANUSCRIPT_CHECKS.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
