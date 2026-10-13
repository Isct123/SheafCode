                                                                                             
from pathlib import Path
import difflib, shutil

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / 'audit_tools/final_iris_review/before_proof_expansion'
appendix = (ROOT / 'audit_tools/final_iris_review/expanded_proofs.tex').read_text(encoding='utf-8')
for name in ['Appendix_Readable_Final', 'Compact_Readable_Final']:
    folder = ROOT / 'SUBMISSION_WORKSPACE/Final_IRIS' / name
    backup = BACKUP / name
    backup.mkdir(parents=True, exist_ok=True)
    for filename in ['SheafCode_IRIS_Final.tex', 'SheafCode_IRIS_Final.pdf',
                     'SheafCode_Final_Overleaf.zip', 'README.md', 'final_check.json']:
        src = folder / filename
        if src.exists() and not (backup / filename).exists():
            shutil.copy2(src, backup / filename)
    path = folder / 'SheafCode_IRIS_Final.tex'
    before = (backup / path.name).read_text(encoding='utf-8')
    body = before[:before.index(r'\appendix')].rstrip()
    if body.endswith(r'\clearpage'):
        body = body[:-len(r'\clearpage')].rstrip()
    body = body.replace(r'\usepackage{amsmath,amssymb,booktabs,graphicx,microtype,tabularx}',
                        r'\usepackage{amsmath,amssymb,amsthm,booktabs,graphicx,microtype,tabularx}')
    original = 'The nonnegative integer $f$ bounds queried stations that may instead return arbitrary vectors.'
    replacement = ('Every vector in each clean-error box is permitted, independently across stations and coordinates. '
                   'The nonnegative integer $f$ bounds queried stations that may instead return arbitrary vectors.')
    assert original in body
    body = body.replace(original, replacement)
    original = 'For a finite scalar scenario bank, let $U$ be the stations not yet queried.'
    replacement = ('For a finite scalar scenario bank with independent closed clean-response intervals, '
                   'query each station at most once. Let $U$ be the stations not yet queried.')
    assert original in body
    body = body.replace(original, replacement)
    original = r'whose supremum is $\min\{G,r_B\}$; neither boundary is included.'
    replacement = (r'whose supremum is $\min\{G,r_B\}$. The separation boundary is excluded; '
                   r'$\rho<G$ is the chosen sufficient strict label-preservation range, '
                   r'and failure at $\rho=G$ is not asserted.')
    assert original in body
    body = body.replace(original, replacement)
    after = body + '\n\n' + appendix
    if name == 'Compact_Readable_Final':
        after = after.replace(
            r'For hour $t$, a \emph{causal} estimate uses only observations available through $t$. This describes its information set, not causal inference about pollution sources. A \emph{retrospective} estimate also uses later observations.',
            r'During an outage, we estimate missing readings using only observations available at the time. We call this a \emph{causal} estimate; the term refers to the information used, without implying causal inference about pollution sources. A \emph{retrospective} estimate also uses later observations.')
    path.write_text(after, encoding='utf-8')
    (folder / 'proof_expansion.diff').write_text(''.join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile='before/SheafCode_IRIS_Final.tex', tofile='after/SheafCode_IRIS_Final.tex')),
        encoding='utf-8')
    print(name, 'expanded seven proofs')
