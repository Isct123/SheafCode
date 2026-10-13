                                                                                
from pathlib import Path
import difflib
import hashlib
import json
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent.parent
ORIGINAL = Path(r'C:\Users\rishi\Downloads\SheafCode_IRIS_Clarity_Final.tex')
OUT = ROOT / 'SUBMISSION_WORKSPACE/Clarity_Fixes'
source = ORIGINAL.read_text(encoding='utf-8-sig')
text = source
changes = []

def replace(name, old, new):
    global text
    assert text.count(old) == 1, (name, text.count(old))
    text = text.replace(old, new, 1)
    changes.append(name)

replace('Define precision blocks before their use',
        'Setting the derivative of this objective to zero gives',
        r'Write $P_{UU}$ for the block relating missing coordinates to each other and $P_{UO}$ for the block relating missing coordinates to observed coordinates. Setting the derivative of this objective to zero gives')
replace('Define station support and its minimum size',
        'The smallest nonzero kernel station support is the smallest component with $q_C>0$, or infinity if none exists.',
        r'For a nonzero state satisfying all homogeneous comparisons, its station support is the set of stations with nonzero vectors. The smallest possible support size is $\min_{C:q_C>0}|C|$, or infinity if no such state exists.')
replace('Define the search optimum',
        'Suppose each unresolved node $N$ has a valid upper bound $B_N$ on every genuine explanation it contains.',
        r'Assuming at least one admissible explanation exists, let $M$ be the largest score over all admissible explanations. Suppose each unresolved node $N$ has a valid upper bound $B_N$ on every genuine explanation it contains.')
replace('Explain dataset roles',
        'Observation-age eligibility depends on the actual submission date.',
        "Observation-age eligibility depends on the actual submission date. In the later checks, a role means a particular station network with its fixed data split, rather than an independent field trial.")
replace('Introduce selected cohort codes',
        'Selected US D1 and UK E1 seeds favour sheaf; Illawarra F1 favours GRIN in all three',
        'Seeds for the selected US cohort D1 (EPA county 04-013) and central UK cohort E1 favour sheaf; the NSW Illawarra cohort F1 favours GRIN in all three')
replace('Simplify scoring wording',
        'Methods share the same truth cells.',
        'Methods are scored on the same observed values hidden for the test.')
replace('Explain the incumbent in plain language',
        'pruned below the incumbent.',
        'pruned below the best verified feasible score found so far.')
replace('Simplify the comparison-matrix description',
        'All nominal full operators have zero kernel;',
        'All full comparison matrices under the stated numerical settings have zero kernel;')
replace('Simplify the environmental claim',
        'The supported benefit is computational monitoring continuity and conditional query planning.',
        'The supported benefit is computational monitoring continuity and planning extra measurements under stated assumptions.')
replace('Explain fault tests and snapshot calibration',
        r'\subsection{Decision and query protocols}',
        r'''For the fault tests, we modify recorded readings to simulate offsets, drift, a sensor stuck at one value, scaling errors and errors in one channel. We compare the reported faulty stations with those we changed, and separately score fault detection, estimated fault count and exact localization. Clean calibration readings set the alarm cutoff; these tests simulate faults rather than identify confirmed instrument failures.

For the snapshot intervals, we hide a station's current readings and use absolute PM2.5 prediction errors from the calibration period to set nominal 90\% interval radii. Fixed intervals pool errors or calibrate them by station; risk-grouped intervals split calibration errors using predicted pollution risk. Hidden test readings do not update the intervals. These empirical prediction intervals are different from the bounded feasible set $Z$: their measured coverage does not, by itself, justify a decision certificate.

\subsection{Decision and query protocols}''')

OUT.mkdir(parents=True, exist_ok=True)
target = OUT / 'SheafCode_IRIS_Clarity_Final_Fixed.tex'
target.write_text(text, encoding='utf-8')
assets = ROOT / 'SUBMISSION_WORKSPACE/Combined_IRIS/Source'
shutil.copy2(assets / 'references.bib', OUT / 'references.bib')
shutil.copytree(assets / 'figures', OUT / 'figures', dirs_exist_ok=True)
(OUT / 'clarity_changes.diff').write_text(''.join(difflib.unified_diff(
    source.splitlines(keepends=True), text.splitlines(keepends=True),
    fromfile=ORIGINAL.name, tofile=target.name)), encoding='utf-8')

numbers = lambda s: set(re.findall(r'(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?![\w.])', s))
assert numbers(source) <= numbers(text), 'An original numeric value disappeared'
labels = lambda s: re.findall(r'\\label\{([^}]+)\}', s)
assert labels(source) == labels(text)
assert re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', source) == re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', text)
assert len(re.findall(r'\\begin\{proposition\}', source)) == len(re.findall(r'\\begin\{proposition\}', text))
assert ORIGINAL.read_text(encoding='utf-8-sig') == source
report = {'changes': changes, 'original_unchanged': True,
          'original_numeric_values_retained': True, 'labels_and_figures_unchanged': True,
          'source_sha256': hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()}
(OUT / 'clarity_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({'file': str(target), 'amendments': len(changes)}))

pdf_path = target.with_suffix('.pdf')
if pdf_path.exists():
    import pymupdf
    log = target.with_suffix('.log').read_text(encoding='utf-8', errors='replace')
    assert 'Overfull' not in log
    assert 'undefined' not in log.lower()
    bib = (OUT / 'references.bib').read_text(encoding='utf-8')
    keys = set(re.findall(r'@\w+\s*\{\s*([^,\s]+)', bib))
    citations = {k.strip() for g in re.findall(r'\\cite\w*(?:\[[^\]]*\])*\{([^}]+)\}', text) for k in g.split(',')}
    assert citations <= keys
    assert set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', text)) <= set(labels(text))
    pdf = pymupdf.open(pdf_path)
    assert all(p.get_text().strip() for p in pdf)
    for p in pdf:
        for word in p.get_text('words'):
            assert word[0] >= -1 and word[1] >= -1 and word[2] <= p.rect.width + 1 and word[3] <= p.rect.height + 1
    report.update({'compilation_passed': True, 'pages_including_references': len(pdf),
                   'citations_and_references_resolve': True, 'no_overfull_boxes': True,
                   'pdf_text_within_page_bounds': True})
    (OUT / 'clarity_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with zipfile.ZipFile(OUT / 'Clarity_Fixed_Overleaf.zip', 'w', zipfile.ZIP_DEFLATED) as package:
        for p in [target, OUT / 'references.bib', *sorted((OUT / 'figures').glob('*.pdf'))]:
            package.write(p, p.relative_to(OUT))
    print(json.dumps({'verified_pdf': str(pdf_path), 'pages': len(pdf)}))
