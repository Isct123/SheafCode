import ast
import json
from pathlib import Path
import numpy as np
import pymupdf

root = Path(__file__).parent / 'code/sheafcode'
bad = []
files = list(root.rglob('*.py'))
for f in files:
    try:
        ast.parse(f.read_text(encoding='utf-8'))
    except Exception as e:
        bad.append({'file': str(f.relative_to(root)), 'error': str(e)})
A = [json.loads(f.read_text()) for f in (root/'results').glob('*/theorems_v4.json')]
C = [json.loads(f.read_text()) for f in (root/'results').glob('*/aci_v4.json')]
sts = [s for c in C for s in c['stations']]
result = {
    'python_files': len(files), 'syntax_errors': bad,
    'hall_sets': sum(a['theorem_A']['total'] for a in A),
    'hall_agreements': sum(a['theorem_A']['agreement'] for a in A),
    'thin_predicted_singles': sum(a['theorem_A']['thin_sheaf_check']['knn_k1_r1']['predicted_nonidentifiable_singles'] for a in A),
    'thin_actual_singles': sum(a['theorem_A']['thin_sheaf_check']['knn_k1_r1']['learned_nonidentifiable_singles'] for a in A),
    'thin_sets': sum(v['sets_checked'] for a in A for v in a['theorem_A']['thin_sheaf_check'].values()),
    'thin_agreements': sum(v['generic_equals_learned'] for a in A for v in a['theorem_A']['thin_sheaf_check'].values()),
    'aci_regions': len(C), 'aci_stations': len(sts),
    'aci_region_macro_coverage': float(np.mean([c['summary']['coverage_aci'] for c in C])),
    'aci_station_macro_coverage': float(np.mean([s['coverage_aci'] for s in sts])),
    'aci_pooled_coverage': sum(s['coverage_aci']*s['n'] for s in sts)/sum(s['n'] for s in sts),
    'aci_within_two_points': sum(abs(s['coverage_aci']-.9)<=.02 for s in sts),
    'aci_mean_of_region_means_of_station_worst_quarters': float(np.mean([c['summary']['worst_quarter_aci'] for c in C])),
    'aci_worst_individual_station_quarter': min(s['worst_quarter_aci'] for s in sts),
    'fault_trials': [{
        'region': a['region'], 'candidates': a['theorem_B']['candidates'],
        'two_fault_two_sigma': next(x for x in a['theorem_B']['results'] if x['t']==2 and x['mag']==2.0),
        'margin': a['theorem_B']['worst_sigma_min_2t'],
    } for a in A],
    'data_files': [str(f.relative_to(root)) for f in (root/'data').rglob('*') if f.is_file()],
}
print(json.dumps(result, indent=2))
(Path(__file__).parent/'artifact_checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
for name in ['SheafCode_PAPER_v4_draft.pdf', 'SheafCode_THEORY_v4.pdf']:
    doc = pymupdf.open(name)
    print(name, 'pages', len(doc), 'embedded images', sum(len(p.get_images()) for p in doc))
    for i in [0, len(doc)-1]:
        doc[i].get_pixmap(matrix=pymupdf.Matrix(1,1)).save(str(Path(__file__).parent/(name+f'.page{i+1}.png')))
