                                                                                   
from pathlib import Path
import csv, hashlib, json, zipfile

B=Path(__file__).resolve().parents[1]
O=B/'Final_Manuscript/Complete_Research'
destination=B/'Final_Manuscript/SheafCode_Complete_Research_Package.zip'
files=set()
for p in O.rglob('*'):
    if p.is_file() and p.name!='PACKAGE_AUDIT.json' and p.suffix.lower() in {'.tex','.bib','.pdf','.csv','.json','.md','.txt','.png'}:
        files.add(p)
files.add(B/'Final_Manuscript/SheafCode_Complete_Manuscript.pdf')
files.add(B/'Final_Manuscript/READ_COMPLETE_MANUSCRIPT.md')
for relative in json.loads((O/'audit/BUILD_INPUTS.json').read_text()):
    files.add(B/relative)
for row in csv.DictReader((O/'audit/EVIDENCE_COVERAGE.csv').open(encoding='utf-8')):
    files.add(B/row['source'])
for name in ['build_complete_research_manuscript.py','complete_research_proofs.tex',
             'complete_research_references.bib','recheck_complete_research.py',
             'audit_complete_manuscript.py','package_complete_research.py']:
    files.add(B/'audit_tools'/name)
for relative in ['Research_Release/validation_results/Supplemental_Research_Results.zip',
                 'Research_Release/joint_decision_results/Additional_Results.zip',
                 'Final_Manuscript/NEW_MATERIAL_REVIEW.md']:
    p=B/relative
    if p.exists():files.add(p)
assert all(p.exists() for p in files)
manifest={str(p.relative_to(B)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(files)}
with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for p in sorted(files):archive.write(p,str(p.relative_to(B)).replace('\\','/'))
    archive.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2))
    archive.writestr('START_HERE.txt',
        'Read Final_Manuscript/SheafCode_Complete_Manuscript.pdf.\n'
        'Read Final_Manuscript/Complete_Research/audit/FINAL_AUDIT.md for scope and freeze guidance.\n'
        'Editable LaTeX: Final_Manuscript/Complete_Research/main.tex.\n'
        'Upload the complete Complete_Research directory for Overleaf editing.\n'
        'Dependencies retain their original relative paths for traceability.\n'
        'Bundled evidence archives contain additional saved results and code; they are not a complete training environment.\n'
        'No further long run is needed to document the included completed results.\n')
with zipfile.ZipFile(destination) as archive:
    assert archive.testzip() is None
    for p,h in manifest.items():
        assert hashlib.sha256(archive.read(p)).hexdigest()==h,p
record=dict(status='passed',files=len(files),archive=str(destination.relative_to(B)),
            bytes=destination.stat().st_size,sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
            scope='CRC and every packaged member SHA-256 verified against its source.')
(O/'audit/PACKAGE_AUDIT.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
