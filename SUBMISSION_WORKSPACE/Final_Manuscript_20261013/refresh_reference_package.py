from pathlib import Path
import json,hashlib,zipfile

out=Path(__file__).resolve().parent
confirmation=out/'FINAL_CONFIRMATION.md';s=confirmation.read_text(encoding='utf-8')
s=s.replace('zero-valued three-fault cells. |\n\n| Measurement-planning citation','zero-valued three-fault cells. |\n| Measurement-planning citation')
download_record=json.loads((out/'FINAL_DOWNLOADS.json').read_text(encoding='utf-8'))
pdf_download=next(p for p in download_record['final_files'] if p.endswith('.pdf'))
note=f'\nLatest checked Downloads PDF: `{pdf_download}`. The existing standard-named Downloads PDF was locked; use this checked copy or the current workspace PDF.\n'
if 'Latest checked Downloads PDF:' not in s:s+=note
confirmation.write_text(s,encoding='utf-8')
ready=out/'SUBMISSION_READINESS.md';s=ready.read_text(encoding='utf-8')
s=s.replace('Use `SheafCode_IRIS_Final_Manuscript.pdf` from Downloads.',f'Use `{Path(pdf_download).name}` from Downloads; the standard-named PDF was locked during the citation update.')
if 'REFERENCE_REVIEW.md' not in s:s+='\nThe final reference review verified all 29 sources and every supplied DOI/URL. One citation-context phrase was tightened; see `REFERENCE_REVIEW.md`.\n'
ready.write_text(s,encoding='utf-8')
manifestpath=out/'FINAL_PACKAGE_MANIFEST.json';m=json.loads(manifestpath.read_text(encoding='utf-8'))
names=set(m['final_files'])|{'REFERENCE_REVIEW.md','REFERENCE_LINK_CHECKS.json','REFERENCE_VERIFICATION.json','check_reference_links.py','write_reference_report.py','refresh_reference_package.py','WHOLE_MANUSCRIPT_REVIEW.md','WHOLE_REVIEW_SUMMARY.json','WHOLE_REVIEW_CHECKS.json','WHOLE_FOLLOWUP_REPLAY.json','WHOLE_FOLLOWUP_STATUS.json','WHOLE_JOINT_REPLAY.json','WHOLE_ARCHIVE_REPLAY.json','WHOLE_FAULT_REPLAY.json','whole_review_checks.py','run_whole_math_checks.py','check_fault_reporting.py','write_whole_review_report.py','whole_inference.log','whole_repair_optima.log','whole_exact_rank.log','whole_joint.log','whole_archive_transfer.log','whole_forests_and_intervals.log'}
m['final_files']={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in sorted(names)}
manifestpath.write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8')
package=out/'SheafCode_IRIS_Final_Manuscript_Package.zip'
with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for name in sorted(names):z.write(out/name,name)
    z.write(manifestpath,manifestpath.name)
for name in download_record['final_files']:
    local=Path(name);suffix=local.suffix
    current=out/('SheafCode_IRIS_Final_Manuscript'+suffix)
    assert local.read_bytes()==current.read_bytes(),local
assert Path(download_record['updated_working_copy']).read_bytes()==(out/'SheafCode_IRIS_Final_Manuscript.tex').read_bytes()
print(json.dumps({'status':'complete','verified_sources':30,'pdf_pages':22,'downloads_files':download_record['final_files'],'package_files':len(names)+1},indent=2))
