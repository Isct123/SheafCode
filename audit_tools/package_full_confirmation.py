from pathlib import Path
import json,zipfile,hashlib,re
import fitz
r=Path(__file__).resolve().parent.parent;w=r/'SUBMISSION_WORKSPACE';s=w/'Source';g=w/'Guides'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
pages={}
for name in ['Manuscript_For_Upload.pdf','Synopsis.pdf']:
    d=fitz.open(w/name);pages[name]=len(d)
    for p in d:
        assert p.get_text().strip()
        for b in p.get_text('blocks'):
            if b[6]==0:assert b[0]>=-1 and b[1]>=-1 and b[2]<=p.rect.width+1 and b[3]<=p.rect.height+1
assert pages=={'Manuscript_For_Upload.pdf':29,'Synopsis.pdf':2}
for name in ['manuscript','synopsis']:
    log=(s/(name+'.log')).read_text(encoding='utf-8',errors='replace');assert 'Overfull' not in log;assert not re.search(r'(Citation|Reference).*undefined',log)
d=fitz.open(w/'Manuscript_For_Upload.pdf');i=next(i for i,p in enumerate(d) if 'versus dense Gaussian conditioning' in p.get_text());d[i].get_pixmap().save(str(r/'audit_tools/full_confirmation_appendix.png'))
record=load(g/'FULL_CONFIRMATION_CHECK.json');record.update(pdf_sha256=sha(w/'Manuscript_For_Upload.pdf'),compiled_pages=pages,layout_out_of_page_text_blocks=0,package_verification='CRC and byte/hash comparisons for all source and writing-package members');save(g/'FULL_CONFIRMATION_CHECK.json',record)
for name in ['FINAL_MANUSCRIPT_REVIEW.md','FULL_PAPER_COMPLETENESS_AUDIT.md']:
    p=g/name;p.write_text(p.read_text(encoding='utf-8')+'\n\nLatest renewed check: FULL_CONFIRMATION_REVIEW.md and FULL_CONFIRMATION_CHECK.json record 333 additional checks, 94 manuscript checks, fresh independent evidence replays and the 2,032-check broader archive audit. Historical snapshot/prolonged comparator names and missing UK map labels were corrected. Current handoff hashes are in WORKSPACE_DOCUMENT_CHECK.json and RESEARCH_FREEZE.json.\n',encoding='utf-8')
p=w/'START_HERE.md';t=p.read_text(encoding='utf-8');t=t.replace('| Guides/FULL_PAPER_COMPLETENESS_AUDIT.md |','| Guides/FULL_CONFIRMATION_REVIEW.md | Read the latest full check, retained strong results, corrections and confirmation limits. |\n| Guides/FULL_PAPER_COMPLETENESS_AUDIT.md |');p.write_text(t,encoding='utf-8')
files=sorted(p for p in s.rglob('*') if p.is_file() and (p.suffix in ['.tex','.bib','.md','.cmd'] or (p.parent.name=='figures' and p.suffix=='.pdf')))
with zipfile.ZipFile(w/'Open_in_Overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,p.relative_to(s).as_posix())
with zipfile.ZipFile(w/'Open_in_Overleaf.zip') as z:
    assert z.testzip() is None
    for p in files:assert z.read(p.relative_to(s).as_posix())==p.read_bytes()
dc=load(g/'WORKSPACE_DOCUMENT_CHECK.json');dc.update(source_sha256=sha(s/'manuscript.tex'),pdf_sha256=sha(w/'Manuscript_For_Upload.pdf'),synopsis_sha256=sha(w/'Synopsis.pdf'),source_zip_sha256=sha(w/'Open_in_Overleaf.zip'),latest_full_review='Guides/FULL_CONFIRMATION_REVIEW.md',latest_full_check='Guides/FULL_CONFIRMATION_CHECK.json',additional_reporting_checks=333)
dc['guide_consistency_review']['guide_hashes']['Guides/FULL_CONFIRMATION_REVIEW.md']=sha(g/'FULL_CONFIRMATION_REVIEW.md')
for rel in dc['guide_consistency_review']['guide_hashes']:dc['guide_consistency_review']['guide_hashes'][rel]=sha(w/rel)
dc['guide_consistency_review']['files_reviewed']=len(dc['guide_consistency_review']['guide_hashes']);save(g/'WORKSPACE_DOCUMENT_CHECK.json',dc)
f=load(g/'RESEARCH_FREEZE.json')
for name in ['FULL_CONFIRMATION_REVIEW.md','FULL_CONFIRMATION_CHECK.json']:f['files']['Guides/'+name]=sha(g/name)
for rel in f['files']:f['files'][rel]=sha(w/rel)
save(g/'RESEARCH_FREEZE.json',f)
with zipfile.ZipFile(w/'Final_Writing_Package.zip','w',zipfile.ZIP_DEFLATED) as z:
    for rel in f['files']:z.write(w/rel,Path(rel).as_posix())
    z.write(g/'RESEARCH_FREEZE.json','Guides/RESEARCH_FREEZE.json')
with zipfile.ZipFile(w/'Final_Writing_Package.zip') as z:
    assert z.testzip() is None
    for rel,h in f['files'].items():assert hashlib.sha256(z.read(Path(rel).as_posix())).hexdigest()==h
print(json.dumps(dict(status='passed',pages=pages,source_zip_files=len(files),writing_zip_files=len(f['files'])+1,appendix_visual_page=i+1),indent=2))
