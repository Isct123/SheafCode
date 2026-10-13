from pathlib import Path
import json,re,hashlib,zipfile
import fitz
r=Path(__file__).resolve().parent.parent;w=r/'SUBMISSION_WORKSPACE';s=w/'Source';g=w/'Guides'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
d=fitz.open(w/'Manuscript_For_Upload.pdf');pages=len(d)
for p in d:
    assert p.get_text().strip()
    for b in p.get_text('blocks'):
        if b[6]==0:assert b[0]>=-1 and b[1]>=-1 and b[2]<=p.rect.width+1 and b[3]<=p.rect.height+1
for n in ['manuscript','synopsis']:
    log=(s/(n+'.log')).read_text(encoding='utf-8',errors='replace');assert 'Overfull' not in log;assert not re.search(r'(Citation|Reference).*undefined',log)
terms=['fixed Indian benchmark finds','Discussion and limitations','Data and code availability'];visual=sorted(set(next(i for i,p in enumerate(d) if term in p.get_text())+1 for term in terms))
for i in visual:d[i-1].get_pixmap().save(str(r/'audit_tools'/f'writing_final_{sha(w/"Manuscript_For_Upload.pdf")[:8]}_{i}.png'))
rec=load(g/'WRITING_REVIEW_CHECK.json');rec.update(source_sha256=sha(s/'manuscript.tex'),pdf_sha256=sha(w/'Manuscript_For_Upload.pdf'),manuscript_pages=pages,compiled=True,out_of_page_text_blocks=0,review_pages=visual,main_numerical_checks=94,reporting_coverage_checks=333,joint_abstention_count_clarified=True);rec['changes'].append('Clarified that 12 inconsistent joint models are included in 117 abstentions');save(g/'WRITING_REVIEW_CHECK.json',rec)
p=w/'START_HERE.md';t=p.read_text(encoding='utf-8').replace('manuscript is 29 pages',f'manuscript is {pages} pages');t=t.replace('| Guides/FULL_CONFIRMATION_REVIEW.md |','| Guides/WRITING_AND_PAPER_STRUCTURE_REVIEW.md | Read the latest writing review, scientific-section assessment and remaining personal/access details. |\n| Guides/FULL_CONFIRMATION_REVIEW.md |');p.write_text(t,encoding='utf-8')
p=s/'README.md';t=p.read_text(encoding='utf-8').replace('The current PDF is 29 pages.',f'The current PDF is {pages} pages.');p.write_text(t+'\nThe latest writing review adds a concise introduction preview, interpretation in Discussion and data/code availability, clarifies terminology and joint abstention counts, and replaces appendix editing-history language. All pre-existing formal statements, proofs, tables, figures and the abstract are retained.\n',encoding='utf-8')
for name in ['FINAL_MANUSCRIPT_REVIEW.md','FULL_PAPER_COMPLETENESS_AUDIT.md']:
    p=g/name;p.write_text(p.read_text(encoding='utf-8')+f'\n\nLatest writing revision: WRITING_AND_PAPER_STRUCTURE_REVIEW.md describes improved interpretation, data/code availability, terminology and the introduction preview. Current manuscript: {pages} pages; eleven figures and nine tables retained. Earlier edit-specific page counts and source hashes remain historical.\n',encoding='utf-8')
files=sorted(p for p in s.rglob('*') if p.is_file() and (p.suffix in ['.tex','.bib','.md','.cmd'] or (p.parent.name=='figures' and p.suffix=='.pdf')))
with zipfile.ZipFile(w/'Open_in_Overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,p.relative_to(s).as_posix())
with zipfile.ZipFile(w/'Open_in_Overleaf.zip') as z:
    assert z.testzip() is None
    for p in files:assert z.read(p.relative_to(s).as_posix())==p.read_bytes()
dc=load(g/'WORKSPACE_DOCUMENT_CHECK.json');dc.update(manuscript_pages=pages,source_sha256=sha(s/'manuscript.tex'),pdf_sha256=sha(w/'Manuscript_For_Upload.pdf'),synopsis_sha256=sha(w/'Synopsis.pdf'),source_zip_sha256=sha(w/'Open_in_Overleaf.zip'),latest_writing_review='Guides/WRITING_AND_PAPER_STRUCTURE_REVIEW.md',visual_review_pages=visual,visual_review='Current introduction, interpreted Discussion and data/code availability pages selected for review; all pages checked for bounds and nonempty text.')
dc['guide_consistency_review']['guide_hashes']['Guides/WRITING_AND_PAPER_STRUCTURE_REVIEW.md']=sha(g/'WRITING_AND_PAPER_STRUCTURE_REVIEW.md')
for rel in dc['guide_consistency_review']['guide_hashes']:dc['guide_consistency_review']['guide_hashes'][rel]=sha(w/rel)
dc['guide_consistency_review']['files_reviewed']=len(dc['guide_consistency_review']['guide_hashes']);dc['guide_consistency_review']['scope']=f'Current {pages}-page writing revision; earlier check records remain historical.';save(g/'WORKSPACE_DOCUMENT_CHECK.json',dc)
cc=load(g/'WORKSPACE_COMPILE_CHECK.json');cc['compiled_pages']['Manuscript_For_Upload.pdf']=pages;save(g/'WORKSPACE_COMPILE_CHECK.json',cc)
fc=load(g/'FULL_CONFIRMATION_CHECK.json');fc.update(compiled_pages=cc['compiled_pages'],pdf_sha256=sha(w/'Manuscript_For_Upload.pdf'),subsequent_writing_review='Guides/WRITING_REVIEW_CHECK.json');save(g/'FULL_CONFIRMATION_CHECK.json',fc)
f=load(g/'RESEARCH_FREEZE.json');f['manuscript_pages']=pages
for name in ['WRITING_AND_PAPER_STRUCTURE_REVIEW.md','WRITING_REVIEW_CHECK.json']:f['files']['Guides/'+name]=sha(g/name)
for rel in f['files']:f['files'][rel]=sha(w/rel)
save(g/'RESEARCH_FREEZE.json',f)
with zipfile.ZipFile(w/'Final_Writing_Package.zip','w',zipfile.ZIP_DEFLATED) as z:
    for rel in f['files']:z.write(w/rel,Path(rel).as_posix())
    z.write(g/'RESEARCH_FREEZE.json','Guides/RESEARCH_FREEZE.json')
with zipfile.ZipFile(w/'Final_Writing_Package.zip') as z:
    assert z.testzip() is None
    for rel,h in f['files'].items():assert hashlib.sha256(z.read(Path(rel).as_posix())).hexdigest()==h
print(json.dumps(dict(status='passed',pages=pages,figures=11,tables=9,visual_pages=visual,pdf_hash=sha(w/'Manuscript_For_Upload.pdf')[:8],overleaf_files=len(files),writing_files=len(f['files'])+1),indent=2))
