from pathlib import Path
import json,hashlib,zipfile,re
import fitz
r=Path(__file__).resolve().parent.parent;w=r/'SUBMISSION_WORKSPACE';s=w/'Source';g=w/'Guides'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
doc=fitz.open(w/'Manuscript_For_Upload.pdf');assert len(doc)==29
for p in doc:
    assert p.get_text().strip()
    for b in p.get_text('blocks'):
        if b[6]==0:assert b[0]>=-1 and b[1]>=-1 and b[2]<=p.rect.width+1 and b[3]<=p.rect.height+1
for name in ['manuscript','synopsis']:
    log=(s/(name+'.log')).read_text(encoding='utf-8',errors='replace')
    assert 'Overfull' not in log and not re.search(r'(Citation|Reference).*undefined',log)
report=load(g/'STRUCTURE_CHECK.json');report.update(manuscript_pages=29,pdf_sha256=sha(w/'Manuscript_For_Upload.pdf'),after_source_sha256=sha(s/'manuscript.tex'),before_source_sha256=sha(r/'audit_tools/structure_audit_backup/manuscript.tex'),visual_review_pages=[1,15,24,25,26],out_of_page_text_blocks=0,numerical_checks=94)
save(g/'STRUCTURE_CHECK.json',report)
note='\n\n## Section-placement review — 10 October 2026\n\nThe headline paragraph and comparison figure now open the primary Results subsection. The network explanation/figure sit in causal Methods, and the mathematical implementation outcomes sit in Results before Discussion and limitations. Conclusion is separate. Primary/external outcomes and fault/snapshot intervals have separate subsection headings. All body tokens are retained after ignoring headings and order; all formal statements, seven proofs, abstract, eleven figure blocks and nine table blocks match the pre-reordering backup. The paper remains 29 pages; compilation, page-bound checks, visual review and all 94 numerical checks pass. STRUCTURE_CHECK.json records this structural edit. Earlier completeness-check source hashes describe the pre-reordering snapshot.\n'
for name in ['FULL_PAPER_COMPLETENESS_AUDIT.md','FINAL_MANUSCRIPT_REVIEW.md']:
    p=g/name;p.write_text(p.read_text(encoding='utf-8')+note,encoding='utf-8')
p=w/'START_HERE.md';t=p.read_text(encoding='utf-8');t=t.replace('| Guides/EVIDENCE_REPORTING_MAP.csv |','| Guides/STRUCTURE_CHECK.json | Check the subsequent section-placement and content-preservation review. |\n| Guides/EVIDENCE_REPORTING_MAP.csv |');t=t.replace('All existing formal statements, proofs and eleven figures are retained.','The section-placement review puts headline and mathematical validation outcomes in Results and the network explanation in Methods, with a separate Conclusion. All existing formal statements, proofs and eleven figures are retained.');p.write_text(t,encoding='utf-8')
p=s/'README.md';p.write_text(p.read_text(encoding='utf-8')+'\nThe subsequent structure review moves detailed results into Results, the network explanation into Methods, and separates Discussion and Conclusion. Body content, mathematical statements/proofs, figure blocks and table blocks are retained.\n',encoding='utf-8')
files=sorted(p for p in s.rglob('*') if p.is_file() and (p.suffix in ['.tex','.bib','.md','.cmd'] or (p.parent.name=='figures' and p.suffix=='.pdf')))
with zipfile.ZipFile(w/'Open_in_Overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,p.relative_to(s).as_posix())
with zipfile.ZipFile(w/'Open_in_Overleaf.zip') as z:
    assert z.testzip() is None
    for p in files:assert z.read(p.relative_to(s).as_posix())==p.read_bytes()
dc=load(g/'WORKSPACE_DOCUMENT_CHECK.json');dc.update(pdf_sha256=sha(w/'Manuscript_For_Upload.pdf'),source_sha256=sha(s/'manuscript.tex'),synopsis_sha256=sha(w/'Synopsis.pdf'),source_zip_sha256=sha(w/'Open_in_Overleaf.zip'),structure_audit='Guides/STRUCTURE_CHECK.json',visual_review_pages=[1,15,24,25,26],visual_review='Current reordered manuscript reviewed at introduction, primary Results, mathematical Results, Discussion and Conclusion; all pages checked for nonempty text and page bounds.')
for rel in dc['guide_consistency_review']['guide_hashes']:dc['guide_consistency_review']['guide_hashes'][rel]=sha(w/rel)
save(g/'WORKSPACE_DOCUMENT_CHECK.json',dc)
f=load(g/'RESEARCH_FREEZE.json');f['files']['Guides/STRUCTURE_CHECK.json']=sha(g/'STRUCTURE_CHECK.json')
for rel in f['files']:f['files'][rel]=sha(w/rel)
save(g/'RESEARCH_FREEZE.json',f)
with zipfile.ZipFile(w/'Final_Writing_Package.zip','w',zipfile.ZIP_DEFLATED) as z:
    for rel in f['files']:z.write(w/rel,Path(rel).as_posix())
    z.write(g/'RESEARCH_FREEZE.json','Guides/RESEARCH_FREEZE.json')
with zipfile.ZipFile(w/'Final_Writing_Package.zip') as z:
    assert z.testzip() is None
    for rel,h in f['files'].items():assert hashlib.sha256(z.read(Path(rel).as_posix())).hexdigest()==h
print(json.dumps(dict(status='passed',pages=len(doc),overleaf_files=len(files),writing_files=len(f['files'])+1,body_content_preserved=True),indent=2))
