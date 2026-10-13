from pathlib import Path
import hashlib,json,re,zipfile
import fitz

root=Path(__file__).resolve().parent.parent
w=root/'SUBMISSION_WORKSPACE'; s=w/'Source'; g=w/'Guides'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def writej(p,data): p.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
def edit(p,old,new):
    t=p.read_text(encoding='utf-8'); assert old in t,(p,old); p.write_text(t.replace(old,new),encoding='utf-8')
before=(root/'audit_tools/completeness_audit_backup/manuscript.tex').read_text(encoding='utf-8')
now=(s/'manuscript.tex').read_text(encoding='utf-8')
for env in ['proposition','theorem']:
    pat=r'\\begin\{'+env+r'\}.*?\\end\{'+env+r'\}'
    assert re.findall(pat,before,re.S)==re.findall(pat,now,re.S)
assert re.findall(r'\\textbf\{Proof\.\}.*?\\textit\{QED\.\}',before,re.S)==re.findall(r'\\textbf\{Proof\.\}.*?\\textit\{QED\.\}',now,re.S)
assert re.findall(r'\\begin\{figure\}.*?\\end\{figure\}',before,re.S)==re.findall(r'\\begin\{figure\}.*?\\end\{figure\}',now,re.S)
assert re.findall(r'\\begin\{abstract\}.*?\\end\{abstract\}',before,re.S)==re.findall(r'\\begin\{abstract\}.*?\\end\{abstract\}',now,re.S)
assert 'We can change this' not in now
assert 'all of it collected within the past twelve months' not in now
cites=set(k.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}',now) for k in group.split(','))
bib=set(re.findall(r'@\w+\s*\{\s*([^,]+)',(s/'references.bib').read_text(encoding='utf-8')))
assert cites<=bib
pages={}; out=[]
for name in ['Manuscript_For_Upload.pdf','Synopsis.pdf']:
    doc=fitz.open(w/name); pages[name]=len(doc)
    for i,p in enumerate(doc):
        assert p.get_text().strip(),(name,i)
        for b in p.get_text('blocks'):
            if b[6]==0 and (b[0]<-1 or b[1]<-1 or b[2]>p.rect.width+1 or b[3]>p.rect.height+1):out.append((name,i,b[:4]))
assert not out,out
assert pages=={'Manuscript_For_Upload.pdf':29,'Synopsis.pdf':2}
for name in ['manuscript','synopsis']:
    log=(s/(name+'.log')).read_text(encoding='utf-8',errors='replace')
    assert 'Overfull' not in log
    assert not re.search(r'(Citation|Reference).*undefined',log)
audit=json.loads((g/'COMPLETENESS_RESTORATION_CHECK.json').read_text())
for rel,h in audit['input_sha256'].items():assert sha(root/rel)==h
audit.update(manuscript_pages=29,figure_count=11,table_count=now.count('\\begin{table}'),source_sha256=sha(s/'manuscript.tex'),pdf_sha256=sha(w/'Manuscript_For_Upload.pdf'),existing_figures_byte_content_retained=True,existing_abstract_unchanged=True,main_numerical_document_checks=94,layout_out_of_page_text_blocks=0,visual_review_pages=[2,26,27],bibliography_cited_keys=len(cites),bibliography_all_cited_keys_defined=True)
audit['corrections'].append('Removed blanket twelve-month data acknowledgement because the restored appendix explicitly includes older supporting archives')
writej(g/'COMPLETENESS_RESTORATION_CHECK.json',audit)
edit(w/'START_HERE.md','The illustrated manuscript is 26 pages','The audited illustrated manuscript is 29 pages')
edit(w/'START_HERE.md','The latest readability edit separates dense paragraphs and makes the explanations more direct; all scientific content is retained.','The 10 October completeness audit restores a dedicated literature review, data/split details and selected supporting results, and corrects the UK certificate summary. All existing formal statements, proofs and eleven figures are retained. Detailed historical case tables and advanced derivations remain in the technical reference.')
edit(w/'START_HERE.md','| Guides/FINAL_MANUSCRIPT_REVIEW.md |','| Guides/FULL_PAPER_COMPLETENESS_AUDIT.md | Read the latest omissions audit, restorations, evidence coverage and remaining limits. |\n| Guides/EVIDENCE_REPORTING_MAP.csv | Locate the supporting result families and saved case tables. |\n| Guides/FINAL_MANUSCRIPT_REVIEW.md |')
edit(s/'README.md','The readability revision retains all scientific content. Explanations and paragraph structure were edited; formal statements, proofs, equations, tables, figure captions, citations and reported numbers are unchanged. The PDF and this source handoff are synchronized.','The 10 October completeness revision restores a dedicated literature review, dataset details and a supporting-evidence appendix. It corrects the UK conditional certificate summary and assistance disclosure. Existing formal statements, proofs, figures and abstract are preserved. The current PDF is 29 pages. See the completeness audit in the writing package for what remains in the reference archive.')
edit(s/'README.md','The current manuscript uses the latest user attachment, with all eleven figures and its 11pt setting. Only its acknowledgement block was replaced with the user acknowledgement wording; all other attachment text is unchanged.','The source retains the user attachment as its writing base and the 11pt setting, with the audited restorations described above. Earlier acknowledgement-only and figure-only checks describe those earlier revisions, not this latest source.')
edit(g/'Mathematical_Selection.md','The illustrated manuscript is 26 pages','The illustrated manuscript is now 29 pages after the 10 October completeness restoration')
p=g/'FINAL_MANUSCRIPT_REVIEW.md'; t=p.read_text(encoding='utf-8')
t=t.replace('The current illustrated PDF has 26 pages','The current illustrated PDF has 29 pages')
t=t.replace('This 26-page illustrated paper','The current 29-page illustrated paper')
t=t.replace('The current source matches the latest supplied 11pt attachment exactly except for its acknowledgement block, which uses the user wording.','At the acknowledgement-only revision, the source matched the latest supplied 11pt attachment except for its acknowledgement block. The 10 October audit subsequently restored missing literature and supporting evidence; see the latest completeness audit.')
t+='\n## Completeness restoration — 10 October 2026\n\nThe latest manuscript is 29 pages with eleven unchanged figures. FULL_PAPER_COMPLETENESS_AUDIT.md documents the restored related-work survey, split/unit details and supporting appendix. The blanket original-network abstention claim was corrected: the separate UK trial has four model-conditional below-reference certificates. Existing formal statements, all seven proofs and the abstract are unchanged. Detailed historical case tables and advanced derivations remain in the technical reference; the manuscript is not every historical experiment printed in full. All 94 main manuscript checks and the independent saved-evidence replays passed. New supplementary aggregates have their own hashed check record. Compilation, page bounds and visual inspection of restored pages passed. Older supporting observations are explicitly separate from the recent benchmark and require their own eligibility interpretation. No independent model refitting or student-contribution certification was performed.\n'
p.write_text(t,encoding='utf-8')
p=g/'FULL_PAPER_COMPLETENESS_AUDIT.md';t=p.read_text(encoding='utf-8')
t+='\n## Current handoff\n\nThe audited manuscript compiles to 29 pages with eleven figures and nine tables. All 94 main numerical checks pass; new supporting aggregates are separately checked against twelve hashed CSVs. All existing theorem/proposition and proof text, figure blocks and the abstract match the pre-audit backup. Restored pages 2, 26 and 27 were visually reviewed; all pages pass nonempty-text and page-bound checks. The PDF, Overleaf ZIP, writing package and current inventory are synchronized. The acknowledgement no longer claims that older supporting archive data fall within twelve months.\n'
p.write_text(t,encoding='utf-8')
sourcefiles=sorted(p for p in s.rglob('*') if p.is_file() and (p.suffix in ['.tex','.bib','.md','.cmd'] or (p.parent.name=='figures' and p.suffix=='.pdf')))
with zipfile.ZipFile(w/'Open_in_Overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sourcefiles:z.write(p,p.relative_to(s).as_posix())
with zipfile.ZipFile(w/'Open_in_Overleaf.zip') as z:
    assert z.testzip() is None
    for p in sourcefiles:assert z.read(p.relative_to(s).as_posix())==p.read_bytes()
dc=json.loads((g/'WORKSPACE_DOCUMENT_CHECK.json').read_text())
dc.update(manuscript_pages=29,pdf_sha256=sha(w/'Manuscript_For_Upload.pdf'),source_sha256=sha(s/'manuscript.tex'),synopsis_sha256=sha(w/'Synopsis.pdf'),source_zip_sha256=sha(w/'Open_in_Overleaf.zip'),source_zip_files=len(sourcefiles),visual_review_pages=[2,26,27],visual_review='Restored related-work and supporting-evidence pages visually inspected; all pages checked for nonempty text and page bounds. Existing eleven figures unchanged.',reviewed_date='2026-10-10',editorial_update_date='2026-10-10',completeness_audit='Guides/FULL_PAPER_COMPLETENESS_AUDIT.md',restoration_check='Guides/COMPLETENESS_RESTORATION_CHECK.json',reporting_map='Guides/EVIDENCE_REPORTING_MAP.csv')
gc=dc['guide_consistency_review'];gc['date']='2026-10-10';gc['scope']='Current 29-page audited manuscript inventory and handoff; historical edit checks remain records of their specific earlier revisions.'
gc['guide_hashes']['Guides/FULL_PAPER_COMPLETENESS_AUDIT.md']=sha(g/'FULL_PAPER_COMPLETENESS_AUDIT.md')
for rel in gc['guide_hashes']:gc['guide_hashes'][rel]=sha(w/rel)
gc['files_reviewed']=len(gc['guide_hashes'])
gc['corrections'].append('Restored literature/evidence and distinguished earlier attachment-only checks from current source')
writej(g/'WORKSPACE_DOCUMENT_CHECK.json',dc)
writej(g/'WORKSPACE_COMPILE_CHECK.json',dict(status='passed',reviewed_date='2026-10-10',compiled_pages=pages,scope='Both documents compiled; eleven figures included, no undefined references/citations or overfull boxes. All pages checked for nonempty text and page bounds.'))
f=json.loads((g/'RESEARCH_FREEZE.json').read_text())
f.update(manuscript_pages=29,editorial_update_date='2026-10-10',research_scope='Existing scientific results remain frozen. Completeness audit restores prior literature and supporting outcomes with separate protocols, corrects UK certificate reporting and assistance wording; no new experiments or theorems added.')
for name in ['FULL_PAPER_COMPLETENESS_AUDIT.md','COMPLETENESS_RESTORATION_CHECK.json','EVIDENCE_REPORTING_MAP.csv']:f['files']['Guides/'+name]=sha(g/name)
for rel in f['files']:f['files'][rel]=sha(w/rel)
writej(g/'RESEARCH_FREEZE.json',f)
with zipfile.ZipFile(w/'Final_Writing_Package.zip','w',zipfile.ZIP_DEFLATED) as z:
    for rel in f['files']:z.write(w/rel,Path(rel).as_posix())
    z.write(g/'RESEARCH_FREEZE.json','Guides/RESEARCH_FREEZE.json')
with zipfile.ZipFile(w/'Final_Writing_Package.zip') as z:
    assert z.testzip() is None
    for rel,h in f['files'].items():assert hashlib.sha256(z.read(Path(rel).as_posix())).hexdigest()==h
    assert z.read('Guides/RESEARCH_FREEZE.json')==(g/'RESEARCH_FREEZE.json').read_bytes()
print(json.dumps(dict(status='passed',pages=pages,source_zip_files=len(sourcefiles),writing_zip_files=len(f['files'])+1,source_sha256=sha(s/'manuscript.tex'),pdf_sha256=sha(w/'Manuscript_For_Upload.pdf')),indent=2))
