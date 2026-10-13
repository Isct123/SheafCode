from pathlib import Path
import re,json,hashlib
from collections import Counter
r=Path(__file__).resolve().parent.parent; w=r/'SUBMISSION_WORKSPACE'; p=w/'Source/manuscript.tex'
before=p.read_text(encoding='utf-8'); backup=r/'audit_tools/structure_audit_backup';backup.mkdir(exist_ok=True)
assert not (backup/'manuscript.tex').exists()
(backup/'manuscript.tex').write_text(before,encoding='utf-8')
(backup/'Manuscript_For_Upload.pdf').write_bytes((w/'Manuscript_For_Upload.pdf').read_bytes())
def take(text,start,end):
    a=text.index(start); b=text.index(end,a); return text[:a]+text[b:],text[a:b]
after,headline=take(before,r'\textbf{Main empirical finding.}',r'A \emph{sheaf}')
after=after.replace(r'\subsection{Primary result and all external reconstruction outcomes}',r'\subsection{Primary reconstruction result}'+'\n'+headline,1)
after,network=take(after,r'\textbf{Reading the network.}',r'\section{Related work')
after=after.replace(r'\subsection{Causal reconstruction and fair comparisons}',r'\subsection{Causal reconstruction and fair comparisons}'+'\n'+network,1)
after,math=take(after,r'\subsection{Do the supporting proofs survive computational implementation?}','Additional experiments include partial channels and snapshots,')
after=after.replace(r'\section{Reliability, limitations and conclusion}',math+r'\section{Discussion and limitations}',1)
after=after.replace('The strongest designated result remains the fixed Indian causal comparison.',r'\section{Conclusion}'+'\n'+'The strongest designated result remains the fixed Indian causal comparison.',1)
after=after.replace('Causal sheaf improves numerically over VAR in 13/20 external cohorts',r'\subsection{External reconstruction outcomes}'+'\n'+'Causal sheaf improves numerically over VAR in 13/20 external cohorts',1)
after=after.replace('No-feedback intervals use a separate calibration block',r'\subsection{Snapshot interval coverage and extreme-event failures}'+'\n'+'No-feedback intervals use a separate calibration block',1)
def content(t):
    t=re.sub(r'\\(?:subsection|section)\*?\{[^}]*\}','',t)
    return Counter(re.findall(r'\S+',t))
assert content(before)==content(after),'Body tokens changed beyond headings/reordering'
for env in ['proposition','theorem','abstract']:
    pattern=r'\\begin\{'+env+r'\}.*?\\end\{'+env+r'\}'
    assert re.findall(pattern,before,re.S)==re.findall(pattern,after,re.S)
assert re.findall(r'\\textbf\{Proof\.\}.*?\\textit\{QED\.\}',before,re.S)==re.findall(r'\\textbf\{Proof\.\}.*?\\textit\{QED\.\}',after,re.S)
for env in ['figure','table']:
    pattern=r'\\begin\{'+env+r'\}.*?\\end\{'+env+r'\}'
    assert sorted(re.findall(pattern,before,re.S))==sorted(re.findall(pattern,after,re.S))
assert after.index(r'\section{Results')<after.index(r'\label{fig:headline}')<after.index(r'\section{Discussion')
assert after.index(r'\section{Results')<after.index(r'\subsection{Do the supporting proofs')<after.index(r'\section{Discussion')
p.write_text(after,encoding='utf-8')
report=dict(status='passed',date='2026-10-10',scope='Section placement, content preservation and handoff; no new scientific claims or experiments.',before_source_sha256=hashlib.sha256(before.encode()).hexdigest(),after_source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),body_tokens_preserved_except_headings=True,all_formal_statements_proofs_abstract_preserved=True,all_11_figure_blocks_and_9_table_blocks_preserved=True,changes=['Moved headline paragraph and comparison figure from introduction to primary Results','Moved network explanation and figure into causal Methods','Moved mathematical implementation evaluation from limitations to Results','Separated Discussion/limitations and Conclusion','Separated primary/external outcomes and fault/snapshot interval subsections'])
(w/'Guides/STRUCTURE_CHECK.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
