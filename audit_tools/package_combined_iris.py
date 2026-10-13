                                                                                
from pathlib import Path
import hashlib, json, re, shutil, zipfile
import pymupdf

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'SUBMISSION_WORKSPACE/Combined_IRIS'
SRC=OUT/'Source'
main=(SRC/'manuscript.tex').read_text(encoding='utf-8')
supp=(SRC/'technical_reference.tex').read_text(encoding='utf-8')
baseline=(OUT/'Inputs/SheafCode_IRIS_Submission_Corrected_Final.tex').read_text(encoding='utf-8')
canonical=(ROOT/'SUBMISSION_WORKSPACE/Source/manuscript.tex').read_text(encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks={}

def demand(name,value):
    checks[name]=bool(value)
    if not value:raise AssertionError(name)

def numeric_literals(s):
                                                                                 
    return set(re.findall(r'(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?![\w.])',s))

base_numbers=numeric_literals(baseline)
demand('baseline_numeric_values_retained_in_main',not(base_numbers-numeric_literals(main)-{'15'}))
demand('baseline_numeric_values_retained_in_package',not(base_numbers-numeric_literals(main+supp)))
formal=lambda t:[re.sub(r'\s+',' ',b).strip() for b in re.findall(r'\\begin\{(?:proposition|theorem)\}.*?\\end\{(?:proposition|theorem)\}',t,re.S)]
demand('six_audited_formal_statements_retained_in_reference',formal(canonical)==formal(supp))
for name in ['manuscript','technical_reference']:
    s=(SRC/(name+'.tex')).read_text(encoding='utf-8')
    bib=(SRC/'references.bib').read_text(encoding='utf-8')
    keys=set(re.findall(r'@\w+\s*\{\s*([^,\s]+)',bib))
    used={k.strip() for group in re.findall(r'\\cite\w*(?:\[[^\]]*\])*\{([^}]+)\}',s) for k in group.split(',')}
    demand(name+'_citation_keys_resolve',used<=keys)
    labels=re.findall(r'\\label\{([^}]+)\}',s)
    references=set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',s))
    demand(name+'_references_resolve',references<=set(labels))
    demand(name+'_labels_unique',len(labels)==len(set(labels)))
    assets=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',s)
    demand(name+'_figure_assets_exist',all((SRC/p).exists() for p in assets))
    log=(SRC/(name+'.log')).read_text(encoding='utf-8',errors='replace')
    demand(name+'_no_overfull_boxes','Overfull' not in log)
    demand(name+'_no_undefined_references',not re.search(r'(Citation|Reference).*undefined',log))
    d=pymupdf.open(SRC/(name+'.pdf'))
    demand(name+'_all_pages_have_text',all(p.get_text().strip() for p in d))
    demand(name+'_text_stays_on_page',all(b[0]>=-1 and b[1]>=-1 and b[2]<=p.rect.width+1 and b[3]<=p.rect.height+1 for p in d for b in p.get_text('blocks') if b[6]==0))
abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',main,re.S).group(1)
demand('abstract_under_300_words',len(abstract.split())<=300)
for token in ['P_{UO}',r'\lambda>0',r'\|e\|_\infty\leq\beta',r'2f+1','outward rounding','initializ','exterior intervals','17.45','including those 12','12.62--20.16','23.2','34.0','5.93']:
    demand('retained_'+token,token in main)
demand('no_internal_page_target','15-page target' not in main)
saved=json.loads((OUT/'INPUT_MANIFEST.json').read_text(encoding='utf-8'))
demand('canonical_manuscript_unchanged',saved['canonical_source_sha256']==sha(ROOT/'SUBMISSION_WORKSPACE/Source/manuscript.tex'))
for rel,h in saved.items():
    if rel.startswith('Inputs/'):demand('input_preserved_'+rel,sha(OUT/rel)==h)

shutil.copy2(SRC/'manuscript.pdf',OUT/'SheafCode_Combined_IRIS.pdf')
shutil.copy2(SRC/'technical_reference.pdf',OUT/'technical_reference.pdf')
main_pages=len(pymupdf.open(SRC/'manuscript.pdf'))
supp_pages=len(pymupdf.open(SRC/'technical_reference.pdf'))
reference_page=next(i+1 for i,p in enumerate(pymupdf.open(SRC/'manuscript.pdf')) if '\nReferences\n' in p.get_text())
report={
    'status':'passed','scope':'Editorial combination and document checks; no new fitting, experiments, peer review or venue approval.',
    'checks':checks,'check_count':len(checks),'main_pages_including_references':main_pages,
    'references_begin_on_page':reference_page,'supplement_pages':supp_pages,'abstract_words':len(abstract.split()),
    'main_propositions':main.count(r'\begin{proposition}'),'main_theorems':main.count(r'\begin{theorem}'),
    'main_figures':main.count(r'\begin{figure}'),'main_tables':main.count(r'\begin{table}'),
    'baseline_numeric_values':sorted(base_numbers),
    'main_source_sha256':sha(SRC/'manuscript.tex'),'main_pdf_sha256':sha(SRC/'manuscript.pdf'),
    'reference_source_sha256':sha(SRC/'technical_reference.tex'),'reference_pdf_sha256':sha(SRC/'technical_reference.pdf'),
    'visual_review':'Reviewed first page, method/normal equations, stability theorem, external outcomes and operational figure; checked all pages for text bounds.',
    'limits':['Underlying experiments not rerun for this editorial task.','The main paper condenses supporting protocols; complete case files remain in the research archive.','Some harmless underfull-box warnings remain, including bibliography URLs.','Author field remains as supplied; acknowledgements and assistance declarations require actual researcher accuracy.'],
}
(OUT/'EDITORIAL_CHECK.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')

readme=f'''# Combined IRIS manuscript

The main paper combines the clearer corrected draft with the PDF's useful mathematical framing, while correcting its inconsistent temporal-map description. It explains why each question leads to the next and uses concise student-readable prose without inventing a research history.

- SheafCode_Combined_IRIS.pdf: main paper, {main_pages} pages including references (references begin on page {reference_page}).
- Source/manuscript.tex: editable main source.
- technical_reference.pdf: companion longer reference, {supp_pages} pages.
- Source/technical_reference.tex: editable companion source.
- Open_in_Overleaf.zip: both sources, bibliography and every required figure; set manuscript.tex as the main document.
- EDITORIAL_CHECK.json: document checks and their scope.
- EDITORIAL_REVIEW.md: what was combined, corrected and retained.

The original Downloads files and canonical workspace manuscript are preserved. Inputs/ records the supplied source and comparison PDF locally. Original result files and models were not changed or rerun. The package is a document handoff, not the entire research-data archive.

The main paper retains all numeric values from the corrected draft, except its removed internal 15-page target. Supporting case-level files are in the project's Research_Release/ archive. Supplementary submission availability must be checked in the portal. Upload a compiled PDF rather than only a .tex file. The author field remains blank as in the supplied source.
'''
(OUT/'README.md').write_text(readme,encoding='utf-8')
(SRC/'README.md').write_text('Set manuscript.tex as the main Overleaf document. Compile technical_reference.tex separately for the companion reference. Both use references.bib and figures/.\n',encoding='utf-8')

source_files=[SRC/'manuscript.tex',SRC/'technical_reference.tex',SRC/'references.bib',SRC/'README.md']+sorted((SRC/'figures').glob('*.pdf'))
with zipfile.ZipFile(OUT/'Open_in_Overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in source_files:z.write(p,p.relative_to(SRC).as_posix())
with zipfile.ZipFile(OUT/'Combined_IRIS_Package.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in source_files:z.write(p,'Source/'+p.relative_to(SRC).as_posix())
    for name in ['SheafCode_Combined_IRIS.pdf','technical_reference.pdf','README.md','EDITORIAL_CHECK.json','EDITORIAL_REVIEW.md']:
        z.write(OUT/name,name)
for name in ['Open_in_Overleaf.zip','Combined_IRIS_Package.zip']:
    with zipfile.ZipFile(OUT/name) as z:demand(name+'_integrity',z.testzip() is None)
print(json.dumps({k:report[k] for k in ['status','check_count','main_pages_including_references','supplement_pages','abstract_words']},indent=2))
