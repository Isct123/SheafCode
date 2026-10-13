                                                                              
import csv, hashlib, json, re, zipfile
from pathlib import Path
import pymupdf as fitz
from PIL import Image, ImageDraw

B = Path(__file__).resolve().parents[1]
O = B / 'Final_Manuscript/Complete_Research'
A = O / 'audit'
checks = []

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def check(name, ok, detail=''):
    checks.append(dict(check=name, passed=bool(ok), detail=detail))

def js(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def run():
    earlier = js(A/'RECHECK.json')
    check('independent_saved_result_recheck', earlier['status']=='passed',
          f"{earlier['total_checks']} checks; {earlier['audited_files']} hashed files")
    for p, h in js(A/'BUILD_INPUTS.json').items():
        path = B/p
        check('unchanged_build_source/'+p, path.exists() and sha(path)==h)
    texfiles = [O/'main.tex', O/'preamble.tex', *sorted((O/'sections').glob('*.tex'))]
    tex = '\n'.join(p.read_text(encoding='utf-8') for p in texfiles)
    bib = (O/'references.bib').read_text(encoding='utf-8')
    keys = set(re.findall(r'@\w+\s*\{\s*([^,]+),', bib))
    cited = set()
    for group in re.findall(r'\\cite\w*\s*(?:\[[^]]*\])*\{([^}]+)\}', tex):
        cited.update(k.strip() for k in group.split(','))
    for key in sorted(cited):
        check('bibliography/'+key, key in keys)
    labels = re.findall(r'\\label\{([^}]+)\}', tex)
    check('unique_equation_table_figure_labels', len(labels)==len(set(labels)))
    for key in re.findall(r'\\(?:eqref|ref|pageref)\{([^}]+)\}', tex):
        check('resolved_reference/'+key, key in labels)
    includes = re.findall(r'\\includegraphics(?:\[([^]]*)\])?\{([^}]+)\}', tex)
    for options, name in includes:
        p = O/name
        check('figure_exists/'+name, p.is_file())
        if p.is_file():
            d = fitz.open(p)
            page = re.search(r'page=(\d+)', options)
            check('figure_page/'+name, d.page_count > 0 and (page is None or int(page[1])<=d.page_count))
            d.close()
    figures = js(A/'FIGURES.json')
    for fig in figures:
        source = B/fig['source']
        copied = O/'figures'/(source.name if fig['label'].startswith('supplied-') else fig['label'].replace(':','_')+'.pdf')
        check('figure_hash/'+fig['label'], sha(source)==fig['copied_sha256'] and copied.exists() and sha(copied)==sha(source))
    check('figure_coverage', len(figures)==len(includes), f'{len(figures)} figure placements')
    coverage = list(csv.DictReader((A/'EVIDENCE_COVERAGE.csv').open(encoding='utf-8')))
    for row in coverage:
        check('evidence_family/'+row['family'], (B/row['source']).exists(), row['manuscript_location'])
    check('complete_evidence_index', len(coverage)==36)
    log = (O/'main.log').read_text(encoding='utf-8',errors='replace')
    for pattern in ['undefined references','undefined citations','Missing character','Overfull \\hbox','Overfull \\vbox','! LaTeX Error']:
        check('TeX_log/'+pattern, pattern.lower() not in log.lower())
    pdf = fitz.open(O/'main.pdf')
    pages = [page.get_text() for page in pdf]
    text = '\n'.join(pages)
    (A/'page_text.txt').write_text(text,encoding='utf-8')
    check('abstract_word_count', len((O/'abstract.txt').read_text().split())==213)
    check('full_numbered_proofs', len(re.findall(r'\\begin\{proposition\}',tex))==15)
    prose_without_urls=re.sub(r'https?://\S+','',text)
    check('no_iteration_labels', re.search(r'\bv[1-9]\d*\b',prose_without_urls,re.I) is None)
    check('no_replacement_glyphs', '\ufffd' not in text)
    for i,(page,page_text) in enumerate(zip(pdf,pages),1):
        check(f'nonblank_page/{i}',len(page_text.strip())>30)
        bad = [word for word in page.get_text('words') if word[0]<-1 or word[1]<-1 or word[2]>page.rect.width+1 or word[3]>page.rect.height+1]
        check(f'no_page_boundary_text_clipping/{i}',not bad,str(bad[:2]) if bad else '')
    for claim in js(A/'CLAIMS.json'):
        printed = f"{claim['value']:.1f}"
        check('headline_display/'+claim['key'],printed in text,printed)
    tables=js(A/'TABLES.json')
                                                                               
                                                                     
    for table in tables:
        check('main_table/'+table['label'],table['label'] in labels)
        for row in table['rows']:
            for value in row:
                value=str(value)
                if re.fullmatch(r'-?\d+(?:\.\d+)?',value):
                    check('table_numeric_display/'+table['label']+'/'+value,value in text)
    for p in sorted((O/'data').glob('*.csv')):
        check('supporting_csv/'+p.name,p.stat().st_size>20)
    check('identical_reader_pdf',sha(O/'main.pdf')==sha(B/'Final_Manuscript/SheafCode_Complete_Manuscript.pdf'))
                                                                                
    preview=A/'preview';preview.mkdir(exist_ok=True)
    for start in range(0,len(pdf),15):
        sheet=Image.new('RGB',(1000,1450),'#cccccc');draw=ImageDraw.Draw(sheet)
        for j in range(start,min(start+15,len(pdf))):
            pix=pdf[j].get_pixmap(matrix=fitz.Matrix(.43,.43),alpha=False)
            im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
            im.thumbnail((320,260))
            col=(j-start)%3;row=(j-start)//3
            sheet.paste(im,(col*333+(320-im.width)//2,row*290+20))
            draw.text((col*333+7,row*290+3),f'Page {j+1}',fill='black')
        sheet.save(preview/f'contact_{start+1}.png')
    for page in [0,8,13,23,48,len(pdf)-1]:
        if page<len(pdf):pdf[page].get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(str(preview/f'page_{page+1}.png'))
    check('all_pages_render',len(pdf)>40)
    summary=dict(status='passed' if all(c['passed'] for c in checks) else 'failed',
        date='2026-10-06',checks=checks,total_checks=len(checks),pages=len(pdf),
        figure_placements=len(figures),numbered_propositions=15,evidence_families=len(coverage),
        abstract_words=213,manuscript_sha256=sha(O/'main.pdf'),
        verification_scope='Internal proof/assumption review; rerun tests; independent saved-result/source audit; compiled-document checks. Not independent peer review or every historical training replay.')
    (A/'FINAL_AUDIT.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='checks'}))
    if summary['status']!='passed':
        print(json.dumps([c for c in checks if not c['passed']],indent=2))
        raise SystemExit(1)

if __name__=='__main__':run()
