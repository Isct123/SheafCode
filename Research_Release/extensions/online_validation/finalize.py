                                                                                 
import sys,re,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
import pymupdf
from PIL import Image,ImageDraw

def main():
    M=ROOT.parent/'Final_Manuscript/Extended_Validation';checks=[]
    def check(name,ok):
        checks.append({'check':name,'passed':bool(ok)})
        if not ok:raise AssertionError(name)
    previous=json.loads((ROOT.parent/'Final_Manuscript/Complete_Research/audit/FINAL_AUDIT.json').read_text())
    check('earlier_reader_PDF_preserved',sha(ROOT.parent/'Final_Manuscript/SheafCode_Complete_Manuscript.pdf')==previous['manuscript_sha256'])
    audit=json.loads((OUT/'audit.json').read_text());check('new_evidence_audit_passed',audit['status']=='passed')
    for relative,h in audit['sources'].items():check('source_snapshot/'+relative,sha(ROOT/relative)==h)
    for relative,h in audit['outputs'].items():check('evidence_snapshot/'+relative,sha(OUT/relative)==h)
    text='\n'.join(p.read_text() for p in [M/'main.tex',M/'preamble.tex',*sorted((M/'sections').glob('*.tex'))])
    labels=re.findall(r'\\label\{([^}]+)\}',text);check('unique_labels',len(labels)==len(set(labels)))
    for label in re.findall(r'\\(?:ref|eqref|pageref)\{([^}]+)\}',text):check('reference/'+label,label in labels)
    keys=set(re.findall(r'@\w+\s*\{\s*([^,]+),',(M/'references.bib').read_text()))
    for group in re.findall(r'\\cite\w*(?:\[[^]]*\])*\{([^}]+)\}',text):
        for key in group.split(','):check('citation/'+key.strip(),key.strip() in keys)
    for options,path in re.findall(r'\\includegraphics(?:\[([^]]*)\])?\{([^}]+)\}',text):
        p=M/path;check('figure/'+path,p.exists())
        with pymupdf.open(p) as doc:
            specified=re.search(r'page=(\d+)',options);check('figure_page/'+path,specified is None or int(specified[1])<=len(doc))
    preview=OUT/'preview';preview.mkdir(exist_ok=True);counts={}
    for filename in ['main','validation_report']:
        doc=pymupdf.open(M/(filename+'.pdf'));counts[filename]=len(doc)
        log=(M/(filename+'.log')).read_text(errors='replace')
        for phrase in ['undefined references','undefined citations','Missing character','Overfull \\hbox','Overfull \\vbox','! LaTeX Error']:
            check(filename+'/log/'+phrase,phrase.lower() not in log.lower())
        pages=[page.get_text() for page in doc];(OUT/(filename+'_page_text.txt')).write_text('\n'.join(pages),encoding='utf-8')
        for i,page in enumerate(doc):
            check(filename+f'/nonblank/{i+1}',len(pages[i].strip())>25)
            words=page.get_text('words');check(filename+f'/text_bounds/{i+1}',all(w[0]>=-1 and w[1]>=-1 and w[2]<=page.rect.width+1 and w[3]<=page.rect.height+1 for w in words))
        for start in range(0,len(doc),15):
            sheet=Image.new('RGB',(1000,1450),'#cccccc');draw=ImageDraw.Draw(sheet)
            for j in range(start,min(start+15,len(doc))):
                pix=doc[j].get_pixmap(matrix=pymupdf.Matrix(.45,.45),alpha=False);im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples);im.thumbnail((320,260))
                col=(j-start)%3;row=(j-start)//3;sheet.paste(im,(col*333+(320-im.width)//2,row*290+20));draw.text((col*333+8,row*290+2),f'Page {j+1}',fill='black')
            sheet.save(preview/f'{filename}_contact_{start+1}.png')
        for page in ([0,8,12,19,60,len(doc)-1] if filename=='main' else list(range(len(doc)))):
            if page<len(doc):doc[page].get_pixmap(matrix=pymupdf.Matrix(1.2,1.2)).save(str(preview/f'{filename}_page_{page+1}.png'))
    check('identical_integrated_reader_pdf',sha(M/'main.pdf')==sha(ROOT.parent/'Final_Manuscript/SheafCode_Validated_Manuscript.pdf'))
    result={'status':'passed','checks':checks,'total_checks':len(checks),'pages':counts,'figures':len(re.findall(r'\\includegraphics',text)),
            'abstract_words':len((M/'abstract.txt').read_text().split()),'integrated_pdf_sha256':sha(M/'main.pdf')}
    save('DOCUMENT_AUDIT.json',result)
                                                                                   
    files={p for directory in [HERE,OUT,M] for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts
           and p.suffix.lower() in {'.py','.md','.tex','.bib','.pdf','.png','.csv','.json','.npz','.zip','.txt'}
           and p.name!='PACKAGE_AUDIT.json'}
    files.add(ROOT.parent/'Run_Online_Validation.cmd')
    files.add(ROOT.parent/'Final_Manuscript/SheafCode_Validated_Manuscript.pdf')
                                                                               
    files.add(ROOT.parent/'Final_Manuscript/SheafCode_Complete_Research_Package.zip')
    manifest={str(p.relative_to(ROOT.parent)).replace('\\','/'):sha(p) for p in sorted(files)}
    archivepath=ROOT.parent/'Final_Manuscript/SheafCode_Validated_Research_Package.zip'
    with zipfile.ZipFile(archivepath,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in sorted(files):archive.write(p,str(p.relative_to(ROOT.parent)).replace('\\','/'))
        archive.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2))
        archive.writestr('START_HERE.txt','Read Final_Manuscript/SheafCode_Validated_Manuscript.pdf.\nRead Research_Release/online_validation_results/ONLINE_VALIDATION_REPORT.md and DOCUMENT_AUDIT.json.\nAll included stages are already completed. No overnight run is required.\nEarlier audited source/evidence are retained in SheafCode_Complete_Research_Package.zip.\nReproduction requires Python scientific dependencies; the pinned interpreter, compiler cache and vendor runtimes are not bundled.\n')
    with zipfile.ZipFile(archivepath) as archive:
        check('package_CRC',archive.testzip() is None)
        for name,h in manifest.items():check('package_hash/'+name,hashlib.sha256(archive.read(name)).hexdigest()==h)
    save('PACKAGE_AUDIT.json',{'status':'passed','file_count':len(files),'archive':str(archivepath.relative_to(ROOT.parent)),
                             'bytes':archivepath.stat().st_size,'sha256':sha(archivepath),'scope':'All members SHA-256 verified and ZIP CRC passed.'})
    print('Final document and package passed',counts,'file count',len(files),flush=True)

if __name__=='__main__':main()
