                                                                                     
from pathlib import Path
import json, re, hashlib
import pymupdf as fitz

ROOT=Path(__file__).resolve().parents[3]
PACKAGE=ROOT/'IRIS_Submission_Recent_Data'

def main():
    meta=json.loads((PACKAGE/'Review_Notes/THREE_NETWORK_DOCUMENT_CHECKS.json').read_text())
    checks=[]; previews=[]
    for record in meta['documents']:
        path=ROOT/record['path']
        checks.append({'name':path.name+'/hash','passed':hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']})
        with fitz.open(path) as doc:
            checks.append({'name':path.name+'/page_count','passed':len(doc)==record['pages']})
            for i,page in enumerate(doc):
                blocks=page.get_text('dict')['blocks']
                for block in blocks:
                    if block['type']!=0:continue
                    x0,y0,x1,y1=block['bbox']
                    checks.append({'name':path.name+f'/page_{i+1}/text_bounds','passed':x0>=0 and y0>=0 and x1<=page.rect.width+.5 and y1<=page.rect.height+.5})
                text=page.get_text()
                checks.append({'name':path.name+f'/page_{i+1}/text_encoding','passed':'\ufffd' not in text})
            if path.name=='SheafCode_Three_Network_Full_Manuscript.pdf':
                indices=[i for i,p in enumerate(doc) if 'UK matched causal reconstruction' in p.get_text() or 'UK no-feedback intervals' in p.get_text() or 'UK availability-aware synthetic fault' in p.get_text() or 'UK joint bounds' in p.get_text()]
                for i in indices[:6]:previews.append((i,doc[i].get_pixmap(matrix=fitz.Matrix(.65,.65))))
    for folder,names in [(PACKAGE/'Three_Network_Source',['three_network_paper','three_network_full','three_network_synopsis']), (PACKAGE/'Full_Manuscript_Source',['full_manuscript_uk'])]:
        for name in names:
            log=(folder/(name+'.log')).read_text(errors='replace')
            checks.append({'name':name+'/resolved_cross_references','passed':not re.search(r'(?:Citation|Reference).{0,180}undefined|multiply defined|undefined references',log,re.I)})
    if not all(c['passed'] for c in checks):raise RuntimeError(json.dumps([c for c in checks if not c['passed']]))
    contact=fitz.open();page=contact.new_page(width=900,height=650)
    for j,(index,pixmap) in enumerate(previews):
        x=(j%3)*300;y=(j//3)*325
        page.insert_text((x+8,y+12),f'Full manuscript page {index+1}',fontsize=9)
        page.insert_image(fitz.Rect(x+5,y+20,x+295,y+320),pixmap=pixmap,keep_proportion=True)
    page.get_pixmap().save(PACKAGE/'Review_Notes/UK_DOCUMENT_CONTACT.png')
    result={'status':'passed','checks':checks,'check_count':len(checks),'scope':'Document preservation, reference resolution, extracted-text page bounds and encoding; rendered contact sheet for visual review'}
    (PACKAGE/'Review_Notes/UK_FINAL_DOCUMENT_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Final UK document audit passed:',len(checks),'checks')

if __name__=='__main__':main()
