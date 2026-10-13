from pathlib import Path
import json, re, hashlib, shutil
import pymupdf as fitz
from PIL import Image, ImageOps, ImageDraw

BASE=Path(__file__).resolve().parents[1];OUT=BASE/'Final_Manuscript'
srcs=list(OUT.glob('*.tex'))+list((OUT/'sections').glob('*.tex'))
source='\n'.join(p.read_text() for p in srcs)
keys=set(re.findall(r'@\w+\s*\{\s*([^,]+)',(OUT/'references.bib').read_text()))
citations={x.strip() for block in re.findall(r'\\cite\w*(?:\[[^]]*\])*\{([^}]+)\}',source) for x in block.split(',')}
labels=re.findall(r'\\label\{([^}]+)\}',source)
refs={x.strip() for block in re.findall(r'\\(?:c|C|eq)?ref\{([^}]+)\}',source) for x in block.split(',')}
check=dict(missing_citation_keys=sorted(citations-keys),missing_reference_labels=sorted(refs-set(labels)),duplicate_labels=sorted({x for x in labels if labels.count(x)>1}),iteration_labels=re.findall(r'\bv[1-9]\b|\bversion\s+\d+|\biteration\s+\d+',source,re.I),missing_includes=[])
for inc in re.findall(r'\\input\{([^}]+)\}',source):
    p=OUT/inc
    if not p.suffix:p=p.with_suffix('.tex')
    if not p.exists():check['missing_includes'].append(inc)
for inc in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',source):
    if not (OUT/inc).exists():check['missing_includes'].append(inc)
for s in ['snapshot','channel','prolonged','certificates','thinning','horizons','secondary','risk_association']:
    assert (OUT/f'figures/{s}.pdf').exists() and (OUT/f'figures/{s}.png').exists(),s
audit=json.loads((OUT/'audit/numerical_audit.json').read_text())
assert not audit['discrepancies']
assert all(h['matched'] for h in audit['input_hashes'])
assert json.loads((OUT/'audit/reconstruction_replay.json').read_text())['all_replays_pass']
render=BASE/'audit_tools/render_final';render.mkdir(exist_ok=True)
check['documents']={}
for name in ['main','synopsis','portal_fields','video_script']:
    doc=fitz.open(OUT/f'{name}.pdf');contents='\n'.join(p.get_text() for p in doc)
    check['documents'][name]=dict(pages=len(doc),unresolved_markers='??' in contents,iteration_labels=re.findall(r'\bv[1-9]\b|\bversion\s+\d+',contents,re.I),out_of_page_text=[])
    thumbs=[]
    for i,p in enumerate(doc):
        image=Image.frombytes('RGB',[p.get_pixmap(matrix=fitz.Matrix(.6,.6)).width,p.get_pixmap(matrix=fitz.Matrix(.6,.6)).height],p.get_pixmap(matrix=fitz.Matrix(.6,.6)).samples)
        image=ImageOps.expand(image,border=(5,22,5,5),fill='white');ImageDraw.Draw(image).text((6,4),f'{name} / page {i+1}',fill='black');thumbs.append(image)
        for b in p.get_text('dict')['blocks']:
            for line in b.get('lines',[]):
                for span in line.get('spans',[]):
                    text=span['text'];x0,y0,x1,y1=span['bbox']
                    if 'REFERENCE EXEMPLAR' in text:continue
                    if x0<-1 or y0<-1 or x1>p.rect.width+1 or y1>p.rect.height+1:check['documents'][name]['out_of_page_text'].append(dict(page=i+1,text=text))
        p.get_pixmap(matrix=fitz.Matrix(1.2,1.2)).save(render/f'{name}_{i+1:02}.png')
    for offset in range(0,len(thumbs),12):
        group=thumbs[offset:offset+12];w=max(t.width for t in group);h=max(t.height for t in group)
        sheet=Image.new('RGB',(w*3,h*((len(group)+2)//3)), '#d8dadd')
        for j,t in enumerate(group):sheet.paste(t,((j%3)*w,(j//3)*h))
        sheet.save(render/f'{name}_contact_{offset//12+1}.png')
    log=(OUT/f'{name}.log').read_text(errors='replace')
    check['documents'][name]['overfull_boxes']=re.findall(r'Overfull[^\n]+',log)
    check['documents'][name]['undefined_reference_warnings']=re.findall(r'[^\n]*(?:undefined|Undefined)[^\n]*',log)
    shutil.copy2(OUT/f'{name}.pdf',OUT/f'preview/{name}.pdf')
assert all(not check[k] for k in ['missing_citation_keys','missing_reference_labels','duplicate_labels','iteration_labels','missing_includes']),check
assert all(not d['unresolved_markers'] and not d['iteration_labels'] and not d['out_of_page_text'] and not d['overfull_boxes'] and not d['undefined_reference_warnings'] for d in check['documents'].values()),check
(OUT/'audit/document_audit.json').write_text(json.dumps(check,indent=2),encoding='utf8')
print(json.dumps(check,indent=2))
