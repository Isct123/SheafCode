from pathlib import Path
import json
import pymupdf as fitz
from PIL import Image, ImageDraw

root=Path(__file__).resolve().parent/'report_recheck'
records=json.loads((root/'inventory.json').read_text())
seen=set()
for i,record in enumerate(records):
    if record['sha256'] in seen:continue
    seen.add(record['sha256'])
    doc=fitz.open(record['file']);thumbs=[]
    for j,page in enumerate(doc):
        pix=page.get_pixmap(matrix=fitz.Matrix(.65,.65))
        im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
        thumb=Image.new('RGB',(im.width,im.height+20),'white');thumb.paste(im,(0,20))
        ImageDraw.Draw(thumb).text((5,3),str(j+1),fill='black');thumbs.append(thumb)
    w=max(t.width for t in thumbs);h=max(t.height for t in thumbs)
    sheet=Image.new('RGB',(w*3,h*((len(thumbs)+2)//3)),'#ddd')
    for j,thumb in enumerate(thumbs):sheet.paste(thumb,((j%3)*w,(j//3)*h))
    sheet.save(root/f'contact_{i}.png')
print('Rendered all seven distinct report/theory PDFs.')
