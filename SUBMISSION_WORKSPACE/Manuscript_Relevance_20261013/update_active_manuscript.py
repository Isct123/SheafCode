from pathlib import Path
import hashlib,json

out=Path(__file__).resolve().parent
target=Path(r'C:\Users\rishi\Downloads\SheafCode_IRIS_Natural_Writing_Final (1).tex')
focused=(out/'SheafCode_Focused_Manuscript.tex').read_bytes()
before='c6bff7ed1dfe01f979da12b329937c83fff76488ead25c331bd6e50549bb9250'
after=hashlib.sha256(focused).hexdigest()
current=hashlib.sha256(target.read_bytes()).hexdigest()
assert current in [before,after],'Active manuscript changed since review; refusing to overwrite new edits.'
if current==before:target.write_bytes(focused)
assert hashlib.sha256(target.read_bytes()).hexdigest()==after
record=dict(status='updated_and_verified',target=str(target),source_sha256=after,backup=str(out/'Before_Relevance_Edit.tex'))
(out/'UPDATED_DOWNLOADS.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,indent=2))
