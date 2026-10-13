from pathlib import Path
import hashlib,json,shutil

out=Path(__file__).resolve().parent
target=Path(r'C:\Users\rishi\Downloads\SheafCode_IRIS_Natural_Writing_Final (1).tex')
source=out/'SheafCode_IRIS_Natural_Writing_Technically_Corrected.tex'
expected='e71427c5aecd907d05545dfac527febf200696a98feb2ddd71f272031b26fd8e'
new=source.read_bytes();newhash=hashlib.sha256(new).hexdigest()
current=hashlib.sha256(target.read_bytes()).hexdigest()
if current not in [expected,newhash]:raise RuntimeError('The active manuscript changed during the audit; refusing to overwrite new edits.')
if current==expected:
    shutil.copy2(target,out/'Original_Active_Downloads_File.tex')
    target.write_bytes(new)
assert hashlib.sha256(target.read_bytes()).hexdigest()==newhash
record={'status':'updated_and_verified','target':str(target),'original_sha256':expected,'corrected_sha256':newhash,'backup':str(out/'Original_Active_Downloads_File.tex'),'other_downloads_tab':'Unchanged; retained as the earlier copy.'}
(out/'UPDATED_DOWNLOADS.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,indent=2))
