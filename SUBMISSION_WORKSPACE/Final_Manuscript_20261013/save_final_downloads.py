from pathlib import Path
import hashlib,json,shutil

out=Path(__file__).resolve().parent
downloads=Path(r'C:\Users\rishi\Downloads')
working=downloads/'SheafCode_IRIS_Cleaned_Heatmap_Restored (4).tex'
tex=out/'SheafCode_IRIS_Final_Manuscript.tex';pdf=tex.with_suffix('.pdf')
original='8b78d30a7886342cdb1169074473f2365b00a26f31c5f01397e9d5d5548ef68a'
finalhash=hashlib.sha256(tex.read_bytes()).hexdigest()
current=hashlib.sha256(working.read_bytes()).hexdigest()
allowed_previous=[original,'9c1a6e546a6ad2d55ecc8fd380da9b8a643f784fb5f40c619ac164b796ba429e','148074b7cb2367024ecc81a01b4a7b9bc43dfc0d249855ea2cfbb44c064dc12d','b026d3c0c6caa766ff8fec82d7de318ceb4d9717a3736904d259f4b3dbadddf9','56692eb53392ee0da04eba1563ad6a5eebedd1e22d1e45b7a9744e6e503b16d0','5bdf9426e6e432ba18d04c4e30519a7b6185e0674e33ddfe4db8f18f835a5bde','f08e44c4e5905dfb5f144b28410c2f11a7b0310d5d3747c23217eaaa75c7fc50','0d52432184b3189c0aef687d8cb54481cb98405a2416dab04edff722696cfe6b',finalhash]
assert current in allowed_previous,'Working copy changed during final review; refusing to overwrite new edits.'
if current==original:
    shutil.copy2(working,out/'Original_Downloads_Working_Copy.tex')
if current!=finalhash:
    working.write_bytes(tex.read_bytes())
written=[]
for source in [tex,pdf]:
    target=downloads/source.name
    if source.suffix=='.tex' and target.exists():
        assert hashlib.sha256(target.read_bytes()).hexdigest() in allowed_previous,'Final Downloads source changed during review; refusing to overwrite new edits.'
    if target.exists() and target.read_bytes()!=source.read_bytes():
        shutil.copy2(target,out/('Previous_Downloads_'+target.name))
    try:
        target.write_bytes(source.read_bytes())
    except PermissionError:
        if source.suffix!='.pdf':raise
        target=downloads/(source.stem+'_Whole_Checked.pdf')
        try:target.write_bytes(source.read_bytes())
        except PermissionError:
            target=downloads/(source.stem+'_Whole_Checked_Final.pdf')
            target.write_bytes(source.read_bytes())
    assert target.read_bytes()==source.read_bytes()
    written.append(str(target))
assert hashlib.sha256(working.read_bytes()).hexdigest()==finalhash
record=dict(status='saved_and_verified',final_files=written,updated_working_copy=str(working),other_numbered_downloads_copies='Unchanged original versions',final_tex_sha256=finalhash,pdf_note='If the standard Downloads PDF is locked by a viewer, the updated PDF is saved with the _Whole_Checked suffix; the workspace PDF is always current.')
(out/'FINAL_DOWNLOADS.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,indent=2))
