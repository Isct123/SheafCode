from pathlib import Path
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parents[1]
folder = root / 'Final_Manuscript'
excluded = {'.log', '.blg', '.aux', '.bbl', '.out', '.toc'}
files = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix not in excluded and p.name != 'PACKAGE_MANIFEST.json')
manifest = {str(p.relative_to(folder)).replace('\\', '/'): {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
manifest_path = folder / 'PACKAGE_MANIFEST.json'
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
archive = root / 'SheafCode_Final_Reference.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in files + [manifest_path]:
        z.write(p, p.relative_to(folder).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert len(z.namelist()) == len(manifest) + 1
    for name, item in manifest.items():
        assert hashlib.sha256(z.read(name)).hexdigest() == item['sha256'], name
    for name in ['main.tex', 'main.pdf', 'synopsis.pdf', 'portal_fields.pdf', 'video_script.pdf', 'FINAL_AUDIT.md', 'rebuild_figures.py']:
        assert name in z.namelist(), name
print(json.dumps({'archive': str(archive), 'files': len(manifest) + 1, 'bytes': archive.stat().st_size, 'integrity': 'passed'}, indent=2))
