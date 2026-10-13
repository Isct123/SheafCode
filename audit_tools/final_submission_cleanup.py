                                                                                         
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'IRIS_Submission_Recent_Data'
REVIEW = PACKAGE / 'Review_Notes'
ARCHIVE = PACKAGE / 'Archived_Drafts_and_Previews'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

audit = json.loads((REVIEW / 'CONCISE_MANUSCRIPT_AUDIT.json').read_text(encoding='utf-8-sig'))
pdf = PACKAGE / 'Upload_Files/SheafCode_Final_Manuscript.pdf'
source = PACKAGE / 'Editable_Manuscript/manuscript.tex'
assert digest(pdf) == audit['pdf_sha256'], 'Final PDF changed after audit'
assert digest(source) == audit['source_sha256'], 'Editable source changed after audit'
checks = []
for name, expected in audit['data_sources'].items():
    path = ROOT / name
    assert digest(path) == expected, f'Result changed after audit: {name}'
    checks.append(name)
assert digest(PACKAGE / 'Upload_Files/SheafCode_Full_Empirical_Manuscript.pdf') == audit['frozen_complete_manuscript_unchanged']
assert audit['external_cohorts'] == 20 and audit['total_neural_seed_rows'] == 12
assert audit['unresolved_references'] == 0 and audit['overfull_text_boxes'] == 0
text = source.read_text(encoding='utf-8')
required = ['16.48', '10.45', '13/20', '18/20', '12/61', '4/15', '239', '414', '58 to 52', '84 to 83', 'corruption', 'undercoverage', 'AI assistance']
for item in required:
    assert item in text, f'Missing selected evidence or limitation: {item}'

keep_uploads = {
    'SheafCode_Final_Manuscript.pdf', 'SheafCode_Concise_Research_Synopsis.pdf',
    'Project_Abstract_Concise.txt', 'Portal_Text_Concise.txt', 'Video_Script_Concise.txt',
                                                                         
    'SheafCode_Full_Empirical_Manuscript.pdf',
}
candidates = [p for p in (PACKAGE / 'Upload_Files').iterdir() if p.is_file() and p.name not in keep_uploads]
candidates += [p for p in REVIEW.iterdir() if p.is_file() and (p.suffix.lower() == '.png' or p.name.endswith('_compile.txt') or p.name in {'EXPANDED_DOCUMENT_CONTACT.pdf', 'WEAKNESS_LAYOUT_PREVIEW.pdf'})]
if (REVIEW / 'concise_preview').exists():
    candidates.append(REVIEW / 'concise_preview')

manifest_path = ARCHIVE / 'CLEANUP_MANIFEST.json'
moved = json.loads(manifest_path.read_text(encoding='utf-8'))['moved'] if manifest_path.exists() else []
for path in candidates:
    resolved = path.resolve()
    resolved.relative_to(PACKAGE.resolve())
    target = ARCHIVE / path.relative_to(PACKAGE)
    target.resolve().relative_to(PACKAGE.resolve())
    if target.exists():
        raise FileExistsError(f'Archive destination already exists: {target}')
    entries = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file())
    entry = {'original': str(path.relative_to(PACKAGE)), 'archived': str(target.relative_to(PACKAGE)),
             'files': [{'path': str(p.relative_to(PACKAGE)), 'sha256': digest(p)} for p in entries]}
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(target))
    for item in entry['files']:
        relative = Path(item['path']).relative_to(path.relative_to(PACKAGE)) if path.name == 'concise_preview' else None
        destination = target / relative if relative is not None else target
        assert digest(destination) == item['sha256']
    moved.append(entry)

ARCHIVE.mkdir(exist_ok=True)
manifest_path.write_text(json.dumps({'policy': 'Reversible archive; no research evidence deleted', 'moved': moved}, indent=2), encoding='utf-8')
report = {'date': '2026-10-08', 'status': 'passed', 'pdf_sha256': digest(pdf),
          'source_sha256': digest(source), 'saved_result_hashes_checked': len(checks),
          'selected_content_checks': required, 'external_cohorts': 20, 'neural_seed_comparisons': 12,
          'archived_items': len(moved), 'scope': 'Document/evidence consistency and reversible cleanup; not a fresh replay of all training runs.'}
(REVIEW / 'FINAL_CONFIRMATION.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
