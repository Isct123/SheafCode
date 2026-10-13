                                                                                    
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'IRIS_Submission_Recent_Data'
NEW = ROOT / 'SUBMISSION_WORKSPACE'
if NEW.exists():
    raise SystemExit('SUBMISSION_WORKSPACE already exists; refusing to overwrite your edits.')
NEW.resolve().relative_to(ROOT.resolve())
SOURCE = NEW / 'Source'
GUIDES = NEW / 'Guides'
SOURCE.mkdir(parents=True)
GUIDES.mkdir()
copies = {
    'Upload_Files/SheafCode_Final_Manuscript.pdf': 'Manuscript.pdf',
    'Upload_Files/SheafCode_Concise_Research_Synopsis.pdf': 'Synopsis.pdf',
    'Upload_Files/Project_Abstract_Concise.txt': 'Abstract.txt',
    'Upload_Files/Portal_Text_Concise.txt': 'Portal_Text.txt',
    'Upload_Files/Video_Script_Concise.txt': 'Video_Script.txt',
    'Editable_Manuscript/manuscript.tex': 'Source/manuscript.tex',
    'Editable_Manuscript/concise_synopsis.tex': 'Source/synopsis.tex',
    'Editable_Manuscript/references.bib': 'Source/references.bib',
    'Review_Notes/NOVELTY_AND_PROOF_DEFENCE.md': 'Guides/Proof_Defence.md',
    'Review_Notes/REMAINING_WEAKNESSES_AND_FIXES.md': 'Guides/Research_Limitations.md',
    'Review_Notes/FINAL_MATH_SELECTION.md': 'Guides/Mathematical_Selection.md',
}
manifest = []
for old_name, new_name in copies.items():
    original, target = OLD / old_name, NEW / new_name
    shutil.copy2(original, target)
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    if actual != hashlib.sha256(original.read_bytes()).hexdigest():
        raise RuntimeError(f'Copy mismatch: {new_name}')
    manifest.append({'working_file': new_name, 'frozen_source': str(original.relative_to(ROOT)), 'initial_sha256': actual})
(SOURCE / 'figures').mkdir()
for original in sorted((OLD / 'Editable_Manuscript/figures').glob('*.pdf')):
    shutil.copy2(original, SOURCE / 'figures' / original.name)
(GUIDES / 'INITIAL_HANDOFF_CHECK.json').write_text(json.dumps({'status': 'passed', 'scope': 'Byte-identical copies of the frozen selected release; hashes describe initial handoff, not subsequent user edits.', 'files': manifest}, indent=2), encoding='utf-8')
print(f'Created {NEW}; {len(copies)} working files and five vector figures copied.')
