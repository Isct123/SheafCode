                                                                                           
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'IRIS_Submission_Recent_Data'
REVIEW = PACKAGE / 'Review_Notes'
REPORT = REVIEW / 'FINAL_PROJECT_VERIFICATION.json'

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def main():
    audit = load(REVIEW / 'CONCISE_MANUSCRIPT_AUDIT.json')
    companions = load(REVIEW / 'CONCISE_COMPANIONS_AUDIT.json')
    paper = PACKAGE / 'Upload_Files/SheafCode_Final_Manuscript.pdf'
    source = PACKAGE / 'Editable_Manuscript/manuscript.tex'
    require(digest(paper) == audit['pdf_sha256'], 'Final paper changed; its document audit must be refreshed.')
    require(digest(source) == audit['source_sha256'], 'Editable manuscript changed; review and refresh its audit before release.')
    for name, expected in audit['data_sources'].items():
        require(digest(ROOT / name) == expected, f'Saved result changed: {name}')
    require(companions['paper_sha256'] == digest(paper), 'Companions describe a different manuscript.')
    require(digest(PACKAGE / 'Upload_Files/SheafCode_Concise_Research_Synopsis.pdf') == companions['synopsis_sha256'], 'Synopsis changed after companion audit.')
    abstract = (PACKAGE / 'Upload_Files/Project_Abstract_Concise.txt').read_text(encoding='utf-8-sig').strip()
    source_abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', source.read_text(encoding='utf-8'), re.S)
    require(source_abstract is not None, 'Abstract missing from editable paper.')
    plain_source_abstract = source_abstract.group(1).replace(r'\%', '%').replace('--', '\u2013')
    require(' '.join(plain_source_abstract.split()) == ' '.join(abstract.split()), 'Portal abstract does not match the manuscript abstract.')
    require(len(abstract.split()) == companions['abstract_words'], 'Abstract word count changed.')
    with zipfile.ZipFile(PACKAGE / 'Supporting_Reference/SheafCode_Editable_Manuscript.zip') as archive:
        require(archive.testzip() is None, 'Editable archive failed CRC validation.')
        for name in ['manuscript.tex', 'references.bib'] + [f'figures/{p.name}' for p in (PACKAGE / 'Editable_Manuscript/figures').glob('*.pdf')]:
            require(archive.read(name) == (PACKAGE / 'Editable_Manuscript' / name).read_bytes(), f'Editable archive has stale content: {name}')
    import pymupdf
    with pymupdf.open(paper) as doc:
        require(len(doc) == audit['pages'], 'Final PDF page count changed.')
        require(all(page.get_text().strip() for page in doc), 'Final PDF contains an empty-text page.')

    scripts = ROOT / 'Research_Release/extensions/decision_repair'
    results = ROOT / 'Research_Release/decision_repair_results'
                                                                                
    temp_parent = ROOT / 'audit_tools/verification_temp'
    temp_parent.mkdir(exist_ok=True)
    temp_parent.resolve().relative_to(ROOT.resolve())
    with tempfile.TemporaryDirectory(prefix='final_', dir=temp_parent) as folder:
        isolated = Path(folder)
        isolated.resolve().relative_to(temp_parent.resolve())
        for name in ['SCENARIO_BANKS.json', 'REPAIR_CONFIGURATIONS.csv', 'ADAPTIVE_POLICIES.json', 'RADIUS_BOUNDARY_WITNESSES.json']:
            shutil.copy2(results / name, isolated / name)
        commands = [
            [sys.executable, str(scripts / 'replay_saved_banks.py'), '--results-dir', str(isolated)],
            [sys.executable, str(scripts / 'replay_adaptive_policies.py'), str(isolated)],
            [sys.executable, str(scripts / 'replay_radius_witnesses.py'), str(isolated)],
        ]
        def replay(command):
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300)
            require(completed.returncode == 0, f'{Path(command[1]).name} failed:\n{completed.stderr}\n{completed.stdout}')
            report = json.loads(completed.stdout)
            require(report.get('status') == 'passed', f'Replay did not pass: {command[1]}')
            return report
        with ThreadPoolExecutor(max_workers=3) as pool:
            replay_reports = list(pool.map(replay, commands))
    for name, expected in audit['data_sources'].items():
        require(digest(ROOT / name) == expected, f'Replay unexpectedly changed saved evidence: {name}')
    report = {'status': 'passed', 'paper_sha256': digest(paper), 'source_sha256': digest(source),
              'result_files_checked': len(audit['data_sources']), 'pages': audit['pages'],
              'replays': replay_reports,
              'scope': 'Release hashes, editable archive, PDF/companions and isolated saved-bank/policy/boundary replays; no retraining, priority or eligibility certification.'}
    REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        REVIEW.mkdir(exist_ok=True)
        REPORT.write_text(json.dumps({'status': 'failed', 'error': str(error)}, indent=2), encoding='utf-8')
        print(f'VERIFICATION FAILED: {error}', file=sys.stderr)
        sys.exit(1)
