                                                                             
from pathlib import Path
import hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'Research_Release/math_integration_results'

def main():
    manifest=json.loads((OUT/'EVIDENCE_MANIFEST.json').read_text())
    for name,h in manifest['files'].items():
        path=ROOT/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=h:raise RuntimeError('Evidence hash mismatch: '+name)
    folder=ROOT/'Research_Release/extensions/math_integration'
    for name in ['test_graph.py','test_search.py','replay.py']:
        result=subprocess.run([sys.executable,str(folder/name)],cwd=ROOT)
        if result.returncode:raise RuntimeError('Verification failed: '+name)
    print('Mathematical experiment evidence and independent replays passed. No retraining or document changes.')

if __name__=='__main__':main()
