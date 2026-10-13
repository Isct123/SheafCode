                                                                                     
from pathlib import Path
import hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'Research_Release/operational_validation_results'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    lock=json.loads((OUT/'PROTOCOL_LOCK.json').read_text())
    for name,value in lock['input_sha256'].items():
        if sha(ROOT/name)!=value:raise RuntimeError(f'Dataset changed: {name}')
    script=ROOT/'Research_Release/extensions/decision_repair/heldout.py'
    if sha(script)!=lock['code_sha256']:raise RuntimeError('Held-out experiment code changed after evaluation')
    if sha(script.with_name('HELDOUT_PROTOCOL.md'))!=lock['protocol_sha256']:raise RuntimeError('Protocol changed after evaluation')
    manifest=json.loads((OUT/'EVIDENCE_MANIFEST.json').read_text())
    for name,value in manifest['files'].items():
        if sha(ROOT/name)!=value:raise RuntimeError(f'Evidence changed: {name}')
    replay=subprocess.run([sys.executable,str(script.with_name('replay_heldout.py')),str(OUT)],cwd=ROOT,capture_output=True,text=True,timeout=120)
    if replay.returncode:raise RuntimeError(replay.stderr+'\n'+replay.stdout)
    print(replay.stdout.strip())
    newer=ROOT/'Research_Release/query_policy_results'
    if (newer/'EVIDENCE_MANIFEST.json').exists():
        recorded=json.loads((newer/'EVIDENCE_MANIFEST.json').read_text())
        for name,value in recorded['files'].items():
            if sha(ROOT/name)!=value:raise RuntimeError(f'Query-policy evidence changed: {name}')
        checked=subprocess.run([sys.executable,str(script.with_name('verify_query_policy.py'))],cwd=ROOT,capture_output=True,text=True,timeout=120)
        if checked.returncode:raise RuntimeError(checked.stderr+'\n'+checked.stdout)
        print(checked.stdout.strip())
    print('Completed evidence verified. No experiments, manuscript rebuild or user-file changes were performed.')

if __name__=='__main__':
    try:main()
    except Exception as error:
        print(f'OPERATIONAL VERIFICATION FAILED: {error}',file=sys.stderr);sys.exit(1)
