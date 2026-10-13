                                                                        
from pathlib import Path
import sys,json,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'Research_Release/joint_certification_results'
    manifest=json.loads((out/'EVIDENCE_MANIFEST.json').read_text())
    for name,h in manifest['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise RuntimeError('Joint follow-up hash mismatch: '+name)
    folder=ROOT/'Research_Release/extensions/math_integration'
    for args in [['replay_joint.py'],['independent_optima.py','--check-only'],['exact_graph_rank.py','--check-only']]:
        result=subprocess.run([sys.executable,str(folder/args[0]),*args[1:]],cwd=ROOT)
        if result.returncode:raise RuntimeError('Joint follow-up verification failed: '+args[0])
    print('Exact joint decisions and independently optimized repair costs verified. No retraining or document changes.')
if __name__=='__main__':main()
