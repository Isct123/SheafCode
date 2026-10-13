                                                                            
import sys,subprocess,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--skip-graph',action='store_true');ap.parse_args()
    jobs=[('hybrid.py','hybrid_summary.json'),('faults.py','fault_scope.json'),('bounds_study.py','refinement_scope.json'),('operator_study.py','operator_scope.json'),('alerts.py','alert_scope.json')]
    for script,result in jobs:
        path=OUT/result
        if path.exists() and json.loads(path.read_text()).get('source_sha256')==digest(HERE/script):
            print('Verified current completed stage:',script,flush=True);continue
        subprocess.run([sys.executable,'-u',str(HERE/script)],check=True,cwd=ROOT)
    source=OUT/'external/observations.csv'
    if not source.exists():subprocess.run([sys.executable,'-u',str(HERE/'acquire_external.py')],check=True,cwd=ROOT)
    result=OUT/'external_summary.json';lock=OUT/'external/evaluation_lock.json'
    if source.exists() and (not result.exists() or not lock.exists() or json.loads(lock.read_text()).get('source_sha256')!=digest(HERE/'external_evaluate.py')):
        subprocess.run([sys.executable,'-u',str(HERE/'external_evaluate.py')],check=True,cwd=ROOT)
    print('All non-neural validation stages available',flush=True)
if __name__=='__main__':main()
