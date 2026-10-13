                                                                          
from pathlib import Path
import sys,os,hashlib,json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
cohort=sys.argv[1]
os.environ['WEAKNESS_NEURAL_COHORT']=cohort
sys.path.insert(0,str(ROOT/'Research_Release/extensions/recent_data'))
import common
target=ROOT/'Research_Release/weakness_resolution_results/neural'
common.OUT=target
original_save=common.save
def routed_save(path,value):
    if path=='neural_progress.json':path=cohort+'/'+path
    original_save(path,value)
common.save=routed_save
source=HERE/'neural_external.py'
text=source.read_text()
text=text.replace("neural.regions=lambda:[r['cohort'] for r in selected]","neural.regions=lambda:[os.environ['WEAKNESS_NEURAL_COHORT']]")
text=text.replace("for item in selected:","for item in [r for r in selected if r['cohort']==os.environ['WEAKNESS_NEURAL_COHORT']]:")
text=text.replace("OUT/'REPLAY.json'","OUT/(os.environ['WEAKNESS_NEURAL_COHORT']+'_REPLAY.json')")
manifest=json.loads((target/'SELECTION.json').read_text())
if cohort not in [r['cohort'] for r in manifest['selections']]:raise RuntimeError('Worker cohort outside fixed selection')
namespace={'__file__':str(source),'__name__':'__main__'}
exec(compile(text,str(source),'exec'),namespace)
folder=target/cohort
(folder/'WORKER_ROUTING.json').write_text(json.dumps({
    'status':'completed','worker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'cohort':cohort,'changes':'Cohort routing and progress/replay destinations only; training grids, case masks, seeds, weights and checkpoint RNG rules unchanged'},indent=2))
