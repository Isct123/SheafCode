                                                                                
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=ROOT/'Research_Release/expanded_validation_results'
def main():
    cohorts=[]
    for marker in ['EPA','AURN','NSW','OpenAQ']:
        path=OUT/(marker+'_acquired.json')
        if not path.exists():raise RuntimeError('Missing acquisition: '+marker)
        cohorts.extend(json.loads(path.read_text())['cohorts'])
    value={'eligible':cohorts,'secondary':[],'scope':'Externally acquired regional cohorts plus explicitly non-independent Indian longer-test resplit'}
    (OUT/'cohorts.json').write_text(json.dumps(value,indent=2))
    sources=[HERE/'PROTOCOL.md',ROOT/'Run_Broad_Empirical_Validation.cmd']+list(HERE.glob('*.py'))
    sources+=list((ROOT/'Research_Release/extensions/recent_data').glob('*.py'))
    sources+=list((ROOT/'Research_Release/extensions/online_validation').glob('*.py'))
    sources+=list((ROOT/'Research_Release/extensions/weakness_resolution').glob('*'))
    sources=[p for p in sources if p.is_file()]
    sources.append(ROOT/'Run_Weakness_Resolution.cmd')
    for directory in ['structured','functional','followup','joint_decisions','operator_uncertainty','validation']:sources+=list((ROOT/'Research_Release/extensions'/directory).glob('*.py'))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (OUT/'IMPLEMENTATION_HASHES.json').write_text(json.dumps({'scope':'Published implementation snapshot, distinct from pre-fit protocol and per-fit source locks','sha256':hashes},indent=2))
    (OUT/'acquisition_status.json').write_text(json.dumps({'status':'completed','cohorts':len(cohorts),'providers':['EPA','AURN','NSW','OpenAQ'],'new_external_cohorts':len(cohorts)-1,'indian_resplit_independent':False},indent=2))
if __name__=='__main__':main()
