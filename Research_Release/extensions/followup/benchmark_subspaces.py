                                                                                    
import argparse,json,sys,time,math
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
from robust_quantity import fault_envelope
from subspace_envelope import compressed_envelope

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='followup_results/subspace_compression');args=ap.parse_args()
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(81);records=[]
    cases=[]
    for n in [8,12,16]:
        A=rng.normal(size=(10,2));b=rng.normal(size=10);h=np.array([1.,.2]);blocks=[np.eye(10)[:,[j%3]]*(j+1) for j in range(n)]
        cases.append(('redundant_synthetic_'+str(n),A,b,h,blocks,3,5.))
    for region in ['delhi2025','bengaluru_raw']:
        D=np.load(ROOT/'results_expanded'/region/'model.npz')['delta'];n=json.loads((ROOT/'results_expanded'/region/'manifest.json').read_text())['region']
        summary=json.loads((ROOT/'results_expanded'/region/'summary.json').read_text());p=summary['n_pollutants'];stations=summary['n_stations']
        A=D[:,:p];b=np.zeros(len(D));h=np.eye(p)[0];blocks=[D[:,v*p:(v+1)*p] for v in range(1,stations)]
        cases.append((region+'_operator_only',A,b,h,blocks,min(2,len(blocks)),1.))
    for name,A,b,h,blocks,t,eps in cases:
        started=time.perf_counter();a=fault_envelope(A,b,h,blocks,t,eps);brute=time.perf_counter()-started
        started=time.perf_counter();c=compressed_envelope(A,b,h,blocks,t,eps);compressed=time.perf_counter()-started
        assert a['status']==c['status']
        if a['status']!='inconsistent':assert np.allclose([a['lower'],a['upper']],[c['lower'],c['upper']],atol=1e-8)
        records.append({'case':name,'support_count':sum(math.comb(len(blocks),k) for k in range(t+1)),'distinct_subspaces':c['distinct_subspaces'],
            'brute_seconds':brute,'compressed_seconds':compressed,'speedup':brute/compressed,'envelope_status':c['status']})
    pd.DataFrame(records).to_csv(out/'benchmark.csv',index=False)
    print(pd.DataFrame(records).to_string(index=False))
    (out/'scope.json').write_text(json.dumps({'scope':'Synthetic redundancy and actual frozen operator checks; not empirical pollution-decision performance',
        'limitations':['Numerical subspace comparison uses tolerances','Distinct spans can remain exponential; no general polynomial speedup','Single timing repetition; not robust runtime inference','First-ever theorem priority unestablished']},indent=2))

if __name__=='__main__':main()
