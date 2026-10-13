                                                                        
import json,math,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
from constrained import constrained_envelope
from budget_dual import dual_envelope

def main():
    out=ROOT/'constrained_results';out.mkdir(exist_ok=True);rows=[];rng=np.random.default_rng(111)
    for n,t in [(6,1),(12,2),(18,3)]:
        for repeat in range(3):
            A=rng.normal(size=(24,3));z=rng.normal(size=3);h=rng.normal(size=3);blocks=[rng.normal(size=(24,1)) for _ in range(n)]
            b=-A@z+blocks[0][:,0]*.5+rng.uniform(-.05,.05,24)
            tic=time.perf_counter();exact=constrained_envelope(A,b,h,blocks,t,.05,1.);exact_time=time.perf_counter()-tic
            tic=time.perf_counter();outer=dual_envelope(A,b,h,blocks,t,.05,1.);dual_time=time.perf_counter()-tic
            assert exact['status']==outer['status']=='bounded'
            assert outer['lower']<=exact['lower']+1e-6 and outer['upper']>=exact['upper']-1e-6
            assert exact['lower']-1e-6<=h@z<=exact['upper']+1e-6
            rows.append({'fault_stations':n,'budget':t,'repeat':repeat,'supports':sum(math.comb(n,k) for k in range(t+1)),
                'exact_seconds':exact_time,'dual_seconds':dual_time,'speedup':exact_time/dual_time,'exact_width':exact['upper']-exact['lower'],
                'dual_width':outer['upper']-outer['lower']})
    f=pd.DataFrame(rows);f.to_csv(out/'budget_dual_benchmark.csv',index=False)
    print(f.groupby(['fault_stations','budget'])[['exact_seconds','dual_seconds','speedup','exact_width','dual_width']].median().to_string())
    (out/'budget_dual_manifest.json').write_text(json.dumps({'scope':'Three repeated synthetic timings per station-count/fault-budget; numerical containment checked against exact support LPs',
        'limitations':['These are synthetic examples, not actual environmental alert throughput','Dual envelope is conservative; strict gaps are tested separately','Established budgeted-uncertainty tools; first-ever priority not established']},indent=2))

if __name__=='__main__':main()
