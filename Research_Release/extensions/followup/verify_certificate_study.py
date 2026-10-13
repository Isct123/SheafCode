                                                                              
import hashlib,json,sys,time
from fractions import Fraction
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
from exact_certificate import verified_upper,rational
from budget_dual import upper_certificate
from constrained import constrained_envelope

def main():
    out=ROOT/'constrained_results';out.mkdir(exist_ok=True);rng=np.random.default_rng(541);rows=[];tic=time.time()
    for trial in range(100):
        A=rng.normal(size=(10,3));z=rng.uniform(-1,1,3);h=rng.normal(size=3);blocks=[rng.normal(size=(10,2)) for _ in range(4)]
        e=rng.uniform(-.5,.5,2);b=-A@z-blocks[trial%4]@e;lo=[-1]*3;hi=[1]*3
        fitted=upper_certificate(A,b,h,blocks,1,.01,.5,lo,hi);assert fitted['status']=='finite'
        w=np.asarray(fitted['w'])+rng.normal(0,1e-9,10)
        verified=verified_upper(A,b,h,blocks,1,.01,.5,w,lo,hi)
        actual=sum((rational(x)*rational(y) for x,y in zip(h,z)),Fraction(0))
        assert actual<=Fraction(verified['upper_rational'])
        exact=constrained_envelope(A,b,h,blocks,1,.01,.5,lo,hi)
        assert verified['upper_outward_float']>=exact['upper']-1e-7
        rows.append({'trial':trial,'verified_upper':verified['upper_outward_float'],'exact_sparse_upper_numerical':exact['upper'],
            'verified_upper_rational':verified['upper_rational'],'residual_correction_rational':verified['residual_correction_rational']})
    pd.DataFrame(rows).to_csv(out/'exact_certificate_checks.csv',index=False)
    (out/'exact_certificate_manifest.json').write_text(json.dumps({'cases':len(rows),'status':'completed','elapsed_seconds':time.time()-tic,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'exact_certificate.py']},
        'scope':'Exact nominal rational evaluation of perturbed floating-point dual vectors; numerical sparse-LP comparison; no new physical coverage evidence'},indent=2))
    print('Exact nominal certificate study completed:',len(rows),'cases',flush=True)

if __name__=='__main__':main()
