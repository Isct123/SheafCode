                                                                               
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
from certificate import functional_interval
from sheafcode.cohort import select_cohort
ap=argparse.ArgumentParser();ap.add_argument('--local',action='store_true');args=ap.parse_args()
out=ROOT/'extension_results'/('functional_local' if args.local else 'functional');out.mkdir(parents=True,exist_ok=True)
results=[];summary={}
for region in ['delhi2025','bengaluru_raw']:
    folder=ROOT/'results'/region;manifest=json.loads((folder/'manifest.json').read_text());ds=select_cohort(region)
    choice=manifest['choices']['sheaf'];ds.set_transform(choice['transform']);Z=ds.Z()
    M=np.load(folder/'model.npz');D=M['delta'];full=np.isfinite(Z).all(axis=(1,2))
    tr=np.flatnonzero((ds.split==0)&full);va=np.flatnonzero((ds.split==1)&full);te=np.flatnonzero((ds.split==2)&full)
    if min(len(tr),len(va),len(te))<30:
        summary[region]={'status':'excluded','reason':'fewer than 30 completely observed vectors in a partition'};continue
    residual=Z[tr].reshape(len(tr),-1)@D.T
    scale=np.maximum(np.std(residual,axis=0),.1);Afull=D/scale[:,None]
    scores=np.linalg.norm(Z[va].reshape(len(va),-1)@Afull.T,axis=1)
    k=int(np.ceil((len(scores)+1)*.95));epsilon=float(np.sort(scores)[min(k,len(scores))-1]) if k<=len(scores) else np.inf
    assert np.isfinite(epsilon)
    local_budgets={}
    if args.local:
        for v in range(ds.n):
            cols=np.arange(v*ds.p,(v+1)*ds.p);rows=np.flatnonzero(np.any(abs(Afull[:,cols])>1e-14,axis=1))
            score=np.linalg.norm(Z[va].reshape(len(va),-1)@Afull[rows].T,axis=1)
            local_budgets[v]=(rows,float(np.sort(score)[k-1]))
    sample=te[np.unique(np.linspace(0,len(te)-1,min(360,len(te))).astype(int))]
    q=ds.pollutants.index('PM2.5');coverage=[];conditional_checks=0
    for t in sample:
        for v in range(ds.n):
            U=np.arange(v*ds.p,(v+1)*ds.p);O=np.setdiff1d(np.arange(ds.n*ds.p),U)
                                                                    
            B=Afull[local_budgets[v][0]] if args.local else Afull
            budget=local_budgets[v][1] if args.local else epsilon
            b=B[:,O]@Z[t].reshape(-1)[O];h=np.zeros(ds.p);h[q]=1
            r=functional_interval(B[:,U],b,h,budget)
            actual=float(Z[t,v,q]);inside=r['status']=='bounded' and r['lower']-1e-9<=actual<=r['upper']+1e-9
            true_residual=float(np.linalg.norm(B@Z[t].reshape(-1)))
            if true_residual<=budget:
                assert inside,'Conditional theorem violated';conditional_checks+=1
            def concentration(x):
                if x is None:return None
                val=x*ds.std[v,q]+ds.mean[v,q]
                if ds.hod is not None:val+=ds.hod[ds.times[t].hour,v,q]
                return float(np.expm1(val)) if choice['transform']!='none' else float(val)
            lower=concentration(r['lower']);upper=concentration(r['upper'])
                                                                                       
            decision='abstain' if lower is None or upper is None else ('above' if lower>60 else ('below' if upper<=60 else 'abstain'))
            truth=bool(ds.X[t,v,q]>60)
            results.append({'region':region,'timestamp':str(ds.times[t]),'station':ds.stations[v],'status':r['status'],'pm25_lower':lower,'pm25_upper':upper,
                'pm25_true':float(ds.X[t,v,q]),'covered':inside,'residual_assumption_met':true_residual<=budget,'threshold_decision':decision,
                'threshold_correct_when_decided':None if decision=='abstain' else bool((decision=='above')==truth)})
    g=pd.DataFrame([x for x in results if x['region']==region]);decided=g[g.threshold_decision!='abstain']
    summary[region]={'status':'complete','train_complete_hours':len(tr),'validation_complete_hours':len(va),'test_complete_hours':len(te),'sampled_test_hours':len(sample),
        'epsilon_validation_95':{ds.stations[v]:b[1] for v,b in local_budgets.items()} if args.local else epsilon,'cases':len(g),'empirical_interval_coverage':float(g.covered.mean()),'decided_fraction':float(len(decided)/len(g)),
        'accuracy_when_decided':float(decided.threshold_correct_when_decided.mean()) if len(decided) else None,'conditional_theorem_checks':conditional_checks,
        'clean_residual_assumption_frequency':float(g.residual_assumption_met.mean()),
        'frozen_model_sha256':hashlib.sha256((folder/'model.npz').read_bytes()).hexdigest()}
    print(region,summary[region],flush=True)
pd.DataFrame(results).to_csv(out/'functional_intervals.csv',index=False)
(out/'summary.json').write_text(json.dumps({'results':summary,'limitations':['complete-vector subset only; missing-context generalization not demonstrated','calibrated residual budget is fixed, with no test feedback','95% rank calibration is exchangeability-conditional, not a distribution-shift guarantee','hourly threshold is illustrative, not daily pollution-standard validity','functional observability and ellipsoid bounds have prior literature; project-specific integration, not claimed first general theorem'],
    'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}},indent=2))
