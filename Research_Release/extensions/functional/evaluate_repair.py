                                                                   

                                                                        
                                                                          
                                                                        
   
import hashlib,itertools,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
from repair import worst_case_repair_halfwidth
from certificate import functional_interval
from sheafcode.cohort import select_cohort
from sheafcode.acquisition import conditional_covariance,choose_repair
from sheafcode.reliable import paired_bootstrap
out=ROOT/'extension_results/worst_case_repair';out.mkdir(parents=True,exist_ok=True)
records=[];configs={};bound_checks=0;theorem_checks=0
for region in ['delhi2025','bengaluru_raw']:
    folder=ROOT/'results'/region;m=json.loads((folder/'manifest.json').read_text());ds=select_cohort(region);ds.set_transform(m['choices']['sheaf']['transform']);Z=ds.Z()
    model=np.load(folder/'model.npz');D=model['delta'];Q=model['L']+float(model['lam'])*model['P'];p=ds.p
    full=np.isfinite(Z).all(axis=(1,2));tr=np.flatnonzero((ds.split==0)&full);va=np.flatnonzero((ds.split==1)&full);te=np.flatnonzero((ds.split==2)&full)
    scale=np.maximum(np.std(Z[tr].reshape(len(tr),-1)@D.T,axis=0),.1);D=D/scale[:,None]
    scores=np.linalg.norm(Z[va].reshape(len(va),-1)@D.T,axis=1);k=int(np.ceil((len(scores)+1)*.95));epsilon=float(np.sort(scores)[k-1])
    hours=te[np.unique(np.linspace(0,len(te)-1,min(180,len(te))).astype(int))];q=ds.pollutants.index('PM2.5')
    attempted=0;excluded=0
    for t in hours:
        pairs=list(itertools.combinations(range(ds.n),2));rng=np.random.default_rng(315+int(t))
        for pi in rng.choice(len(pairs),min(4,len(pairs)),replace=False):
            attempted+=1
            dark=list(pairs[pi]);U=np.concatenate([np.arange(v*p,(v+1)*p) for v in dark]);O=np.setdiff1d(np.arange(ds.n*p),U)
            x=Z[t].reshape(-1);A=D[:,U];b=D[:,O]@x[O];h=np.zeros(2*p);h[[q,p+q]]=.5
                                                                   
            blocks={v:np.arange(j*p,(j+1)*p) for j,v in enumerate(dark)}
            worst={v:worst_case_repair_halfwidth(A,b,h,epsilon,R) for v,R in blocks.items()}
            feasible=all(r['status']!='inconsistent' for r in worst.values())
            if not feasible:excluded+=1;continue
            chosen=min(worst,key=lambda v:worst[v]['halfwidth'])
            C=conditional_covariance(Q,U);trace_choice,_=choose_repair(C,blocks)
            variance_choice=max(blocks,key=lambda v:float(np.diag(C)[blocks[v]].sum()))
            errors={};truth=float(h@x[U]);clean_met=np.linalg.norm(D@x)<=epsilon
            for v,R in blocks.items():
                                                                                
                remain=np.setdiff1d(U,U[R]);observed=np.setdiff1d(np.arange(ds.n*p),remain)
                y=x.copy();y[remain]=-np.linalg.solve(Q[np.ix_(remain,remain)],Q[np.ix_(remain,observed)]@x[observed])
                errors[v]=float((h@y[U]-truth)**2)
                K=np.setdiff1d(np.arange(len(U)),R)
                interval=functional_interval(A[:,K],b+A[:,R]@x[U[R]],h[K],epsilon)
                if interval['status']=='bounded':
                    assert interval['halfwidth']<=worst[v]['halfwidth']+1e-8
                    if clean_met:
                        remaining_truth=float(h[K]@x[U[K]])
                        assert interval['lower']-1e-8<=remaining_truth<=interval['upper']+1e-8
                        theorem_checks+=1
            pattern=f'{dark[0]}-{dark[1]}-{int(t)}'
            for name,value in [('worst_case',errors[chosen]),('gaussian_trace',errors[trace_choice]),('variance_only',errors[variance_choice]),('random_expected',float(np.mean(list(errors.values()))))]:
                records.append({'region':region,'day':str(ds.times[t].date()),'station':'dark-pair model PM aggregate','pattern':pattern,'method':name,'mse':value,
                    'residual_assumption_met':bool(clean_met),'selected_halfwidth':worst[chosen]['halfwidth'],'target_units':'average standardized model PM coordinate, not physical concentration'})
            bound_checks+=1
    configs[region]={'epsilon_validation':epsilon,'sampled_hours':len(hours),'attempted_pairs':attempted,'excluded_inconsistent_pairs':excluded,'model_sha256':hashlib.sha256((folder/'model.npz').read_bytes()).hexdigest()}
R=pd.DataFrame(records);R.to_csv(out/'trials.csv',index=False)
summary={region:{ref:paired_bootstrap(g,'worst_case',ref) for ref in ['gaussian_trace','variance_only','random_expected']} for region,g in R.groupby('region')}
(out/'summary.json').write_text(json.dumps({'comparisons':summary,'config':configs,'candidate_sets_evaluated':bound_checks,'conditional_actual_reading_theorem_checks':theorem_checks,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'scope':'exploratory new policy evaluated on previously inspected raw archives; full-vector subset; infeasible residual cases explicitly excluded; standardized model PM aggregate; common frozen point estimator'},indent=2))
print(json.dumps(summary,indent=2),flush=True)
