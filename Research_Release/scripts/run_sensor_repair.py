                                                                       

                                                                         
                                                                       
                                                                           
                                                                        
   
import hashlib,itertools,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sheafcode.cohort import select_cohort
from sheafcode.reliable import paired_bootstrap
from sheafcode.acquisition import conditional_covariance,choose_repair
records=[];audit=[];configs=[]
for folder in sorted((ROOT/'results').iterdir()):
    if not (folder/'manifest.json').exists():continue
    manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['status']!='complete':continue
    ds=select_cohort(folder.name);ds.set_transform('none');std=ds.std.copy()
    ds.set_transform(manifest['choices']['sheaf']['transform']);Z=ds.Z()
    model=np.load(folder/'model.npz');Q=model['L']+float(model['lam'])*model['P'];p=ds.p
    te=ds.rows(2);hours=te[np.unique(np.linspace(0,len(te)-1,min(240,len(te))).astype(int))]
                                                                            
    pairs=0
    for t in hours:
        eligible=np.flatnonzero(np.isfinite(Z[t]).all(axis=1))
        if len(eligible)<2:continue
                                                                              
        all_pairs=list(itertools.combinations(eligible,2));rng=np.random.default_rng(611+int(t))
        ids=rng.choice(len(all_pairs),min(4,len(all_pairs)),replace=False)
        for i in ids:
            dark=list(all_pairs[i]);row=Z[t].reshape(-1).copy();target=np.concatenate([np.arange(v*p,(v+1)*p) for v in dark])
            row[target]=np.nan;U=np.flatnonzero(~np.isfinite(row));O=np.flatnonzero(np.isfinite(row))
            C=conditional_covariance(Q,U);pos={int(c):j for j,c in enumerate(U)}
            blocks={v:[pos[int(c)] for c in range(v*p,(v+1)*p)] for v in dark}
                                                                            
                                                                           
            w=np.zeros(len(U));w[[pos[int(c)] for c in target]]=1
            selected,gains=choose_repair(C,blocks,w)
            variance_choice=max(blocks,key=lambda v:float(np.diag(C)[blocks[v]].sum()))
                                                                               
            losses={};model_losses={}
            for repair in dark:
                x=row.copy();rr=np.arange(repair*p,(repair+1)*p);x[rr]=Z[t,repair]
                uu=np.flatnonzero(~np.isfinite(x));oo=np.flatnonzero(np.isfinite(x))
                x[uu]=-np.linalg.solve(Q[np.ix_(uu,uu)],Q[np.ix_(uu,oo)]@x[oo])
                raw=ds.raw_from_Z(x.reshape(1,ds.n,p),[t],list(range(ds.n)))[0]
                loss=float(np.mean(((raw[dark]-ds.X[t,dark])/std[dark])**2))
                losses[repair]=loss;model_losses[repair]=float(w@np.diag(C)-gains[repair])
            assert model_losses[selected]<=min(model_losses.values())+1e-9
            pattern=f'{ds.stations[dark[0]]}|{ds.stations[dark[1]]}|{int(t)}'
            for name,value in [('certificate',losses[selected]),('variance_only',losses[variance_choice]),('random_expected',float(np.mean(list(losses.values())))),('outcome_oracle',min(losses.values()))]:
                records.append({'region':folder.name,'day':str(ds.times[t].date()),'station':'two-station set','pattern':pattern,'method':name,'mse':value,
                    'repair':ds.stations[selected] if name=='certificate' else '', 'selection_uses_hidden_values':name=='outcome_oracle'})
            pairs+=1
    audit.append({'region':folder.name,'one_repair_model_optimality_checks':pairs,'passed':True})
    configs.append({'region':folder.name,'hours_sampled':len(hours),'pairs_per_hour_max':4,'model_sha256':hashlib.sha256((folder/'model.npz').read_bytes()).hexdigest()})
    print(folder.name,'repair evaluation complete:',pairs,flush=True)
R=pd.DataFrame(records);R.to_csv(ROOT/'docs/sensor_repair_trials.csv',index=False)
summary={r:{ref:paired_bootstrap(g,'certificate',ref) for ref in ['random_expected','variance_only']} for r,g in R.groupby('region')}
(ROOT/'docs/sensor_repair_summary.json').write_text(json.dumps({'comparisons':summary,'checks':audit,'configs':configs,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'scope':'single repair of two simultaneous artificial dark stations, frozen spatial sheaf; equal model-coordinate weights; evaluation raw training-standardized concentration; no claim of calibrated covariance or field impact'},indent=2))
