                                                                             

                                                                    
                                                                   
   
import hashlib,json,sys,warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore',message='Mean of empty slice',category=RuntimeWarning)
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
from run import completed_training,fit_model
from sheafcode.cohort import select_cohort
from sheafcode.masked import MaskedSheaf
from sheafcode.graph import knn_edges
from sheafcode.data import haversine_km
from sheafcode.benchmark import seasonal_predict
from sheafcode.reliable import CachedSpatial,paired_bootstrap
out=ROOT/'extension_results/horizon_stress';out.mkdir(parents=True,exist_ok=True)
records=[];configs={}
for region in ['delhi2025','bengaluru_raw']:
    core=ROOT/'results_expanded'/region;m=json.loads((core/'manifest.json').read_text());assert m['status']=='complete'
    choices=m['choices'];ds=select_cohort(region);ds.set_transform('none');std=ds.std.copy();s=choices['sheaf'];ds.set_transform(s['transform']);Z=ds.Z()
    tr=ds.rows(0);D=haversine_km(ds.coords);edges=knn_edges(D,s['k']);ts=choices['temporal_sheaf']['causal']['t_scale']
    sh=MaskedSheaf(Z[tr],edges,ds.p,'reg',temporal=True,t_scale=ts,stalk='cca');lam=s['lam']
    spatial=CachedSpatial(sh.L+lam*sh.P,ds.n,ds.p)
    kchoice=json.loads((ROOT/'extension_results/strong_baselines'/(region+'_choices.json')).read_text())['causal']
    ds.set_transform(kchoice['transform']);Zk=ds.Z();X=completed_training(Zk[tr]);kal=fit_model(X,ds.n,ds.p,kchoice['ridge'],kchoice['observation_noise'],kchoice['innovation_shrink'])
    ds.set_transform(s['transform'])
    starts=[]
    for day in pd.unique(ds.times[ds.split==2].floor('D')):
        rows=np.flatnonzero((ds.split==2)&(ds.times.floor('D')==day))
        if len(rows)!=24:continue
        t=int(rows[0])
        if t>=24 and t+72<=len(Z) and ds.split[t-24]==2 and ds.split[t+71]==2:starts.append(t)
    starts=[starts[i] for i in np.unique(np.linspace(0,len(starts)-1,min(12,len(starts))).astype(int))]
    configs[region]={'starts':[str(ds.times[t]) for t in starts],'all_stations':ds.stations,'parameters_frozen_from_24_hour_validation':True,'checkpoint_rule':'hour 1, then each six hours; eligibility >= half of full-outage target cells observed'}
    for start in starts:
        day=str(ds.times[start].date())
        for duration in [24,72]:
            for history in [0,24]:
                ix=np.arange(start-history,start+duration);check=np.unique(np.r_[0,np.arange(5,duration,6)]).astype(int)
                targetrows=np.arange(start,start+duration)[check]
                for v in range(ds.n):
                    truth=ds.X[start:start+duration,v]
                    if np.isfinite(truth).sum()<duration*ds.p/2:continue
                    hide=np.zeros((len(ix),ds.n),bool);hide[history:,v]=True
                    B=Z[ix];pred=np.array([sh.reconstruct_mask(B[:history+c+1],hide[:history+c+1],lam)[-1,v] for c in check])
                    spatialpred=spatial.predict(B[history:],[v])[0][check,0]
                    seasonal=seasonal_predict(B,ds.times[ix],v,history)[check]
                    hidden=Zk[ix].copy();hidden[history:,v]=np.nan
                    kp=kal.smooth(hidden,causal=True)[history+check,v]
                    predictions={'temporal_sheaf':pred,'spatial_sheaf':spatialpred,'seasonal_persistence':seasonal}
                    rawpred={name:ds.raw_from_Z(pr[:,None,:],targetrows,[v])[:,0] for name,pr in predictions.items()}
                    ds.set_transform(kchoice['transform']);rawpred['full_covariance_var']=ds.raw_from_Z(kp[:,None,:],targetrows,[v])[:,0];ds.set_transform(s['transform'])
                    good=np.isfinite(truth[check]);assert all(np.isfinite(pr[good]).all() for pr in rawpred.values())
                    if not good.any():continue
                    for name,pr in rawpred.items():records.append({'region':region,'day':day,'station':ds.stations[v],'history_hours':history,'outage_hours':duration,'method':name,
                        'mse':float(np.mean(((pr-truth[check])/std[v])[good]**2)),'observed_checkpoint_cells':int(good.sum()),'checkpoint_count':len(check)})
        pd.DataFrame(records).to_csv(out/'cases.csv',index=False)
        print(region,day,'24/72-hour and zero/24-hour-history stress complete',flush=True)
R=pd.DataFrame(records);summary={}
for (region,history,duration),g in R.groupby(['region','history_hours','outage_hours']):
    summary[f'{region}:history{history}:outage{duration}']={ref:paired_bootstrap(g,'temporal_sheaf',ref) for ref in ['full_covariance_var','spatial_sheaf','seasonal_persistence']}
(out/'summary.json').write_text(json.dumps({'comparisons':summary,'config':configs,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'limitations':['checkpoint benchmark, not all-hour RMSE','24-hour-tuned parameters deliberately transferred to 72 hours and zero-history cases','internal exploratory archive test, not blinded validation','shared starting days are paired cases, not independent station replications']},indent=2))
print('Horizon stress completed',flush=True)
