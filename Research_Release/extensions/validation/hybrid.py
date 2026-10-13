                                                                                  
import time
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from common import *
from sheafcode.masked import MaskedSheaf
from sheafcode.benchmark import seasonal_predict
sys.path.insert(0,str(ROOT/'extensions/strong_baselines'))
from run import completed_training,fit_model

def main():
    started=time.time();rows=[];hourly=[];settings={}
    for region in REGIONS:
        ds,m,folder=cohort(region);ds.set_transform('none');std=ds.std.copy()
        full=json.loads((ROOT/'extension_results/strong_baselines'/f'{region}_choices.json').read_text())['causal']
        ds.set_transform(full['transform']);Zvar=ds.Z().copy();var=fit_model(completed_training(Zvar[ds.rows(0)]),ds.n,ds.p,full['ridge'],full['observation_noise'],full['innovation_shrink'])
        ds.set_transform(m['choices']['sheaf']['transform']);Z=ds.Z();model=np.load(folder/'model.npz')
        from sheafcode.data import haversine_km
        from sheafcode.graph import knn_edges
        edges=knn_edges(haversine_km(ds.coords),m['choices']['sheaf']['k'])
        sh=MaskedSheaf(Z[ds.rows(0)],edges,ds.p,'reg',temporal=True,t_scale=m['choices']['temporal_sheaf']['causal']['t_scale'],stalk='cca')
        def predictions(ix,v):
            hidden=Z[ix].copy();hidden[24:,v]=np.nan;mask=np.zeros((48,ds.n),bool);mask[24:,v]=True
            s=np.array([sh.reconstruct_mask(hidden[:h+1],mask[:h+1],float(model['lam']))[-1,v] for h in range(24,48)])
            sheaf=ds.raw_from_Z(s[:,None],ix[24:],[v])[:,0]
            B=Zvar[ix].copy();B[24:,v]=np.nan
            ds.set_transform(full['transform']);vv=ds.raw_from_Z(var.smooth(B,causal=True)[24:,v][:,None],ix[24:],[v])[:,0]
            ds.set_transform(m['choices']['sheaf']['transform'])
            seasonal=ds.raw_from_Z(seasonal_predict(hidden,ds.times[ix],v,24)[:,None],ix[24:],[v])[:,0]
            return np.stack([sheaf,vv,seasonal],axis=0)
        val=[]
        for day,ix in dayblocks(ds,1,8):
            for v in range(ds.n):
                truth=ds.X[ix[24:],v];ok=np.isfinite(truth)
                if ok.sum()<12*ds.p:continue
                val.append((predictions(ix,v),truth,ok,v))
        weights=[np.array([a,b,10-a-b])/10 for a in range(11) for b in range(11-a)]
        losses=[np.mean([np.mean(((np.einsum('m,mhp->hp',w,pr)-truth)/std[v])[ok]**2) for pr,truth,ok,v in val]) for w in weights]
        w=weights[int(np.argmin(losses))];settings[region]={'weights':w.tolist(),'components':['temporal_sheaf','full_covariance_var','seasonal'],'validation_cases':len(val),'validation_mse':float(min(losses)),'model_sha256':digest(folder/'model.npz')}
        save('hybrid_choices.json',settings)
        allowed=set(map(tuple,pd.read_csv(folder/'prolonged.csv')[['day','station']].drop_duplicates().astype(str).values))
        for day,ix in dayblocks(ds,2,None):
            for v,station in enumerate(ds.stations):
                if (day,station) not in allowed:continue
                truth=ds.X[ix[24:],v];ok=np.isfinite(truth);pr=predictions(ix,v)
                for name,pred in zip(settings[region]['components']+['hybrid'],list(pr)+[np.einsum('m,mhp->hp',w,pr)]):
                    if not np.isfinite(pred[ok]).all():raise RuntimeError('Nonfinite prediction')
                    rows.append({'region':region,'day':day,'station':station,'method':name,'mse':float(np.mean(((pred-truth)/std[v])[ok]**2))})
                    for h,t in enumerate(ix[24:]):
                        for q,pollutant in enumerate(ds.pollutants):
                            if ok[h,q]:hourly.append({'region':region,'day':day,'timestamp':str(ds.times[t]),'station':station,'method':name,'pollutant':pollutant,'truth':truth[h,q],'prediction':pred[h,q],'outage_hour':h+1})
            print(region,'hybrid',day,round(time.time()-started,1),'seconds',flush=True)
    f=pd.DataFrame(rows);f.to_csv(OUT/'hybrid_cases.csv',index=False);pd.DataFrame(hourly).to_csv(OUT/'hybrid_hourly.csv',index=False)
    summary={r:{method:paired_interval(f[f.region==r],'hybrid',method) for method in ['temporal_sheaf','full_covariance_var','seasonal']} for r in REGIONS}
    save('hybrid_summary.json',{'comparisons':summary,'source_sha256':digest(__file__),'scope':'Internal exploratory causal 24h outage comparisons. Blend selected on validation only; examined test archives; fixed legacy component settings.'})
if __name__=='__main__':main()
