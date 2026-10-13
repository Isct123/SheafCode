                                                                          
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
from sheafcode.reliable import CachedSpatial
from sheafcode.masked import MaskedSheaf
from sheafcode.data import haversine_km
from sheafcode.graph import knn_edges
from sheafcode.benchmark import seasonal_predict
sys.path.insert(0,str(ROOT/'extensions/strong_baselines'))
from run import completed_training,fit_model
def main():
    folder=OUT/'external';source=folder/'observations.csv'
    if not source.exists():save('external_summary.json',{'status':'awaiting_compatible_observations'});return
    ds,m,modeldir=cohort('delhi2025');weights=json.loads((OUT/'hybrid_choices.json').read_text())['delhi2025']['weights']
    fullpath=ROOT/'extension_results/strong_baselines/delhi2025_choices.json';full=json.loads(fullpath.read_text())['causal']
    lock={'data_sha256':digest(source),'model_sha256':digest(modeldir/'model.npz'),'full_covariance_choices_sha256':digest(fullpath),'hybrid_choices_sha256':digest(OUT/'hybrid_choices.json'),'source_sha256':digest(__file__),'input_policy':'PM2.5 and PM10 mass concentrations only; gas channels unavailable pending archived unit provenance reconciliation','minimum_surviving_pm_stations':2,'period_selection':'Entire acquired non-overlapping period; no outcome-based truncation'}
    lockpath=folder/'evaluation_lock.json'
    if lockpath.exists() and json.loads(lockpath.read_text())!=lock:raise RuntimeError('External evaluation lock changed; preserve original evaluation')
    lockpath.write_text(json.dumps(lock,indent=2))
    f=pd.read_csv(source);f['timestamp']=pd.to_datetime(f.timestamp)
    if f.timestamp.min()<=ds.times.max():raise RuntimeError('Overlap with examined archive')
    f=f[f.pollutant.isin(['PM2.5','PM10'])];wide=f.pivot(index='timestamp',columns=['station','pollutant'],values='value').sort_index()
    times=pd.date_range(wide.index.min(),wide.index.max(),freq='h');cols=pd.MultiIndex.from_product([ds.stations,ds.pollutants]);X=wide.reindex(index=times,columns=cols).to_numpy().reshape(-1,ds.n,ds.p)
    q=ds.pollutants.index('PM2.5');ds.set_transform('none');std=ds.std.copy()
    ds.set_transform(full['transform']);var=fit_model(completed_training(ds.Z()[ds.rows(0)]),ds.n,ds.p,full['ridge'],full['observation_noise'],full['innovation_shrink'])
    def standardized(transform):
        ds.set_transform(transform);Y=ds._fwd(X)-ds.mean
        if ds.hod is not None:Y-=ds.hod[times.hour]
        return Y/ds.std
    Zvar=standardized(full['transform']);choice=m['choices']['sheaf'];Z=standardized(choice['transform']);model=np.load(modeldir/'model.npz');decoder=CachedSpatial(model['L']+model['lam']*model['P'],ds.n,ds.p)
    sh=MaskedSheaf(ds.Z()[ds.rows(0)],knn_edges(haversine_km(ds.coords),choice['k']),ds.p,'reg',temporal=True,t_scale=m['choices']['temporal_sheaf']['causal']['t_scale'],stalk='cca')
    def raw(pred,ix,v,transform):
        ds.set_transform(transform);Y=pred*ds.std[v]+ds.mean[v]
        if ds.hod is not None:Y+=ds.hod[times[ix].hour,v]
        return ds._inv(Y)
    records=[];hourly=[]
    for day in pd.unique(times.floor('D')):
        ix=np.flatnonzero(times.floor('D')==day)
        if len(ix)!=24 or ix[0]<24:continue
        whole=np.arange(ix[0]-24,ix[-1]+1)
        for v,station in enumerate(ds.stations):
            truth=X[ix,v,q];valid=np.isfinite(truth)
            survive=np.isfinite(X[ix,:,q]).sum(1)-valid.astype(int);good=valid&(survive>=2)
            if good.sum()<12:continue
            hidden=Z[whole].copy();hidden[24:,v]=np.nan;mask=np.zeros((48,ds.n),bool);mask[24:,v]=True
            pred=np.array([sh.reconstruct_mask(hidden[:h+1],mask[:h+1],float(model['lam']))[-1,v] for h in range(24,48)])
            s=raw(pred,ix,v,choice['transform']);B=Zvar[whole].copy();B[24:,v]=np.nan;vv=raw(var.smooth(B,causal=True)[24:,v],ix,v,full['transform'])
            seasonal=raw(seasonal_predict(hidden,times[whole],v,24),ix,v,choice['transform']);ds.set_transform(choice['transform'])
            snapshot=raw(decoder.predict(hidden[24:],[v])[0][:,0],ix,v,choice['transform']);others=X[ix,:,q].copy();others[:,v]=np.nan
            counts=np.isfinite(others).sum(1);city=np.divide(np.nansum(others,axis=1),counts,out=np.full(24,np.nan),where=counts>0)
            preds={'temporal_sheaf':s[:,q],'full_covariance_var':vv[:,q],'seasonal':seasonal[:,q],'hybrid':weights[0]*s[:,q]+weights[1]*vv[:,q]+weights[2]*seasonal[:,q],'spatial_sheaf':snapshot[:,q],'city_mean':city}
            for name,estimate in preds.items():
                if not np.isfinite(estimate[good]).all():raise RuntimeError('Nonfinite external prediction')
                records.append({'day':str(day.date()),'station':station,'method':name,'mse':float(np.mean(((estimate-truth)/std[v,q])[good]**2)),'raw_mse':float(np.mean((estimate-truth)[good]**2)),'cells':int(good.sum())})
                for j,t in enumerate(ix):
                    if good[j]:hourly.append({'timestamp':str(times[t]),'station':station,'method':name,'truth':truth[j],'prediction':estimate[j],'outage_hour':j+1})
    if not records:save('external_summary.json',{'status':'insufficient_compatible_context','acquired_stations':f.station.nunique()});return
    cases=pd.DataFrame(records);cases.to_csv(folder/'cases.csv',index=False);pd.DataFrame(hourly).to_csv(folder/'hourly.csv',index=False)
    comparisons={name:paired_interval(cases,'temporal_sheaf',name) for name in ['full_covariance_var','seasonal','city_mean']}
    save('external_summary.json',{'status':'evaluated','comparison':comparisons,'cases':len(cases)//6,'days':cases.day.nunique(),'stations':cases.station.nunique(),'period':[str(times.min()),str(times.max())],'normalized_unit':'Training PM2.5 standard deviation; each station/day weighted equally','scope':'Newly acquired non-overlapping same-network period; prior user access unverifiable. PM-only inputs, all methods same context and history. No retuning. Short period and partial network; same instruments, no independent-instrument replication or field intervention. Calendar-day intervals exploratory.'})
                                                                               
                                                     
    choices=json.loads((OUT/'alert_choices.json').read_text())['delhi2025_24_0.0']['cutoffs'];a=pd.DataFrame(hourly);alarm=[]
    for name,method in [('spatial_sheaf','sheaf'),('city_mean','city_mean')]:
        g=a[a.method==name];positive=g.truth>60;alert=g.prediction>choices[method]
        alarm.append({'method':name,'cases':len(g),'positive_cases':int(positive.sum()),'negative_cases':int((~positive).sum()),'sensitivity':float(alert[positive].mean()) if positive.any() else None,'fpr':float(alert[~positive].mean()) if (~positive).any() else None,'cutoff':choices[method]})
    pd.DataFrame(alarm).to_csv(folder/'frozen_alert_summary.csv',index=False)
    print('External frozen evaluation complete',comparisons,flush=True)
if __name__=='__main__':main()
