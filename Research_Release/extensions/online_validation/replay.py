                                                                                
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
from reference import QuadraticDecoder,FullVAR,complete_training

def raw_delhi():
    path=ROOT/'data/raw/delhi_2025_openaq_cpcb/cpcb_hourly.parquet'
    raw=pd.read_parquet(path);raw['timestamp']=pd.to_datetime(raw['datetime'])
    pollutants=['PM2.5','PM10','NO2','O3']
    raw=raw.rename(columns={'pm2_5':'PM2.5','pm10':'PM10','no2':'NO2','o3':'O3'})
    grouped=raw.groupby(['timestamp','station'])[pollutants].mean()
    times=pd.date_range(raw.timestamp.min(),raw.timestamp.max(),freq='h')
    stations=sorted(raw.station.unique());X=grouped.reindex(pd.MultiIndex.from_product([times,stations])).to_numpy().reshape(len(times),len(stations),4)
    split=np.zeros(len(times),int);split[int(.6*len(times)):int(.8*len(times))]=1;split[int(.8*len(times)):]=2
    tr=X[split==0];eligible=((np.isfinite(tr).mean(0)>=.5)&(np.isfinite(tr).sum(0)>=720)).all(1)
    stations=[s for s,keep in zip(stations,eligible) if keep];X=X[:,eligible]
    meta=raw.groupby('station')[['lat','lon']].first().loc[stations].to_numpy()
    mu=np.nanmean(X[split==0],axis=0);sd=np.nanstd(X[split==0],axis=0,ddof=1)
    return X,times,split,stations,meta,mu,sd,path

def geographic(coords,k=3):
    r=np.radians(coords);lat=r[:,0];lon=r[:,1]
    a=np.sin((lat[:,None]-lat)/2)**2+np.cos(lat[:,None])*np.cos(lat)*np.sin((lon[:,None]-lon)/2)**2
    dist=6371*2*np.arcsin(np.sqrt(np.clip(a,0,1)))
    return sorted({tuple(sorted((u,int(v)))) for u in range(len(coords)) for v in np.argsort(dist[u])[1:k+1]})

def main():
    X,times,split,stations,coords,mu,sd,path=raw_delhi();Z=(X-mu)/sd
    folder=ROOT/'results_expanded/delhi2025';model=np.load(folder/'model.npz');manifest=json.loads((folder/'manifest.json').read_text())
    varpath=ROOT/'extension_results/strong_baselines/delhi2025_choices.json';varsettings=json.loads(varpath.read_text())['causal']
    lock('replay_lock.json',{'raw_sha256':sha(path),'model_sha256':sha(folder/'model.npz'),
         'var_settings_sha256':sha(varpath),'source_sha256':sha(__file__),'reference_sha256':sha(HERE/'reference.py'),
         'protocol_sha256':sha(HERE/'PROTOCOL.md')})
    saved=np.load(folder/'dataset.npz');checks={}
    checks['raw_tensor_matches_saved']=X.shape==saved['X'].shape and np.allclose(X,saved['X'],equal_nan=True,atol=0,rtol=0)
    checks['split_matches_saved']=np.array_equal(split,saved['split'])
    checks['stations_match']=stations==manifest['cohort']['stations']
    checks['times_match']=np.array_equal(times.to_numpy().astype('datetime64[us]').astype('int64'),saved['times'])
    ref=QuadraticDecoder(Z[split==0],geographic(coords),lam=1,temporal_scale=2)
    checks['spatial_gram_matches']=np.allclose(ref.D.T@ref.D,model['L'],atol=1e-9,rtol=1e-9)
    checks['prior_matches']=np.allclose(ref.P,model['P'],atol=1e-9,rtol=1e-9)
    if varsettings['transform']!='none':raise RuntimeError('Independent replay currently requires the frozen raw transform')
    var=FullVAR(complete_training(Z[split==0]),len(stations),4,varsettings['ridge'],varsettings['observation_noise'],varsettings['innovation_shrink'])
    if not all(checks.values()):save('replay_summary.json',{'status':'failed','checks':checks});raise RuntimeError(checks)
    old=pd.read_csv(ROOT/'extension_results/strong_baselines/delhi2025_cases.csv')
    old=old[(old['mode']=='causal')&old.method.isin(['temporal_sheaf','full_covariance_var'])]
    destination=OUT/'replay_cases.csv';done=pd.read_csv(destination) if destination.exists() else pd.DataFrame()
    rows=done.to_dict('records');completed=set(zip(done.day,done.station)) if len(done) else set()
    for (day,station),group in old.groupby(['day','station']):
        if (day,station) in completed:continue
        v=stations.index(station);ix=np.flatnonzero(times.floor('D')==pd.Timestamp(day));whole=np.arange(ix[0]-24,ix[-1]+1)
        truth=Z[ix,v];mask=np.isfinite(truth);hidden=Z[whole].copy();hidden[24:,v]=np.nan
        predictions={'temporal_sheaf':ref.causal(hidden,v),'full_covariance_var':var.filter(hidden)[24:,v]}
        for name,pred in predictions.items():
            mse=float(np.mean((pred-truth)[mask]**2));prior=float(group[group.method==name].mse.iloc[0])
            rows.append({'day':day,'station':station,'method':name,'mse':mse,'saved_mse':prior,'absolute_difference':abs(mse-prior),'observed_cells':int(mask.sum())})
        pd.DataFrame(rows).to_csv(destination,index=False)
        if len(rows)%70==0:print('Independent replay station-days',len(rows)//2,flush=True)
    frame=pd.DataFrame(rows);checks['all_399_cases_replayed']=len(frame)==798
    checks['individual_losses_match']=bool(np.allclose(frame.mse,frame.saved_mse,atol=1e-9,rtol=1e-8))
    result={'status':'passed' if all(checks.values()) else 'failed','checks':checks,'max_case_mse_discrepancy':float(frame.absolute_difference.max()),
            'comparison':paired(frame,'temporal_sheaf','full_covariance_var'),
            'scope':'Second implementation from supplied raw parquet, recorded settings and same instruments; no new data, independent laboratory or priority certification.'}
    save('replay_summary.json',result)
    if result['status']!='passed':raise RuntimeError(result)
    print('Independent raw headline replay passed',result['comparison'],flush=True)

if __name__=='__main__':main()
