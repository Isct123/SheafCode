                                                                              

                                                                           
                                                                   
   
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from sheafcode.cohort import select_cohort
from sheafcode.reliable import CachedSpatial

def main():
    out=ROOT/'followup_results/external';out.mkdir(parents=True,exist_ok=True)
    inputs=ROOT/'external_validation';plan=inputs/'provenance.json';data=inputs/'observations.csv'
    if not plan.exists() or not data.exists():
        (out/'status.json').write_text(json.dumps({'status':'awaiting_external_evidence','required':['external_validation/provenance.json','external_validation/observations.csv'],
            'reason':'No independent data are supplied. This stage cannot establish external validity.'},indent=2));return
    provenance=json.loads(plan.read_text());region=provenance['region']
    if region not in ['delhi2025','bengaluru_raw']:raise ValueError('Supported raw cohorts only')
    if not provenance.get('source_url') or not provenance.get('acquired_at') or provenance.get('previously_used_for_development') is not False:
        raise ValueError('Document source, acquisition time and previous development use')
    folder=ROOT/'results_expanded'/region;m=json.loads((folder/'manifest.json').read_text());ds=select_cohort(region)
                                                                
    lock={'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'model_sha256':hashlib.sha256((folder/'model.npz').read_bytes()).hexdigest(),
        'provenance':provenance,'human_source_verification_required':True}
    (out/'evaluation_lock.json').write_text(json.dumps(lock,indent=2))
    frame=pd.read_csv(data);required={'timestamp','station','pollutant','value'}
    if not required.issubset(frame.columns):raise ValueError('Long format requires '+str(required))
    frame['timestamp']=pd.to_datetime(frame.timestamp)
    if frame.timestamp.dt.tz is not None:raise ValueError('Use timezone-naive local timestamps consistent with cohort')
    if ((frame.timestamp>=ds.times.min())&(frame.timestamp<=ds.times.max())).any():raise ValueError('External period overlaps examined archive')
    if frame.duplicated(['timestamp','station','pollutant']).any():raise ValueError('Duplicate observations')
    wide=frame.pivot(index='timestamp',columns=['station','pollutant'],values='value').sort_index()
    cols=pd.MultiIndex.from_product([ds.stations,ds.pollutants]);X=wide.reindex(columns=cols).to_numpy().reshape(-1,ds.n,ds.p)
    ds.set_transform(m['choices']['sheaf']['transform']);model=np.load(folder/'model.npz')
    transformed=ds._fwd(X)-ds.mean
    if ds.hod is not None:transformed-=ds.hod[wide.index.hour]
    Z=transformed/ds.std;decoder=CachedSpatial(model['L']+model['lam']*model['P'],ds.n,ds.p)
    records=[]
    for v in range(ds.n):
        pred=decoder.predict(Z,[v])[0][:,0]*ds.std[v]+ds.mean[v]
        if ds.hod is not None:pred+=ds.hod[wide.index.hour,v]
        pred=ds._inv(pred)
        for t in range(len(X)):
            good=np.isfinite(X[t,v])
            if not good.any():continue
            others=X[t,:, :].copy();others[v]=np.nan
            fallback=np.nanmean(others,axis=0);common=good&np.isfinite(fallback)
            if common.any():records.append({'timestamp':str(wide.index[t]),'station':ds.stations[v],'cells':int(common.sum()),
                'sheaf_mse':float(np.mean(((pred[t]-X[t,v])/ds.std[v])[common]**2)),
                'city_mean_mse':float(np.mean(((fallback-X[t,v])/ds.std[v])[common]**2))})
    if not records:raise ValueError('No eligible external comparisons')
    pd.DataFrame(records).to_csv(out/'cases.csv',index=False)
    (out/'status.json').write_text(json.dumps({'status':'evaluated','cases':len(records),
        'scope':'Fixed model, artificial single-station outages on a non-overlapping supplied period; source independence requires verification; no environmental intervention'},indent=2))

if __name__=='__main__':main()
