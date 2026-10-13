                                                                         
import io,zipfile,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
from reference import QuadraticDecoder,FullVAR,complete_training,graph_correlation,seasonal

def load_archive(path):
    outer=zipfile.ZipFile(path);inner=zipfile.ZipFile(io.BytesIO(outer.read('PRSA2017_Data_20130301-20170228.zip')))
    frames=[]
    for name in sorted(inner.namelist()):
        if not name.endswith('.csv'):continue
        f=pd.read_csv(inner.open(name));f['timestamp']=pd.to_datetime(f[['year','month','day','hour']]);frames.append(f)
    raw=pd.concat(frames);pollutants=['PM2.5','PM10','NO2','O3'];stations=sorted(raw.station.unique())
    times=pd.date_range(raw.timestamp.min(),raw.timestamp.max(),freq='h')
    grouped=raw.groupby(['timestamp','station'])[pollutants].mean()
    X=grouped.reindex(pd.MultiIndex.from_product([times,stations])).to_numpy().reshape(len(times),len(stations),4)
    X[X<0]=np.nan;split=np.zeros(len(times),int);split[int(.6*len(times)):int(.8*len(times))]=1;split[int(.8*len(times)):]=2
    full=X[split==0];keep=((np.isfinite(full).mean(0)>=.5)&(np.isfinite(full).sum(0)>=720)).all(1)
    excluded=[s for s,k in zip(stations,keep) if not k];stations=[s for s,k in zip(stations,keep) if k];X=X[:,keep]
    if len(stations)<3:raise RuntimeError('Insufficient training-eligible stations')
    mu=np.nanmean(X[split==0],0);sd=np.nanstd(X[split==0],0,ddof=1)
    return X,times,split,stations,excluded,mu,sd

def main():
    path=OUT/'beijing_original.zip'
    acquired=json.loads((OUT/'ACQUISITION_LOCK.json').read_text())
    if acquired['protocol_sha256']!=sha(HERE/'PROTOCOL.md'):raise RuntimeError('Acquisition protocol changed')
    lock('beijing_lock.json',{'raw_sha256':sha(path),'source_sha256':sha(__file__),
         'reference_sha256':sha(HERE/'reference.py'),'protocol_sha256':sha(HERE/'PROTOCOL.md'),
         'split':[.6,.2,.2],'channels':['PM2.5','PM10','NO2','O3'],'test_day_stride':7,
         'sheaf':{'k':3,'lam':1,'t_scale':2,'graph':'training mean absolute pollutant correlation'}})
    X,times,split,stations,excluded,mu,sd=load_archive(path);Z=(X-mu)/sd;n=len(stations)
    edges=graph_correlation(Z[split==0]);sheaf=QuadraticDecoder(Z[split==0],edges)
    training=complete_training(Z[split==0]);allval=day_blocks(times,split,1)
    val=[allval[i] for i in np.unique(np.linspace(0,len(allval)-1,8).astype(int))]
    valcases=[]
    for day,ix in val:
        for v in range(n):
            truth=Z[ix[24:],v];good=np.isfinite(truth)
            if good.sum()>=48:valcases.append((day,v,ix,good))
    choices_path=OUT/'beijing_choices.json'
    if choices_path.exists():choice=json.loads(choices_path.read_text())
    else:
        validation=[]
        for ridge in [1.,100.]:
            for noise in [.05,.5]:
                for shrink in [0.,.2,1.]:
                    model=FullVAR(training,n,4,ridge,noise,shrink);loss=[]
                    for day,v,ix,good in valcases:
                        hidden=Z[ix].copy();hidden[24:,v]=np.nan;pred=model.filter(hidden)[24:,v]
                        loss.append(np.mean((pred-Z[ix[24:],v])[good]**2))
                    validation.append({'ridge':ridge,'noise':noise,'shrink':shrink,'validation_mse':float(np.mean(loss))})
                    print('Beijing validation',validation[-1],flush=True)
        pd.DataFrame(validation).to_csv(OUT/'beijing_validation_grid.csv',index=False)
        choice=min(validation,key=lambda x:x['validation_mse']);save('beijing_choices.json',choice)
    var=FullVAR(training,n,4,choice['ridge'],choice['noise'],choice['shrink'])
    destination=OUT/'beijing_cases.csv';done=pd.read_csv(destination) if destination.exists() else pd.DataFrame()
    rows=done.to_dict('records');completed=set(zip(done.day,done.station)) if len(done) else set();skipped=[]
    test=day_blocks(times,split,2,stride=7)
    for day,ix in test:
        for v,station in enumerate(stations):
            if (day,station) in completed:continue
            truth=Z[ix[24:],v];good=np.isfinite(truth)
            if good.sum()<48:skipped.append({'day':day,'station':station,'observed_cells':int(good.sum())});continue
            hidden=Z[ix].copy();hidden[24:,v]=np.nan
            others=np.delete(hidden[24:],v,axis=1);counts=np.isfinite(others).sum(1)
            city=np.divide(np.nansum(others,axis=1),counts,out=np.zeros((24,4)),where=counts>0)
            predictions={'temporal_sheaf':sheaf.causal(hidden,v),'full_covariance_var':var.filter(hidden)[24:,v],
                         'seasonal':seasonal(hidden,v),'city_mean':city}
            for method,pred in predictions.items():
                if not np.isfinite(pred).all():raise RuntimeError('Invalid prediction')
                qgood=good[:,0];raw_error=(pred[:,0]-truth[:,0])*sd[v,0]
                rows.append({'day':day,'station':station,'method':method,'mse':float(np.mean((pred-truth)[good]**2)),
                             'pm25_raw_mse':float(np.mean(raw_error[qgood]**2)) if qgood.any() else np.nan,
                             'observed_cells':int(good.sum()),'pm25_cells':int(qgood.sum())})
            pd.DataFrame(rows).to_csv(destination,index=False)
        print('Beijing test completed day',day,'station-days',len(rows)//4,flush=True)
    frame=pd.DataFrame(rows);pd.DataFrame(skipped).to_csv(OUT/'beijing_excluded_cases.csv',index=False)
    result={'status':'completed','stations':stations,'excluded_stations':excluded,'edges':edges,'period':[str(times.min()),str(times.max())],
            'training_hours':int((split==0).sum()),'validation_hours':int((split==1).sum()),'test_hours':int((split==2).sum()),
            'test_days_planned':len(test),'cases':len(frame)//4,'validation_choices':choice,
            'comparisons':{name:paired(frame,'temporal_sheaf',name) for name in ['full_covariance_var','seasonal','city_mean']},
            'pm25_raw_comparisons':{name:paired(frame.assign(mse=frame.pm25_raw_mse),'temporal_sheaf',name) for name in ['full_covariance_var','seasonal','city_mean']},
            'source_url':'https://archive.ics.uci.edu/dataset/501/beijing','dataset_doi':'10.24432/C5RK5G','license':'CC BY 4.0',
            'scope':'Protocol-fixed additional-country network; newly trained CCA/correlation graph on Beijing training only. Not zero-shot Delhi maps or a previously unseen dataset priority claim. No test-driven tuning. Prior user access unverifiable.'}
    save('beijing_summary.json',result);print('Beijing completed',result['comparisons'],flush=True)

if __name__=='__main__':main()
