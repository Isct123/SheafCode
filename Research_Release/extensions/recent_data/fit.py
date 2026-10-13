import sys,time,pickle,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
def prepare():
    f=pd.read_csv(DATA/'observations.csv',dtype={'station':str},parse_dates=['timestamp']);times=pd.date_range('2026-01-01','2026-10-05 23:00',freq='h');split=np.zeros(len(times),int);split[int(.6*len(times)):int(.8*len(times))]=1;split[int(.8*len(times)):]=2
                                                                                       
    cut=int(.7*len(times));split[(np.arange(len(times))>=cut)&(split==1)]=3
    mapping=json.loads((DATA/'station_mapping.json').read_text());eligible=[];excluded={}
    for region in mapping:
        ff=f[f.region==region];stations=[str(v['location_id']) for v in mapping[region]];pols=['PM2.5','PM10','O3','NO2_ppb']
        if len(stations)<3:excluded[region]='Fewer than three metadata-eligible stations';continue
        index=pd.MultiIndex.from_product([times,stations],names=['timestamp','station']);table=ff.pivot(index=['timestamp','station'],columns='pollutant',values='value').reindex(index).reindex(columns=pols)
        full=table.to_numpy().reshape(len(times),len(stations),4);tr=full[split==0];valid=(np.isfinite(tr).mean(0)>=.5)&(np.isfinite(tr).sum(0)>=720);choices=[]
        for size in range(2,5):
            for cols in itertools.combinations(range(4),size):
                if 0 not in cols:continue
                keep=valid[:,cols].all(1)
                if keep.sum()>=3:choices.append((int(keep.sum()*size),size,tuple(cols),keep))
        if not choices:excluded[region]='No training-only subset with three stations and two channels';continue
        _,_,cols,keep=sorted(choices,key=lambda a:(-a[0],-a[1],a[2]))[0];ss=np.array(stations)[keep];X=full[:,keep][:,:,cols];mu=np.nanmean(X[split==0],0);sd=np.nanstd(X[split==0],0,ddof=1)
        if np.any(sd<1e-8):excluded[region]='Degenerate training channel';continue
        folder=OUT/region;folder.mkdir(exist_ok=True);np.savez_compressed(folder/'dataset.npz',X=X,Z=(X-mu)/sd,mu=mu,sd=sd,split=split,times=times.to_numpy().astype('datetime64[ns]'),stations=ss,channels=np.array(pols)[list(cols)])
        manifest={'region':region,'stations':ss.tolist(),'channels':np.array(pols)[list(cols)].tolist(),'units':['ppb' if pols[c]=='NO2_ppb' else 'ug/m3' for c in cols],'dates':['2026-01-01','2026-10-05'],'split_counts':{str(i):int((split==i).sum()) for i in [0,1,3,2]},'training_only_selection':True,'excluded_station_ids':np.array(stations)[~keep].tolist(),'minimum_training_hours':720,'observations_sha256':sha(DATA/'observations.csv'),'protocol_sha256':sha(HERE/'PROTOCOL.md'),'availability_amendment_sha256':sha(HERE/'AVAILABILITY_AMENDMENT.md')}
        save(region+'/dataset_manifest.json',manifest);eligible.append(region)
    save('cohorts.json',{'eligible':eligible,'excluded':excluded})
    if not eligible:raise RuntimeError('No recent eligible network; acquisition cannot support proposed task')
def main():
    prepare()
    for region in regions():
        ds=load(region);Z=ds['Z'];tr=Z[ds['split']==0];n,p=tr.shape[1:];edges=graph_correlation(tr);lock(region+'/fit_lock.json',{'source_sha256':sha(__file__),'common_sha256':sha(HERE/'common.py'),'reference_sha256':sha(ROOT/'extensions/online_validation/reference.py'),'dataset_sha256':sha(OUT/region/'dataset.npz'),'choices':'fixed grids, first validation half only'})
        if (OUT/region/'models.pkl').exists():print(region,'models already complete',flush=True);continue
        val=[]
        for day,ix in subset_days(blocks(ds,1),12):
            for v in range(n):
                good=np.isfinite(Z[ix[24:],v])
                if good.sum()>=12*p:val.append((day,ix,v,good))
        if not val:raise RuntimeError(region+' no model selection cases')
        trials=[];best=None
        for lam,ts in itertools.product([.1,1.],[.5,2.]):
            model=QuadraticDecoder(tr,edges,lam,ts);scores=[]
            for day,ix,v,good in val:
                x=Z[ix].copy();x[24:,v]=np.nan;pred=model.causal(x,v);scores.append(float(np.mean((pred-Z[ix[24:],v])[good]**2)))
            item={'method':'sheaf','lam':lam,'temporal_scale':ts,'validation_mse':float(np.mean(scores))};trials.append(item)
            if best is None or item['validation_mse']<best[0]['validation_mse']:best=(item,model)
            print(region,'sheaf choice',item,flush=True);frame_save(region+'/validation_grid.csv',pd.DataFrame(trials))
        training=complete_training(tr);vbest=None
        for ridge,noise,shrink in itertools.product([1.,100.],[.05,.5],[0.,.2,1.]):
            model=FullVAR(training,n,p,ridge,noise,shrink);scores=[]
            for day,ix,v,good in val:
                x=Z[ix].copy();x[24:,v]=np.nan;pred=model.filter(x)[24:,v];scores.append(float(np.mean((pred-Z[ix[24:],v])[good]**2)))
            item={'method':'var','ridge':ridge,'noise':noise,'shrink':shrink,'validation_mse':float(np.mean(scores))};trials.append(item)
            if vbest is None or item['validation_mse']<vbest[0]['validation_mse']:vbest=(item,model)
            print(region,'VAR choice',item,flush=True);frame_save(region+'/validation_grid.csv',pd.DataFrame(trials))
        covariance=np.cov(training,rowvar=False)+.001*np.eye(n*p);gaussian=np.linalg.inv(covariance);sheaf=best[1];sheaf.precisions={}
        fitted={'sheaf':sheaf,'var':vbest[1],'gaussian':gaussian,'edges':edges};temp=OUT/region/'models.tmp'
        with temp.open('wb') as h:pickle.dump(fitted,h)
        temp.replace(OUT/region/'models.pkl');save(region+'/model_choices.json',{'sheaf':best[0],'var':vbest[0],'validation_cases':len(val),'edges':edges,'model_sha256':sha(OUT/region/'models.pkl'),'scope':'Both point models selected using first validation half; second half untouched for calibration'})
        print(region,'model fitting complete',flush=True)
if __name__=='__main__':main()
