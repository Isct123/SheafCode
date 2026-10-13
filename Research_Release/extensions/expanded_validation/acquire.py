                                                                             
from pathlib import Path
import sys,json,hashlib,subprocess,zipfile,shutil,itertools,concurrent.futures
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=ROOT/'Research_Release/expanded_validation_results';RAW=OUT/'raw'
sys.path.insert(0,str(ROOT/'audit_tools/uk_packages'))

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2),encoding='utf-8')
def download(name,url,cache=None):
    p=RAW/name
    if not p.exists():
        if cache and cache.exists():shutil.copy2(cache,p)
        else:
            tmp=p.with_suffix('.download');subprocess.run(['curl.exe','--fail','--location','--silent','--show-error','--max-time','900','--retry','2','--continue-at','-','--output',str(tmp),url],check=True);tmp.replace(p)
    return p,{'file':str(p.relative_to(ROOT)),'url':url,'sha256':sha(p),'bytes':p.stat().st_size}
def timeline(start,end):
    times=pd.date_range(start,end+' 23:00',freq='h');ids=np.arange(len(times));split=np.select([ids<int(.6*len(times)),ids<int(.7*len(times)),ids<int(.8*len(times))],[0,1,3],default=2);return times,split
def qualified(x,split):
    t=x[split==0];return bool((np.isfinite(t).mean(0)>=.5).all() and (np.isfinite(t).sum(0)>=720).all())
def store(name,X,times,split,stations,channels,units,sources,details):
    folder=OUT/name;folder.mkdir(exist_ok=True);X=np.array(X,dtype=float,copy=True);X[~np.isfinite(X)]=np.nan;X[X<0]=np.nan
    tr=X[split==0];mu=np.nanmean(tr,0);sd=np.nanstd(tr,0,ddof=1)
    if not np.isfinite(sd).all() or (sd<1e-8).any():raise RuntimeError('Degenerate training channels: '+name)
    np.savez_compressed(folder/'dataset.npz',X=X,Z=(X-mu)/sd,mu=mu,sd=sd,split=split,times=times.to_numpy(),stations=np.array(stations,dtype=str),channels=np.array(channels))
    save(name+'/dataset_manifest.json',{'region':name,'stations':list(map(str,stations)),'channels':channels,'units':units,'dates':[str(times[0]),str(times[-1])],'sources':sources,'training_only_selection':True,'split_counts':{str(i):int((split==i).sum()) for i in [0,1,3,2]},'protocol_sha256':sha(HERE/'PROTOCOL.md'),**details})
    rows=[]
    for v,s in enumerate(stations):
        for j,c in enumerate(channels):
            for t in np.flatnonzero(np.isfinite(X[:,v,j])):rows.append({'timestamp':str(times[t]),'station':str(s),'pollutant':c,'value':float(X[t,v,j]),'unit':units[j]})
    pd.DataFrame(rows).to_csv(folder/'source_tensor_observations.csv',index=False)
    return name

def epa():
    sources=[];jobs=[]
    for year,code in itertools.product([2025,2026],[88101,88502,44201]):
        name=f'hourly_{code}_{year}.zip';cache=ROOT/'Research_Release/recent_data_results/epa_la/raw'/name
        jobs.append((name,'https://aqs.epa.gov/aqsweb/airdata/'+name,cache))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        acquired=list(pool.map(lambda job:download(*job),jobs))
    frames=[];cols=['State Code','County Code','Site Num','POC','Date GMT','Time GMT','Sample Measurement','Units of Measure','Latitude','Longitude','Qualifier']
    for (path,source),(_,_,_) in zip(acquired,jobs):
        sources.append(source);code=int(path.stem.split('_')[1])
        with zipfile.ZipFile(path) as z:
            for filename in z.namelist():
                if not filename.endswith('.csv'):continue
                for f in pd.read_csv(z.open(filename),usecols=cols,chunksize=200000,low_memory=False):
                    f=f[f['Date GMT'].between('2025-10-14','2026-03-31')].copy()
                    if len(f):
                        f['timestamp']=pd.to_datetime(f['Date GMT']+' '+f['Time GMT']);f['parameter']=code
                        frames.append(f.drop(columns=['Date GMT','Time GMT']))
        print('EPA national source parsed',path.name,flush=True)
    f=pd.concat(frames,ignore_index=True);del frames
    times,split=timeline('2025-10-14','2026-03-31');tr_end=times[split==0][-1];selected=[];excluded=[]
    for (state,county),g in f.groupby(['State Code','County Code'],sort=True):
        if (int(state),int(county))==(6,37):continue
        chosen=[];arrays=[];monitors=[];qualifier_counts=g.Qualifier.value_counts(dropna=False).to_dict()
        for site,gg in g.groupby('Site Num',sort=True):
            vectors=[];identities=[]
            for codes,unit in [([88101,88502],'Micrograms/cubic meter (LC)'),([44201],'Parts per million')]:
                possibilities=[]
                for (code,poc),hh in gg[gg.parameter.isin(codes)&gg['Units of Measure'].eq(unit)&gg['Sample Measurement'].ge(0)].groupby(['parameter','POC'],sort=True):
                                                                                                                   
                    s=hh.groupby('timestamp')['Sample Measurement'].mean().reindex(times);train=s.loc[:tr_end]
                    if train.notna().mean()>=.5 and train.notna().sum()>=720:possibilities.append((int(code),int(poc),s,int(hh.timestamp.duplicated().sum())))
                if not possibilities:break
                code,poc,s,duplicates=min(possibilities,key=lambda x:(x[0],x[1]));vectors.append(s.to_numpy());identities.append({'parameter':code,'POC':poc,'duplicate_same_monitor_rows_averaged':duplicates})
            if len(vectors)==2:chosen.append(str(int(site)));arrays.append(np.stack(vectors,axis=1));monitors.append(identities)
        if len(chosen)<3:
            if len(chosen):excluded.append({'source':'EPA','state':int(state),'county':int(county),'eligible_stations':len(chosen),'reason':'Fewer than three training-eligible colocated two-channel sites'})
            continue
        if len(selected)>=12:continue
        name=f'epa_{int(state):02d}_{int(county):03d}'
        g.to_csv(OUT/(name+'_raw_observations.csv'),index=False)
        name=store(name,np.stack(arrays[:4],axis=1),times,split,chosen[:4],['PM2.5','O3_ppm'],['ug/m3','ppm'],sources,{'provider':'EPA','state_code':int(state),'county_code':int(county),'all_training_eligible_sites':chosen,'chosen_monitors':monitors[:4],'source_qualifiers':{str(k):int(v) for k,v in qualifier_counts.items()},'timezone':'UTC','native_unit_scope':'PM mass and ozone volume units preserved; no conversion between them'})
        selected.append(name);print('EPA county selected',name,'sites',chosen[:4],flush=True)
    save('epa_candidates.json',{'selected':selected,'excluded':excluded,'sources':sources,'selection':'First twelve qualifying state/county groups; no test-score selection'})
    return selected

def uk():
    import rdata,warnings
    warnings.filterwarnings('ignore',message='Missing constructor for R class')
    path,source=download('AURN_metadata.RData','https://uk-air.defra.gov.uk/openair/R_data/AURN_metadata.RData',ROOT/'Research_Release/uk_validation_results/raw/AURN_metadata.RData')
    meta=rdata.read_rda(path)['AURN_metadata'];meta.to_csv(RAW/'aurn_metadata.csv',index=False)
    active=meta[(pd.to_datetime(meta.start_date)<=pd.Timestamp('2026-01-01'))&(meta.end_date.eq('ongoing')|meta.end_date.isna()|(pd.to_datetime(meta.end_date,errors='coerce')>=pd.Timestamp('2026-01-01')))]
    boxes={'north':(54,56.5,-4,0),'central':(52,54,-4,1),'southwest':(50,52,-5.5,-1),'southeast':(50,52,-1,1.5)};times,split=timeline('2026-01-01','2026-06-30');selected=[];audit=[]
    for label,(a,b,c,d) in boxes.items():
        g=active[active.latitude.ge(a)&active.latitude.lt(b)&active.longitude.ge(c)&active.longitude.lt(d)]
        g=g[~(g.latitude.between(51.2,51.8)&g.longitude.between(-.6,.35))]
        candidates=[code for code,gg in g.groupby('site_id') if {'PM2.5','PM10'}<=set(gg.parameter)]
        chosen=[];arrays=[];sources=[source];quality=[]
        for code in sorted(candidates):
            try:path,info=download(code+'_2026.RData','https://uk-air.defra.gov.uk/openair/R_data/'+code+'_2026.RData',ROOT/'Research_Release/uk_validation_results/raw'/(code+'_2026.RData'));sources.append(info);f=rdata.read_rda(path)[code+'_2026'].copy()
            except Exception as exc:audit.append({'box':label,'station':code,'reason':str(exc)});continue
            f.columns=[str(x).lower() for x in f.columns];dates=pd.to_datetime(f.date,unit='s',utc=True).dt.tz_localize(None) if pd.api.types.is_numeric_dtype(f.date) else pd.to_datetime(f.date,utc=True).dt.tz_localize(None)
            if dates.duplicated().any():raise RuntimeError('UK duplicate hours')
            f.index=dates;X=f.reindex(times).reindex(columns=['pm2.5','pm10']).to_numpy(dtype=float,copy=True);X[~np.isfinite(X)]=np.nan;X[X<0]=np.nan
            if qualified(X,split):
                chosen.append(str(code));arrays.append(X)
                quality.extend([{'station':str(code),'parameter':row.parameter,'ratified_to':str(row.ratified_to)} for _,row in g[(g.site_id==code)&g.parameter.isin(['PM2.5','PM10'])].iterrows()])
            if len(chosen)==4:break
        if len(chosen)<3:audit.append({'box':label,'reason':'Fewer than three training-eligible sites','eligible':chosen});continue
        name=store('uk_'+label,np.stack(arrays,axis=1),times,split,chosen,['PM2.5','PM10'],['ug/m3']*2,sources,{'provider':'AURN','box':[a,b,c,d],'quality':quality,'timezone':'UTC','timestamp_convention':'hour beginning','prior_uk_sites_excluded':True})
        selected.append(name);print('UK geographic cohort selected',name,chosen,flush=True)
    save('uk_candidates.json',{'selected':selected,'exclusions':audit});return selected

def india():
    folder=ROOT/'Research_Release/recent_data_results/delhi';ds=np.load(folder/'dataset.npz',allow_pickle=False);times=pd.DatetimeIndex(ds['times']);split=np.select([times<pd.Timestamp('2026-04-01'),times<pd.Timestamp('2026-05-01'),times<pd.Timestamp('2026-06-01')],[0,1,3],default=2);X=ds['X'];valid=(np.isfinite(X[split==0]).mean(0)>=.5)&(np.isfinite(X[split==0]).sum(0)>=720);options=[]
    for size in range(2,X.shape[2]+1):
        for channels in itertools.combinations(range(X.shape[2]),size):
            if 0 not in channels:continue
            keep=valid[:,channels].all(1)
            if keep.sum()>=3:options.append((int(keep.sum()*size),size,channels,keep))
    if not options:return []
    _,_,channels,keep=sorted(options,key=lambda x:(-x[0],-x[1],x[2]))[0];source={'file':str((folder/'dataset.npz').relative_to(ROOT)),'sha256':sha(folder/'dataset.npz'),'url':'Preserved OpenAQ original source ingestion; not independent observations'}
    name=store('india_long_test',X[:,keep][:,:,channels],times,split,ds['stations'][keep].tolist(),ds['channels'][list(channels)].tolist(),['ppb' if 'ppb' in c else 'ug/m3' for c in ds['channels'][list(channels)]],[source],{'provider':'OpenAQ','scope':'Exploratory longer-test resplit of previously accessed observations; not independent validation','temporal_protocol':'Train Jan-Mar, select Apr, calibrate May, test June-Oct'})
    return [name]

def main():
    RAW.mkdir(parents=True,exist_ok=True);protocol=sha(HERE/'PROTOCOL.md');lock=OUT/'acquisition_lock.json'
    if lock.exists() and json.loads(lock.read_text())['protocol_sha256']!=protocol:raise RuntimeError('Expansion protocol changed')
    save('acquisition_lock.json',{'protocol_sha256':protocol,'acquisition_source_sha256':sha(Path(__file__))})
    selected=[];issues=[]
    for name,function in [('EPA',epa),('AURN',uk),('OpenAQ longer-test sensitivity',india)]:
        marker=OUT/(name.split()[0]+'_acquired.json')
        if marker.exists():selected+=json.loads(marker.read_text())['cohorts'];continue
        try:cohorts=function();selected+=cohorts;save(marker.name,{'cohorts':cohorts})
        except Exception as exc:issues.append({'provider':name,'error':str(exc)});save('acquisition_issues.json',issues);raise
    save('cohorts.json',{'eligible':selected,'secondary':[]});save('acquisition_status.json',{'status':'completed','cohorts':len(selected),'issues':issues})

if __name__=='__main__':main()
