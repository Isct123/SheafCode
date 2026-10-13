import sys,subprocess,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
def main():
    folder=OUT/'epa_la';raw=folder/'raw';raw.mkdir(parents=True,exist_ok=True)
    identity={'protocol_sha256':sha(HERE/'EPA_PROTOCOL.md'),'dates':['2026-01-01','2026-03-31'],'state':'06','county':'037','parameters':[88101,44201,42602]}
    previous=json.loads((folder/'acquisition_lock.json').read_text()) if (folder/'acquisition_lock.json').exists() else {}
    for k,v in identity.items():
        if k in previous and previous[k]!=v:raise RuntimeError('EPA acquisition protocol changed')
    save('epa_la/acquisition_lock.json',identity|{'source_sha256':sha(__file__),'source_history':list(dict.fromkeys(previous.get('source_history',[])+([previous['source_sha256']] if previous.get('source_sha256') else [])+[sha(__file__)])),'schema_note':'EPA hourly files omit Sample Duration; official hourly-file documentation excludes other sample durations. Parser corrected before any EPA outcome evaluation.'});frames=[];sources=[]
    for code in [88101,44201,42602]:
        path=raw/f'hourly_{code}_2026.zip';url='https://aqs.epa.gov/aqsweb/airdata/'+path.name
        if not path.exists():
            temp=path.with_suffix('.download');r=subprocess.run(['curl.exe','--silent','--show-error','--location','--fail','--max-time','120','--retry','2','--output',str(temp),url],capture_output=True,text=True)
            if r.returncode:raise RuntimeError(r.stderr)
            with zipfile.ZipFile(temp) as z:z.namelist()
            temp.replace(path)
        sources.append({'url':url,'sha256':sha(path),'bytes':path.stat().st_size})
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if not name.endswith('.csv'):continue
                for f in pd.read_csv(z.open(name),chunksize=200000,low_memory=False):
                    f=f[(f['State Code']==6)&(f['County Code']==37)].copy()
                    if not len(f):continue
                    f['timestamp']=pd.to_datetime(f['Date GMT']+' '+f['Time GMT']);f=f[f.timestamp.between('2026-01-01','2026-03-31 23:00')]
                    if not len(f):continue
                    f['parameter']=code;frames.append(f[['timestamp','Site Num','POC','parameter','Sample Measurement','Units of Measure','Latitude','Longitude','Date of Last Change']])
        print('EPA acquired/parsed',code,flush=True)
    if not frames:raise RuntimeError('No compatible 2026 LA observations')
    f=pd.concat(frames);frame_save('epa_la/source_observations.csv',f);save('epa_la/sources.json',sources);times=pd.date_range('2026-01-01','2026-03-31 23:00',freq='h');split=np.zeros(len(times),int);split[int(.6*len(times)):int(.8*len(times))]=1;split[int(.8*len(times)):]=2;split[(np.arange(len(times))>=int(.7*len(times)))&(split==1)]=3
    tr_end=times[split==0][-1];chosen=[];site_codes=sorted(f['Site Num'].unique());channels=['PM2.5','O3_ppm','NO2_ppb'];codes=[88101,44201,42602];unitmap={88101:'Micrograms/cubic meter (LC)',44201:'Parts per million',42602:'Parts per billion'};cols=[]
    for site in site_codes:
        ff=f[f['Site Num']==site];vectors=[];pocs=[]
        for code in codes:
            gg=ff[(ff.parameter==code)&(ff['Units of Measure']==unitmap[code])&(ff['Sample Measurement']>=0)];options=[]
            for poc,g in gg.groupby('POC'):
                s=g.groupby('timestamp')['Sample Measurement'].mean().reindex(times);a=s.loc[:tr_end]
                if a.notna().sum()>=720 and a.notna().mean()>=.5:options.append((poc,s))
            if not options:break
            poc,s=min(options,key=lambda x:x[0]);vectors.append(s.to_numpy());pocs.append(int(poc))
        if len(vectors)==3 and len(chosen)<6:chosen.append({'site_code':int(site),'pocs':pocs});cols.append(np.stack(vectors,axis=1))
    if len(chosen)<3:save('epa_la/availability_status.json',{'status':'insufficient_training_eligible_sites','eligible':chosen});print('EPA inapplicable:',len(chosen),'sites',flush=True);return
    X=np.stack(cols,axis=1);mu=np.nanmean(X[split==0],0);sd=np.nanstd(X[split==0],0,ddof=1);stations=np.array([str(s['site_code']) for s in chosen]);np.savez_compressed(folder/'dataset.npz',X=X,Z=(X-mu)/sd,mu=mu,sd=sd,split=split,times=times.to_numpy(),stations=stations,channels=np.array(channels));save('epa_la/dataset_manifest.json',{'region':'epa_la','dates':['2026-01-01','2026-03-31'],'stations':stations.tolist(),'channels':channels,'units':['ug/m3','ppm','ppb'],'selected_monitor_pocs':chosen,'split_counts':{str(i):int((split==i).sum()) for i in [0,1,3,2]},'protocol_sha256':sha(HERE/'EPA_PROTOCOL.md'),'source_hashes':sources,'scope':'Secondary recent US network; native units, UTC timeline, source reporting delay/preliminary status possible'})
    registry=json.loads((OUT/'cohorts.json').read_text());registry['secondary']=['epa_la'];save('cohorts.json',registry)
                                                                                  
    import fit;fit.prepare=lambda:None;fit.regions=lambda:['epa_la'];fit.main()
    import evaluate;evaluate.regions=lambda:['epa_la'];evaluate.main()
    save('epa_la/availability_status.json',{'status':'completed_recent_secondary_network','training_eligible_sites':len(chosen),'source_sha256':sha(__file__)})
if __name__=='__main__':main()
