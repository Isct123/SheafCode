                                                                            

                                                                              
                                                                              
   
import sys,urllib.request,urllib.error,gzip,io,concurrent.futures
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
BASE='https://openaq-data-archive.s3.amazonaws.com/'
def fetch(location,date,folder):
    key=f'records/csv.gz/locationid={location}/year={date:%Y}/month={date:%m}/location-{location}-{date:%Y%m%d}.csv.gz'
    path=folder/f'{location}_{date:%Y%m%d}.csv.gz'
    if path.exists():data=path.read_bytes()
    else:
        try:data=urllib.request.urlopen(BASE+key,timeout=20).read()
        except urllib.error.HTTPError as e:
            if e.code in [403,404]:return None
            raise
        path.write_bytes(data)
    return pd.read_csv(io.BytesIO(gzip.decompress(data))),{'key':key,'sha256':digest(path)}
def main():
    folder=OUT/'external';folder.mkdir(exist_ok=True);raw=folder/'download';raw.mkdir(exist_ok=True)
    ds,m,modeldir=cohort('delhi2025');start=pd.Timestamp('2026-09-12');end=pd.Timestamp('2026-10-05')
    if start<=ds.times.max():raise RuntimeError('Holdout overlap')
    lock={'period':[str(start.date()),str(end.date())],'model_sha256':digest(modeldir/'model.npz'),'frozen_manifest_sha256':digest(modeldir/'manifest.json'),'source_sha256':digest(__file__),'station_coords':ds.coords.to_dict(),'scope':'New acquisition after frozen analysis; same instruments and network; broader prior user access cannot be independently verified'}
    lockpath=folder/'acquisition_lock.json'
    if lockpath.exists():
        existing=json.loads(lockpath.read_text())
        for key in ['period','model_sha256','frozen_manifest_sha256','station_coords']:
            if existing[key]!=lock[key]:raise RuntimeError('External acquisition settings changed')
        history=existing.get('acquisition_source_history',[existing['source_sha256']])
        lock['acquisition_source_history']=list(dict.fromkeys(history+[lock['source_sha256']]))
    lockpath.write_text(json.dumps(lock,indent=2))
    def probe(location):
        r=fetch(location,start,raw)
        if r is None:return None
        f,meta=r;lat=float(f.lat.iloc[0]);lon=float(f.lon.iloc[0]);d=np.linalg.norm(ds.coords.to_numpy()-[lat,lon],axis=1);v=int(np.argmin(d))
        return (location,v,float(d[v]),str(f.location.iloc[0]),meta) if d[v]<.005 else None
                                                                             
                                                                    
    ids=sorted(set(range(8100,8301))|set(range(1,101))|{10486,8714,10485,10487,10488})
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:matches=[x for x in pool.map(probe,ids) if x is not None]
    mapping={}
    for location,v,dist,name,meta in matches:
        if v not in mapping or dist<mapping[v]['distance_degrees']:mapping[v]={'location_id':location,'distance_degrees':dist,'source_name':name,'station':ds.stations[v]}
    (folder/'station_mapping.json').write_text(json.dumps(mapping,indent=2))
    print('External matching:',len(mapping),'of',ds.n,'stations',flush=True)
    jobs=[(x['location_id'],date) for x in mapping.values() for date in pd.date_range(start,end)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:received=list(pool.map(lambda job:fetch(*job,raw),jobs))
    frames=[];sources=[];reverse={x['location_id']:x['station'] for x in mapping.values()};parameter={'pm25':'PM2.5','pm10':'PM10','no2':'NO2','no2_mass':'NO2','o3':'O3','o3_mass':'O3'};rejected={}
    for result in received:
        if result is None:continue
        f,meta=result;sources.append(meta)
        for (param,unit),g in f.groupby(['parameter','units']):
            if param not in parameter:continue
                                                                               
                                                               
            if unit not in ['µg/m³','μg/m³','ug/m3','µg/m3']:
                rejected[param+' '+str(unit)]=rejected.get(param+' '+str(unit),0)+len(g);continue
            times=pd.to_datetime(g.datetime,utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None).dt.floor('h')
            frames.append(pd.DataFrame({'timestamp':times,'station':g.location_id.map(reverse),'pollutant':parameter[param],'value':pd.to_numeric(g.value,errors='coerce'),'sensor_id':g.sensors_id}))
    if not frames:
        (folder/'status.json').write_text(json.dumps({'status':'unavailable_compatible_observations','matched_stations':len(mapping),'downloaded_files':len(sources),'rejected_units':rejected},indent=2));return
    f=pd.concat(frames);f=f[(f.timestamp>ds.times.max())&(f.value>=0)]
                                                                            
                                                                                 
    duplicates=int(f.duplicated(['timestamp','station','pollutant']).sum());f=f.groupby(['timestamp','station','pollutant'],as_index=False).value.mean()
    f.to_csv(folder/'observations.csv',index=False)
    (folder/'provenance.json').write_text(json.dumps({'region':'delhi2025','source_url':'https://docs.openaq.org/aws/about','acquired_at':pd.Timestamp.now(tz='UTC').isoformat(),'previously_used_for_development':False,'prior_user_access_not_verifiable':True,'same_instruments_as_training':True,'sources':sources,'duplicate_channel_rows_averaged':duplicates,'rejected_units':rejected,'matching_rule':'Coordinate distance <0.005 degrees; minimum distance among available matching IDs; no outcome-based selection'},indent=2))
    (folder/'status.json').write_text(json.dumps({'status':'acquired','station_count':f.station.nunique(),'rows':len(f),'source_files':len(sources),'period':[str(f.timestamp.min()),str(f.timestamp.max())],'scope':'New same-network non-overlapping temporal holdout; not independent instruments'},indent=2))
if __name__=='__main__':main()
