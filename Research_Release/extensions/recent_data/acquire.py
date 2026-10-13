import os,sys,json,hashlib,gzip,io,subprocess,concurrent.futures,time,itertools
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
PROBE=pd.Timestamp(sys.argv[1] if len(sys.argv)>1 else '2026-01-15')
OUT=BASE/'recent_data_results'
if len(sys.argv)>1:OUT=OUT/'availability_expansion'
RAW=OUT/'raw'
OUT.mkdir(exist_ok=True);RAW.mkdir(exist_ok=True)
URL='https://openaq-data-archive.s3.amazonaws.com/'
IDS=sorted(set(range(1,101))|set(range(8100,8301))|{8714,10485,10486,10487,10488})
BOXES={'delhi':(28.3,29.,76.8,77.6),'bengaluru':(12.7,13.3,77.3,77.9),'hyderabad':(17.1,17.7,78.1,78.8)}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,data):
    p=OUT/p;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,indent=2),encoding='utf-8');t.replace(p)
def fetch(job):
    loc,date=job;key=f'records/csv.gz/locationid={loc}/year={date:%Y}/month={date:%m}/location-{loc}-{date:%Y%m%d}.csv.gz';p=RAW/f'{loc}_{date:%Y%m%d}.csv.gz'
    if p.exists():
        try:return p,{'location_id':loc,'date':str(date.date()),'url':URL+key,'status':'cached','sha256':sha(p)}
        except Exception:pass
    temp=p.with_suffix('.download')
    cmd=['curl.exe','--silent','--show-error','--location','--max-time','35','--retry','2','--retry-delay','1','--write-out','%{http_code}','--output',str(temp),URL+key]
    r=subprocess.run(cmd,capture_output=True,text=True)
    code=r.stdout.strip();meta={'location_id':loc,'date':str(date.date()),'url':URL+key,'status':code,'returncode':r.returncode}
    if r.returncode==0 and code=='200':
        try:
            with gzip.open(temp,'rb') as f:f.read(100)
            temp.replace(p);meta.update(status='downloaded',sha256=sha(p));return p,meta
        except Exception as e:meta['parse_error']=str(e)
    if temp.exists():temp.unlink()
    if r.stderr:meta['error']=r.stderr[-300:]
    return None,meta
def read(p):return pd.read_csv(p)
def region(f):
    lat=float(f.lat.iloc[0]);lon=float(f.lon.iloc[0])
    return next((name for name,(a,b,c,d) in BOXES.items() if a<=lat<=b and c<=lon<=d),None)
def main():
    lock={'protocol_sha256':sha(HERE/'PROTOCOL.md'),'source_sha256':sha(__file__),'start':'2026-01-01','end':'2026-10-05','probe':str(PROBE.date()),'ids':IDS,'boxes':BOXES,'maximum_stations':6}
    lp=OUT/'ACQUISITION_LOCK.json'
    if lp.exists() and json.loads(lp.read_text())!=json.loads(json.dumps(lock)):raise RuntimeError('Acquisition protocol changed')
    save('ACQUISITION_LOCK.json',lock)
    mapping_path=OUT/'station_mapping.json'
    if mapping_path.exists():mapping=json.loads(mapping_path.read_text())
    else:
        mapping={k:[] for k in BOXES};probe=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for i,(p,m) in enumerate(pool.map(fetch,[(x,PROBE) for x in IDS])):
                probe.append(m)
                if p:
                    f=read(p);name=region(f)
                    if name:
                        item={'location_id':m['location_id'],'source_name':str(f.location.iloc[0]),'lat':float(f.lat.iloc[0]),'lon':float(f.lon.iloc[0])}
                        if len(mapping[name])<6 and not any((item['lat']-v['lat'])**2+(item['lon']-v['lon'])**2<(.002**2) for v in mapping[name]):mapping[name].append(item)
                if i%40==0:print('Metadata probes',i+1,'selected',{k:len(v) for k,v in mapping.items()},flush=True)
        save('probe_records.json',probe);save('station_mapping.json',mapping)
    print('Recent station mapping',mapping,flush=True)
    dates=pd.date_range('2026-01-01','2026-10-05');jobs=[(v['location_id'],day) for vs in mapping.values() for v in vs for day in dates]
    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i,(p,m) in enumerate(pool.map(fetch,jobs)):
            records.append(m)
            if i%100==0:save('acquisition_progress.json',{'status':'running','done':i+1,'total':len(jobs),'valid':sum(r['status'] in ['cached','downloaded'] for r in records)});print('Daily acquisition',i+1,'/',len(jobs),flush=True)
    save('acquisition_records.json',records)
    channel={('pm25','ug/m3'):'PM2.5',('pm10','ug/m3'):'PM10',('o3','ug/m3'):'O3',('no2','ppb'):'NO2_ppb'}
    def normalize(u):return str(u).replace('µ','u').replace('μ','u').replace('³','3').replace(' ','')
    reverse={v['location_id']:(name,str(v['location_id'])) for name,vs in mapping.items() for v in vs};frames=[];rejected={}
    for m in records:
        if m['status'] not in ['cached','downloaded']:continue
        p=RAW/f"{m['location_id']}_{pd.Timestamp(m['date']):%Y%m%d}.csv.gz";f=read(p)
        for (param,unit),g in f.groupby(['parameter','units']):
            pol=channel.get((param,normalize(unit)))
            if pol is None:
                rejected[str(param)+' '+str(unit)]=rejected.get(str(param)+' '+str(unit),0)+len(g);continue
            ts=pd.to_datetime(g.datetime,utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None).dt.floor('h');val=pd.to_numeric(g.value,errors='coerce')
            name,station=reverse[m['location_id']]
            frames.append(pd.DataFrame({'region':name,'station':station,'timestamp':ts,'pollutant':pol,'value':val,'sensor_id':g.sensors_id}))
    if not frames:raise RuntimeError('No compatible recent observations')
    f=pd.concat(frames,ignore_index=True);bad=(f.value<0)|(~f.timestamp.between(pd.Timestamp('2026-01-01'),pd.Timestamp('2026-10-05 23:59:59')))|f.value.isna()
    excluded=int(bad.sum());f=f[~bad];g=f.groupby(['region','station','timestamp','pollutant']).agg(value=('value','mean'),source_rows=('value','size'),distinct_sensors=('sensor_id','nunique')).reset_index();g.to_csv(OUT/'observations.csv',index=False)
    summary={'status':'completed','period':[str(g.timestamp.min()),str(g.timestamp.max())],'valid_rows':len(g),'excluded_values_or_dates':excluded,'rejected_parameters_units':rejected,'station_counts':g.groupby('region').station.nunique().to_dict(),'duplicate_averaging':'All compatible source rows within station/hour/channel averaged; source counts retained','raw_objects':sum(m['status'] in ['cached','downloaded'] for m in records),'acquired_utc':pd.Timestamp.now(tz='UTC').isoformat()}
    save('acquisition_summary.json',summary);print(json.dumps(summary,ensure_ascii=True,indent=2),flush=True)
if __name__=='__main__':main()
