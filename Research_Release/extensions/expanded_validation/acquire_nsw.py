                                                                              
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import acquire as a
import json,subprocess,concurrent.futures,re
import numpy as np
import pandas as pd
URL='https://data.airquality.nsw.gov.au/api/Data/get_Observations'
def request(region,sites):
    label=re.sub('[^a-z0-9]+','_',region.lower()).strip('_')
    req=a.RAW/('nsw_'+label+'_request.json');out=a.RAW/('nsw_'+label+'_response.json')
    body={'Parameters':['PM2.5','PM10'],'Sites':sites,'StartDate':'2026-01-01','EndDate':'2026-07-01','Categories':['Averages'],'Subcategories':['Hourly'],'Frequency':['Hourly average']}
    req.write_text(json.dumps(body,indent=2))
    if not out.exists():
        temp=out.with_suffix('.download')
        subprocess.run(['curl.exe','--fail','--location','--silent','--show-error','--compressed','--max-time','900','--retry','1','--header','Content-Type: application/json','--data-binary','@'+str(req),'--output',str(temp),URL],check=True)
        value=json.loads(temp.read_text(encoding='utf-8-sig'))
        if not isinstance(value,list):raise RuntimeError('Unexpected NSW response '+region)
        temp.replace(out)
    rows=json.loads(out.read_text(encoding='utf-8-sig'));data=[]
    for row in rows:
        p=row['Parameter'];units=p['Units'].replace('μ','µ').replace('³','3').replace(' ','')
        if p['Category']!='Averages' or p['SubCategory']!='Hourly' or p['Frequency']!='Hourly average':raise RuntimeError('Wrong NSW averaging convention')
        if units not in ['µg/m3','ug/m3']:raise RuntimeError('Unexpected NSW PM units '+units)
        hour=int(row['Hour'])
        if hour<1 or hour>24:raise RuntimeError('Unexpected NSW hour')
        timestamp=pd.Timestamp(row['Date'])+pd.Timedelta(hours=hour-1)
        data.append({'timestamp':timestamp,'station':int(row['Site_Id']),'channel':p['ParameterCode'],'value':row['Value']})
    return region,label,sites,pd.DataFrame(data),{'file':str(out.relative_to(a.ROOT)),'url':URL,'sha256':a.sha(out),'request_file':str(req.relative_to(a.ROOT)),'request_sha256':a.sha(req)}
def main():
    a.RAW.mkdir(parents=True,exist_ok=True)
    path=a.RAW/'nsw_sites.json';sites=json.loads(path.read_text());groups={}
    for s in sites:
        if s['Latitude'] is not None and s['Longitude'] is not None:groups.setdefault(s['Region'].strip(),[]).append(int(s['Site_Id']))
    jobs=[(r,sorted(set(v))) for r,v in sorted(groups.items()) if len(set(v))>=3]
    results=[];errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(request,*job):job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:results.append(future.result());print('NSW response saved',futures[future][0],flush=True)
            except Exception as exc:errors.append({'region':futures[future][0],'error':str(exc)})
    if errors:
        a.save('nsw_acquisition_errors.json',errors);raise RuntimeError('Cannot certify first eligible regions while source requests are missing; resume acquisition')
    times,split=a.timeline('2026-01-01','2026-06-30');selected=[];decisions=[]
    for region,label,candidates,f,source in sorted(results,key=lambda x:x[0]):
        arrays=[];chosen=[]
        for station in candidates:
            if f.empty:continue
            g=f[f.station==station]
            if g.duplicated(['timestamp','channel']).any():raise RuntimeError('Duplicate NSW hourly observations')
            X=g.pivot(index='timestamp',columns='channel',values='value').reindex(index=times,columns=['PM2.5','PM10']).to_numpy(dtype=float,copy=True);X[X<0]=np.nan
            if a.qualified(X,split):chosen.append(str(station));arrays.append(X)
        decisions.append({'region':region,'eligible':chosen,'reason':'selected' if len(chosen)>=3 and len(selected)<6 else 'fewer_than_three_eligible' if len(chosen)<3 else 'beyond_fixed_six_group_cap'})
        if len(chosen)>=3 and len(selected)<6:
            selected.append(a.store('nsw_'+label,np.stack(arrays[:4],axis=1),times,split,chosen[:4],['PM2.5','PM10'],['ug/m3']*2,[source,{'file':str(path.relative_to(a.ROOT)),'url':'https://data.airquality.nsw.gov.au/api/Data/get_SiteDetails','sha256':a.sha(path)}],{'provider':'NSW','source_region':region,'all_training_eligible_sites':chosen,'timezone':'Fixed AEST (UTC+10), not daylight-saving local Sydney time','timestamp_convention':'Hour 1 denotes 00:00-01:00; timestamp is beginning of that interval','official_time_reference':'https://www.airquality.nsw.gov.au/air-quality-data-services/data-explorer','source_quality':'Public official API as retrieved; ratification status not asserted'}))
    a.save('NSW_acquired.json',{'cohorts':selected});a.save('nsw_candidates.json',{'decisions':decisions})
if __name__=='__main__':main()
