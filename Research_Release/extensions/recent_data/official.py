import sys,re,subprocess,concurrent.futures
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
import pymupdf
sys.path.insert(0,str(ROOT));from sheafcode.cpcb_index import concentration_to_index
def main():
    region='delhi';ds=load(region);m=models(region);dates=pd.date_range('2026-08-14','2026-10-02',freq='7D');folder=OUT/region/'official';folder.mkdir(exist_ok=True)
    identity={'dates':[str(x.date()) for x in dates],'dataset_sha256':sha(OUT/region/'dataset.npz'),'scope':'Selected hotspot comparison using two-PM proxy only; not official AQI or GRAP'}
    previous=json.loads((OUT/region/'official_lock.json').read_text()) if (OUT/region/'official_lock.json').exists() else {}
    for k,v in identity.items():
        if k in previous and previous[k]!=v:raise RuntimeError('Official alignment protocol changed')
    save(region+'/official_lock.json',identity|{'source_sha256':sha(__file__),'source_history':list(dict.fromkeys(previous.get('source_history',[])+([previous['source_sha256']] if previous.get('source_sha256') else [])+[sha(__file__)]))})
    records=[];parsed=[]
    for date in dates:
        p=folder/f'{date:%Y%m%d}_detail.pdf';url=f'https://cpcb.nic.in/uploads/AQM/AQ-NCR-{date:%d%m%Y}.pdf'
        if not p.exists():
            temp=p.with_suffix('.download');r=subprocess.run(['curl.exe','--silent','--show-error','--location','--fail','--max-time','20','--output',str(temp),url],capture_output=True,text=True)
            if r.returncode:
                records.append({'date':str(date.date()),'url':url,'status':'unavailable','error':r.stderr[-250:]});continue
            if not temp.read_bytes().startswith(b'%PDF'):records.append({'date':str(date.date()),'url':url,'status':'not_pdf'});continue
            temp.replace(p)
        doc=pymupdf.open(p);text='\n'.join(page.get_text() for page in doc);doc.close();p.with_suffix('.txt').write_text(text,encoding='utf-8');expected=[f'{date:%B} {date.day}, {date.year}',date.strftime('%B %d, %Y'),f'{date:%b} {date.day}, {date.year}',date.strftime('%b %d, %Y')];verified=any(x in text for x in expected);records.append({'date':str(date.date()),'url':url,'status':'downloaded','sha256':sha(p),'date_verified':verified})
        if not verified:continue
        pattern=r'(?P<station>[^\n]+, Delhi - (?:DPCC|CPCB))\s*\n(?P<index>\d{1,3})\s*\n(?P<category>Very Poor|Satisfactory|Moderate|Severe|Poor|Good)\s*\n(?P<prominent>[^\n]+)'
        for match in re.finditer(pattern,text):parsed.append({'date':str(date.date()),'station_name':match['station'].strip(),'official_aqi':int(match['index']),'official_category':match['category'],'prominent_pollutant':match['prominent'].strip(),'source_sha256':sha(p)})
    save(region+'/official_acquisition.json',records);frame_save(region+'/official_station_records.csv',pd.DataFrame(parsed));mapping={v['source_name'].rsplit('-',1)[0]:str(v['location_id']) for v in json.loads((DATA/'station_mapping.json').read_text())['delhi']};rows=[];rejected=[]
    def aqi_category(x):return ['Good','Satisfactory','Moderate','Poor','Very Poor','Severe'][int(np.searchsorted([50,100,200,300,400],x,side='left'))]
    for record in parsed:
        station=mapping.get(record['station_name'])
        if station is None or station not in ds['stations']:rejected.append(record|{'reason':'Station outside eligible recent cohort'});continue
        v=int(np.flatnonzero(ds['stations']==station)[0]);end=pd.Timestamp(record['date'])+pd.Timedelta(hours=16);ix=np.flatnonzero((ds['times']>=end-pd.Timedelta(hours=24))&(ds['times']<end))
        if len(ix)!=24 or ix[0]<24 or not (ds['split'][ix[0]-24:ix[-1]+1]==2).all():rejected.append(record|{'reason':'No matching fully held-out prefix/window'});continue
        truth=ds['X'][ix,v,:2];good=np.isfinite(truth)
        if good.sum(0).min()<18:rejected.append(record|{'reason':'Fewer than eighteen observed hours in a PM channel'});continue
        whole=np.arange(ix[0]-24,ix[-1]+1);x=ds['Z'][whole].copy();x[24:,v]=np.nan;sheaf=m['sheaf'].causal(x,v)*ds['sd'][v]+ds['mu'][v];city=raw_city(x[24:],v,ds['mu'],ds['sd'])*ds['sd'][v]+ds['mu'][v]
        for method,pred in [('observed_PM_proxy',ds['X'][ix,v]),('sheaf_PM_proxy',sheaf),('city_PM_proxy',city)]:
            scores=[]
            for q,pol in enumerate(['PM2.5','PM10']):scores.append(float(concentration_to_index(pol,np.array([np.mean(pred[good[:,q],q])]))[0]))
            proxy=max(scores);rows.append(record|{'station':station,'method':method,'proxy_aqi':proxy,'category_matches':aqi_category(proxy)==record['official_category'],'scope':'Selected two-PM daily proxy on common observed hours; not validated official AQI'})
    frame_save(region+'/official_proxy_cases.csv',pd.DataFrame(rows));save(region+'/official_rejected_alignments.json',rejected);summary={'status':'completed','requested_documents':len(dates),'downloaded_documents':sum(r['status']=='downloaded' for r in records),'parsed_station_entries':len(parsed),'eligible_station_days':len(rows)//3,'agreements':pd.DataFrame(rows).groupby('method').category_matches.sum().astype(int).to_dict() if rows else {},'scope':'Selected hotspots, approximate two-PM proxy; no representative accuracy, official AQI or actual GRAP recovery claim'};save(region+'/official_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
