                                                                               
import sys,urllib.request,urllib.error,concurrent.futures,re,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
import pymupdf

def acquire():
    ds,m,folder=local('delhi2025');days=day_blocks(ds.times,ds.split,2,stride=7)
    dates=[pd.Timestamp(d) for d,_ in days]
                                                                           
                                                                              
    controls=[pd.Timestamp('2025-03-26'),pd.Timestamp('2025-08-25')]
    settings={'dates':[str(d.date()) for d in dates],
         'historical_alignment_control_dates':[str(d.date()) for d in controls],
         'source_sha256':sha(__file__),'manifest_sha256':sha(folder/'manifest.json')}
    lockpath=OUT/'official_acquisition_lock.json'
    if lockpath.exists():
        original=json.loads(lockpath.read_text())
        for key in ['dates','historical_alignment_control_dates','manifest_sha256']:
            if original[key]!=settings[key]:raise RuntimeError('Official evaluation dates or model changed')
        save('official_transport.json',{'source_sha256':sha(__file__),'method':'Windows curl with native Schannel certificate validation','original_source_sha256':original['source_sha256']})
        old=OUT/'official_acquisition.json'
        if old.exists() and not (OUT/'official_initial_transport_errors.json').exists():
            save('official_initial_transport_errors.json',json.loads(old.read_text()))
    else:lock('official_acquisition_lock.json',settings)
    directory=OUT/'official';directory.mkdir(exist_ok=True)
    def download(task):
        date,kind=task;tag=f'{date:%Y%m%d}_{kind}';path=directory/(tag+'.pdf')
        results=[]
        if path.exists():return {'date':str(date.date()),'kind':kind,'status':'downloaded','path':str(path.relative_to(OUT)),'sha256':sha(path)}
        stem=f'uploads/AQM/AQ-NCR-{date:%d%m%Y}.pdf' if kind=='detail' else f'upload/Downloads/AQI_Bulletin_{date:%Y%m%d}.pdf'
        for domain in ['cpcb.nic.in','cpcb.gov.in']:
            url='https://'+domain+'/'+stem
            try:
                temp=path.with_suffix('.download')
                result=subprocess.run(['curl.exe','--silent','--show-error','--location','--fail','--max-time','12','--output',str(temp),url],capture_output=True,text=True)
                if result.returncode:raise RuntimeError(result.stderr.strip())
                data=temp.read_bytes()
                if not data.startswith(b'%PDF'):raise ValueError('Response is not a PDF')
                temp.replace(path)
                return {'date':str(date.date()),'kind':kind,'status':'downloaded','url':url,'path':str(path.relative_to(OUT)),'sha256':sha(path)}
            except Exception as error:results.append({'url':url,'error':str(error)})
        return {'date':str(date.date()),'kind':kind,'status':'unavailable','attempts':results}
    tasks=[(d,kind) for d in dates for kind in ['detail','city']]+[(d,'detail') for d in controls]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        records=[]
        for record in pool.map(download,tasks):
            record['is_test_date']=record['date'] in [str(d.date()) for d in dates]
            records.append(record);save('official_acquisition.json',records)
            print('Official record',record['date'],record['kind'],record['status'],flush=True)
    return records

def extract():
    records=json.loads((OUT/'official_acquisition.json').read_text());stationrows=[];cityrows=[];issues=[]
    for record in records:
        if record['status']!='downloaded':continue
        doc=pymupdf.open(OUT/record['path']);texts=[p.get_text(sort=True) for p in doc]
        (OUT/record['path']).with_suffix('.txt').write_text('\n'.join(texts),encoding='utf-8')
                                                                           
                                                                       
        ordered='\n'.join(p.get_text() for p in doc)
        date=pd.Timestamp(record['date']);expected=[date.strftime('%B %d, %Y'),date.strftime('%B %-d, %Y') if sys.platform!='win32' else f'{date:%B} {date.day}, {date.year}']
        expected.extend([date.strftime('%b %d, %Y'),f'{date:%b} {date.day}, {date.year}'])
        verified=any(x in ordered for x in expected) or record['kind']=='city' and (date.strftime('%d/%m/%Y') in ordered or date.strftime('%d-%m-%Y') in ordered)
        if not verified:
            issues.append({'date':record['date'],'kind':record['kind'],'issue':'Publication date not parsed; source text retained, no performance use'});continue
        pattern=r'(?P<station>[^\n]+, Delhi - (?:DPCC|CPCB))\s*\n(?P<index>\d{1,3})\s*\n(?P<category>Very Poor|Satisfactory|Moderate|Severe|Poor|Good)\s*\n(?P<prominent>[^\n]+)'
        for match in re.finditer(pattern,ordered):
            stationrows.append({'date':record['date'],'station':match['station'].strip(),'official_aqi':int(match['index']),
                                'official_category':match['category'],'prominent_pollutant':match['prominent'],
                                'source_path':record['path'],'is_test_date':record['is_test_date'],'date_verified':verified,'excerpt':match[0]})
        match=re.search(r'Delhi\s+(Good|Satisfactory|Moderate|Very Poor|Poor|Severe)\s+(\d{1,3})',ordered)
        if match:cityrows.append({'date':record['date'],'city':'Delhi','official_category':match[1],'official_aqi':int(match[2]),'source_path':record['path'],'is_test_date':record['is_test_date']})
    pd.DataFrame(stationrows).to_csv(OUT/'official_station_records.csv',index=False)
    pd.DataFrame(cityrows).to_csv(OUT/'official_city_records.csv',index=False)
    save('official_parse_issues.json',issues)
    return stationrows,cityrows

def alignment():
    from replay import raw_delhi,geographic
    from reference import QuadraticDecoder
    stationrows,cityrows=extract();X,times,split,stations,coords,mu,sd,_=raw_delhi();Z=(X-mu)/sd
    ref=QuadraticDecoder(Z[split==0],geographic(coords));rows=[];rejected=[]
    from sheafcode.cpcb_index import concentration_to_index
    for record in stationrows:
        if record['station'] not in stations:
            rejected.append(record|{'reason':'Station outside raw-data cohort'});continue
        if not record['is_test_date']:
            rejected.append(record|{'reason':'Historical alignment control precedes test; no held-out performance evaluation'});continue
        end=pd.Timestamp(record['date'])+pd.Timedelta(hours=16)
        ix=np.flatnonzero((times>=end-pd.Timedelta(hours=24))&(times<end))
        if len(ix)!=24 or ix[0]<24 or not (split[ix]==2).all():
            rejected.append(record|{'reason':'No matching 24-hour held-out window'});continue
        v=stations.index(record['station']);truth=X[ix,v];good=np.isfinite(truth)
                                                                         
                                                                                
                                                                           
        if good.sum(0).min()<18:
            rejected.append(record|{'reason':'Fewer than 18 observed hours in a selected pollutant'});continue
        whole=np.arange(ix[0]-24,ix[-1]+1);hidden=Z[whole].copy();hidden[24:,v]=np.nan
        reconstructed=ref.causal(hidden,v)*sd[v]+mu[v]
        others=np.delete(X[ix],v,axis=1);counts=np.isfinite(others).sum(1)
        fallback=np.divide(np.nansum(others,axis=1),counts,out=np.full((24,4),np.nan),where=counts>0)
        if np.isfinite(fallback).sum(0).min()<18 or np.isfinite(fallback[-8:,3]).sum()<6:
            rejected.append(record|{'reason':'Insufficient common fallback averaging coverage'});continue
        for method,values in [('observed_proxy',truth),('sheaf_proxy',reconstructed),('city_fallback_proxy',fallback)]:
            scores=[]
            for q,pollutant in enumerate(['PM2.5','PM10','NO2','O3']):
                concentration=np.nanmean(values[-8:,q]) if pollutant=='O3' else np.nanmean(values[:,q])
                scores.append(float(concentration_to_index(pollutant,concentration)))
            estimate=max(scores);category=int(np.searchsorted([50,100,200,300,400],estimate,side='left'))
            official=int(np.searchsorted([50,100,200,300,400],record['official_aqi'],side='left'))
            rows.append({'date':record['date'],'station':record['station'],'method':method,'proxy_aqi':estimate,'official_aqi':record['official_aqi'],
                         'category_matches':category==official,'source_path':record['source_path'],'averaging_window':f'{times[ix[0]]} to {times[ix[-1]]}',
                         'scope':'Selected-pollutant approximate proxy, not official AQI or GRAP invocation'})
    pd.DataFrame(rows).to_csv(OUT/'official_proxy_comparison.csv',index=False)
    save('official_rejected_alignments.json',rejected)
    records=json.loads((OUT/'official_acquisition.json').read_text())
    save('official_summary.json',{'status':'completed_availability_and_alignment_audit',
        'requested_files':len(records),'downloaded_files':sum(r['status']=='downloaded' for r in records),
        'test_date_downloaded_files':sum(r['status']=='downloaded' and r['is_test_date'] for r in records),
        'parsed_station_rows':len(stationrows),'parsed_city_rows':len(cityrows),'eligible_test_station_days':len(rows)//3,
        'rejected_station_alignments':len(rejected),
        'scope':'Public CPCB bulletin alignment; station rows in these detailed reports are selected hotspot listings, not a representative sample. Unavailable records and incompatible station/period/averaging are retained. Historical control records are not held-out performance. Zero eligible pairs means no official-decision validation claim. Selected four-pollutant proxy with final-window ozone averaging is approximate; gas units and omitted pollutants are not independently reconciled. Proxy comparison does not recover official AQI or actual GRAP actions.'})

if __name__=='__main__':
    if '--acquire' in sys.argv:acquire()
    else:alignment()
