                                                                                 
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *

def interval_difference(f,a,b,condition=None):
    q=f if condition is None else f[condition]
    if not len(q):return {'cases':0,'difference_percentage_points':None,'ci95':None}
    p=q.pivot(index=['day','timestamp','station'],columns='method',values='alarm')[[a,b]].dropna()
    if not len(p):return {'cases':0,'difference_percentage_points':None,'ci95':None}
    d=p[a].astype(float)-p[b].astype(float);dates=pd.to_datetime(p.index.get_level_values('day'));bins=(dates-dates.min()).days//7
    g=np.array([[len(x),x.sum()] for _,x in d.groupby(bins)]);rng=np.random.default_rng(8043)
    boot=[100*(z:=g[rng.integers(0,len(g),len(g))].sum(0))[1]/z[0] for _ in range(2000)]
    return {'cases':len(p),'calendar_bins':len(g),'difference_percentage_points':100*d.mean(),'ci95':np.quantile(boot,[.025,.975]).tolist() if len(g)>=5 else None}

def main():
    for region in regions()+['epa_la']:
        ds=load(region);m=models(region);folder=OUT/region
        alerts=pd.read_csv(folder/'hourly_alert_predictions.csv',dtype={'station':str})
        result={}
        for baseline in ['var_causal','city_mean','seasonal']:
            result[baseline]={'sensitivity_difference':interval_difference(alerts,'sheaf_causal',baseline,alerts.positive),'false_alarm_difference':interval_difference(alerts,'sheaf_causal',baseline,~alerts.positive)}
        save(region+'/paired_alert_summary.json',result)
        f=pd.read_csv(folder/'block_interval_cases.csv',dtype={'station':str});f['correct_when_decided']=((f.lower>60)==(f.truth>60))
        daily=f.groupby(['context_loss','method','day','station']).agg(all_covered=('covered','all'),target_hours=('covered','size'),full_set=('full_set','all')).reset_index()
        frame_save(region+'/block_interval_day_cases.csv',daily)
        frame_save(region+'/block_interval_day_summary.csv',daily.groupby(['context_loss','method']).agg(station_days=('all_covered','size'),simultaneous_coverage=('all_covered','mean'),full_set_fraction=('full_set','mean')).reset_index())
                                                                        
        status=json.loads((folder/'additional_status.json').read_text());status['hybrid_rows']=len(pd.read_csv(folder/'hybrid_cases.csv'));save(region+'/additional_status.json',status)
                                                                                       
                                                                                     
        X=ds['X'];Z=ds['Z'];rows=[];patterns=[]
        for day,ix in blocks(ds,2,history=0):
            for v,station in enumerate(ds['stations']):
                mask=~np.isfinite(X[ix,v]);pm_missing=mask[:,0]
                if pm_missing.any() and np.isfinite(X[ix,v,0]).any():patterns.append((day,v,mask))
        for source_day,v,mask in patterns:
            candidates=[(day,ix) for day,ix in blocks(ds,2,history=0) if day!=source_day and np.isfinite(X[ix,v]).all()]
            if not candidates:continue
            day,ix=candidates[len(rows)%len(candidates)];x=Z[ix].copy();x[:,v][mask]=np.nan
            pred=static_solve(x,m['sheaf'].S)[:,v,0]*ds['sd'][v,0]+ds['mu'][v,0]
            city=raw_city(x,v,ds['mu'],ds['sd'])[:,0]*ds['sd'][v,0]+ds['mu'][v,0]
            for j in np.flatnonzero(mask[:,0]):
                rows.append({'source_mask_day':source_day,'evaluation_day':day,'station':str(ds['stations'][v]),'timestamp':str(ds['times'][ix[j]]),'truth':X[ix[j],v,0],'sheaf_estimate':pred[j],'city_estimate':city[j],'other_target_channels_available':int(np.isfinite(x[j,v]).sum()),'mask_mismatch':False})
        ff=pd.DataFrame(rows,columns=['source_mask_day','evaluation_day','station','timestamp','truth','sheaf_estimate','city_estimate','other_target_channels_available','mask_mismatch']);frame_save(region+'/matched_missing_mask_cases.csv',ff)
        summary={'cases':len(ff),'source_patterns':len(patterns),'evaluated_patterns':ff.groupby(['source_mask_day','station']).ngroups if len(ff) else 0,'scope':'Exploratory transplantation of actual within-station pollutant/hour masks to a different fully observed test station-day; surviving context remains natural. Does not verify the original dark cells or remove complete-day selection bias.'}
        if len(ff):
            for k in ['sheaf','city']:summary[k+'_pm25_rmse']=np.sqrt(np.mean((ff[k+'_estimate']-ff.truth)**2))
            summary['gain_percent']=100*(1-summary['sheaf_pm25_rmse']/summary['city_pm25_rmse'])
        save(region+'/matched_missing_mask_summary.json',summary)
        save(region+'/final_analysis_status.json',{'status':'completed','scope':'Post-primary descriptive analyses; no parameter selection on test outcomes','matched_mask_rows':len(ff),'paired_alerts':result})
        print(region,'paired alerts, simultaneous intervals and actual-mask validation completed',flush=True)
if __name__=='__main__':main()
