import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
from evaluate import predict
def category(x):return int(np.searchsorted([30.,60.,90.,120.,250.],x,side='left'))
def stream(ds,m,part):
    rows=[];daily=[]
    for day,ix in blocks(ds,part):
        for v,station in enumerate(ds['stations']):
            truth=ds['X'][ix[24:],v,0];good=np.isfinite(truth)
            if good.sum()<12:continue
            preds=predict(ds['Z'][ix],v,24,m,ds,False)
            for method,pred in preds.items():
                raw=pred[:,0]*ds['sd'][v,0]+ds['mu'][v,0]
                for j,t in enumerate(ix[24:]):
                    if good[j]:rows.append({'part':part,'day':day,'timestamp':str(ds['times'][t]),'station':str(station),'method':method,'prediction':raw[j],'truth':truth[j],'positive':truth[j]>60})
                if good.sum()>=18:
                    prior=ds['X'][ix[:24],v,0];previous=np.nanmean(prior) if np.isfinite(prior).sum()>=18 else np.nan
                    daily.append({'day':day,'station':str(station),'method':method,'truth_mean':np.mean(truth[good]),'prediction_mean':np.mean(raw[good]),'truth_category':category(np.mean(truth[good])),'predicted_category':category(np.mean(raw[good])),'previous_day_mean':previous,'target_hours':int(good.sum()),'aggregation':'Common observed target hours; PM2.5-only category proxy, not official all-pollutant AQI'})
    return pd.DataFrame(rows),pd.DataFrame(daily)
def main():
    for region in regions():
        ds=load(region);m=models(region);cal,_=stream(ds,m,3);test,daily=stream(ds,m,2);choices={};summary=[]
        for method,g in cal.groupby('method'):
            cutoff=quantile(g.loc[~g.positive,'prediction'],.05);choices[method]={'cutoff':cutoff,'validation_negatives':int((~g.positive).sum()),'validation_positives':int(g.positive.sum())}
            q=test[test.method==method].copy();q['alarm']=q.prediction>cutoff;test.loc[q.index,'alarm']=q.alarm
            pos=q.positive;summary.append({'method':method,'cases':len(q),'positive_cases':int(pos.sum()),'negative_cases':int((~pos).sum()),'sensitivity':q.loc[pos,'alarm'].mean() if pos.any() else None,'false_alarm_rate':q.loc[~pos,'alarm'].mean() if (~pos).any() else None,'precision':q.loc[q.alarm,'positive'].mean() if q.alarm.any() else None})
        frame_save(region+'/hourly_alert_predictions.csv',test);frame_save(region+'/alert_summary.csv',pd.DataFrame(summary));save(region+'/alert_cutoffs.json',choices)
        if len(daily):
            matched=daily[daily.previous_day_mean.notna()];summ=[]
            for method,g in matched.groupby('method'):
                prev=[category(v) for v in g.previous_day_mean];summ.append({'method':method,'common_station_days':len(g),'category_accuracy':np.mean(g.truth_category==g.predicted_category),'previous_day_accuracy':np.mean(g.truth_category==prev),'poor_or_worse_truth_days':int((g.truth_category>=3).sum()),'missed_poor_rate':np.mean(g.loc[g.truth_category>=3,'predicted_category']<3) if (g.truth_category>=3).any() else None})
            frame_save(region+'/pm25_daily_predictions.csv',daily);frame_save(region+'/pm25_daily_summary.csv',pd.DataFrame(summ))
                                                                                             
        snapshots=[];Z=ds['Z'];Q=m['sheaf'].S
        for t in np.flatnonzero(ds['split']==2):
            pred=static_solve(Z[t:t+1],Q)[0];city_preds=[raw_city(Z[t:t+1],v,ds['mu'],ds['sd'])[0,0] for v in range(len(ds['stations']))]
            for v,station in enumerate(ds['stations']):
                if np.isfinite(Z[t,v,0]):continue
                snapshots.append({'day':str(ds['times'][t].date()),'timestamp':str(ds['times'][t]),'station':str(station),'estimate':pred[v,0]*ds['sd'][v,0]+ds['mu'][v,0],'city_estimate':city_preds[v]*ds['sd'][v,0]+ds['mu'][v,0],'truth_unavailable':True})
        frame_save(region+'/natural_missing_pm25_estimates.csv',pd.DataFrame(snapshots))
        save(region+'/environment_status.json',{'status':'completed','calibration_alert_rows':len(cal),'test_alert_rows':len(test),'daily_proxy_rows':len(daily),'unverified_missing_cell_rows':len(snapshots),'reference_threshold':'Hourly PM2.5 >60 ug/m3; illustrative, not official daily standard','decision_scope':'PM2.5-only daily proxy, not CPCB AQI or GRAP. No actual pollution intervention or health benefit measured.'})
        print(region,'environmental comparisons',len(test),'hour rows',len(daily),'daily rows',flush=True)
if __name__=='__main__':main()
