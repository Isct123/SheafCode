import sys,itertools,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
def score(pred,truth,sd):
    good=np.isfinite(truth)
    if not np.isfinite(pred).all():raise RuntimeError('Nonfinite predictions')
    raw=(pred[:,0]-truth[:,0])*sd[0];q=good[:,0]
    return {'mse':float(np.mean((pred-truth)[good]**2)),'pm25_raw_mse':float(np.mean(raw[q]**2)) if q.any() else np.nan,'target_cells':int(good.sum()),'pm25_cells':int(q.sum())}
def predict(x,v,hist,models,ds,allmethods=True):
    sheaf=models['sheaf'];var=models['var'];hidden=x.copy();hidden[hist:,v]=np.nan
    ans={'sheaf_causal':sheaf.causal(hidden,v,start=hist),'var_causal':var.filter(hidden)[hist:,v],'seasonal':seasonal(hidden,v,start=hist),'city_mean':raw_city(hidden[hist:],v,ds['mu'],ds['sd'])}
    if allmethods:
        ans.update(sheaf_retrospective=sheaf.solve(hidden)[hist:,v],spatial_sheaf=static_solve(hidden[hist:],sheaf.S)[:,v],gaussian=static_solve(hidden[hist:],models['gaussian'])[:,v])
    return ans
def main():
    for region in regions():
        ds=load(region);m=models(region);Z=ds['Z'];n,p=Z.shape[1:];lock(region+'/evaluation_lock.json',{'source_sha256':sha(__file__),'common_sha256':sha(HERE/'common.py'),'dataset_sha256':sha(OUT/region/'dataset.npz'),'model_sha256':sha(OUT/region/'models.pkl')})
        dest=OUT/region/'outage_cases.csv';old=pd.read_csv(dest,dtype={'station':str}) if dest.exists() else pd.DataFrame();rows=old.to_dict('records');done=set(zip(old.day,old.station)) if len(old) else set();excluded=[];planned=blocks(ds,2)
        primary={day for day,_ in planned[::3]}
        for day,ix in planned:
            for v,station in enumerate(ds['stations']):
                if (day,str(station)) in done:continue
                truth=Z[ix[24:],v];good=np.isfinite(truth)
                if good.sum()<12*p:excluded.append({'day':day,'station':str(station),'cells':int(good.sum())});continue
                preds=predict(Z[ix],v,24,m,ds)
                for method,pred in preds.items():rows.append({'region':region,'day':day,'station':str(station),'method':method,'primary':day in primary,**score(pred,truth,ds['sd'][v])})
                frame_save(region+'/outage_cases.csv',pd.DataFrame(rows))
            print(region,'test day',day,'station-days',len(rows)//7,flush=True)
        frame_save(region+'/outage_exclusions.csv',pd.DataFrame(excluded));f=pd.DataFrame(rows);comps={}
        for scope,ff in [('primary_every_third_day',f[f.primary]),('all_eligible_test_days',f)]:
            comps[scope]={name:paired(ff,'sheaf_causal',name) for name in ['var_causal','seasonal','city_mean','gaussian','spatial_sheaf']}
            comps[scope]['pm25_physical']={name:paired(ff,'sheaf_causal',name,value='pm25_raw_mse') for name in ['var_causal','seasonal','city_mean']}
            comps[scope]['retrospective_vs_causal']={'scope':'Different information sets; descriptive only','gain':paired(ff,'sheaf_retrospective','sheaf_causal')}
        save(region+'/reconstruction_summary.json',comps)
                                                                      
        rows=[];test=np.flatnonzero(ds['split']==2)[::24]
        for t in test:
            for v,station in enumerate(ds['stations']):
                for mask_type in ['station','pm25_channel']:
                    truth=Z[t:t+1,v];good=np.isfinite(truth)
                    if mask_type=='pm25_channel':good[:,1:]=False
                    if not good.any():continue
                    x=Z[t:t+1].copy()
                    if mask_type=='station':x[:,v]=np.nan
                    else:x[:,v,0]=np.nan
                    preds={'spatial_sheaf':static_solve(x,m['sheaf'].S)[:,v],'gaussian':static_solve(x,m['gaussian'])[:,v],'city_mean':raw_city(x,v,ds['mu'],ds['sd'])}
                    for method,pred in preds.items():rows.append({'day':str(ds['times'][t].date()),'station':str(station),'timestamp':str(ds['times'][t]),'mask_type':mask_type,'method':method,'mse':float(np.mean((pred-truth)[good]**2)),'cells':int(good.sum())})
        frame_save(region+'/snapshot_channel_cases.csv',pd.DataFrame(rows))
                                                                               
        stress=[];rng=np.random.default_rng(881)
        for history,horizon,loss in itertools.product([0,24],[24,72,168],[0.,.2,.5]):
            for day,ix in subset_days(blocks(ds,2,history=history,horizon=horizon),6):
                for v,station in enumerate(ds['stations']):
                    x=Z[ix].copy();drop=rng.random((n,p))<loss;drop[v]=False;x[:,drop]=np.nan;truth=Z[ix[history:],v];good=np.isfinite(truth)
                    if good.sum()<max(12,horizon//2)*p:continue
                    for method,pred in predict(x,v,history,m,ds,False).items():stress.append({'day':day,'station':str(station),'history':history,'horizon':horizon,'context_loss':loss,'method':method,**score(pred,truth,ds['sd'][v])})
            frame_save(region+'/history_horizon_context_cases.csv',pd.DataFrame(stress));print(region,'stress',history,horizon,loss,flush=True)
        save(region+'/evaluation_status.json',{'status':'completed','outage_rows':len(f),'snapshot_rows':len(rows),'stress_rows':len(stress),'all_outcomes_retained':True})
if __name__=='__main__':main()
