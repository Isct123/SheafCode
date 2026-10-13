                                                                               
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
from sheafcode.reliable import CachedSpatial
def negative_cutoff(scores,truth,alpha=.05):
    scores=np.asarray(scores,float);truth=np.asarray(truth,bool);neg=np.sort(scores[~truth]);k=int(np.ceil((len(neg)+1)*(1-alpha)))
    if not len(neg) or k>len(neg):return np.inf
    return float(neg[k-1])
def episodes(truth,alert,valid):
                                                                          
                                                                               
    out=[];i=0
    while i<len(truth):
        if not valid[i] or not truth[i]:i+=1;continue
        start=i
        while i<len(truth) and valid[i] and truth[i]:i+=1
        hit=np.flatnonzero(alert[start:i]);out.append({'start':start,'length_hours':i-start,'detected':bool(len(hit)),'delay_hours':int(hit[0]) if len(hit) else None,'left_truncated':start==0,'right_truncated':i==len(truth)})
    return out
def main():
    rng=np.random.default_rng(922);rows=[];eventrows=[];settings={}
    for region in REGIONS:
        ds,m,folder=cohort(region);ds.set_transform(m['choices']['sheaf']['transform']);Z=ds.Z();model=np.load(folder/'model.npz');decoder=CachedSpatial(model['L']+model['lam']*model['P'],ds.n,ds.p);q=ds.pollutants.index('PM2.5')
        for horizon in [24,72,168]:
            for rate in [0.,.2,.5]:
                parts={}
                for part in [1,2]:
                    ids=ds.rows(part);cases=[]
                    for start in range(0,len(ids)-horizon+1,horizon):
                        ix=ids[start:start+horizon]
                        for v in range(ds.n):
                            hidden=Z[ix].copy();hidden[:,v]=np.nan
                            candidates=[(w,c) for w in range(ds.n) if w!=v for c in range(ds.p)];k=int(round(rate*len(candidates)))
                            for j in rng.choice(len(candidates),k,replace=False):w,c=candidates[j];hidden[:,w,c]=np.nan
                            pred=decoder.predict(hidden,[v])[0];point=ds.raw_from_Z(pred,ix,[v])[:,0,q]
                            other=ds.raw_from_Z(hidden,ix,list(range(ds.n)))[:,:,q]
                            count=np.isfinite(other).sum(1);fallback=np.divide(np.nansum(other,axis=1),count,out=np.full(len(other),np.nan),where=count>0)
                            truth=ds.X[ix,v,q];common=np.isfinite(truth)&np.isfinite(point)&np.isfinite(fallback)
                            for method,estimate in [('sheaf',point),('city_mean',fallback)]:
                                cases.append({'ix':ix,'v':v,'method':method,'prediction':estimate,'truth':truth,'common':common,'actual_hidden_context_fraction':k/len(candidates)})
                    parts[part]=cases
                cutoffs={}
                for method in ['sheaf','city_mean']:
                    cal=[c for c in parts[1] if c['method']==method];scores=np.concatenate([c['prediction'][c['common']] for c in cal]);truth=np.concatenate([c['truth'][c['common']]>60 for c in cal]);cutoffs[method]=negative_cutoff(scores,truth)
                key=f'{region}_{horizon}_{rate}';settings[key]={'cutoffs':cutoffs,'model_sha256':digest(folder/'model.npz'),'calibration_blocks':len(parts[1])//2};save('alert_choices.json',settings)
                for c in parts[2]:
                    alert=c['prediction']>cutoffs[c['method']];positive=c['truth']>60
                    for j,t in enumerate(c['ix']):
                        if not c['common'][j]:continue
                        rows.append({'region':region,'horizon':horizon,'missing_rate':rate,'actual_context_hidden_fraction':c['actual_hidden_context_fraction'],'method':c['method'],'day':str(ds.times[t].date()),'timestamp':str(ds.times[t]),'station':ds.stations[c['v']],'truth':c['truth'][j],'prediction':c['prediction'][j],'positive':positive[j],'alert':alert[j],'outage_hour':j+1})
                    for event in episodes(positive,alert,c['common']):
                        eventrows.append({'region':region,'horizon':horizon,'missing_rate':rate,'method':c['method'],'station':ds.stations[c['v']],'timestamp':str(ds.times[c['ix'][event['start']]]),**event})
                print(region,'operational alerts',horizon,rate,'complete',flush=True)
    f=pd.DataFrame(rows);f.to_csv(OUT/'alert_cases.csv',index=False);e=pd.DataFrame(eventrows);e.to_csv(OUT/'alert_episodes.csv',index=False);summary=[]
    for key,g in f.groupby(['region','horizon','missing_rate','method']):
        pos=g[g.positive];neg=g[~g.positive];ev=e[(e.region==key[0])&(e.horizon==key[1])&(e.missing_rate==key[2])&(e.method==key[3])];hit=ev[ev.detected]
        summary.append({'region':key[0],'horizon':key[1],'missing_rate':key[2],'method':key[3],'cases':len(g),'positive_cases':len(pos),'negative_cases':len(neg),'sensitivity':float(pos.alert.mean()) if len(pos) else None,'false_positive_rate':float(neg.alert.mean()) if len(neg) else None,'positive_episodes':len(ev),'episode_detection_rate':float(ev.detected.mean()) if len(ev) else None,'median_delay_detected_hours':float(hit.delay_hours.median()) if len(hit) else None,'truncated_episodes':int((ev.left_truncated|ev.right_truncated).sum())})
    pd.DataFrame(summary).to_csv(OUT/'alert_summary.csv',index=False)
                                                                              
    comparisons=[];rng=np.random.default_rng(1902)
    for key,g in f.groupby(['region','horizon','missing_rate']):
        p=g.pivot(index=['timestamp','station','positive'],columns='method',values='alert').dropna().reset_index();p['week']=pd.to_datetime(p.timestamp).dt.to_period('W-SUN').astype(str)
        groups=[]
        for _,w in p.groupby('week'):
            positive=w.positive.to_numpy(bool);a=w.sheaf.to_numpy(bool);b=w.city_mean.to_numpy(bool)
            groups.append([positive.sum(),(~positive).sum(),(a&positive).sum(),(b&positive).sum(),(a&~positive).sum(),(b&~positive).sum()])
        arr=np.asarray(groups,float);total=arr.sum(0);draw=[]
        for _ in range(1000):
            v=arr[rng.integers(0,len(arr),len(arr))].sum(0)
            if v[0]>0 and v[1]>0:draw.append([(v[2]-v[3])/v[0],(v[4]-v[5])/v[1]])
        ci=np.quantile(draw,[.025,.975],axis=0) if draw else np.full((2,2),np.nan)
        comparisons.append({'region':key[0],'horizon':key[1],'missing_rate':key[2],'weeks':len(arr),'sensitivity_difference':(total[2]-total[3])/total[0] if total[0] else None,'sensitivity_ci_low':ci[0,0],'sensitivity_ci_high':ci[1,0],'fpr_difference':(total[4]-total[5])/total[1] if total[1] else None,'fpr_ci_low':ci[0,1],'fpr_ci_high':ci[1,1],'valid_draws':len(draw)})
    pd.DataFrame(comparisons).to_csv(OUT/'alert_paired_week_intervals.csv',index=False)
    save('alert_scope.json',{'source_sha256':digest(__file__),'scope':'Validation-fixed deterministic hourly illustrative PM2.5>60 cutoffs; test outcomes only score; 24/72/168h target outages and persistent context-channel losses; same eligible cases/masks; calendar-week paired bootstrap exploratory, no familywise claim. Episode delays conditional on detection; boundary-truncated episodes explicitly counted. Previously examined archives, no official AQI/GRAP actions or emissions/health benefit.'})
if __name__=='__main__':main()
