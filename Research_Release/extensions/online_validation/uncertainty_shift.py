                                                                            
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
from sheafcode.reliable import CachedSpatial

def quantile(x,alpha=.1):
    values=np.sort(np.asarray(x));k=int(np.ceil((len(values)+1)*(1-alpha)))
    return float(values[k-1]) if k<=len(values) and len(values) else float('inf')

def cluster_ci(g):
    arr=np.array([[len(w),w.covered.sum()] for _,w in g.groupby('week')]);rng=np.random.default_rng(825)
    if not len(arr):return [None,None]
    draws=arr[rng.integers(0,len(arr),(1000,len(arr)))].sum(1)
    return np.quantile(draws[:,1]/draws[:,0],[.025,.975]).tolist()

def main():
    lock('uncertainty_lock.json',{'source_sha256':sha(__file__),'seed':9218,'alpha':.1,
         'models':{r:sha(ROOT/'results_expanded'/r/'model.npz') for r in ['delhi2025','bengaluru_raw']},
         'calibration':'Final chronological half of validation; original point-model selection had used validation, so no independent split-conformal guarantee.'})
    frames=[];settings={};rng=np.random.default_rng(9218)
    for region in ['delhi2025','bengaluru_raw']:
        ds,m,folder=local(region);Z=ds.Z();model=np.load(folder/'model.npz');q=ds.pollutants.index('PM2.5')
        decoder=CachedSpatial(model['L']+model['lam']*model['P'],ds.n,ds.p,cache_limit=4096)
        validation=ds.rows(1);cal=validation[len(validation)//2:];test=ds.rows(2)
        train95=np.nanquantile(ds.X[ds.split==0,:,q],.95,axis=0)
        for rate in [0.,.2,.5]:
            destination=OUT/f'uncertainty_{region}_{rate:g}_cases.csv'
                                                                           
            context={part:Z[ix].copy() for part,ix in [('cal',cal),('test',test)]}
            for name in context:
                mask=rng.random(context[name].shape)<rate;context[name][mask]=np.nan
            records=[];scores=[];localcases=[]
            for v,station in enumerate(ds.stations):
                outputs={}
                for name,indices in [('cal',cal),('test',test)]:
                    pred,gain=decoder.predict(context[name],[v]);point=ds.raw_from_Z(pred,indices,[v])[:,0,q]
                    truth=ds.X[indices,v,q];ok=np.isfinite(truth)
                    outputs[name]=(point,truth,ok,gain[:,0,q],indices)
                point,truth,ok,gain,_=outputs['cal'];error=np.abs(point[ok]-truth[ok]);scores.extend(error.tolist())
                fixed=quantile(error);scaled=quantile(error/gain[ok]);settings[f'{region}_{rate}_{v}']={'calibration_cases':int(ok.sum()),'station_quantile':fixed,'scaled_quantile':scaled,'train_extreme_pm25':float(train95[v])}
                point,truth,ok,gain,indices=outputs['test'];ids=np.flatnonzero(ok)
                localcases.append((station,point,truth,gain,indices,ids,v,fixed,scaled))
            pooled=quantile(scores)
            for station,point,truth,gain,indices,ids,v,fixed,scaled in localcases:
                for method,width in [('pooled_fixed',np.full(len(indices),pooled)),('station_fixed',np.full(len(indices),fixed)),('station_availability',scaled*gain)]:
                    selected=indices[ids];lower=point[ids]-width[ids];upper=point[ids]+width[ids]
                    records.append(pd.DataFrame({'region':region,'timestamp':ds.times[selected].astype(str),'station':station,'extra_missing_rate':rate,'method':method,
                        'truth':truth[ids],'prediction':point[ids],'lower':lower,'upper':upper,'width':2*width[ids],
                        'covered':(lower<=truth[ids])&(truth[ids]<=upper),'decided':(lower>60)|(upper<=60),
                        'decision_correct':np.where(lower>60,truth[ids]>60,np.where(upper<=60,truth[ids]<=60,False)),
                        'calendar_quarter':ds.times[selected].to_period('Q').astype(str),
                        'week':ds.times[selected].to_period('W-SUN').astype(str),'train_defined_extreme':truth[ids]>train95[v]}))
            frame=pd.concat(records,ignore_index=True);frame.to_csv(destination,index=False);frames.append(frame)
            print(region,'no-feedback shift cases',rate,len(frame),flush=True)
    save('uncertainty_choices.json',settings);summary=[]
    for frame in frames:
        for key,g in frame.groupby(['region','extra_missing_rate','method']):
            groups=[('all','all',g)]
            groups.extend(('quarter',str(k),q) for k,q in g.groupby('calendar_quarter'))
            groups.extend(('train_extreme',str(k),q) for k,q in g.groupby('train_defined_extreme'))
            for kind,label,q in groups:
                ci=cluster_ci(q);summary.append(dict(zip(['region','extra_missing_rate','method'],key))|{'group_type':kind,'group':label,
                    'cases':len(q),'stations':q.station.nunique(),'weeks':q.week.nunique(),'coverage':q.covered.mean(),
                    'coverage_ci_low':ci[0],'coverage_ci_high':ci[1],'mean_width':q.width.mean(),
                    'decided_fraction':q.decided.mean(),'decided_cases':int(q.decided.sum()),
                    'accuracy_when_decided':q.loc[q.decided,'decision_correct'].mean() if q.decided.any() else None,
                    'infinite_interval_fraction':float((~np.isfinite(q.width)).mean())})
    pd.DataFrame(summary).to_csv(OUT/'uncertainty_shift_summary.csv',index=False)
    save('uncertainty_scope.json',{'status':'completed','case_rows':sum(len(f) for f in frames),'methods':3,
        'scope':'All observed test PM2.5 target cells scored under artificial whole-station snapshot outages and fixed additional context losses. Final half-validation calibration, no outage label feedback. Original validation-selected point model reuses validation; intervals are empirical, not an independent split-conformal guarantee. Extreme groups use training 95th percentiles but test truth for retrospective stratification. Weekly intervals exploratory.'})

if __name__=='__main__':main()
