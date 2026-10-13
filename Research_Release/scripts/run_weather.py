                                                                                

                                                                           
                                                                         
                                                                          
                                                             
   
import hashlib,json,sys,warnings
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sheafcode.cohort import select_cohort
from sheafcode.data import RAW4_FILES,haversine_km
from sheafcode.graph import knn_edges
from sheafcode.masked import MaskedSheaf
from sheafcode.baselines2 import GaussianConditional
from sheafcode.reliable import CachedSpatial,paired_bootstrap
warnings.filterwarnings('ignore',category=RuntimeWarning)
out=ROOT/'results/weather';out.mkdir(exist_ok=True)
ds=select_cohort('bengaluru_raw');n,p=ds.n,ds.p;ds.set_transform('none');Z=ds.Z();std=ds.std.copy()
tr=ds.rows(0);va=ds.rows(1);te=ds.rows(2);rng=np.random.default_rng(105)
fields=['AT','RH','WS','WD','RF','SR'];per={}
for f in sorted((ROOT/'data/raw/bengaluru_4station_cpcb_raw').glob('*.csv')):
    key=next((k for k in RAW4_FILES if f.name.lower().startswith(k)),None)
    if key is None or RAW4_FILES[key] not in ds.stations:continue
    d=pd.read_csv(f,na_values=['NA','None',''])
    cols={c.split(' (')[0]:c for c in d.columns}
    w=pd.DataFrame(index=pd.to_datetime(d.Timestamp,format='%Y-%m-%d %H:%M:%S'))
    for name in fields:
        w[name]=pd.to_numeric(d[cols[name]],errors='coerce').to_numpy() if name in cols else np.nan
    per.setdefault(RAW4_FILES[key],[]).append(w)
weather=np.full((len(ds.times),n,len(fields)),np.nan)
for v,s in enumerate(ds.stations):
    if s not in per:continue
    w=pd.concat(per[s]).sort_index();w=w[~w.index.duplicated(keep='first')].reindex(ds.times)
    weather[:,v]=w[fields].to_numpy()
                                                                   
bounds=[(-5,55),(0,100),(0,40),(0,360),(0,300),(0,1500)]
for j,(lo,hi) in enumerate(bounds):weather[:,:,j][(weather[:,:,j]<lo)|(weather[:,:,j]>hi)]=np.nan
coverage=np.isfinite(weather[tr]).mean(0)
pd.DataFrame(coverage,index=ds.stations,columns=fields).to_csv(out/'training_weather_coverage.csv')
D=haversine_km(ds.coords);edges=knn_edges(D,min(2,n-1))
choice=json.loads((ROOT/'results/bengaluru_raw/frozen_choices.json').read_text())['sheaf']
ds.set_transform(choice['transform']);Zm=ds.Z();lam=choice['lam']
base=MaskedSheaf(Zm[tr],edges,p,'reg',stalk='cca')
base_decoder=CachedSpatial(base.L+lam*base.P,n,p)
gchoice=json.loads((ROOT/'results/bengaluru_raw/frozen_choices.json').read_text())['gaussian']
ds.set_transform(gchoice['transform']);Zg=ds.Z()
gauss=GaussianConditional(Zg[tr],.1)
cov=(1-gchoice['shrink'])*gauss.S+gchoice['shrink']*np.diag(np.diag(gauss.S))
gaussian_decoder=CachedSpatial(np.linalg.inv(cov),n,p,gauss.mu)
ds.set_transform(choice['transform'])
regime_records=[];decisions=[];theory_checks=0;validation=[]
def loss(pr,truth,v):
    good=np.isfinite(truth);return float(np.mean(((pr-truth)/std[v])[good]**2)) if good.any() else np.inf
def days(rows,count):
    dates=ds.times[rows].floor('D');groups=[]
    for day in pd.unique(dates):
        ix=rows[dates==day]
        if len(ix)==24:groups.append((str(pd.Timestamp(day).date()),ix))
    return [groups[i] for i in np.unique(np.linspace(0,len(groups)-1,min(count,len(groups))).astype(int))]
vblocks=days(va,24);tblocks=days(te,60)
for v in range(n):
                                                                            
    others=[u for u in range(n) if u!=v]
    wind=np.nanmedian(weather[:,others,fields.index('WS')],axis=1)
    finite=np.isfinite(wind[tr])
    if finite.sum()<720:continue
    median=float(np.nanmedian(wind[tr]));iqr=float(np.nanquantile(wind[tr],.75)-np.nanquantile(wind[tr],.25))
    width=max(iqr,.1);observed_context=np.isfinite(wind)
    wind=np.where(observed_context,wind,median)
    low=tr[wind[tr]<=median];high=tr[wind[tr]>median]
    if min(len(low),len(high))<720:continue
    shlow=MaskedSheaf(Zm[low],edges,p,'reg',stalk='cca');shhigh=MaskedSheaf(Zm[high],edges,p,'reg',stalk='cca')
    Qlo=shlow.L;Qhi=shhigh.L;regularizer=lam*base.P
    best=(np.inf,None)
    for sharpness in [.5,1.,2.]:
        losses=[]
        for day,ix in vblocks:
            preds=[]
            for t in ix:
                alpha=float(1/(1+np.exp(-np.clip(sharpness*(wind[t]-median)/width,-30,30))))
                decoder=CachedSpatial((1-alpha)*Qlo+alpha*Qhi+regularizer,n,p)
                pr=decoder.predict(Zm[t:t+1],[v])[0]
                preds.append(ds.raw_from_Z(pr,[t],[v])[0,0])
            losses.append(loss(np.array(preds),ds.X[ix,v],v))
        eligible=[x for x in losses if np.isfinite(x)]
        score=float(np.mean(eligible)) if eligible else np.inf
        validation.append({'station':ds.stations[v],'sharpness':sharpness,'validation_mse':score,'eligible_validation_days':len(eligible)})
        if score<best[0]:best=(score,sharpness)
                                                               
    sharpness=best[1]
    if sharpness is None:continue
    for day,ix in tblocks:
        weather_preds=[]
        for t in ix:
            alpha=float(1/(1+np.exp(-np.clip(sharpness*(wind[t]-median)/width,-30,30))))
            Q=(1-alpha)*Qlo+alpha*Qhi+regularizer
            pr=CachedSpatial(Q,n,p).predict(Zm[t:t+1],[v])[0]
            weather_preds.append(ds.raw_from_Z(pr,[t],[v])[0,0])
                                                                                     
            if theory_checks<100:
                row=Zm[t].reshape(-1).copy();row[v*p:(v+1)*p]=np.nan
                U=np.flatnonzero(~np.isfinite(row));O=np.flatnonzero(np.isfinite(row))
                beta=.5;Qb=(1-beta)*Qlo+beta*Qhi+regularizer
                xa=row.copy();xb=row.copy()
                xa[U]=-np.linalg.solve(Q[np.ix_(U,U)],Q[np.ix_(U,O)]@row[O])
                xb[U]=-np.linalg.solve(Qb[np.ix_(U,U)],Qb[np.ix_(U,O)]@row[O])
                forcing=((Qlo-Qhi)@(xb))*(alpha-beta)
                identity=np.linalg.solve(Q[np.ix_(U,U)],forcing[U])
                if not np.allclose(xa[U]-xb[U],identity,atol=1e-8):raise AssertionError('Context perturbation identity failed')
                prior_lower=lam*np.linalg.eigvalsh(base.P[np.ix_(U,U)])[0]
                bound=np.linalg.norm(forcing[U])/prior_lower
                if np.linalg.norm(xa[U]-xb[U])>bound+1e-8:raise AssertionError('Context sensitivity bound failed')
                theory_checks+=1
        ordinary=ds.raw_from_Z(base_decoder.predict(Zm[ix],[v])[0],ix,[v])[:,0]
        ds.set_transform(gchoice['transform'])
        gaussian=ds.raw_from_Z(gaussian_decoder.predict(Zg[ix],[v])[0],ix,[v])[:,0]
        ds.set_transform(choice['transform'])
        for method,pr in [('weather_conditioned_sheaf',np.array(weather_preds)),('ordinary_sheaf',ordinary),('gaussian',gaussian)]:
            score=loss(pr,ds.X[ix,v],v)
            if not np.isfinite(score):continue
            regime_records.append({'day':day,'station':ds.stations[v],'method':method,'mse':score,
                'wind_context_observed_hours':int(observed_context[ix].sum())})
            if 'PM2.5' in ds.pollutants:
                q=ds.pollutants.index('PM2.5');good=np.isfinite(ds.X[ix,v,q])
                if good.sum()>=16:
                    actual=float(ds.X[ix[good],v,q].mean());estimate=float(pr[good,q].mean())
                    decisions.append({'day':day,'station':ds.stations[v],'method':method,'true_pm25':actual,'pred_pm25':estimate,
                        'miss':int(actual>60 and estimate<=60),'wrong':int((actual>60)!=(estimate>60)),
                        'absolute_error':abs(actual-estimate)})
R=pd.DataFrame(regime_records);R.to_csv(out/'weather_reconstruction.csv',index=False)
A=pd.DataFrame(decisions);A.to_csv(out/'weather_thresholds.csv',index=False)
pd.DataFrame(validation).to_csv(out/'weather_validation.csv',index=False)

                                                                    
                                                                           
forecast=[];fchoices=[]
calendar=np.stack([np.sin(2*np.pi*ds.times.hour/24),np.cos(2*np.pi*ds.times.hour/24),
                   np.sin(2*np.pi*ds.times.dayofyear/365.25),np.cos(2*np.pi*ds.times.dayofyear/365.25)],1)
def features(M):
    train=M[tr];mu=np.nanmedian(train,axis=0);mu=np.nan_to_num(mu)
    scale=np.nanstd(train,axis=0);scale=np.where(np.isfinite(scale)&(scale>1e-6),scale,1.)
    finite=np.isfinite(M);X=np.where(finite,(M-mu)/scale,0.)
    return np.column_stack([np.ones(len(M)),X,(~finite).astype(float),calendar])
weather_flat=weather.reshape(len(ds.times),-1)
                                                                             
wf=weather.copy();angle=np.deg2rad(wf[:,:,fields.index('WD')]);speed=wf[:,:,fields.index('WS')]
wf[:,:,fields.index('WD')]=-speed*np.sin(angle)
wf=np.concatenate([wf,(-speed*np.cos(angle))[:,:,None]],axis=2).reshape(len(ds.times),-1)
train_weather_cols=np.isfinite(wf[tr]).mean(0)>=.3
wp=wf[:,train_weather_cols]
for target in ['PM2.5','AT']:
    y=ds.X[:,:,ds.pollutants.index('PM2.5')] if target=='PM2.5' else weather[:,:,fields.index('AT')]
    raw_inputs=ds.X.reshape(len(ds.times),-1) if target=='PM2.5' else weather[:,:,fields.index('AT')]
    baseline_inputs=np.column_stack([raw_inputs,np.roll(y,24,axis=0)])
    baseline_inputs[:24,-n:]=np.nan
    richer=np.column_stack([baseline_inputs,wp])
    X0=features(baseline_inputs);X1=features(richer)
    for v in range(n):
        means=float(np.nanmean(y[tr,v]));yscale=float(np.nanstd(y[tr,v]));yscale=max(yscale,1e-6)
                                                                              
                                                                                 
        sets={part:np.flatnonzero((ds.split[:-1]==part)&(ds.split[1:]==part)&np.isfinite(y[1:,v])&np.isfinite(y[:-1,v])) for part in [0,1,2]}
        if min(len(sets[k]) for k in sets)<200:continue
        for name,X in [('ridge_without_weather',X0),('ridge_with_weather',X1)]:
            best=(np.inf,None,None)
            for ridge in [1.,10.,100.,1000.]:
                ii=sets[0];a=X[ii];label=(y[ii+1,v]-means)/yscale
                penalty=np.eye(a.shape[1])*ridge;penalty[0,0]=0
                coef=np.linalg.solve(a.T@a+penalty,a.T@label)
                jj=sets[1];pr=X[jj]@coef*yscale+means
                mse=float(np.mean((pr-y[jj+1,v])**2))
                if mse<best[0]:best=(mse,coef,ridge)
            fchoices.append({'target':target,'station':ds.stations[v],'method':name,'ridge':best[2],'validation_mse':best[0]})
            jj=sets[2];pr=X[jj]@best[1]*yscale+means
            table=pd.DataFrame({'day':ds.times[jj+1].date.astype(str),'squared_error':(pr-y[jj+1,v])**2})
            for day,g in table.groupby('day'):
                forecast.append({'day':day,'station':ds.stations[v],'target':target,'method':name,'mse':float(g.squared_error.mean()),'hours':len(g)})
        jj=sets[2];pr=y[jj,v];ok=np.isfinite(pr)
        table=pd.DataFrame({'day':ds.times[jj[ok]+1].date.astype(str),'squared_error':(pr[ok]-y[jj[ok]+1,v])**2})
        for day,g in table.groupby('day'):forecast.append({'day':day,'station':ds.stations[v],'target':target,'method':'persistence','mse':float(g.squared_error.mean()),'hours':len(g)})
F=pd.DataFrame(forecast);F.to_csv(out/'one_hour_forecasts.csv',index=False)
pd.DataFrame(fchoices).to_csv(out/'forecast_validation.csv',index=False)
summary={'study':'exploratory new meteorology analysis on previously examined Bengaluru archive',
    'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'stations':ds.stations,'meteorology':fields,
    'context_theorem_identity_and_bound_checks':theory_checks,
    'weather_reconstruction':paired_bootstrap(R,'weather_conditioned_sheaf','ordinary_sheaf') if len(R) else {'n':0},
    'weather_reconstruction_vs_gaussian':paired_bootstrap(R,'weather_conditioned_sheaf','gaussian') if len(R) else {'n':0},
    'threshold_error':paired_bootstrap(A,'weather_conditioned_sheaf','ordinary_sheaf',value='absolute_error') if len(A) else {'n':0},
    'one_hour_forecasting':{target:{ref:paired_bootstrap(F[F.target==target],'ridge_with_weather',ref) for ref in ['ridge_without_weather','persistence']} for target in ['PM2.5','AT']},
    'limitations':['one metropolitan area; three selected stations','physical concentration interpretation applies only to raw records',
      'weather-context reconstruction is still pollution reconstruction, not forecasting','the forecast model is ridge regression, not sheaf forecasting',
      'no comparison to operational numerical weather prediction; no field intervention or demonstrated pollution reduction']}
(out/'weather_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2),flush=True)
