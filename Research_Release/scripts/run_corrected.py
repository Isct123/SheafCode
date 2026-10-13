                                                                                 
import argparse
import hashlib
import itertools
import json
import os
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from sheafcode.cohort import select_cohort
from sheafcode.masked import MaskedSheaf
from sheafcode.baselines2 import GaussianConditional,Kriging2,IDW2
from sheafcode.data import haversine_km
from sheafcode.graph import knn_edges
from sheafcode.reliable import CachedSpatial,AdaptiveQuantile,paired_bootstrap
from sheafcode.benchmark import StableVAR,seasonal_predict

warnings.filterwarnings('ignore',category=RuntimeWarning)
ap=argparse.ArgumentParser()
ap.add_argument('--region',required=True)
ap.add_argument('--days',type=int,default=24)
ap.add_argument('--stations',type=int,default=12)
ap.add_argument('--run-days',type=int,default=12)
ap.add_argument('--run-stations',type=int,default=6)
ap.add_argument('--seed',type=int,default=37)
ap.add_argument('--output',default='results')
args=ap.parse_args(); started=time.time()
out=ROOT/args.output/args.region; out.mkdir(parents=True,exist_ok=True)

def write(name,obj):
    (out/name).write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x),allow_nan=False),encoding='utf-8')

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def log(s):
    print(f'[{args.region}] {s} ({time.time()-started:.0f}s)',flush=True)

def finite_json(x):
    if isinstance(x,dict):return {str(k):finite_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [finite_json(v) for v in x]
    if isinstance(x,np.ndarray):return finite_json(x.tolist())
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    return x

code={str(p.relative_to(ROOT)):digest(p) for folder in ['sheafcode','scripts'] for p in (ROOT/folder).glob('*.py')}
data={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'data/raw').rglob('*') if p.is_file()}
manifest={'region':args.region,'args':vars(args),'started_utc':pd.Timestamp.now(tz='UTC').isoformat(),
          'protocol_sha256':digest(ROOT/'docs/EXPERIMENT_PROTOCOL.md'),'code_sha256':code,'data_sha256':data,
          'status':'running','previous_test_data_seen':True}
write('manifest.json',manifest)
try:
    ds=select_cohort(args.region)
except ValueError as e:
    manifest.update(status='excluded',reason=str(e)); write('manifest.json',manifest); log(str(e)); sys.exit(0)
n,p=ds.n,ds.p; rng=np.random.default_rng(args.seed)
stations=list(range(n)) if n<=args.stations else sorted(rng.choice(n,args.stations,replace=False).tolist())
selection=dict(ds.selection); selection['coverage']={str(k):v for k,v in selection['coverage'].items()}
write('cohort.json',selection)
np.savez_compressed(out/'dataset.npz',X=ds.X,split=ds.split,times=ds.times.asi8)
log(f'cohort {n} stations x {p} pollutants; {len(ds.times)} consecutive hours')
D=haversine_km(ds.coords); tr=ds.rows(0); va=ds.rows(1); te=ds.rows(2)
ds.set_transform('none'); raw_mean=ds.mean.copy(); raw_std=ds.std.copy(); Zraw=ds.Z().copy()
spaces={}
for transform in ['none','diurnal']:
    ds.set_transform(transform)
    spaces[transform]={'Z':ds.Z().copy(),'mean':ds.mean.copy(),'std':ds.std.copy(),'hod':None if ds.hod is None else ds.hod.copy()}

def raw(pred,rows,E,transform):
    s=spaces[transform]; val=pred*s['std'][E]+s['mean'][E]
    if s['hod'] is not None:val+=s['hod'][ds.times[rows].hour.to_numpy()][:,E,:]
    return np.expm1(val) if transform!='none' else val

def score(pred,rows,E,transform):
    truth=ds.X[rows][:,E]; good=np.isfinite(truth)
    estimate=raw(pred,rows,E,transform)
    good &= np.isfinite(estimate)
    if not good.any():return np.inf
    return float(np.mean(((estimate-truth)/raw_std[E])[good]**2))

vrows=va[np.unique(np.linspace(0,len(va)-1,min(192,len(va))).astype(int))]
vstations=stations[:min(6,len(stations))]
vlog=[]; models={}; chosen={}
best=(np.inf,None,None,None)
for transform,space in spaces.items():
    for k in sorted(set([min(3,n-1),min(5,n-1)])):
        edges=knn_edges(D,k)
        sh=MaskedSheaf(space['Z'][tr],edges,p,'reg',stalk='cca')
        for lam in [.1,1.,10.]:
            decoder=CachedSpatial(sh.L+lam*sh.P,n,p)
            loss=np.mean([score(decoder.predict(space['Z'][vrows],[v])[0],vrows,[v],transform) for v in vstations])
            rec={'method':'sheaf','transform':transform,'k':k,'lam':lam,'validation_mse':float(loss)}; vlog.append(rec)
            if loss<best[0]:best=(loss,sh,decoder,rec)
_,sh,decoder,c=best; transform=c['transform']; Z=spaces[transform]['Z']; lam=c['lam']; edges=sh.edges
models['sheaf']=(transform,lambda B,E:decoder.predict(B,E)[0]); chosen['sheaf']=c
bestg=(np.inf,None,None,None)
for tg,space in spaces.items():
    base=GaussianConditional(space['Z'][tr],.1)
                                                                                     
    for shrink in [.05,.2,.5]:
        S=(1-shrink)*base.S+shrink*np.diag(np.diag(base.S))
        dec=CachedSpatial(np.linalg.inv(S),n,p,base.mu)
        loss=np.mean([score(dec.predict(space['Z'][vrows],[v])[0],vrows,[v],tg) for v in vstations])
        rec={'method':'gaussian','transform':tg,'shrink':shrink,'validation_mse':float(loss)}; vlog.append(rec)
        if loss<bestg[0]:bestg=(loss,dec,tg,rec)
_,gauss,tg,g=bestg
models['gaussian']=(tg,lambda B,E:gauss.predict(B,E)[0]); chosen['gaussian']=g
for name in ['kriging','idw']:
    best=(np.inf,None,None,None)
    for t,space in spaces.items():
        model=Kriging2(space['Z'][tr],D) if name=='kriging' else IDW2(D)
        loss=np.mean([score(model.reconstruct(space['Z'][vrows],[v]),vrows,[v],t) for v in vstations])
        rec={'method':name,'transform':t,'validation_mse':float(loss)}; vlog.append(rec)
        if loss<best[0]:best=(loss,model,t,rec)
    _,model,t,rec=best; models[name]=(t,model.reconstruct); chosen[name]=rec
log(f'spatial choices frozen: {c}; Gaussian {g}')

def blocks(part,count,history=0):
    rows=ds.rows(part); dates=ds.times[rows].floor('D'); ans=[]
    for day in pd.unique(dates):
        ix=rows[dates==day]
        if len(ix)!=24:continue
        if history:
            if ix[0]<history or ds.split[ix[0]-history]!=part:continue
            ix=np.arange(ix[0]-history,ix[-1]+1)
        ans.append((str(pd.Timestamp(day).date()),ix))
    ids=np.unique(np.linspace(0,len(ans)-1,min(count,len(ans))).astype(int)) if ans else []
    return [ans[i] for i in ids]

vb=blocks(1,4,24); tb=blocks(2,args.run_days,24); runstations=stations[:min(args.run_stations,len(stations))]
                                                                        
def eligible(ix,v):return np.isfinite(ds.X[ix[24:],v]).sum()>=12*p

temporal_choices={}; temporal_models={}
for mode in ['retrospective','causal']:
    best=(np.inf,None,None)
    for ts in [.5,2.,8.]:
        st=MaskedSheaf(Z[tr],edges,p,'reg',temporal=True,t_scale=ts,stalk='cca')
        losses=[]
        for day,ix in vb:
            for v in runstations[:min(3,len(runstations))]:
                if not eligible(ix,v):continue
                B=Z[ix]; hide=np.zeros((48,n),bool); hide[24:,v]=True
                if mode=='retrospective':pred=st.reconstruct_mask(B,hide,lam)[24:,v][:,None,:]
                else:
                                                                                    
                    horizons=[24,28,32,36,40,47]
                    pr=np.array([st.reconstruct_mask(B[:h+1],hide[:h+1],lam)[-1,v] for h in horizons])[:,None,:]
                    losses.append(score(pr,ix[horizons],[v],transform)); continue
                losses.append(score(pred,ix[24:],[v],transform))
        loss=float(np.mean(losses)) if losses else np.inf
        vlog.append({'method':'temporal_sheaf','mode':mode,'t_scale':ts,'validation_mse':loss})
        if loss<best[0]:best=(loss,st,ts)
    if best[1] is None:raise ValueError('No eligible validation prolonged runs')
    temporal_models[mode]=best[1]; temporal_choices[mode]={'t_scale':best[2],'validation_mse':best[0]}

kalchoices={}; kalmodels={}
for ridge in [1.,100.]:
    for noise in [.05,.5]:
        kal=StableVAR(Z[tr],ridge,noise)
        for mode in ['retrospective','causal']:
            losses=[]
            for day,ix in vb:
                for v in runstations[:min(3,len(runstations))]:
                    if not eligible(ix,v):continue
                    B=Z[ix].copy(); B[24:,v]=np.nan
                    pr=kal.smooth(B,causal=mode=='causal')[24:,v][:,None,:]
                    losses.append(score(pr,ix[24:],[v],transform))
            loss=float(np.mean(losses)) if losses else np.inf
            rec={'ridge':ridge,'observation_noise':noise,'validation_mse':loss}; vlog.append({'method':'stable_var','mode':mode,**rec})
            if mode not in kalchoices or loss<kalchoices[mode]['validation_mse']:
                kalchoices[mode]=rec; kalmodels[mode]=kal
pd.DataFrame(vlog).to_csv(out/'validation.csv',index=False)
chosen.update(temporal_sheaf=temporal_choices,stable_var=kalchoices)
write('frozen_choices.json',finite_json(chosen))
np.savez_compressed(out/'model.npz',delta=sh.delta,L=sh.L,P=sh.P,lam=lam)
manifest.update(cohort=selection,target_stations=[ds.stations[v] for v in stations],choices=chosen,
                choices_frozen_utc=pd.Timestamp.now(tz='UTC').isoformat())
manifest['cohort']['coverage']={str(k):v for k,v in manifest['cohort']['coverage'].items()}
write('manifest.json',finite_json(manifest)); log('all validation choices frozen; entering evaluation')

records=[]; decisions=[]; priorities=[]; exposure=[]; interval=[]
testblocks=blocks(2,args.days)
                                                                         
                                                                               
for v in stations:
    pred_val,gain_val=decoder.predict(Z[va],[v]); pred_test,gain_test=decoder.predict(Z[te],[v])
    pred_val=raw(pred_val,va,[v],transform)[:,0]; pred_test=raw(pred_test,te,[v],transform)[:,0]
    gain_val=gain_val[:,0]; gain_test=gain_test[:,0]
                                                                                
    yv=ds.X[va,v]; yt=ds.X[te,v]; ev=abs(pred_val-yv)/raw_std[v]; et=abs(pred_test-yt)/raw_std[v]
    fixed_q=[]; unit=[]; scaled=[]; constant=[]
    for q in range(p):
        ok=np.isfinite(ev[:,q]); hist=ev[ok,q]
        if len(hist)<50:fixed_q.append(None); unit.append(None); scaled.append(None); constant.append(None); continue
        fixed_q.append(AdaptiveQuantile(hist).quantile(adaptive=False))
        unit.append(AdaptiveQuantile(hist)); scaled.append(AdaptiveQuantile(hist/gain_val[ok,q])); constant.append(AdaptiveQuantile(hist/7))
    hits={name:[] for name in ['fixed_no_labels','adaptive_revealed','adaptive_mask_scale','constant_scale_control']}
    widths={name:[] for name in hits}; times={name:[] for name in hits}
    full={name:0 for name in hits}; empty={name:0 for name in hits}
    for t in range(len(te)):
        thresholds=[]; normalized=[]
        for q in range(p):
            if unit[q] is None:thresholds.append(None);normalized.append(None);continue
                                                                 
            qu=unit[q].quantile(); qs=scaled[q].quantile(); qc=constant[q].quantile()
            normalized.append((qu,qs,qc))
            thresholds.append({'fixed_no_labels':fixed_q[q],'adaptive_revealed':unit[q].quantile(),
                               'adaptive_mask_scale':qs*gain_test[t,q],
                               'constant_scale_control':qc*7})
        for q in range(p):
            if thresholds[q] is None or not np.isfinite(et[t,q]):continue
            for name,threshold in thresholds[q].items():
                hit=et[t,q]<=threshold
                if name=='adaptive_mask_scale':hit=et[t,q]/gain_test[t,q]<=normalized[q][1]
                if name=='constant_scale_control':hit=et[t,q]/7<=normalized[q][2]
                hits[name].append(int(hit)); widths[name].append(threshold*raw_std[v,q]); times[name].append(t)
                full[name]+=int(np.isposinf(threshold)); empty[name]+=int(np.isneginf(threshold))
            unit[q].update(et[t,q],thresholds[q]['adaptive_revealed'])
            scaled[q].update(et[t,q]/gain_test[t,q],normalized[q][1])
            constant[q].update(et[t,q]/7,normalized[q][2])
    for name in hits:
        if not hits[name]:continue
        h=np.array(hits[name]); w=np.array(widths[name]); tt=np.array(times[name]); quarters=np.minimum(3,tt*4//len(te))
        quarter=[float(h[quarters==j].mean()) for j in range(4) if (quarters==j).any()]
        interval.append({'station':ds.stations[v],'method':name,'n':len(h),'coverage':float(h.mean()),
                         'worst_calendar_quarter':min(quarter),'full_sets':full[name],'empty_sets':empty[name],
                         'mean_finite_halfwidth_raw_mixed_units':float(w[np.isfinite(w)&(w>=0)].mean()) if (np.isfinite(w)&(w>=0)).any() else None,
                         'quarter_coverage':json.dumps(quarter)})
    if not np.array_equal(hits['adaptive_revealed'],hits['constant_scale_control']):raise AssertionError('Scale cancellation control failed')
    log(f'interval stream {ds.stations[v]} complete')

for day,ix in testblocks:
    truth=ds.X[ix]
    for v in stations:
        predictions={}
        for name,(t,fn) in models.items():
            pred=fn(spaces[t]['Z'][ix],[v]); predictions[name]=raw(pred,ix,[v],t)[:,0]
        predictions['city_mean']=np.nanmean(np.delete(truth,v,axis=1),axis=1)
        prev=ix-24
                                                               
        predictions['yesterday']=ds.X[prev,v] if prev.min()>=0 else np.full((24,p),np.nan)
        for name,pred in predictions.items():
            ok=np.isfinite(truth[:,v])&np.isfinite(pred)
            if ok.sum()<12:continue
            errors=(pred-truth[:,v])/raw_std[v]
            records.append({'day':day,'station':ds.stations[v],'pattern':'single_station','method':name,'mse':float(np.mean(errors[ok]**2)),
                            'n_cells':int(ok.sum())})
        if 'PM2.5' not in ds.pollutants:continue
        q=ds.pollutants.index('PM2.5'); good=np.isfinite(truth[:,v,q])
        if good.sum()<16:continue
                                                                         
        usable={name:pr for name,pr in predictions.items() if np.isfinite(pr[good,q]).all()}
        actual=float(truth[good,v,q].mean())
        for name,pr in usable.items():
            estimate=float(pr[good,q].mean())
            decisions.append({'day':day,'station':ds.stations[v],'method':name,'true_pm25':actual,'pred_pm25':estimate,
                              'observed_hours':int(good.sum()),'true_exceed':int(actual>60),'pred_exceed':int(estimate>60),
                              'true_poor_threshold':int(actual>90),'pred_poor_threshold':int(estimate>90),
                              'absolute_error':abs(actual-estimate),'mse':(actual-estimate)**2})
    if 'PM2.5' in ds.pollutants:
        q=ds.pollutants.index('PM2.5')
        full=sh.reconstruct_mask(Z[ix],np.zeros((24,n),bool),lam)
        direct=raw(full,ix,list(range(n)),transform)
                                                                           
        gspace=spaces[chosen['gaussian']['transform']]
        M=gspace['Z'][ix].reshape(24,-1).copy(); filled=M.copy()
        for h in range(24):
            U=np.flatnonzero(~np.isfinite(M[h])); O=np.flatnonzero(np.isfinite(M[h]))
            if len(U):filled[h,U]=gauss.mean[U]-np.linalg.solve(gauss.Q[np.ix_(U,U)],gauss.Q[np.ix_(U,O)])@(M[h,O]-gauss.mean[O])
        gp=raw(filled.reshape(24,n,p),ix,list(range(n)),chosen['gaussian']['transform'])
        for v in range(n):
            count=int(np.isfinite(truth[:,v,q]).sum())
            if count>=12:continue
            exposure.append({'day':day,'station':ds.stations[v],'pm25_observed_hours':count,
                             'estimated_mean_sheaf':float(direct[:,v,q].mean()),'estimated_mean_gaussian':float(gp[:,v,q].mean()),
                             'classification_agreement_at_60':bool((direct[:,v,q].mean()>60)==(gp[:,v,q].mean()>60)),
                             'status':'unverified model estimate; not measured exposure'})
    log(f'snapshot/decision day {day} complete')

prolonged=[]
for day,ix in tb:
    for v in runstations:
        if not eligible(ix,v):continue
        B=Z[ix]; hide=np.zeros((48,n),bool); hide[24:,v]=True
        for mode in ['retrospective','causal']:
            st=temporal_models[mode]
            if mode=='retrospective':pr=st.reconstruct_mask(B,hide,lam)[24:,v]
            else:pr=np.array([st.reconstruct_mask(B[:h+1],hide[:h+1],lam)[-1,v] for h in range(24,48)])
            hidden=B.copy(); hidden[24:,v]=np.nan
            preds={'temporal_sheaf':pr,'stable_var':kalmodels[mode].smooth(hidden,causal=mode=='causal')[24:,v],
                   'spatial_sheaf':decoder.predict(B[24:],[v])[0][:,0],
                   'seasonal_persistence':seasonal_predict(B,ds.times[ix],v,24)}
            for name,pred in preds.items():
                err=score(pred[:,None,:],ix[24:],[v],transform)
                if np.isfinite(err):prolonged.append({'day':day,'station':ds.stations[v],'mode':mode,'method':name,'mse':err,
                                                     'observed_cells':int(np.isfinite(ds.X[ix[24:],v]).sum()),'history_hours':24,'outage_hours':24})
    log(f'prolonged day {day} complete')

S=pd.DataFrame(records); P=pd.DataFrame(prolonged); I=pd.DataFrame(interval); A=pd.DataFrame(decisions)
for name,frame in [('snapshots.csv',S),('prolonged.csv',P),('intervals.csv',I),('environment.csv',A),('unmeasured_estimates.csv',pd.DataFrame(exposure))]:
    frame.to_csv(out/name,index=False)

summary={'region':args.region,'n_stations':n,'n_pollutants':p,'pollutants':ds.pollutants,'sampled_target_stations':len(stations),
         'snapshot':{ref:paired_bootstrap(S,'sheaf',ref) for ref in ['gaussian','kriging','idw','city_mean','yesterday']},
         'prolonged':{},'intervals':{},'environment':{},'environment_source':'direct raw concentration' if args.region in ['delhi2025','bengaluru_raw'] else 'inverted sub-index archive; concentration interpretation conditional on provenance'}
for mode in ['retrospective','causal']:
    sub=P[P['mode']==mode]
    summary['prolonged'][mode]={ref:paired_bootstrap(sub,'temporal_sheaf',ref) for ref in ['stable_var','spatial_sheaf','seasonal_persistence']}
for name,g in I.groupby('method'):
    summary['intervals'][name]={'stations':len(g),'pooled_coverage':float(np.average(g.coverage,weights=g.n)),
        'station_macro_coverage':float(g.coverage.mean()),'lowest_station_quarter':float(g.worst_calendar_quarter.min()),
        'within_two_points':int((abs(g.coverage-.9)<=.02).sum()),'full_sets':int(g.full_sets.sum()),'empty_sets':int(g.empty_sets.sum()),'n_cells':int(g.n.sum())}
if len(A):
                                                                   
    core=['sheaf','gaussian','kriging','idw','city_mean']
    tab=A[A.method.isin(core)].groupby(['day','station']).method.nunique()
    keys=set(tab[tab==len(core)].index)
    A=A[[tuple(x) in keys for x in A[['day','station']].to_numpy()]]
    A.to_csv(out/'environment_common.csv',index=False)
    for name,g in A.groupby('method'):
        positives=int(g.true_exceed.sum()); negatives=int((g.true_exceed==0).sum())
        summary['environment'][name]={'station_days':len(g),'true_exceedances':positives,'non_exceedances':negatives,
            'accuracy':float((g.true_exceed==g.pred_exceed).mean()),
            'miss_rate':float(((g.true_exceed==1)&(g.pred_exceed==0)).sum()/positives) if positives else None,
            'false_alarm_rate':float(((g.true_exceed==0)&(g.pred_exceed==1)).sum()/negatives) if negatives else None,
            'mean_absolute_error':float(g.absolute_error.mean())}
                                                                      
    for day,g in A.groupby('day'):
        for name,h in g.groupby('method'):
            k=max(1,int(np.ceil(.25*len(h))))
            selected=h.nlargest(k,'pred_pm25')
            capture=float(selected.true_exceed.sum())
            priorities.append({'day':day,'method':name,'stations':len(h),'repair_budget':k,'exceedances_found':capture,
                'random_expected':float(k*h.true_exceed.mean()),'actual_exceedances':int(h.true_exceed.sum())})
    pd.DataFrame(priorities).to_csv(out/'repair_priority.csv',index=False)
write('summary.json',finite_json(summary))
manifest.update(status='complete',ended_utc=pd.Timestamp.now(tz='UTC').isoformat(),elapsed_seconds=time.time()-started)
write('manifest.json',finite_json(manifest)); log('corrected run complete')
