                                                                                     
from pathlib import Path
import sys,os,json,pickle,hashlib,time,math,runpy,collections,re
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np,pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];RESEARCH=ROOT/'Research_Release'
sys.path.insert(0,str(RESEARCH/'extensions/recent_data'))
import common,final_confirmation as replay
checks=[];inputs={};started=time.time()
def check(name,ok,detail=None):
    if not ok:raise AssertionError((name,detail))
    checks.append({'check':name,'passed':True,'detail':detail})
def read_json(path):
    inputs[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding='utf-8'))
def frame(path,**kw):
    inputs[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path,**kw)
def close(a,b):return bool(np.isclose(a,b,rtol=1e-9,atol=1e-10,equal_nan=True))
folders=[RESEARCH/'recent_data_results/delhi',RESEARCH/'recent_data_results/epa_la',RESEARCH/'uk_validation_results/uk_aurn']+sorted(p.parent for p in (RESEARCH/'expanded_validation_results').glob('*/dataset.npz'))
inference_cases=0;folder_reports=[]
for folder in folders:
    label=str(folder.relative_to(RESEARCH));path=folder/'dataset.npz'
    inputs[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path,allow_pickle=False) as d:ds={k:d[k] for k in d.files}
    ds['times']=pd.DatetimeIndex(ds['times']);train=ds['X'][ds['split']==0]
    check(label+'/train_mean',np.allclose(ds['mu'],np.nanmean(train,0)))
    check(label+'/train_scale',np.allclose(ds['sd'],np.nanstd(train,0,ddof=1)))
    check(label+'/normalization',np.allclose(ds['Z'],(ds['X']-ds['mu'])/ds['sd'],equal_nan=True))
    check(label+'/native_nonnegative',np.all(ds['X'][np.isfinite(ds['X'])]>=0))
    check(label+'/chronology',all(ds['times'][ds['split']==a].max()<ds['times'][ds['split']==b].min() for a,b in [(0,1),(1,3),(3,2)]))
    check(label+'/hourly_calendar',ds['times'].equals(pd.date_range(ds['times'][0],ds['times'][-1],freq='h')))
    if not (folder/'outage_cases.csv').exists():
        check(label+'/unavailable_has_no_test_truth',not np.isfinite(ds['X'][ds['split']==2]).any())
        folder_reports.append({'role':label,'status':'no_usable_test'});continue
    choices=read_json(folder/'model_choices.json');model_path=folder/'models.pkl'
    digest=hashlib.sha256(model_path.read_bytes()).hexdigest();inputs[str(model_path.relative_to(ROOT))]=digest
    check(label+'/frozen_model_hash',digest==choices['model_sha256'])
    with model_path.open('rb') as stream:m=pickle.load(stream)
    check(label+'/training_graph',m['edges']==common.graph_correlation(ds['Z'][ds['split']==0]))
    rebuilt=common.QuadraticDecoder(ds['Z'][ds['split']==0],m['edges'],choices['sheaf']['lam'],choices['sheaf']['temporal_scale'])
    check(label+'/train_only_maps_and_prior',np.array_equal(rebuilt.D,m['sheaf'].D) and np.array_equal(rebuilt.P,m['sheaf'].P))
    check(label+'/train_only_temporal_maps',all(np.array_equal(a,c) and np.array_equal(b,d) for lag in [1,24] for (a,b),(c,d) in zip(rebuilt.temporal[lag],m['sheaf'].temporal[lag])))
    grid=frame(folder/'validation_grid.csv')
    for method in ['sheaf','var']:
        check(label+'/selection_minimum/'+method,close(grid.loc[grid.method==method,'validation_mse'].min(),choices[method]['validation_mse']))
    f=frame(folder/'outage_cases.csv',dtype={'station':str});ixmap=dict(common.blocks(ds,2));Q={W:replay.objective(m['sheaf'],W) for W in range(25,49)}
    case_count=0
    for (day,station),g in f.groupby(['day','station']):
        ix=ixmap[day];v=list(map(str,ds['stations'])).index(station);x=ds['Z'][ix].copy();x[24:,v]=np.nan
        truth=ds['Z'][ix[24:],v];good=np.isfinite(truth)
        check(label+'/'+day+'/'+station+'/same_truth_cells',g.target_cells.eq(good.sum()).all() and good.sum()>=12*truth.shape[1])
        predictions={'sheaf_causal':np.array([replay.solve(Q[j+1],x[:j+1])[-1,v] for j in range(24,48)]),'sheaf_retrospective':replay.solve(Q[48],x)[24:,v],'var_causal':replay.filter_var(m['var'],x)[24:,v]}|replay.controls(x,v,ds,m)
        for method,pred in predictions.items():
            row=g[g.method==method].iloc[0];mse=np.mean((pred-truth)[good]**2)
            pm=good[:,0];physical=np.mean(((pred[:,0]-truth[:,0])*ds['sd'][v,0])[pm]**2) if pm.any() else np.nan
            check(label+'/'+day+'/'+station+'/'+method,close(mse,row.mse) and close(physical,row.pm25_raw_mse))
        case_count+=1
    saved=read_json(folder/'reconstruction_summary.json')
    for scope,subset in [('primary_every_third_day',f[f.primary]),('all_eligible_test_days',f)]:
        for method in ['var_causal','seasonal','city_mean','gaussian','spatial_sheaf']:
            r=common.paired(subset,'sheaf_causal',method);s=saved[scope][method]
            check(label+'/'+scope+'/'+method+'/recomputed_interval',close(r['gain_percent'],s['gain_percent']) and r['cases']==s['cases'] and ((r['ci95'] is None and s['ci95'] is None) or np.allclose(r['ci95'],s['ci95'])))
    inference_cases+=case_count;folder_reports.append({'role':label,'station_days':case_count,'methods':7,'status':'passed'})
    print(label,'independent replay passed',case_count,'station-days',round(time.time()-started,1),'seconds',flush=True)

                                                    
aggregate=collections.defaultdict(lambda:[0,0])
for folder in sorted(p.parent for p in (RESEARCH/'weakness_resolution_results/uncertainty_decisions').glob('*/interval_cases.csv')):
    if folder.name=='india_long_test':continue
    ds=np.load(RESEARCH/'expanded_validation_results'/folder.name/'dataset.npz')
    stations=ds['stations'].astype(str);q75=dict(zip(stations,np.nanquantile(ds['X'][ds['split']==0,:,0],.75,axis=0)));q95=dict(zip(stations,np.nanquantile(ds['X'][ds['split']==0,:,0],.95,axis=0)))
    f=frame(folder/'interval_cases.csv',dtype={'station':str},float_precision='round_trip');f=f[f.context_loss.eq(0)&f.method.isin(['station_fixed','risk_station'])]
    cal=frame(RESEARCH/'expanded_validation_results'/folder.name/'interval_calibration_predictions.csv',dtype={'station':str},float_precision='round_trip');cal=cal[cal.loss.eq(0)].copy()
    cal['risk']=np.maximum(cal.prediction,cal.city_prediction)>cal.station.map(q75);cal['error']=abs(cal.truth-cal.prediction)
    check(folder.name+'/risk_definition',f.high_risk.eq(np.maximum(f.prediction,f.city_prediction)>f.station.map(q75)).all())
    check(folder.name+'/extreme_definition',f.extreme.eq(f.truth>f.station.map(q95)).all())
    for (method,station),g in f.groupby(['method','station']):
        for risk in ([False,True] if method=='risk_station' else [None]):
            c=cal[cal.station.eq(station)];target=g
            if risk is not None:c=c[c.risk.eq(risk)];target=g[g.high_risk.eq(risk)]
            scores=np.sort(c.error.to_numpy());rank=math.ceil(.9*(len(scores)+1));radius=scores[rank-1] if rank<=len(scores) else np.inf
            check(folder.name+'/'+method+'/'+station+'/'+str(risk)+'/radius',np.allclose(target.radius,radius))
        check(folder.name+'/'+method+'/'+station+'/coverage',g.covered.eq(abs(g.truth-g.prediction)<=g.radius).all())
    for method,g in f.groupby('method'):
        for name,subset in [('All',g),('Extreme',g[g.extreme])]:
            aggregate[method,name][0]+=len(subset);aggregate[method,name][1]+=int(subset.covered.sum())
for (method,group),(count,covered) in aggregate.items():
    expected={('station_fixed','All'):(49996,87.5),('risk_station','All'):(49996,88.9),('station_fixed','Extreme'):(2309,25.2),('risk_station','Extreme'):(2309,45.7)}[method,group]
    check('interval_totals/'+method+'/'+group,count==expected[0] and round(100*covered/count,1)==expected[1],{'readings':count,'coverage':100*covered/count})

                                                                                 
alerts=frame(RESEARCH/'recent_data_results/delhi/hourly_alert_predictions.csv');cutoffs=read_json(RESEARCH/'recent_data_results/delhi/alert_cutoffs.json')
for method in ['sheaf_causal','var_causal']:
    g=alerts[alerts.method==method];cutoff=cutoffs[method]['cutoff']
    check('alerts/'+method+'/operating_cutoff',g.alarm.eq(g.prediction>cutoff).all(),cutoff)
    check('alerts/'+method+'/truth_threshold',g.positive.eq(g.truth>60).all())

banks=read_json(RESEARCH/'decision_repair_results/SCENARIO_BANKS.json')
duplicates={name:len(b['pm25_vectors'])-len({tuple(row) for row in b['pm25_vectors']}) for name,b in banks.items()}
check('finite_bank/no_duplicate_vectors',not any(duplicates.values()),duplicates)

report={'status':'passed','checks':len(checks),'all_replayed_station_days':inference_cases,'roles':folder_reports,'checks_detail':checks,'input_sha256':inputs,'elapsed_seconds':time.time()-started,'scope':'Fresh inference replay using frozen fitted parameters, training-only map reconstruction, chronological and calibration checks, raw-loss and interval reaggregation. Does not independently retrain VAR, forests or neural models, prove worldwide priority or validate physical error budgets.'}
(OUT/'WHOLE_REVIEW_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['checks_detail','input_sha256','roles']},indent=2),flush=True)
