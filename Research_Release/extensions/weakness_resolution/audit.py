                                                                                   
from pathlib import Path
import sys,json,pickle,hashlib,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'Research_Release/weakness_resolution_results';SOURCE=ROOT/'Research_Release/expanded_validation_results'
sys.path.insert(0,str(HERE))
import nonlinear as rf
import numpy as np,pandas as pd

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def independent_features(ds,ix,v):
                                                                                     
    z=ds['Z'];n,p=z.shape[1:];result=[]
    past=z[ix[:24],v];last=np.zeros(p);known=np.zeros(p)
    for c in range(p):
        obs=np.flatnonzero(np.isfinite(past[:,c]))
        if len(obs):last[c]=past[obs[-1],c];known[c]=1
    onehot=np.eye(n)[v]
    for k in range(24,48):
        rows=[]
        for offset in [0,1,24]:
            a=z[ix[k-offset]].copy()
            if k-offset>=24:a[v]=np.nan
            rows.append(a.reshape(-1))
        time=ds['times'][ix[k]]
        clock=[np.sin(time.hour*2*np.pi/24),np.cos(time.hour*2*np.pi/24),
               np.sin(time.dayofweek*2*np.pi/7),np.cos(time.dayofweek*2*np.pi/7),(k-23)/24]
        result.append(np.concatenate([*[np.nan_to_num(a,nan=0.) for a in rows],
                                      *[np.isfinite(a).astype(float) for a in rows],last,known,onehot,clock]))
    return np.asarray(result)

def main():
    checks=[]
    def check(label,value):
        checks.append({'check':label,'passed':bool(value)})
        if not value:raise AssertionError(label)
    expected={p.parent.name for p in SOURCE.glob('*/outage_cases.csv')}
    actual={p.parent.name for p in (OUT/'nonlinear').glob('*/status.json')}
    check('nonlinear_all_evaluable_cohorts',expected==actual)
    for folder in sorted(p.parent for p in (OUT/'nonlinear').glob('*/status.json')):
        name=folder.name;status=json.loads((folder/'status.json').read_text());identity=status['identity']
        for key,path in [('source_sha256',HERE/'nonlinear.py'),('protocol_sha256',HERE/'PROTOCOL.md'),
                         ('dataset_sha256',SOURCE/name/'dataset.npz'),('reference_cases_sha256',SOURCE/name/'outage_cases.csv')]:
            check(name+'/'+key,identity[key]==sha(path))
        check(name+'/model_hash',status['model_sha256']==sha(folder/'models.pkl'))
        check(name+'/cases_hash',status['cases_sha256']==sha(folder/'cases.csv'))
        d=np.load(SOURCE/name/'dataset.npz',allow_pickle=False)
        ds={k:d[k] for k in d.files}|{'times':pd.DatetimeIndex(d['times'])}
        with (folder/'models.pkl').open('rb') as stream:models=pickle.load(stream)
        saved=pd.read_csv(folder/'cases.csv',dtype={'station':str}).set_index(['day','station'])
        reference=pd.read_csv(SOURCE/name/'outage_cases.csv',dtype={'station':str})
        ref=reference[reference.method.eq('sheaf_causal')].set_index(['day','station'])
        check(name+'/matched_case_keys',set(saved.index)==set(ref.index))
        for day,station,v,ix,f,truth in rf.cases(ds,2):
            independent=independent_features(ds,ix,v)
            check(name+'/'+day+'/'+station+'/feature_replay',np.allclose(independent,f,rtol=0,atol=1e-14))
            prediction=np.column_stack([model.predict(independent) for model in models])
            good=np.isfinite(truth);pm=good[:,0];row=saved.loc[(day,station)]
            check(name+'/'+day+'/'+station+'/loss_replay',np.isclose(np.mean((prediction-truth)[good]**2),row.mse,rtol=1e-10))
            physical=np.mean(((prediction[:,0]-truth[:,0])*ds['sd'][v,0])[pm]**2) if pm.any() else np.nan
            check(name+'/'+day+'/'+station+'/physical_loss',np.isclose(physical,row.pm25_raw_mse,rtol=1e-10,equal_nan=True))
            check(name+'/'+day+'/'+station+'/same_cells',int(good.sum())==row.target_cells==ref.loc[(day,station)].target_cells)
                                                                                  
        day,station,v,ix,f,truth=rf.cases(ds,2)[0]
        for k in range(24,48):
            modified=dict(ds);modified['Z']=ds['Z'].copy()
            modified['Z'][ix[k]+1:]=887766.;modified['Z'][ix[24:],v]=443322.
            check(name+'/causal_hour_'+str(k-24),np.array_equal(rf.features(modified,ix,v)[k-24],f[k-24]))
        selection=pd.read_csv(folder/'selection.csv');best=selection.sort_values('selection_mse').iloc[0]
        check(name+'/validation_choice',best.leaf==status['leaf'] and np.isclose(best.selection_mse,status['selection_mse']))
        print('Nonlinear checkpoint audit',name,flush=True)
    summary=pd.read_csv(OUT/'uncertainty_decisions/INTERVAL_SUMMARY.csv')
    for folder in sorted(p.parent for p in (OUT/'uncertainty_decisions').glob('*/interval_cases.csv')):
        frame=pd.read_csv(folder/'interval_cases.csv',dtype={'station':str},float_precision='round_trip')
        for (loss,method),original in frame.groupby(['context_loss','method']):
            check(folder.name+'/interval_cover_arithmetic',np.array_equal(original.covered.to_numpy(),(abs(original.truth-original.prediction)<=original.radius).to_numpy()))
            check(folder.name+'/interval_width_arithmetic',np.allclose(original.width,2*original.radius))
            for group,g in [('all',original),('extreme',original[original.extreme]),('predicted_high_risk',original[original.high_risk])]:
                rows=summary[summary.cohort.eq(folder.name)&summary.context_loss.eq(loss)&summary.method.eq(method)&summary.group.eq(group)]
                check(folder.name+'/interval_summary_key',len(rows)==1)
                row=rows.iloc[0]
                check(folder.name+'/interval_count',len(g)==row.cells)
                check(folder.name+'/interval_coverage',np.isclose(g.covered.mean(),row.coverage,equal_nan=True))
                check(folder.name+'/full_interval_fraction',np.isclose(g.full_set.mean(),row.full_set_fraction,equal_nan=True))
                quarters=g.assign(quarter=pd.to_datetime(g.timestamp).dt.to_period('Q').astype(str)).groupby(['station','quarter']).covered.agg(['size','mean'])
                if len(quarters):
                    check(folder.name+'/true_worst_quarter',np.isclose(quarters['mean'].min(),row.minimum_station_quarter_coverage))
                    check(folder.name+'/true_worst_quarter_count',int(quarters.loc[quarters['mean'].idxmin(),'size'])==row.minimum_station_quarter_cells)
        status=json.loads((folder/'status.json').read_text())
        check(folder.name+'/interval_code_hash',status['identity']['source_sha256']==sha(HERE/'uncertainty_decisions.py'))
        for filename,digest in status['identity']['input_sha256'].items():check(folder.name+'/interval_input/'+filename,digest==sha(SOURCE/folder.name/filename))
        d=np.load(SOURCE/folder.name/'dataset.npz',allow_pickle=False)
        thresholds=dict(zip(d['stations'].astype(str),np.nanquantile(d['X'][d['split']==0,:,0],.75,axis=0)))
        check(folder.name+'/risk_training_only_thresholds',all(np.isclose(thresholds[k],v) for k,v in status['risk_thresholds'].items()))
        check(folder.name+'/risk_pre_outcome_group',np.array_equal(frame.high_risk,(np.maximum(frame.prediction,frame.city_prediction)>frame.station.map(thresholds))))
    alert_summary=json.loads((OUT/'uncertainty_decisions/ALERT_SUMMARY.json').read_text())
    for folder in sorted(p.parent for p in SOURCE.glob('*/hourly_alert_predictions.csv')):
        f=pd.read_csv(folder/'hourly_alert_predictions.csv')
        for method,g in f.groupby('method'):
            result=next(r for r in alert_summary if r['cohort']==folder.name and r['method']==method)
            y=g.positive.astype(bool);positives=int(y.sum());negatives=len(y)-positives
            check(folder.name+'/roc_class_counts',result['positive_hours']==positives and result['negative_hours']==negatives)
            if positives and negatives:
                ranks=g.prediction.rank(method='average')
                auc=(ranks[y].sum()-positives*(positives+1)/2)/(positives*negatives)
                check(folder.name+'/roc_rank_sum_replay',np.isclose(auc,result['auc'],rtol=0,atol=1e-12))
            else:check(folder.name+'/roc_undefined',result['status'].startswith('undefined'))
    guard=json.loads((OUT/'selection_guard/SUMMARY.json').read_text())
    check('selector_all_cohorts',{r['cohort'] for r in guard}==expected)
    for record in guard:
        name=record['cohort'];choices=json.loads((SOURCE/name/'model_choices.json').read_text())
        forest=json.loads((OUT/'nonlinear'/name/'status.json').read_text())
        scores={'sheaf_causal':choices['sheaf']['validation_mse'],'var_causal':choices['var']['validation_mse'],'random_forest_causal':forest['selection_mse']}
        check(name+'/selector_validation_scores',scores==record['validation_scores'])
        chosen=min(scores,key=lambda key:(scores[key],key))
        check(name+'/selector_no_test_choice',chosen==record['chosen_method'] and record['test_used_for_choice'] is False)
        selected=pd.read_csv(OUT/'selection_guard'/(name+'_cases.csv'),dtype={'station':str})
        source=pd.read_csv(OUT/'nonlinear'/name/'cases.csv',dtype={'station':str}) if chosen=='random_forest_causal' else pd.read_csv(SOURCE/name/'outage_cases.csv',dtype={'station':str})
        source=source[source.method.eq(chosen)]
        joined=source.merge(selected,on=['day','station'],suffixes=('_original','_selected'),validate='one_to_one')
        check(name+'/selector_prediction_replay',len(joined)==len(source)==len(selected) and np.allclose(joined.mse_original,joined.mse_selected))
    width_status=json.loads((OUT/'width_control/status.json').read_text())
    check('width_control_source',width_status['source_sha256']==sha(HERE/'width_control.py'))
    check('width_control_protocol',width_status['protocol_sha256']==sha(HERE/'WIDTH_CONTROL_PROTOCOL.md'))
    for file in sorted((OUT/'width_control').glob('*_cases.csv')):
        name=file.name.removesuffix('_cases.csv')
        try:f=pd.read_csv(file,dtype={'station':str},float_precision='round_trip')
        except pd.errors.EmptyDataError:continue
        d=np.load(SOURCE/name/'dataset.npz',allow_pickle=False)
        thresholds=dict(zip(d['stations'].astype(str),np.nanquantile(d['X'][d['split']==0,:,0],.75,axis=0)))
        cal=pd.read_csv(SOURCE/name/'interval_calibration_predictions.csv',dtype={'station':str})
        cal['risk']=np.maximum(cal.prediction,cal.city_prediction)>cal.station.map(thresholds)
        cal['error']=abs(cal.truth-cal.prediction)
        for (loss,station),g in f.groupby(['context_loss','station']):
            c=cal[cal.loss.eq(loss)&cal.station.eq(station)]
                                                                                           
            q=[]
            for risk in [False,True]:
                scores=np.sort(c.loc[c.risk.eq(risk),'error'].to_numpy())
                index=int(np.ceil((len(scores)+1)*.9))-1
                check(name+'/width_calibration_support',0<=index<len(scores))
                q.append(scores[index])
            radius=float(np.where(c.risk,q[1],q[0]).mean())
            check(name+'/calibration_width_control',np.allclose(g.control_radius,radius,rtol=0,atol=1e-12))
            check(name+'/width_control_arithmetic',np.array_equal(g.control_covered,(abs(g.truth-g.prediction)<=g.control_radius)))
    if '--nonneural' in sys.argv:
        (OUT/'NONNEURAL_AUDIT.json').write_text(json.dumps({'status':'passed','check_count':len(checks),'checks':checks},indent=2))
        print('Nonneural audit passed',len(checks),'checks',flush=True)
        return
    neural=OUT/'neural';selection=json.loads((neural/'SELECTION.json').read_text())
    check('neural/protocol',selection['protocol_sha256']==sha(HERE/'PROTOCOL.md'))
    check('neural/wrapper',selection['wrapper_sha256']==sha(HERE/'neural_external.py'))
    for item in selection['selections']:
        name=item['cohort'];folder=neural/name
        if '--wait-neural' in sys.argv:
            deadline=time.monotonic()+900
            while not (neural/(name+'_REPLAY.json')).exists():
                if time.monotonic()>deadline:raise RuntimeError('Neural replay still incomplete: '+name)
                print('Awaiting final neural checkpoint replay',name,flush=True)
                time.sleep(15)
        replay=json.loads((neural/(name+'_REPLAY.json')).read_text())
        check(name+'/neural_replay',replay['status']=='passed' and len(replay['checks'])==3 and all(r['passed'] for r in replay['checks']))
        choices=json.loads((folder/'grin_choices.json').read_text())
        check(name+'/all_neural_seeds',sorted(c['seed'] for c in choices)==[0,1,2])
        for choice in choices:
            grid=[]
            for width in [32,64]:
                history=pd.read_csv(folder/f"grin_seed{choice['seed']}_width{width}_training.csv")
                best=history.sort_values(['validation_mse','epoch']).iloc[0]
                grid.append((float(best.validation_mse),width,int(best.epoch)))
            selected=min(grid,key=lambda x:x[0])
            check(name+'/neural_validation_choice_'+str(choice['seed']),selected[1]==choice['width'] and selected[2]==choice['epoch'] and np.isclose(selected[0],choice['validation_mse']))
        for filename in ['dataset.npz','models.pkl','outage_cases.csv']:check(name+'/neural_frozen_'+filename,sha(folder/filename)==sha(SOURCE/name/filename))
        check(name+'/neural_source',json.loads((folder/'neural_lock.json').read_text())['source_sha256']==sha(ROOT/'Research_Release/extensions/recent_data/neural.py'))
    temporal=pd.read_csv(OUT/'fault_temporal/SUMMARY.csv')
    statuses=json.loads((OUT/'fault_temporal/status.json').read_text())
    check('temporal_completed',statuses['status']=='completed')
    for record in statuses['cohorts']:
        if record['status']=='completed':
            check(record['cohort']+'/temporal_source',record['source_sha256']==sha(HERE/'fault_temporal.py'))
            check(record['cohort']+'/temporal_protocol',record['protocol_sha256']==sha(HERE/'FAULT_TEMPORAL_PROTOCOL.md'))
    for folder in sorted(p.parent for p in (OUT/'fault_temporal').glob('*/cases.csv')):
        f=pd.read_csv(folder/'cases.csv')
        keys=[set(map(tuple,g[['episode','timestamp','kind','true_count']].to_numpy())) for _,g in f.groupby('method')]
        check(folder.name+'/temporal_matched_rows',all(k==keys[0] for k in keys))
        for key,g in f.groupby(['method','kind','true_count']):
            row=temporal[temporal.cohort.eq(folder.name)&temporal.method.eq(key[0])&temporal.kind.eq(key[1])&temporal.true_count.eq(key[2])].iloc[0]
            check(folder.name+'/temporal_summary',len(g)==row.hours and np.isclose(g.alarm.mean(),row.alarm_rate))
    value={'status':'passed','check_count':len(checks),'checks':checks,
           'scope':'Internal checkpoint, causal input, source identity and aggregate audit; no external proof review, field effect or priority certification',
           'implementation_sha256':{str(p.relative_to(ROOT)):sha(p) for p in HERE.glob('*') if p.is_file()}}
    (OUT/'AUDIT.json').write_text(json.dumps(value,indent=2))
    print('Weakness follow-up audit passed',len(checks),'checks',flush=True)

if __name__=='__main__':main()
