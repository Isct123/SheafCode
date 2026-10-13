                                                                        

                                                                     
                                                                       
   
import hashlib,itertools,json,sys,os
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'matplotlib_cache'))
from sheafcode.cohort import select_cohort
from sheafcode.baselines2 import GaussianConditional
from sheafcode.reliable import CachedSpatial,paired_bootstrap
REGIONS=['delhi','ncr','mumbai','kolkata','hyderabad','bengaluru','india','delhi2025','bengaluru_raw']
tables=[];checks=[];overlap=[];information={}
def check(name,condition,details=None):
    checks.append({'check':name,'passed':bool(condition),'details':details})
    if not condition:raise AssertionError(name)

def comparison(frame,method,ref,value='mse'):
    keys=['day','station']
    t=frame.pivot_table(index=keys,columns='method',values=value).dropna(subset=[method,ref])[[method,ref]].reset_index()
    if len(t)==0:return {'n':0}
    delta=t[ref]-t[method]
    result=paired_bootstrap(frame,method,ref,value=value)
    result['station_cluster_ci95']=paired_bootstrap(frame,method,ref,value=value,unit='station')['difference_ci95']
    day=t.assign(delta=delta).groupby('day').delta.agg(['sum','count']).sort_index()
    sums=day['sum'].to_numpy();counts=day['count'].to_numpy();N=len(sums)
    rng=np.random.default_rng(911)
                                                                              
    boot=[];length=min(3,N)
    for _ in range(2000):
        starts=rng.integers(0,max(N-length+1,1),int(np.ceil(N/length)))
        idx=np.concatenate([np.arange(s,s+length) for s in starts])[:N]
        boot.append(float(sums[idx].sum()/counts[idx].sum()))
    result['ordered_three_sampled_day_block_ci95']=np.quantile(boot,[.025,.975]).tolist()
    observed=float(sums.sum())
    if N<=16:
        signs=np.array(list(itertools.product([-1.,1.],repeat=N)))
        p=float(np.mean(signs@sums>=observed-1e-12));scheme='exact day-cluster sign randomization'
    else:
        signs=rng.choice([-1.,1.],(19999,N))
        p=float((1+np.sum(signs@sums>=observed-1e-12))/20000);scheme='Monte Carlo day-cluster sign randomization'
    result.update(nominal_one_sided_p=p,randomization=scheme,
        assumption='conditional sign symmetry across day clusters; not guaranteed under arbitrary dependence',
        descriptive_difference=float(delta.mean()))
    return result

for r in REGIONS:
    out=ROOT/'results'/r
    check(r+' manifest exists',(out/'manifest.json').exists())
    m=json.loads((out/'manifest.json').read_text())
    check(r+' finalized status',m['status'] in ['complete','excluded'])
    if m['status']=='excluded':continue
    for path,h in m['code_sha256'].items():check(r+' source '+path,hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h)
    summary=json.loads((out/'summary.json').read_text())
    P=pd.read_csv(out/'prolonged.csv');S=pd.read_csv(out/'snapshots.csv');I=pd.read_csv(out/'intervals.csv')
    check(r+' finite primary errors',np.isfinite(S.mse).all() and np.isfinite(P.mse).all())
    check(r+' interval unit/constant cancellation',np.allclose(I[I.method=='adaptive_revealed'].sort_values('station').coverage,
          I[I.method=='constant_scale_control'].sort_values('station').coverage,rtol=0,atol=0))
    for name,g in I.groupby('method'):
        expected=float(np.average(g.coverage,weights=g.n))
        check(r+' interval '+name,abs(expected-summary['intervals'][name]['pooled_coverage'])<1e-12)
    for mode in ['causal','retrospective']:
        sub=P[P['mode']==mode]
        piv=sub.pivot_table(index=['day','station'],columns='method',values='mse').dropna(subset=['temporal_sheaf','stable_var'])
        means=piv[['temporal_sheaf','stable_var']].mean()
        c=summary['prolonged'][mode]['stable_var']
        check(r+' '+mode+' gain',abs((1-np.sqrt(means.iloc[0]/means.iloc[1]))-c['rmse_relative_gain'])<1e-12)
                                                                           
    ds=select_cohort(r);ds.set_transform('none');std=ds.std.copy()
    choice=json.loads((out/'frozen_choices.json').read_text())['gaussian']
    ds.set_transform(choice['transform']);Z=ds.Z();base=GaussianConditional(Z[ds.split==0],.1)
    shrink=choice['shrink']
                                                                            
                                                                              
                                                                      
    C=(1-shrink)*base.S+shrink*np.diag(np.diag(base.S))
    decoder=CachedSpatial(np.linalg.inv(C),ds.n,ds.p,base.mu)
    baseline=[]
    for (day,station),g in P.groupby(['day','station']):
        rows=np.flatnonzero((ds.split==2)&(ds.times.floor('D')==pd.Timestamp(day)))
        if len(rows)!=24:raise AssertionError('Invalid prolonged calendar')
        v=ds.stations.index(station)
        pred=decoder.predict(Z[rows],[v])[0]
        pred=ds.raw_from_Z(pred,rows,[v])[:,0]
        truth=ds.X[rows,v];observed=np.isfinite(truth)
        check(r+' Gaussian finite '+day+station,np.isfinite(pred[observed]).all())
        error=float(np.mean(((pred-truth)/std[v])[observed]**2))
        for mode in ['causal','retrospective']:
            baseline.append({'day':day,'station':station,'mode':mode,'method':'gaussian','mse':error})
    extended=pd.concat([P,pd.DataFrame(baseline)],ignore_index=True)
    extended.to_csv(out/'prolonged_with_gaussian.csv',index=False)
    for mode in ['causal','retrospective']:
        for ref in ['stable_var','gaussian','spatial_sheaf','seasonal_persistence']:
            c=comparison(extended[extended['mode']==mode],'temporal_sheaf',ref)
            tables.append({'region':r,'mode':mode,'reference':ref,'analysis':'supplementary stronger comparator/dependence check',**c})
    fault=pd.read_csv(out/'fault_trials.csv');fs=json.loads((out/'fault_structure_summary.json').read_text())
    check(r+' fault model hash',hashlib.sha256((out/'model.npz').read_bytes()).hexdigest()==fs['input_model_sha256'])
    check(r+' actual-mask theorem check',(fault.loc[fault.corrected_bound_sufficient==1,'ls_known_count_exact']==1).all())
    check(r+' actual fault matrix recorded','actual_mask_margin' in fault.columns)
    for rec in fs['results']:
        g=fault[(fault.fault_count==rec['fault_count'])&(fault.magnitude==rec['magnitude'])&(fault.direction==rec['direction'])]
        check(r+' fault sample size '+str(rec['fault_count'])+str(rec['magnitude'])+rec['direction'],len(g)==rec['n_hours'])
        for metric in ['ls_alarm','heuristic_alarm','ls_exact_end_to_end','heuristic_exact_end_to_end','ls_known_count_exact','heuristic_known_count_exact']:
            check(r+' fault arithmetic '+metric,abs(g[metric].mean()-rec[metric])<1e-12)
    information[r]={'stations':ds.stations,'coords':ds.coords.to_numpy().tolist(),
                    'times_start':str(ds.times[0]),'times_end':str(ds.times[-1]),
                    'source':'archive' if r not in ['delhi2025','bengaluru_raw'] else r}
    print(r,'supplementary comparison and numerical audit complete',flush=True)

                                                                        
                                                                    
for mode in ['causal','retrospective']:
    for reference in ['stable_var','gaussian','spatial_sheaf','seasonal_persistence']:
        group=[x for x in tables if x['mode']==mode and x['reference']==reference and x.get('n',0)]
        ordered=sorted(group,key=lambda x:x['nominal_one_sided_p']);previous=0.
        for j,x in enumerate(ordered):
            adjusted=max(previous,min(1.,(len(ordered)-j)*x['nominal_one_sided_p']));previous=adjusted
            x['holm_across_networks_p']=adjusted
from sheafcode.data import haversine_km
for i,a in enumerate(information):
    for b in list(information)[i+1:]:
        A,B=information[a],information[b];ca=np.array(A['coords']);cb=np.array(B['coords'])
        distances=haversine_km(pd.DataFrame(np.vstack([ca,cb]),columns=['latitude','longitude']))[:len(ca),len(ca):]
        overlap.append({'a':a,'b':b,'shared_ids':len(set(A['stations'])&set(B['stations'])),
            'a_sites_within_200m_of_b':int((distances.min(1)<.2).sum()),'same_source':A['source']==B['source'],
            'calendar_overlap_hours':max(0,int((min(pd.Timestamp(A['times_end']),pd.Timestamp(B['times_end']))-max(pd.Timestamp(A['times_start']),pd.Timestamp(B['times_start']))).total_seconds()/3600)+1)})
(ROOT/'docs/overlap_with_coordinates.json').write_text(json.dumps(overlap,indent=2),encoding='utf-8')
(ROOT/'docs/final_audit.json').write_text(json.dumps({'checks':checks,'passed':len(checks),'failed':0,'supplementary_comparisons':tables},indent=2),encoding='utf-8')
pd.DataFrame(tables).to_csv(ROOT/'docs/dependence_sensitivity.csv',index=False)
print('FINAL NUMERICAL AUDIT:',len(checks),'checks passed',flush=True)
