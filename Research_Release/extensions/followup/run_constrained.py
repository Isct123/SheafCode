                                                                                   
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
from constrained import constrained_envelope
from robust_quantity import fault_envelope
from sheafcode.cohort import select_cohort

def rank_quantile(values,alpha=.05):
    values=np.asarray(values,float);k=int(np.ceil((len(values)+1)*(1-alpha)))
    if k>len(values):raise ValueError('Insufficient validation calibration')
    return float(np.sort(values)[k-1])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--hours',type=int,default=40);args=ap.parse_args()
    out=ROOT/'constrained_results';out.mkdir(exist_ok=True);rng=np.random.default_rng(710);records=[];configs={};checks=0;start=time.time()
    for region in ['delhi2025','bengaluru_raw']:
        folder=ROOT/'results_expanded'/region;m=json.loads((folder/'manifest.json').read_text());ds=select_cohort(region)
        ds.set_transform(m['choices']['sheaf']['transform']);Z=ds.Z();D=np.load(folder/'model.npz')['delta'];q=ds.pollutants.index('PM2.5')
        complete=np.isfinite(Z).all(axis=(1,2));tr=np.flatnonzero((ds.split==0)&complete)
                                                                                
                                                                           
                                                                                 
        test_rows=np.flatnonzero(ds.split==2);cut=int(test_rows[len(test_rows)//2]);calibration=(ds.split==2)&(np.arange(len(Z))<cut)
        evaluation=(ds.split==2)&(np.arange(len(Z))>=cut);va=np.flatnonzero(calibration&complete)
        scale=np.maximum(np.std(Z[tr].reshape(len(tr),-1)@D.T,axis=0),.1);D=D/scale[:,None]
        cal_residual=Z[va].reshape(len(va),-1)@D.T;box_eps=rank_quantile(np.max(abs(cal_residual),axis=1));norm_eps=rank_quantile(np.linalg.norm(cal_residual,axis=1))
        configs[region]={'box_budget':box_eps,'norm_budget':norm_eps,'model_sha256':hashlib.sha256((folder/'model.npz').read_bytes()).hexdigest(),
            'calibration_start':str(ds.times[test_rows[0]]),'evaluation_start':str(ds.times[cut]),'sampled_hours_per_horizon':args.hours,'temporal':{}}
        for horizon in [24,72]:
            end=np.flatnonzero(evaluation&complete);end=end[(end>=horizon)&evaluation[np.maximum(0,end-horizon)]]
            sampled=end[np.unique(np.linspace(0,len(end)-1,min(args.hours,len(end))).astype(int))]
            for v in range(ds.n):
                train=tr[(tr>=horizon)&(ds.split[np.maximum(0,tr-horizon)]==0)&np.isfinite(Z[np.maximum(0,tr-horizon),v]).all(axis=1)]
                delta_scale=np.maximum(np.std(Z[train,v]-Z[train-horizon,v],axis=0),.1)
                valid=va[(va>=horizon)&calibration[np.maximum(0,va-horizon)]&np.isfinite(Z[np.maximum(0,va-horizon),v]).all(axis=1)]
                resid=np.max(abs(Z[valid].reshape(len(valid),-1)@D.T),axis=1)
                drift=np.max(abs((Z[valid,v]-Z[valid-horizon,v])/delta_scale),axis=1)
                joint_eps=rank_quantile(np.maximum(resid,drift));configs[region]['temporal'][str(horizon)+'_'+str(v)]={'joint_budget':joint_eps,'training_increment_scale':delta_scale.tolist(),'validation_cases':len(valid)}
                U=np.arange(v*ds.p,(v+1)*ds.p);O=np.setdiff1d(np.arange(ds.n*ds.p),U);neighbors=[w for w in range(ds.n) if w!=v]
                blocks=[D[:,w*ds.p:(w+1)*ds.p] for w in neighbors];h=np.eye(ds.p)[q]
                for t in sampled:
                    if not np.isfinite(Z[t-horizon,v]).all():continue
                    target=Z[t,v];fullres=D@Z[t].reshape(-1);prior=Z[t-horizon,v]
                    zero=-ds.mean[v].copy()
                    if ds.hod is not None:zero-=ds.hod[ds.times[t].hour,v]
                    physical_lower=zero/ds.std[v]
                    lower=np.maximum(physical_lower,prior-joint_eps*delta_scale);upper=prior+joint_eps*delta_scale
                    for scenario in ['clean','positive_bias','negative_bias','random_bias']:
                        observed=Z[t].reshape(-1).copy();fault=neighbors[(int(t)+v)%len(neighbors)]
                        e=np.zeros(ds.p)
                        if scenario=='positive_bias':e[:]=2
                        if scenario=='negative_bias':e[:]=-2
                        if scenario=='random_bias':e=rng.uniform(-2,2,ds.p)
                        observed[fault*ds.p:(fault+1)*ds.p]+=e;b=D[:,O]@observed[O]
                        outputs={
                            'unrestricted_fault':fault_envelope(D[:,U],b,h,blocks,1,norm_eps),
                            'bounded_fault':constrained_envelope(D[:,U],b,h,blocks,1,box_eps,2.,physical_lower),
                            'joint_budget_no_history':constrained_envelope(D[:,U],b,h,blocks,1,joint_eps,2.,physical_lower)}
                        outputs['joint_temporal']=constrained_envelope(D[:,U],b,h,blocks,1,joint_eps,2.,lower,upper) if np.all(lower<=upper) else {'status':'inconsistent'}
                        assumptions={
                            'unrestricted_fault':np.linalg.norm(fullres)<=norm_eps,
                            'bounded_fault':np.max(abs(fullres))<=box_eps and np.all(target>=physical_lower),
                            'joint_budget_no_history':np.max(abs(fullres))<=joint_eps and np.all(target>=physical_lower),
                            'joint_temporal':max(np.max(abs(fullres)),np.max(abs((target-prior)/delta_scale)))<=joint_eps and np.all(target>=physical_lower)}
                        def raw(x):
                            value=x*ds.std[v,q]+ds.mean[v,q]
                            if ds.hod is not None:value+=ds.hod[ds.times[t].hour,v,q]
                            return float(np.expm1(value)) if ds.transform!='none' else float(value)
                        for method,r in outputs.items():
                            covered=r['status']!='inconsistent' and r['lower']-1e-6<=target[q]<=r['upper']+1e-6
                            if assumptions[method]:assert covered,(region,t,v,scenario,method,r);checks+=1
                            L=raw(r['lower']) if r['status']!='inconsistent' else None;H=raw(r['upper']) if r['status']!='inconsistent' else None
                            decision='abstain' if L is None else ('above' if L>60 else ('below' if H<=60 else 'abstain'))
                            positive=bool(ds.X[t,v,q]>60)
                            records.append({'region':region,'timestamp':str(ds.times[t]),'station':ds.stations[v],'horizon':horizon,'scenario':scenario,'method':method,'status':r['status'],
                                'covered':covered,'assumptions_met':bool(assumptions[method]),'lower':L,'upper':H,'truth':float(ds.X[t,v,q]),'positive':positive,'decision':decision,
                                'correct_decision':None if decision=='abstain' else bool((decision=='above')==positive)})
                print(region,horizon,ds.stations[v],'complete',round(time.time()-start,1),'seconds',flush=True)
                                                                               
        (out/(region+'_config.json')).write_text(json.dumps(configs[region],indent=2))
    f=pd.DataFrame(records);f.to_csv(out/'cases.csv',index=False);summary=[]
    for keys,g in f.groupby(['region','horizon','scenario','method']):
        d=g[g.decision!='abstain'];pos=g[g.positive];neg=g[~g.positive]
        width=g.upper-g.lower;finite=width[np.isfinite(width)]
        summary.append({'region':keys[0],'horizon':keys[1],'scenario':keys[2],'method':keys[3],'cases':len(g),'coverage':float(g.covered.mean()),
            'assumption_frequency':float(g.assumptions_met.mean()),'inconsistent_fraction':float((g.status=='inconsistent').mean()),'decided_fraction':len(d)/len(g),
            'decision_accuracy':float(d.correct_decision.mean()) if len(d) else None,'positive_cases':len(pos),'negative_cases':len(neg),
            'positive_alert_sensitivity':float((pos.decision=='above').mean()) if len(pos) else None,'false_positive_rate':float((neg.decision=='above').mean()) if len(neg) else None,
            'mean_finite_width':float(finite.mean()) if len(finite) else None})
    s=pd.DataFrame(summary);s.to_csv(out/'summary.csv',index=False)
    manifest={'conditional_containment_checks':checks,'elapsed_seconds':time.time()-start,'config':configs,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'constrained.py']},
        'scope':'Complete-vector internal archives; calibration on earlier archive-test half, evaluation on later half after model selection; artificial 24/72-hour outages scored at sampled endpoints; target last observed before outage, never during it; zero/one unknown bounded surviving-station fault; hourly illustrative threshold',
        'limitations':['Joint rank calibration needs exchangeability for its finite-sample interpretation; no arbitrary-shift guarantee','Fault bound two standardized units is a declared scenario, not fitted proof of real-world sensor bias limits',
        'Linear-programming primal tolerances are checked; this is numerical, not an exact-arithmetic certificate','Pointwise calibration is not simultaneous coverage across all stations or a full outage','No official AQI, health or emissions intervention demonstrated']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('Constrained study complete:',len(f),'rows,',checks,'conditional checks',flush=True)

if __name__=='__main__':main()
