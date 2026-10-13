                                                                            
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
from robust_quantity import fault_envelope
from sheafcode.cohort import select_cohort

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--hours',type=int,default=120);ap.add_argument('--output',default='followup_results/robust_quantity');args=ap.parse_args()
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True);records=[];checks=0;config={}
    for region in ['delhi2025','bengaluru_raw']:
        folder=ROOT/'results_expanded'/region;m=json.loads((folder/'manifest.json').read_text());ds=select_cohort(region)
        ds.set_transform(m['choices']['sheaf']['transform']);Z=ds.Z();model=np.load(folder/'model.npz');D=model['delta']
        complete=np.isfinite(Z).all(axis=(1,2));tr=np.flatnonzero((ds.split==0)&complete);va=np.flatnonzero((ds.split==1)&complete);te=np.flatnonzero((ds.split==2)&complete)
        if min(len(tr),len(va),len(te))<20:raise ValueError('Insufficient complete vectors')
        scale=np.maximum(np.std(Z[tr].reshape(len(tr),-1)@D.T,axis=0),.1);D=D/scale[:,None]
        scores=np.linalg.norm(Z[va].reshape(len(va),-1)@D.T,axis=1);k=int(np.ceil((len(scores)+1)*.95))
        if k>len(scores):raise ValueError('Finite residual calibration unavailable')
        epsilon=float(np.sort(scores)[k-1]);q=ds.pollutants.index('PM2.5')
        config[region]={'validation_budget':epsilon,'model_sha256':hashlib.sha256((folder/'model.npz').read_bytes()).hexdigest(),'validation_complete_hours':len(va)}
        sample=te[np.unique(np.linspace(0,len(te)-1,min(args.hours,len(te))).astype(int))]
        for t in sample:
            for v in range(ds.n):
                U=np.arange(v*ds.p,(v+1)*ds.p);O=np.setdiff1d(np.arange(ds.n*ds.p),U);candidates=[w for w in range(ds.n) if w!=v]
                blocks=[D[:,np.arange(w*ds.p,(w+1)*ds.p)] for w in candidates]
                h=np.zeros(ds.p);h[q]=1
                for scenario in ['clean','positive_bias','negative_bias']:
                    observed=Z[t].reshape(-1).copy();fault=candidates[(int(t)+v)%len(candidates)]
                    if scenario!='clean':observed[fault*ds.p:(fault+1)*ds.p]+=2 if scenario=='positive_bias' else -2
                                                                               
                    r=fault_envelope(D[:,U],D[:,O]@observed[O],h,blocks,1,epsilon)
                    truth=float(Z[t,v,q]);assumption=np.linalg.norm(D@Z[t].reshape(-1))<=epsilon
                    covered=r['status']!='inconsistent' and r['lower']-1e-8<=truth<=r['upper']+1e-8
                    if assumption:assert covered;checks+=1
                    def raw(x):
                        value=x*ds.std[v,q]+ds.mean[v,q]
                        if ds.hod is not None:value+=ds.hod[ds.times[t].hour,v,q]
                        return float(np.expm1(value)) if ds.transform!='none' else float(value)
                    lo=raw(r['lower']) if r['status']!='inconsistent' else None;hi=raw(r['upper']) if r['status']!='inconsistent' else None
                    decision='abstain' if lo is None else ('above' if lo>60 else ('below' if hi<=60 else 'abstain'))
                    records.append({'region':region,'timestamp':str(ds.times[t]),'station':ds.stations[v],'scenario':scenario,'status':r['status'],
                        'covered':covered,'residual_assumption_met':bool(assumption),'pm25_lower':lo,'pm25_upper':hi,'decision':decision,
                        'decision_correct':None if decision=='abstain' else bool((decision=='above')==(ds.X[t,v,q]>60)),'feasible_supports':r.get('feasible_supports',0)})
        print(region,'robust quantity complete',flush=True)
    frame=pd.DataFrame(records);frame.to_csv(out/'cases.csv',index=False);rows=[]
    for key,g in frame.groupby(['region','scenario']):
        d=g[g.decision!='abstain'];rows.append({'region':key[0],'scenario':key[1],'cases':len(g),'coverage':float(g.covered.mean()),
            'bounded_fraction':float((g.status=='bounded').mean()),'decided_fraction':len(d)/len(g),'accuracy_when_decided':float(d.decision_correct.mean()) if len(d) else None})
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
    (out/'manifest.json').write_text(json.dumps({'config':config,'conditional_containment_checks':checks,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'robust_quantity.py']},
        'limitations':['Complete-vector subset; one hidden station and at most one unknown faulty surviving station','Injected equal-channel plus/minus two standardized units; not arbitrary real drift validation',
        '95% validation residual budget is empirical; no arbitrary distribution-shift guarantee','Illustrative hourly PM threshold; no official AQI or environmental intervention','Sparse-support envelope may be unbounded or mostly abstain; report this without filtering']},indent=2))

if __name__=='__main__':main()
