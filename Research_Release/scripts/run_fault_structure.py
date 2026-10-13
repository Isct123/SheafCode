                                                                                       
import argparse,hashlib,itertools,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sheafcode.cohort import select_cohort
from sheafcode.reliable import structural_rank,union_margin

ap=argparse.ArgumentParser();ap.add_argument('--region',required=True)
ap.add_argument('--hours',type=int,default=120);ap.add_argument('--output',default='results')
args=ap.parse_args();out=ROOT/args.output/args.region
if not (out/'model.npz').exists():
    print('No qualifying completed model; fault study excluded',flush=True);sys.exit(0)
tic=time.time();rng=np.random.default_rng(73)
ds=select_cohort(args.region); choices=json.loads((out/'frozen_choices.json').read_text())
ds.set_transform(choices['sheaf']['transform']);Z=ds.Z();tr=ds.rows(0);va=ds.rows(1);te=ds.rows(2)
model=np.load(out/'model.npz');D=model['delta'];n,p=ds.n,ds.p
support=(abs(D)>1e-14).astype(int)
                                                                             
X=np.nan_to_num(Z[tr].reshape(len(tr),-1)); valid=(np.isfinite(Z[tr].reshape(len(tr),-1)).astype(int)@support.T)==support.sum(1)
res=X@D.T;counts=valid.sum(0)
mu=np.divide((res*valid).sum(0),counts,out=np.zeros(D.shape[0]),where=counts>0)
variance=np.divide(((res-mu)**2*valid).sum(0),np.maximum(counts-1,1))
sd=np.sqrt(np.maximum(variance,.01));D=D/sd[:,None];mu=mu/sd
G=D.T@D
sv=np.linalg.svd(D,compute_uv=False);global_margin=float(sv[-1]) if len(sv)==n*p else 0.
margin={'certified_full_matrix_lower_bound':global_margin,
        'interpretation':'lower bound valid for every column subset, including disconnected supports; may be vacuous'}
if n<=12:margin['exhaustive']=union_margin(D,p,4)
rankrows=[]
for size in range(1, min(n,4)+1):
    total=__import__('math').comb(n,size)
    sets=list(itertools.combinations(range(n),size)) if total<=5000 else sorted({tuple(sorted(rng.choice(n,size,replace=False))) for _ in range(5000)})
    for S in sets:
        cols=np.concatenate([np.arange(v*p,(v+1)*p) for v in S])
        M=D[:,cols];s=np.linalg.svd(M,compute_uv=False)
        learned=int((s>1e-8*max(s.max(),1e-300)).sum())
        rankrows.append({'size':size,'support':'+'.join(ds.stations[v] for v in S),'structural_support_rank':structural_rank(M),
                         'learned_rank':learned,'columns':len(cols),'scope':'exhaustive' if total<=5000 else 'sampled'})
pd.DataFrame(rankrows).to_csv(out/'structural_rank.csv',index=False)
cache={}
def candidates(x):
    observed=np.isfinite(x.reshape(-1));validrows=(support@observed.astype(int))==support.sum(1)
    cand=np.flatnonzero(np.isfinite(x).all(axis=1));key=(validrows.tobytes(),cand.tobytes())
    if key in cache:return cache[key]
                                                                         
    M=D[validrows];baseline=mu[validrows];gram=M.T@M
    groups={}
    for k in [1,2]:
        ss=list(itertools.combinations(cand.tolist(),k))
        if not ss:continue
        cols=np.array([np.concatenate([np.arange(v*p,(v+1)*p) for v in S]) for S in ss])
        sub=gram[cols[:,:,None],cols[:,None,:]]
        inverse=np.linalg.pinv(sub,rcond=1e-10)
        groups[k]=(ss,cols,inverse)
    incident=(np.reshape(np.abs(M),(len(M),n,p)).sum(2)>0).astype(float)
    if len(cache)>=32:cache.clear()
    cache[key]=(M,baseline,groups,cand,incident)
    return cache[key]

def decode(x):
    M,baseline,groups,cand,incident=candidates(x)
    s=M@np.nan_to_num(x.reshape(-1))-baseline;b=M.T@s
    scores={};best={}
    for k,(ss,cols,inv) in groups.items():
        bs=b[cols];gain=np.einsum('bi,bij,bj->b',bs,inv,bs)
        ix=int(np.argmax(gain));scores[k]=max(float(gain[ix]),0.);best[k]=tuple(ss[ix])
    energy=(incident.T@(s*s))/np.maximum(incident.sum(0),1)
    order=sorted(cand.tolist(),key=lambda v:(-energy[v],v))
    heur={k:float(sum(energy[v] for v in order[:k])) for k in scores}
    return scores,best,heur,order,s

def choose(scores,best,penalty):
    k=max([0]+list(scores),key=lambda k:0. if k==0 else scores[k]-k*penalty)
    return () if k==0 or scores[k]-k*penalty<=0 else best[k]

def hours(rows,count):
    ids=np.unique(np.linspace(0,len(rows)-1,min(count,len(rows))).astype(int))
    return rows[ids]
cal=[];hcal=[]
for t in hours(va,args.hours):
    gain,best,heur,order,s=decode(Z[t])
    if not gain:continue
    cal.append(max(gain[k]/k for k in gain));hcal.append(max(heur[k]/k for k in heur))
if len(cal)<20:raise ValueError('Insufficient clean calibration hours')
def quantile(a):
    a=sorted(a);k=int(np.ceil((len(a)+1)*.95));return a[min(k-1,len(a)-1)]
penalty=quantile(cal);hpenalty=quantile(hcal)
(out/'fault_calibration.json').write_text(json.dumps({'validation_hours':len(cal),'ls_penalty':penalty,'heuristic_penalty':hpenalty,
    'calibrated_before_test':pd.Timestamp.now(tz='UTC').isoformat(),'candidate_universe':'all fully observed stations, identical for both decoders'}),encoding='utf-8')
rows=[]
for t in hours(te,args.hours):
    x=Z[t];fully=np.flatnonzero(np.isfinite(x).all(axis=1))
    if len(fully)<2:continue
    clean=decode(x);base_r=clean[-1];day=str(ds.times[t].date())
                                                                           
                                                                         
    observed_operator=candidates(x)[0]
    candidate_cols=np.concatenate([np.arange(v*p,(v+1)*p) for v in fully])
    actual_sv=np.linalg.svd(observed_operator[:,candidate_cols],compute_uv=False)
    actual_margin=float(actual_sv[-1]) if len(actual_sv)==len(candidate_cols) else 0.
    for true_k in [0,1,2]:
        magnitudes=[0.] if true_k==0 else [1.,2.,3.]
        for magnitude in magnitudes:
            directions=['clean'] if true_k==0 else ['positive','negative','random']
            for direction in directions:
                F=tuple(sorted(rng.choice(fully,true_k,replace=False).tolist())) if true_k else ()
                y=x.copy();faults=[]
                for v in F:
                    shift=np.ones(p) if direction=='positive' else -np.ones(p) if direction=='negative' else rng.normal(size=p)
                    shift=shift/np.linalg.norm(shift)*np.sqrt(p)*magnitude
                    y[v]+=shift;faults.append(shift)
                gain,best,heur,order,s=decode(y)
                found=choose(gain,best,penalty)
                hbest={k:tuple(sorted(order[:k])) for k in heur};hfound=choose(heur,hbest,hpenalty)
                bound=bool(true_k and min(np.linalg.norm(f) for f in faults)*actual_margin>2*np.linalg.norm(base_r))
                known=tuple(best.get(true_k,()))
                if bound and known!=F:raise AssertionError('Corrected robust sufficient bound violated')
                rows.append({'day':day,'hour':str(ds.times[t]),'fault_count':true_k,'magnitude':magnitude,'direction':direction,
                    'fault_support':'+'.join(ds.stations[v] for v in F),'ls_support':'+'.join(ds.stations[v] for v in found),
                    'heuristic_support':'+'.join(ds.stations[v] for v in hfound),'candidate_stations':len(fully),
                    'ls_alarm':int(bool(found)),'heuristic_alarm':int(bool(hfound)),
                    'ls_exact_end_to_end':int(found==F),'heuristic_exact_end_to_end':int(hfound==F),
                    'ls_known_count_exact':int(known==F),'heuristic_known_count_exact':int(tuple(sorted(order[:true_k]))==F),
                    'corrected_bound_sufficient':int(bound),'actual_mask_margin':actual_margin,'clean_residual_norm':float(np.linalg.norm(base_r))})
R=pd.DataFrame(rows);R.to_csv(out/'fault_trials.csv',index=False)
summary=[]
for key,g in R.groupby(['fault_count','magnitude','direction']):
    summary.append({'fault_count':int(key[0]),'magnitude':float(key[1]),'direction':key[2],'n_hours':len(g),
        **{c:float(g[c].mean()) for c in ['ls_alarm','heuristic_alarm','ls_exact_end_to_end','heuristic_exact_end_to_end','ls_known_count_exact','heuristic_known_count_exact']},
        'sufficient_bound_trials':int(g.corrected_bound_sufficient.sum())})
(out/'fault_structure_summary.json').write_text(json.dumps({'region':args.region,'margin':margin,'results':summary,
    'rank_sets':len(rankrows),'rank_agreements':sum(r['learned_rank']==r['structural_support_rank'] for r in rankrows),
    'input_model_sha256':hashlib.sha256((out/'model.npz').read_bytes()).hexdigest(),
    'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'elapsed_seconds':time.time()-tic},indent=2),encoding='utf-8')
print(args.region,'fault/structure complete',len(R),'trials',round(time.time()-tic,1),'seconds',flush=True)
