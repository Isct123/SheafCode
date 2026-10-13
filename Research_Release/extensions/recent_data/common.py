import os,sys,json,hashlib,pickle,itertools
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'recent_data_results';DATA=OUT/'availability_expansion'
sys.path.insert(0,str(ROOT/'extensions/online_validation'))
from reference import QuadraticDecoder,FullVAR,complete_training,graph_correlation,seasonal
sys.path.insert(0,str(HERE))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
    if isinstance(v,(np.integer,)):return int(v)
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    if isinstance(v,(np.bool_,)):return bool(v)
    return v
def save(p,v):
    p=OUT/p;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(clean(v),indent=2),encoding='utf-8');t.replace(p)
def lock(p,v):
    target=OUT/p;v=clean(v)
    if target.exists() and json.loads(target.read_text())!=v:raise RuntimeError('Frozen stage mismatch: '+str(p))
    save(p,v)
def frame_save(p,f):
    p=OUT/p;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');f.to_csv(t,index=False);t.replace(p)
def load(region):
    d=np.load(OUT/region/'dataset.npz',allow_pickle=False)
    return {k:d[k] for k in d.files}|{'times':pd.DatetimeIndex(d['times'])}
def models(region):
    with (OUT/region/'models.pkl').open('rb') as f:return pickle.load(f)
def regions():return json.loads((OUT/'cohorts.json').read_text())['eligible']
def blocks(ds,part,stride=1,history=24,horizon=24):
    times=ds['times'];split=ds['split'];ans=[]
    for d in pd.unique(times[split==part].floor('D')):
        i=int(times.searchsorted(d));a=i-history;b=i+horizon
        if a>=0 and b<=len(times) and np.all(split[a:b]==part):ans.append((str(pd.Timestamp(d).date()),np.arange(a,b)))
    return ans[::stride]
def subset_days(xs,count):return [xs[i] for i in np.unique(np.linspace(0,len(xs)-1,min(count,len(xs))).astype(int))]
def quantile(scores,alpha=.1):
    x=np.asarray(scores);x=x[np.isfinite(x)];k=int(np.ceil((len(x)+1)*(1-alpha)))
    return float(np.partition(x,k-1)[k-1]) if k<=len(x) and k>0 else (float('inf') if k>len(x) else -float('inf'))
def paired(f,a,b,value='mse',draws=2000,days=7):
    p=f.pivot(index=['day','station'],columns='method',values=value)
    if a not in p or b not in p:return {'cases':0}
    p=p[[a,b]].dropna();date=pd.to_datetime(p.index.get_level_values('day'))
    if not len(p):return {'cases':0}
    bins=((date-date.min()).days//days);g=np.asarray([[len(v),v[a].sum(),v[b].sum()] for _,v in p.groupby(bins)])
    def gain(x):return 100*(1-np.sqrt(x[1]/x[2]))
    rng=np.random.default_rng(3091);boot=[gain(g[rng.integers(0,len(g),len(g))].sum(0)) for _ in range(draws)]
    return {'cases':len(p),'dates':len(set(date)),'calendar_bins':len(g),'bin_days':days,'gain_percent':gain(g.sum(0)),'ci95':np.quantile(boot,[.025,.975]).tolist() if len(g)>=5 else None,'scope':'Exploratory paired calendar-bin percentile interval; fewer than five bins suppress interval'}
def raw_city(x,v,mu,sd):
    raw=x*sd+mu;other=np.delete(raw,v,axis=1);counts=np.isfinite(other).sum(1)
                                                                         
    avg=np.divide(np.nansum(other,axis=1),counts,out=np.broadcast_to(mu[v],(len(x),len(mu[v]))).copy(),where=counts>0)
    return (avg-mu[v])/sd[v]
def static_solve(x,Q):
    shape=x.shape;y=x.reshape(len(x),-1).copy()
    for t,row in enumerate(y):
        U=np.flatnonzero(~np.isfinite(row));O=np.flatnonzero(np.isfinite(row))
        if len(U):row[U]=np.linalg.solve(Q[np.ix_(U,U)],-Q[np.ix_(U,O)]@row[O])
    return y.reshape(shape)
