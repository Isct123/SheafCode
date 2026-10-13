import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'online_validation_results'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    def clean(x):
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
        if isinstance(x,(np.integer,)):return int(x)
        if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
        if isinstance(x,(np.bool_,)):return bool(x)
        return x
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(clean(obj),indent=2),encoding='utf-8');temp.replace(p)

def lock(name,settings):
    p=OUT/name
    if p.exists() and json.loads(p.read_text())!=settings:
        raise RuntimeError('Frozen extension settings changed: '+name)
    save(name,settings)

def day_blocks(times,split,part,stride=1):
    rows=np.flatnonzero(split==part);day=times[rows].floor('D');blocks=[]
    for d in pd.unique(day):
        ix=rows[day==d]
        if len(ix)==24 and ix[0]>=24 and split[ix[0]-24]==part:
            blocks.append((str(pd.Timestamp(d).date()),np.arange(ix[0]-24,ix[-1]+1)))
    return blocks[::stride]

def paired(frame,method,reference,cluster_days=7,draws=2000):
    p=frame.pivot(index=['day','station'],columns='method',values='mse')[[method,reference]].dropna()
    if p.empty:return {'cases':0}
    dates=pd.to_datetime(p.index.get_level_values('day'));cluster=((dates-dates.min()).days//cluster_days)
    arr=np.array([[len(g),g[method].sum(),g[reference].sum()] for _,g in p.groupby(cluster)])
    rng=np.random.default_rng(9071)
    def gain(v):return 100*(1-np.sqrt(v[1]/v[2]))
    boot=[gain(arr[rng.integers(0,len(arr),len(arr))].sum(0)) for _ in range(draws)]
    total=arr.sum(0)
    return {'cases':len(p),'days':len(set(p.index.get_level_values('day'))),'clusters':len(arr),
            'cluster_days':cluster_days,'method_mse':total[1]/total[0],'reference_mse':total[2]/total[0],
            'gain_percent':gain(total),'ci95':np.quantile(boot,[.025,.975]).tolist(),
            'scope':'Exploratory calendar-cluster percentile interval; not a guarantee under arbitrary dependence.'}

def local(region):
    from sheafcode.cohort import select_cohort
    folder=ROOT/'results_expanded'/region;m=json.loads((folder/'manifest.json').read_text())
    for p,h in m['code_sha256'].items():
        if sha(ROOT/p)!=h:raise RuntimeError('Frozen core source changed: '+p)
    ds=select_cohort(region);ds.set_transform(m['choices']['sheaf']['transform'])
    return ds,m,folder
