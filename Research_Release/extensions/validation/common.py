import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'extensions/followup'))
OUT=ROOT/'validation_results';OUT.mkdir(exist_ok=True)
REGIONS=['delhi2025','bengaluru_raw']
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):
    def clean(x):
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
        if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
        if isinstance(x,(int,np.integer)):return int(x)
        return x
    (OUT/name).write_text(json.dumps(clean(obj),indent=2),encoding='utf-8')
def cohort(region):
    from sheafcode.cohort import select_cohort
    folder=ROOT/'results_expanded'/region;m=json.loads((folder/'manifest.json').read_text())
    for name,sha in m['code_sha256'].items():
        if digest(ROOT/name)!=sha:raise RuntimeError('Frozen source changed: '+name)
    ds=select_cohort(region)
    return ds,m,folder
def dayblocks(ds,part,count=12):
    rows=ds.rows(part);days=ds.times[rows].floor('D');blocks=[]
    for day in pd.unique(days):
        ix=rows[days==day]
        if len(ix)==24 and ix[0]>=24 and ds.split[ix[0]-24]==part:blocks.append((str(pd.Timestamp(day).date()),np.arange(ix[0]-24,ix[-1]+1)))
    if count is None:return blocks
    return [blocks[i] for i in np.unique(np.linspace(0,len(blocks)-1,min(count,len(blocks))).astype(int))]
def paired_interval(frame,a,b,draws=1000):
    p=frame.pivot(index=['day','station'],columns='method',values='mse')[[a,b]].dropna()
    if not len(p):return {'cases':0}
    grouped=[g.to_numpy() for _,g in p.groupby(level='day')];rng=np.random.default_rng(511)
    def gain(v):return 100*(1-np.sqrt(v[:,0].mean()/v[:,1].mean()))
    bootstrap=[gain(np.concatenate([grouped[j] for j in rng.integers(0,len(grouped),len(grouped))])) for _ in range(draws)]
    return {'cases':len(p),'days':len(grouped),'rmse_improvement_percent':gain(p.to_numpy()),'paired_day_ci':np.quantile(bootstrap,[.025,.975]).tolist()}
