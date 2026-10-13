                                                                             
import hashlib,itertools,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
from holonomy import structured_components,outage_rank
from sheafcode.cohort import select_cohort
from sheafcode.graph import knn_edges
from sheafcode.data import haversine_km
out=ROOT/'extension_results/structured';out.mkdir(parents=True,exist_ok=True)
results={};checks=[]
for folder in sorted((ROOT/'results').iterdir()):
    if not (folder/'model.npz').exists():continue
    m=json.loads((folder/'manifest.json').read_text());ds=select_cohort(folder.name);D=np.load(folder/'model.npz')['delta']
    edges=knn_edges(haversine_km(ds.coords),m['choices']['sheaf']['k'])
    r=structured_components(D,ds.n,ds.p,edges);assert r['applicable'],r
    actual=int(np.sum(np.linalg.svd(D,compute_uv=False)>r['rank_absolute_tolerance']))
    assert actual==r['predicted_rank'];rng=np.random.default_rng(472);sets={tuple(range(ds.n))}
    for k in range(1,min(4,ds.n)+1):
        if __import__('math').comb(ds.n,k)<=1000:sets.update(itertools.combinations(range(ds.n),k))
        else:sets.update(tuple(sorted(rng.choice(ds.n,k,replace=False))) for _ in range(1000))
    for S in sorted(sets,key=lambda x:(len(x),x)):
        cols=np.concatenate([np.arange(v*ds.p,(v+1)*ds.p) for v in S]);sv=np.linalg.svd(D[:,cols],compute_uv=False)
        observed=int(np.sum(sv>r['rank_absolute_tolerance']));predicted=outage_rank(r,S)
        assert observed==predicted,(folder.name,S,observed,predicted)
        checks.append({'region':folder.name,'stations':list(S),'predicted_rank':predicted,'observed_rank':observed})
    r['block_distance']=r['block_distance'] if np.isfinite(r['block_distance']) else 'infinity'
    r['model_sha256']=hashlib.sha256((folder/'model.npz').read_bytes()).hexdigest();results[folder.name]=r
    print(folder.name,'actual-map rank verified:',len(sets),'sets; kernel',r['kernel_dimension'],flush=True)
pd.DataFrame(checks).to_csv(out/'rank_checks.csv',index=False)
(out/'summary.json').write_text(json.dumps({'results':results,'rank_agreements':len(checks),'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
    'scope':'exact invertible-map theorem; numerical verification uses stated tolerances; sampled large outage families, not exhaustive 2^n enumeration'},indent=2))
