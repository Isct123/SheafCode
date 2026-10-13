import itertools,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from holonomy import structured_components,outage_rank

def operator(n,p,edges,maps):
    rows=[]
    for (u,v),(Au,Av) in zip(edges,maps):
        B=np.zeros((len(Au),n*p));B[:,u*p:(u+1)*p]=Au;B[:,v*p:(v+1)*p]=-Av;rows.append(B)
    return np.vstack(rows) if rows else np.empty((0,n*p))

def test_identity_cycles_are_not_independent_generic_maps():
    edges=[(0,1),(1,2),(0,2)];maps=[(np.eye(2),np.eye(2))]*3;D=operator(3,2,edges,maps)
    r=structured_components(D,3,2,edges)
    assert r['kernel_dimension']==2 and r['block_distance']==3
    assert outage_rank(r,[0,1])==4 and outage_rank(r,[0,1,2])==4

def test_actual_rank_for_every_outage_disconnected_random_and_tied_maps():
    rng=np.random.default_rng(91);n=6;p=2;edges=[(0,1),(1,2),(0,2),(3,4)]
    for tied in [True,False]:
        maps=[(np.eye(p),np.eye(p)) if tied else (np.eye(p)+.2*rng.normal(size=(p,p)),np.eye(p)+.2*rng.normal(size=(p,p))) for e in edges]
        D=operator(n,p,edges,maps);r=structured_components(D,n,p,edges)
        assert r['predicted_rank']==np.linalg.matrix_rank(D)
        for k in range(1,n+1):
            for S in itertools.combinations(range(n),k):
                cols=np.concatenate([np.arange(v*p,(v+1)*p) for v in S])
                assert outage_rank(r,S)==np.linalg.matrix_rank(D[:,cols])

def test_extra_tied_rows_are_retained_in_root_constraints():
    edges=[(0,1)];Au=np.vstack([np.eye(2),np.eye(2)]);Av=np.vstack([2*np.eye(2),np.eye(2)])
    D=operator(2,2,edges,[(Au,Av)]);r=structured_components(D,2,2,edges,[4])
    assert r['predicted_rank']==4 and np.isinf(r['block_distance'])

def test_singular_and_thinned_maps_are_excluded():
    D=operator(2,2,[(0,1)],[(np.eye(2)[:1],np.eye(2)[:1])])
    assert not structured_components(D,2,2,[(0,1)],[1])['applicable']

def test_regularized_amplification_need_not_improve_when_rows_are_added():
    old=np.array([[.1]]);new=np.array([[.1],[.5]])
    so=np.linalg.svd(old,compute_uv=False);sn=np.linalg.svd(new,compute_uv=False)
    assert sn[-1]>so[-1]
    assert np.max(sn/(sn**2+1))>np.max(so/(so**2+1))
