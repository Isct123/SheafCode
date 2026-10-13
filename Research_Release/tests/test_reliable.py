import itertools
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sheafcode.reliable import AdaptiveQuantile, CachedSpatial, structural_rank, union_margin


def test_adaptive_arbitrary_sequence_bound_and_full_sets():
    model=AdaptiveQuantile([1.],gamma=.005)
    mx=1.; full=0
    for _ in range(1000):
        q=model.quantile(); full+=int(np.isposinf(q))
        mx*=1.5
        model.update(mx,q)
        assert -.005-1e-12 <= model.level <= 1.005+1e-12
    bound=(.9+.005)/(model.n*.005)
    assert abs(model.errors/model.n-.1)<=bound+1e-12
    assert full>0


def test_corrected_minimum_block_bound_and_old_counterexample():
    D=np.array([[1.,-1.,0.],[0.,1.,-1.],[-2.,0.,1.]])
    f=np.array([100.,.001,0.]); r=D[:,2]; s=D@f+r
    sigma=np.linalg.svd(D,compute_uv=False)[-1]
    def residual(S,s):
        return np.linalg.norm(s-D[:,S]@np.linalg.lstsq(D[:,S],s,rcond=None)[0])
    assert np.linalg.norm(f)*sigma>2*np.linalg.norm(r)
    assert residual([0,2],s)<residual([0,1],s)
    assert min(abs(f[:2]))*sigma<2*np.linalg.norm(r)
    f=np.array([100.,100.,0.]); s=D@f+r
    assert min(abs(f[:2]))*sigma>2*np.linalg.norm(r)
    assert min(itertools.combinations(range(3),2),key=lambda S:residual(list(S),s))==(0,1)


def test_matching_and_disconnected_margin():
    D=np.array([[1.,1.,0.],[0.,0.,1.]])
    assert structural_rank(D)==2
    m=union_margin(D,1,3)
    assert m['2']['sets_checked']==3
    assert m['3']['sigma_min']==0


def test_missing_equations_require_actual_mask_margin():
    D=np.eye(3)
    masked=D[:2]
    full_margin=np.linalg.svd(D,compute_uv=False)[-1]
    singular=np.linalg.svd(masked,compute_uv=False)
    masked_margin=float(singular[-1]) if len(singular)==masked.shape[1] else 0.
    assert full_margin==1.
    assert masked_margin==0.
                                                                            
    assert np.allclose(masked@np.array([0.,0.,100.]),0.)


def test_target_erasure_and_natural_gap_cache_invariance():
    rng=np.random.default_rng(5); A=rng.normal(size=(12,12)); Q=A.T@A+np.eye(12)
    model=CachedSpatial(Q,4,3)
    X=rng.normal(size=(10,4,3)); X[2,2,1]=np.nan
    a,_=model.predict(X,[1]); X[:,1,:]=1e8
    b,_=model.predict(X,[1]); assert np.allclose(a,b)


def test_constant_score_scale_is_exact_control():
    a=AdaptiveQuantile([1.,2.,3.]); b=AdaptiveQuantile([1/7,2/7,3/7])
    for s in [4.,1.,2.,8.,.2]*10:
        qa=a.quantile(); qb=b.quantile()
        assert np.isclose(qa,qb*7) or (np.isinf(qa) and np.isinf(qb))
        a.update(s,qa); b.update(s/7,qb)
