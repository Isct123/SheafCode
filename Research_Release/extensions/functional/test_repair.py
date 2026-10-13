import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from certificate import functional_interval
from repair import worst_case_repair_halfwidth

def test_sharp_repair_bound_is_attained_and_dominates_feasible_repairs():
    rng=np.random.default_rng(920)
    for _ in range(100):
        A=rng.normal(size=(5,7));b=rng.normal(size=5);h=A.T@rng.normal(size=5)
        eps=2.;R=np.sort(rng.choice(7,2,replace=False));K=np.setdiff1d(np.arange(7),R)
        bound=worst_case_repair_halfwidth(A,b,h,eps,R)
        attained=functional_interval(A[:,K],b+A[:,R]@np.array(bound['attaining_repair_reading']),h[K],eps)
        assert abs(attained['halfwidth']-bound['halfwidth'])<1e-9
        z0=-np.linalg.pinv(A)@b
        for _ in range(5):
            v=np.linalg.pinv(A)@rng.normal(size=5);v*=rng.uniform(0,eps)/np.linalg.norm(A@v)
            z=z0+v;after=functional_interval(A[:,K],b+A[:,R]@z[R],h[K],eps)
            assert after['status']=='bounded' and after['halfwidth']<=bound['halfwidth']+1e-9

def test_repair_can_remove_unbounded_functional_ambiguity():
    A=np.array([[1.,0.]]);b=np.zeros(1);h=np.array([0.,1.])
    assert functional_interval(A,b,h,1.)['status']=='unbounded'
    assert worst_case_repair_halfwidth(A,b,h,1.,[0])['status']=='unbounded'
    assert worst_case_repair_halfwidth(A,b,h,1.,[1])['halfwidth']==0.

def test_full_repair_and_infeasibility():
    assert worst_case_repair_halfwidth(np.eye(2),np.zeros(2),np.ones(2),1.,[0,1])['halfwidth']==0.
    assert worst_case_repair_halfwidth(np.zeros((1,2)),np.ones(1),np.ones(2),0.,[0])['status']=='inconsistent'
