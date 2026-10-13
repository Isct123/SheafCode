import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pytest
from certificate import functional_interval

def test_functional_identifiable_although_full_vector_is_not():
    A=np.array([[1.,0.,0.],[0.,1.,0.]])
    r=functional_interval(A,np.array([-2.,-3.]),np.array([1.,0.,0.]),.5)
    assert r['rank']==2 and r['status']=='bounded'
    np.testing.assert_allclose([r['lower'],r['upper']],[1.5,2.5])
    assert functional_interval(A,np.array([-2.,-3.]),np.array([0.,0.,1.]),.5)['status']=='unbounded'

def test_empty_feasible_set_and_bad_inputs_are_explicit():
    assert functional_interval(np.array([[1.],[0.]]),np.array([0.,2.]),np.ones(1),1.)['status']=='inconsistent'
    with pytest.raises(ValueError):functional_interval(np.eye(2),np.ones(2),np.ones(2),-1.)

def test_bound_is_attained_on_rank_deficient_random_problems():
    rng=np.random.default_rng(810)
    for _ in range(100):
        A=rng.normal(size=(4,7));b=rng.normal(size=4);h=A.T@rng.normal(size=4);eps=float(rng.uniform(.1,2.))
        r=functional_interval(A,b,h,eps);assert r['status']=='bounded'
        pinv=np.linalg.pinv(A);center=-pinv@b;v=pinv@pinv.T@h
        direction=v/np.linalg.norm(A@v)
        for sign,key in [(-1,'lower'),(1,'upper')]:
            z=center+sign*eps*direction
            assert abs(np.linalg.norm(A@z+b)-eps)<1e-9
            assert abs(h@z-r[key])<1e-9
