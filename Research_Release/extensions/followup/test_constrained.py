import numpy as np
from constrained import constrained_envelope

def test_bounded_faults_restore_finite_quantity_range():
    A=np.eye(2);b=np.array([-4.,-2.]);h=np.array([1.,0.]);B=[np.array([[1.],[0.]])]
    r=constrained_envelope(A,b,h,B,1,.1,2.)
    assert np.allclose([r['lower'],r['upper']],[1.9,6.1])
    s=constrained_envelope(A,b,h,B,1,.1,1.)
    assert s['lower']>=r['lower'] and s['upper']<=r['upper']

def test_temporal_box_tightens_and_endpoints_attain():
    A=np.eye(2);b=np.array([-4.,-2.]);h=np.array([1.,0.]);B=[np.array([[1.],[0.]])]
    r=constrained_envelope(A,b,h,B,1,.1,2.,[3.5,1.],[4.5,3.])
    assert np.allclose([r['lower'],r['upper']],[3.5,4.5])

def test_invalid_assumption_can_exclude_truth():
                                                        
    r=constrained_envelope(np.ones((1,1)),np.array([-3.]),np.ones(1),[np.ones((1,1))],1,0.,.1)
    assert not r['lower']<=1.<=r['upper']

def test_random_feasible_truth_is_contained():
    rng=np.random.default_rng(92)
    for _ in range(30):
        A=rng.normal(size=(7,2));blocks=[rng.normal(size=(7,2)) for _ in range(2)]
        z=rng.uniform(-1,1,2);e=rng.uniform(-.5,.5,2);noise=rng.uniform(-.01,.01,7);b=-A@z-blocks[0]@e+noise;h=rng.normal(size=2)
        r=constrained_envelope(A,b,h,blocks,1,.01,.5,[-1.,-1.],[1.,1.])
        assert r['lower']-1e-7<=h@z<=r['upper']+1e-7
