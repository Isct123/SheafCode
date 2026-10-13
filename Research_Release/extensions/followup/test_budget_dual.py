import numpy as np
from budget_dual import dual_envelope
from constrained import constrained_envelope

def test_dual_outer_bounds_exact_sparse_envelope():
    rng=np.random.default_rng(78)
    for _ in range(15):
        A=rng.normal(size=(6,2));z=rng.normal(size=2);blocks=[rng.normal(size=(6,1)) for _ in range(3)]
        b=-A@z+blocks[0][:,0]*.2;h=rng.normal(size=2)
        exact=constrained_envelope(A,b,h,blocks,1,.1,.5)
        dual=dual_envelope(A,b,h,blocks,1,.1,.5)
        assert dual['lower']<=exact['lower']+1e-6
        assert dual['upper']>=exact['upper']-1e-6

def test_fault_budget_zero_and_box_only():
    A=np.eye(2);b=np.array([-3.,-4.]);h=np.array([1.,0.]);blocks=[np.eye(2)]
    r=dual_envelope(A,b,h,blocks,0,.1,2.)
    assert np.allclose([r['lower'],r['upper']],[2.9,3.1])
    r=dual_envelope(np.zeros((1,2)),np.zeros(1),h,[],0,.1,0.,[1.,2.],[3.,4.])
    assert np.allclose([r['lower'],r['upper']],[1.,3.])

def test_convexification_gap_is_real_not_exact_sparse_claim():
    A=np.ones((2,1));b=-np.ones(2);h=np.ones(1)
    blocks=[np.eye(2)[:,[0]],np.eye(2)[:,[1]]]
    exact=constrained_envelope(A,b,h,blocks,1,0.,2.)
    dual=dual_envelope(A,b,h,blocks,1,0.,2.)
    assert np.allclose([exact['lower'],exact['upper']],[1.,1.])
    assert np.allclose([dual['lower'],dual['upper']],[0.,2.])
