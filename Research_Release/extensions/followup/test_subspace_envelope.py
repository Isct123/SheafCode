import numpy as np
from robust_quantity import fault_envelope
from subspace_envelope import compressed_envelope

def test_duplicate_and_scaled_blocks_compress_without_changing_bounds():
    rng=np.random.default_rng(91);A=rng.normal(size=(8,2));b=rng.normal(size=8);h=np.array([1.,-.5])
    blocks=[np.eye(8)[:,[j%2]]*(j+1) for j in range(10)]
    a=fault_envelope(A,b,h,blocks,3,4.)
    c=compressed_envelope(A,b,h,blocks,3,4.)
    assert c['distinct_subspaces']==4
    assert np.allclose([a['lower'],a['upper']],[c['lower'],c['upper']])

def test_matches_exhaustive_envelope_on_random_and_redundant_blocks():
    rng=np.random.default_rng(12)
    for _ in range(40):
        A=rng.normal(size=(9,2));b=rng.normal(size=9);h=rng.normal(size=2)
        base=[rng.normal(size=(9,1)) for _ in range(3)];blocks=base+[2*base[0],base[1]+base[2]]
        a=fault_envelope(A,b,h,blocks,2,4.);c=compressed_envelope(A,b,h,blocks,2,4.)
        assert a['status']==c['status']
        assert np.allclose([a['lower'],a['upper']],[c['lower'],c['upper']],atol=1e-8)

def test_unbounded_and_inconsistent_cases():
    A=np.eye(2);blocks=[np.array([[1.],[0.]])];h=np.array([1.,0.])
    assert compressed_envelope(A,np.zeros(2),h,blocks,1,0.)['status']=='unbounded'
    assert compressed_envelope(np.zeros((2,1)),np.ones(2),np.ones(1),[],0,0.)['status']=='inconsistent'
