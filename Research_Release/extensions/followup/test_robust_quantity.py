import numpy as np
from robust_quantity import fault_envelope,uniformly_identifiable

def test_quantity_identifiable_with_nonunique_full_state():
    A=np.array([[1.,0.],[0.,0.]]);B=[np.array([[0.],[1.]])];h=np.array([1.,0.])
    assert uniformly_identifiable(A,h,B,1)['identifiable']
    r=fault_envelope(A,np.array([-4.,9.]),h,B,1,.2)
    assert r['status']=='bounded'
    assert np.allclose([r['lower'],r['upper']],[3.8,4.2])
    assert not uniformly_identifiable(A,np.array([0.,1.]),B,1)['identifiable']

def test_fault_can_destroy_identification():
    A=np.eye(2);B=[np.array([[1.],[0.]])];h=np.array([1.,0.])
    r=fault_envelope(A,np.zeros(2),h,B,1,0.)
    assert r['status']=='unbounded'
    assert not uniformly_identifiable(A,h,B,1)['identifiable']

def test_true_quantity_contained_without_knowing_support():
    rng=np.random.default_rng(31)
    for _ in range(100):
        A=rng.normal(size=(10,3));blocks=[rng.normal(size=(10,2)) for _ in range(3)]
        z=rng.normal(size=3);e=rng.normal(size=2);noise=.01*rng.normal(size=10);h=rng.normal(size=3)
        b=-A@z-blocks[1]@e+noise
        r=fault_envelope(A,b,h,blocks,1,np.linalg.norm(noise))
        assert r['lower']-1e-8<=h@z<=r['upper']+1e-8
                                                                               
        clean=fault_envelope(A,b,h,blocks,0,np.linalg.norm(noise))
        if clean['status']=='bounded':assert r['lower']<=clean['lower']+1e-8 and r['upper']>=clean['upper']-1e-8
