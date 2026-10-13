import numpy as np
import pytest
from theory import observed_margin,mismatch_budget,minimum_reveals

def test_missing_rows_destroy_uniform_fault_identifiability():
    A=np.eye(4)
    assert observed_margin(A,1,1)['sigma']==1
    assert observed_margin(A[:-1],1,1)['sigma']==0

def test_uniform_fault_stability_bound():
    rng=np.random.default_rng(7);A=rng.normal(size=(12,6))
    sigma=observed_margin(A,2,1)['sigma']
    for _ in range(100):
        e=np.zeros(6);f=np.zeros(6)
        e[:2]=rng.normal(size=2);f[2:4]=rng.normal(size=2)
        assert np.linalg.norm(A@(e-f))+1e-10>=sigma*np.linalg.norm(e-f)

def test_mismatch_budget_contains_perturbed_residual():
    rng=np.random.default_rng(13)
    for _ in range(100):
        A=rng.normal(size=(7,4));E=.03*rng.normal(size=(7,4));z=rng.normal(size=4)
        b=-(A+E)@z
        budget=mismatch_budget(0,np.linalg.norm(E,2),np.linalg.norm(z))
        assert np.linalg.norm(A@z+b)<=budget+1e-10
    with pytest.raises(ValueError):mismatch_budget(0,-1,2)

def test_minimum_reveals():
    A=np.array([[1.,1.,0.]])
    assert minimum_reveals(A,np.array([1.,1.,0.]))['cardinality']==0
    assert minimum_reveals(A,np.array([1.,0.,0.]))['cardinality']==1
    assert minimum_reveals(A,np.array([0.,0.,1.]))['coordinates']==[2]
