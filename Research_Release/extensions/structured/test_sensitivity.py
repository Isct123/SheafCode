import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pytest
from sensitivity import weighted_gains

def test_learned_prior_metric_matches_direct_operator_and_general_bound():
    rng=np.random.default_rng(313)
    for _ in range(100):
        A=rng.normal(size=(9,4));B=rng.normal(size=(4,4));P=B.T@B+.2*np.eye(4);lam=.7
        gains=weighted_gains(A,P,lam);eig,V=np.linalg.eigh(P);sqrt=(V*np.sqrt(eig))@V.T;invsqrt=(V/np.sqrt(eig))@V.T
        G=A.T@A+lam*P;operator=sqrt@np.linalg.solve(G,A.T)
        assert abs(np.linalg.norm(operator,2)-gains['residual_gain_in_prior_metric'])<1e-10
        r=rng.normal(size=9);px=rng.normal(size=4);error=-np.linalg.solve(G,A.T@r+lam*px)
        bound=gains['residual_gain_in_prior_metric']*np.linalg.norm(r)+gains['prior_bias_gain']*np.linalg.norm(invsqrt@px)
        assert np.linalg.norm(sqrt@error)<=bound+1e-10

def test_more_mismatched_edges_can_increase_actual_error():
    old=np.array([[.1,-.1]]);new=np.array([[.1,-.1],[.5,-.5]])
                                                                    
    errors=[]
    for D in [old,new]:
        A=D[:,:1];b=D[:,1:]
        z=-np.linalg.solve(A.T@A+np.eye(1),A.T@b)
        errors.append(abs(float(z[0,0])))
    assert errors[1]>errors[0]

def test_null_directions_and_invalid_priors():
    g=weighted_gains(np.array([[1.,0.]]),np.eye(2),1.)
    assert g['prior_bias_gain']==1. and not g['above_decreasing_branch']
    with pytest.raises(ValueError):weighted_gains(np.eye(2),np.diag([1.,-1.]),1.)
