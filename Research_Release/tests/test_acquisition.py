import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from sheafcode.acquisition import conditional_covariance,reveal_block,trace_reduction,choose_repair

def test_station_reveal_matches_direct_conditioning_and_psd_gain():
    rng=np.random.default_rng(349)
    for _ in range(60):
        A=rng.normal(size=(12,12));Q=A.T@A+.1*np.eye(12)
        U=np.sort(rng.choice(12,8,replace=False));C=conditional_covariance(Q,U)
        R=np.sort(rng.choice(8,3,replace=False));keep=np.setdiff1d(np.arange(8),R)
        after=reveal_block(C,R)
        np.testing.assert_allclose(after[np.ix_(keep,keep)],conditional_covariance(Q,U[keep]),atol=1e-10)
        assert np.linalg.eigvalsh(C-after)[0]>-1e-10
        w=rng.uniform(size=8)
        formula=float(np.trace((w[:,None]*C[:,R])@np.linalg.solve(C[np.ix_(R,R)],C[R,:])))
        assert abs(formula-trace_reduction(C,R,w))<1e-10

def test_one_repair_is_best_model_choice_not_an_empirical_guarantee():
    C=np.array([[1.,0.,0.],[0.,3.,.5],[0.,.5,2.]])
    selected,gains=choose_repair(C,{'a':[0],'b':[1],'c':[2]})
    assert selected=='b' and gains['b']>gains['c']>gains['a']

def test_invalid_weights_duplicate_coordinates_and_singular_precision_rejected():
    with pytest.raises(ValueError):trace_reduction(np.eye(2),[0],[-1,1])
    with pytest.raises(ValueError):conditional_covariance(np.eye(2),[0,0])
    with pytest.raises(np.linalg.LinAlgError):conditional_covariance(np.zeros((2,2)),[0,1])
