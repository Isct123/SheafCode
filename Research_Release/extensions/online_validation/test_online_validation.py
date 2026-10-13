import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from reference import QuadraticDecoder,FullVAR,graph_correlation,seasonal
from exact_math import rank,nullvector,mv
from uncertainty_shift import quantile

def fixture():
    rng=np.random.default_rng(100);training=rng.normal(size=(400,3,2))
    training[:,1]=.6*training[:,0]+.4*training[:,1]
    return training,rng.normal(size=(48,3,2))

def test_independent_quadratic_matches_first_order_equation():
    training,Z=fixture();model=QuadraticDecoder(training,[(0,1),(1,2)])
    Z[24:,1]=np.nan;Z[20,0,1]=np.nan;pred=model.solve(Z)
    U=np.flatnonzero(~np.isfinite(Z.ravel()));O=np.flatnonzero(np.isfinite(Z.ravel()));Q=model.precision(48)
    residual=Q[U][:,U]@pred.ravel()[U]+Q[U][:,O]@Z.ravel()[O]
    assert np.max(abs(residual))<1e-9
    assert np.array_equal(pred.ravel()[O],Z.ravel()[O])

def test_causal_prediction_ignores_hidden_target_truth_and_future():
    training,Z=fixture();model=QuadraticDecoder(training,[(0,1),(1,2)])
    a=model.causal(Z,1);changed=Z.copy();changed[24:,1]=100000
    assert np.allclose(a,model.causal(changed,1))
    changed[36:,0]=100000
    assert np.allclose(a[:12],model.causal(changed,1)[:12])

def test_reference_kalman_causality_and_observation_preservation():
    training,Z=fixture();var=FullVAR(training.reshape(400,-1),3,2)
    Z[24:,1]=np.nan;a=var.filter(Z);changed=Z.copy();changed[36:,0]=100000
    assert np.allclose(a[:36],var.filter(changed)[:36])
    good=np.isfinite(Z);assert np.array_equal(a[good],Z[good])

def test_correlation_graph_is_defined_using_supplied_training_only():
    training,_=fixture();edges=graph_correlation(training,k=1)
    assert (0,1) in edges
    assert all(u<v for u,v in edges)

def test_exact_rank_kernel_with_rational_dependence():
    M=[[1,2,3],[2,4,6]];assert rank(M)==1
    vector=nullvector(M);assert mv(M,vector)==[0,0]

def test_small_sample_conformal_boundary_is_infinite():
    assert np.isinf(quantile([1],alpha=.1))
    assert quantile(range(20),alpha=.1)==18

def test_seasonal_never_uses_target_inside_outage():
    _,Z=fixture();a=seasonal(Z,1);Z[24:,1]=1e9
    assert np.array_equal(a,seasonal(Z,1))
