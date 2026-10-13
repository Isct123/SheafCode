                                                                               
import numpy as np

def weighted_gains(delta_unknown,prior,lam):
    A=np.asarray(delta_unknown,float);P=np.asarray(prior,float)
    if A.ndim!=2 or P.shape!=(A.shape[1],A.shape[1]) or not np.allclose(P,P.T):raise ValueError('Invalid dimensions or nonsymmetric prior')
    if not np.isfinite(A).all() or not np.isfinite(P).all() or not np.isfinite(lam) or lam<=0:raise ValueError('Finite inputs and positive lambda required')
    eig,V=np.linalg.eigh(P)
    if not len(eig) or eig[0]<=0:raise ValueError('Prior must be positive definite')
    invsqrt=(V/np.sqrt(eig))@V.T;T=A@invsqrt;s=np.linalg.svd(T,compute_uv=False)
    smin=float(s[-1]) if len(s)==A.shape[1] else 0.
    gain=float(np.max(s/(s*s+lam))) if len(s) else 0.
    return {'residual_gain_in_prior_metric':gain,'prior_bias_gain':float(lam/(smin*smin+lam)),
        'whitened_sigma_min':smin,'above_decreasing_branch':smin>=np.sqrt(lam),
        'metric':'||P_UU^(1/2) e_U||; bias input ||P_UU^(-1/2)(P x)_U||',
        'scope':'deterministic norm gains, not calibrated real-world error intervals'}
