import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from run import completed_training,fit_model

def test_correlated_innovations_and_stability():
    rng=np.random.default_rng(824);Z=rng.normal(size=(600,3,2));Z[:,:,1]=Z[:,:,0]+.1*Z[:,:,1]
    Z[::5,0,0]=np.nan;X=completed_training(Z)
    assert np.isfinite(X).all()
    np.testing.assert_array_equal(X.reshape(Z.shape)[np.isfinite(Z)],Z[np.isfinite(Z)])
    full=fit_model(X,3,2,1.,.05,0.);diag=fit_model(X,3,2,1.,.05,1.)
    assert np.linalg.eigvalsh(full.Q)[0]>0 and np.max(abs(np.linalg.eigvals(full.A)))<=.995+1e-10
    assert abs(full.Q[0,1])>.1 and diag.Q[0,1]==0

def test_causal_filter_prefix_invariance():
    rng=np.random.default_rng(8);X=rng.normal(size=(300,6));model=fit_model(X,3,2,100.,.05,.2)
    Z=rng.normal(size=(24,3,2));Z[12:,0]=np.nan
    changed=Z.copy();changed[18:]+=100
    np.testing.assert_allclose(model.smooth(Z,causal=True)[:18],model.smooth(changed,causal=True)[:18])
