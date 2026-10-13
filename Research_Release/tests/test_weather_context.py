                                                                           
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np

def test_context_identity_and_bound_under_partial_masks():
    rng=np.random.default_rng(734)
    for _ in range(100):
        A=rng.normal(size=(15,9));B=rng.normal(size=(15,9));C=rng.normal(size=(9,9))
        L0=A.T@A;L1=B.T@B;P=C.T@C+.2*np.eye(9);lam=.3
        a,b=rng.uniform(size=2);Qa=(1-a)*L0+a*L1+lam*P;Qb=(1-b)*L0+b*L1+lam*P
        U=np.sort(rng.choice(9,rng.integers(1,9),replace=False));O=np.setdiff1d(np.arange(9),U)
        xa=rng.normal(size=9);xb=xa.copy()
        xa[U]=-np.linalg.solve(Qa[np.ix_(U,U)],Qa[np.ix_(U,O)]@xa[O])
        xb[U]=-np.linalg.solve(Qb[np.ix_(U,U)],Qb[np.ix_(U,O)]@xb[O])
        force=(a-b)*((L0-L1)@xb)[U]
        delta=xa[U]-xb[U]
        np.testing.assert_allclose(delta,np.linalg.solve(Qa[np.ix_(U,U)],force),atol=1e-10)
        assert np.linalg.norm(delta)<=np.linalg.norm(force)/(lam*np.linalg.eigvalsh(P[np.ix_(U,U)])[0])+1e-10
