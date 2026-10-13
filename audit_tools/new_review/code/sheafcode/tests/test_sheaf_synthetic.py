                                                                            
                                                                     
                                                                
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sheafcode.sheaf import Sheaf, cca_maps, learn_sheaf              
from sheafcode.cpcb_index import concentration_to_index, index_to_concentration              

rng = np.random.default_rng(0)


def make_exact_sheaf(n=6, p=3, r=2, edges=None):
    edges = edges or [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (1, 4)]
    A = [(rng.normal(size=(r, p)), rng.normal(size=(r, p))) for _ in edges]
    W = [np.ones(r) for _ in edges]
    return Sheaf(n, p, edges, A, [np.ones(r)] * len(edges), W)


def test_global_sections_reconstruct_exactly():
    sh = make_exact_sheaf()
                        
    U, s, Vt = np.linalg.svd(sh.delta)
    null = Vt[np.sum(s > 1e-10):].T
    assert null.shape[1] == sh.kernel_dimension() > 0
    x = null @ rng.normal(size=null.shape[1])
    Z = x.reshape(sh.n, sh.p)
    for erased in [[0], [2], [0, 3], [1, 4]]:
        cert = sh.certificate(erased)
        zhat, resid = sh.reconstruct(Z, erased)
        if cert["identifiable"]:
            assert np.allclose(zhat, Z[erased], atol=1e-9), (erased, zhat, Z[erased])
            assert resid < 1e-9
        else:
                                                                                    
            assert not np.allclose(zhat, Z[erased], atol=1e-6) or cert["sigma_min"] < 1e-10


def test_rank_certificate_matches_uniqueness():
    sh = make_exact_sheaf(n=5, p=2, r=1, edges=[(0, 1), (1, 2), (2, 3), (3, 4)])                        
    cert = sh.certificate([1, 2])
                                                                                            
    assert not cert["identifiable"] and cert["rank"] <= 3


def test_batch_matches_single():
    sh = make_exact_sheaf()
    Z = rng.normal(size=(7, sh.n, sh.p))
    single = np.stack([sh.reconstruct(Z[t], [1, 3])[0] for t in range(7)])
    batch = sh.reconstruct_batch(Z, [1, 3])
    assert np.allclose(single, batch, atol=1e-8)


def test_cca_recovers_planted_relation():
    T, p = 5000, 3
    Zu = rng.normal(size=(T, p))
    Bu = rng.normal(size=(1, p)); Bv = rng.normal(size=(1, p))
    shared = Zu @ Bu.T
                                                   
    Zv = rng.normal(size=(T, p))
    Zv = Zv - (Zv @ Bv.T) @ Bv / (Bv @ Bv.T) + shared @ Bv / (Bv @ Bv.T) + 0.01 * rng.normal(size=(T, p))
    Au, Av, rho = cca_maps(Zu, Zv, 1)
    assert rho[0] > 0.99
                                                                          
    cu = np.corrcoef((Zu @ Au.T).ravel(), shared.ravel())[0, 1]
    assert abs(cu) > 0.99


def test_learn_sheaf_unit_variance_and_weights():
    T, n, p = 2000, 4, 3
    Z = rng.normal(size=(T, n, p))
    sh = learn_sheaf(Z, [(0, 1), (1, 2), (2, 3)], r=2, weighting="invvar")
    Au, Av = sh.A[0]
    v = (Z[:, 0, :] @ Au.T).var(axis=0, ddof=1)
    assert np.allclose(v, 1.0, atol=0.1)
    assert sh.delta.shape == (6, n * p)


def test_cpcb_index_roundtrip():
    for pol in ["PM2.5", "PM10", "NO2", "O3", "CO"]:
        top = {"PM2.5": 370, "PM10": 500, "NO2": 500, "O3": 700, "CO": 50}[pol]
        c = np.linspace(0.01, top, 500)
        assert np.allclose(index_to_concentration(pol, concentration_to_index(pol, c)), c, atol=1e-9)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
