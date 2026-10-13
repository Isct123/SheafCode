                                                    

                                                                    
                                                                       
                                                                        
                                                 
                                                                       
                                                                         
                                                           
                                                                        
                                                                        
   
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from sheafcode.masked import MaskedSheaf              
from sheafcode.baselines2 import IDW2, ScalarLaplacian2, Kriging2, LowRank2, GaussianConditional              
from sheafcode.kalman import KalmanVAR              

rng = np.random.default_rng(3)
T, n, p = 3000, 7, 3
coords = rng.uniform(size=(n, 2)) * 20


def make_data():
    base = rng.normal(size=(T, 1, p))
    Z = base + 0.6 * rng.normal(size=(T, n, p)) @ rng.normal(size=(p, p)) / np.sqrt(p)
    for t in range(1, T):
        Z[t] = 0.7 * Z[t - 1] + 0.3 * Z[t]
    Z = (Z - Z.mean(0)) / Z.std(0)
    Zg = Z.copy(); Zg[rng.random(Z.shape) < 0.08] = np.nan                    
    return Z, Zg


Z, Zg = make_data()
Ztr = Zg[:2000]
edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.6]
sheaf = MaskedSheaf(Ztr, edges, 2, "reg")
sheaf_t = MaskedSheaf(Ztr, edges, 2, "reg", temporal=True, t_scale=4.0, lags=(1, 24))
D = np.sqrt(((coords[:, None] - coords[None]) ** 2).sum(-1))
models = {
    "vector_sheaf": lambda W, E: sheaf.reconstruct(W, E, 1.0),
    "st_sheaf": lambda W, E: sheaf_t.reconstruct_mask(W, np.isin(np.arange(n), E)[None].repeat(len(W), 0), 1.0)[:, E, :],
    "idw": lambda W, E: IDW2(D).reconstruct(W, E),
    "scalar_laplacian": lambda W, E: ScalarLaplacian2(D, edges).reconstruct(W, E),
    "kriging": lambda W, E: Kriging2(Ztr, D).reconstruct(W, E),
    "lowrank": lambda W, E: LowRank2(Ztr, 4).reconstruct(W, E),
    "gaussian_cond": lambda W, E: GaussianConditional(Ztr).reconstruct(W, E),
}


def perturb(W, E, scale=50.0):
    W2 = W.copy()
    W2[:, E, :] = np.where(np.isfinite(W2[:, E, :]), W2[:, E, :] + scale * rng.normal(size=W2[:, E, :].shape), np.nan)
    return W2


def test_leakage_station_outage():
    W = Zg[2100:2148]
    for E in ([0], [1, 4], [2, 3, 5]):
        for name, fn in models.items():
            a = fn(W, E); b = fn(perturb(W, E), E)
            assert np.allclose(np.nan_to_num(a), np.nan_to_num(b), atol=1e-8), f"{name} leaks hidden values for E={E}"
    print("leakage (station outage): ok for", list(models))


def test_leakage_single_channel():
    W = Zg[2200:2248]
    v, q = 3, 1
    for prior in ("station_cov", "identity"):
        Wc = W.copy(); Wc[:, v, q] = np.nan
        hide = np.zeros((len(W), n), bool)
        a = sheaf.reconstruct_mask(Wc, hide, 1.0, prior=prior)[:, v, q]
        Wp = W.copy(); Wp[:, v, q] = 1e3                                                    
        Wp[:, v, q] = np.nan
        b = sheaf.reconstruct_mask(Wp, hide, 1.0, prior=prior)[:, v, q]
        assert np.allclose(a, b)
                                                                                                         
        full = sheaf.reconstruct_mask(Wc, hide, 1.0, prior=prior)
        assert np.isfinite(full).all()
    print("leakage (single channel): ok")


def test_leakage_prolonged_and_kalman():
    W = Zg[2300:2300 + 24 + 48].copy()
    v = 2
    hide = np.zeros((len(W), n), bool); hide[24:, v] = True
    a = sheaf_t.reconstruct_mask(W, hide, 1.0)[24:, v, :]
    b = sheaf_t.reconstruct_mask(perturb(W, [v]), hide, 1.0)[24:, v, :]                                                              
    W2 = W.copy(); W2[24:, v, :] = 1e3
    c = sheaf_t.reconstruct_mask(W2, hide, 1.0)[24:, v, :]
    assert np.allclose(a, c), "space-time sheaf leaks hidden values"
    assert not np.allclose(a, b), "history perturbation should change the answer (sanity of the test)"
    kal = KalmanVAR(np.nan_to_num(Ztr[:800]), k_complete=4)
    Wk = W.copy(); Wk[24:, v, :] = np.nan
    ka = kal.smooth(Wk)[24:, v, :]
    Wk2 = W.copy(); Wk2[24:, v, :] = 1e3; Wk2[24:, v, :] = np.nan
    kb = kal.smooth(Wk2)[24:, v, :]
    assert np.allclose(ka, kb), "Kalman leaks"
    print("leakage (prolonged, Kalman): ok")


def test_certificates_data_independent_and_permutation_invariant():
    E = [1, 4]
    c_a = sheaf.certificate(E, 1.0); s_a, g_a = sheaf.c2star(E, 1.0)
                                                                                                     
    c_b = sheaf.certificate(E, 1.0); s_b, g_b = sheaf.c2star(E, 1.0)
    assert c_a["rank"] == c_b["rank"] and np.isclose(c_a["sigma_min"], c_b["sigma_min"]) and np.isclose(s_a, s_b)
                                                         
    perm = rng.permutation(n); inv = np.argsort(perm)
    Ztr_p = Ztr[:, perm, :]
    edges_p = [(int(inv[u]), int(inv[v])) for u, v in edges]
    sheaf_p = MaskedSheaf(Ztr_p, edges_p, 2, "reg")
    E_p = sorted(int(inv[v]) for v in E)
    c_p = sheaf_p.certificate(E_p, 1.0); s_p, g_p = sheaf_p.c2star(E_p, 1.0)
    assert c_p["rank"] == c_a["rank"], "C1 not permutation invariant"
    assert np.isclose(c_p["sigma_min"], c_a["sigma_min"], rtol=1e-6), "C2 not permutation invariant"
    assert np.isclose(s_p, s_a, rtol=1e-6), "C2* not permutation invariant"
    assert sheaf_p.kernel_dimension() == sheaf.kernel_dimension()
                                 
    W = Zg[2500:2524]
    a = sheaf.reconstruct(W, E, 1.0)
    b = sheaf_p.reconstruct(W[:, perm, :], E_p, 1.0)
    order = np.argsort([inv[v] for v in E])
    assert np.allclose(np.nan_to_num(a[:, order, :]), np.nan_to_num(b), atol=1e-8), "reconstruction not permutation equivariant"
    print("invariance (permutation, data-independence): ok")


def test_nothing_hidden_and_c3():
    W = Zg[2600:2624]
    hide = np.zeros((len(W), n), bool)
    full = sheaf.reconstruct_mask(W, hide, 1.0)
    m = np.isfinite(W)
    assert np.allclose(full[m], W[m]), "observed cells must be returned unchanged"
    assert np.isfinite(full).all(), "natural gaps must be filled"
    c3 = sheaf.predicted_rmse([0], 1.0, Z[:2000])
    c3e = sheaf.predicted_rmse_empirical([0], 1.0, Ztr, max_rows=300)
    assert np.isfinite(c3) and np.isfinite(c3e) and c3 > 0 and c3e > 0
    print(f"sanity: ok (C3_complete={c3:.3f}, C3_empirical={c3e:.3f})")


if __name__ == "__main__":
    test_leakage_station_outage()
    test_leakage_single_channel()
    test_leakage_prolonged_and_kalman()
    test_certificates_data_independent_and_permutation_invariant()
    test_nothing_hidden_and_c3()
    print("all leakage / invariance tests passed")
