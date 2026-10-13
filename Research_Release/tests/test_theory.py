                                                          
                                                                 
                                                                     
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from sheafcode.masked import MaskedSheaf              
from sheafcode.sheaf import cca_maps              

rng = np.random.default_rng(1)


def random_data(T=4000, n=8, p=3, ar=0.6):
    base = rng.normal(size=(T, 1, p))
    Z = base + 0.5 * rng.normal(size=(T, n, p)) @ rng.normal(size=(p, p)) / np.sqrt(p)
    for t in range(1, T):
        Z[t] = ar * Z[t - 1] + (1 - ar) * Z[t]
    return (Z - Z.mean(0)) / Z.std(0)


def test_prop2_error_identity_and_bound():
    Z = random_data()
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if rng.random() < 0.5]
    sh = MaskedSheaf(Z[:3000], edges, 3, "reg")
    lam = 1.0
    for E in ([0], [2, 5], [1, 3, 6]):
        cols = sh.cols(E)
        for t in range(3000, 3020):
            x = Z[t].reshape(-1)
            z = sh.reconstruct(Z[t:t + 1], E, lam, prior="identity")[0].reshape(-1)
            e = z - x[cols]
            dE = sh.delta[:, cols]
            G = dE.T @ dE + lam * np.eye(len(cols))
            e_formula = -np.linalg.solve(G, dE.T @ (sh.delta @ x) + lam * x[cols])
            assert np.allclose(e, e_formula, atol=1e-9), "identity (2) fails"
            assert np.linalg.norm(e) <= sh.deterministic_bound(E, lam, Z[t]) + 1e-9, "bound (3) violated"


def test_prop3_monotone_singular_values():
    Z = random_data()
    edges = [(i, i + 1) for i in range(7)]
    sh1 = MaskedSheaf(Z[:3000], edges, 3, "reg")
    sh2 = MaskedSheaf(Z[:3000], edges + [(0, 4), (2, 6), (1, 7)], 3, "reg")
    for E in ([0], [3, 4], [1, 5, 6]):
        s1 = np.linalg.svd(sh1.delta[:, sh1.cols(E)], compute_uv=False)
        s2 = np.linalg.svd(sh2.delta[:, sh2.cols(E)], compute_uv=False)
        assert np.all(s2 + 1e-12 >= s1), "adding rows lowered a singular value"


def test_prop4_predicted_rmse_matches_training_errors():
    Z = random_data()
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if rng.random() < 0.5]
    sh = MaskedSheaf(Z[:3000], edges, 3, "reg")
    E = [2, 5]
    pred = sh.predicted_rmse(E, 1.0, Z[:3000])
    errs = []
    for t in range(0, 3000, 7):
        z = sh.reconstruct(Z[t:t + 1], E, 1.0)[0]
        errs.append(np.mean((z - Z[t, E, :]) ** 2))
    direct = float(np.sqrt(np.mean(errs)))
    assert abs(pred - direct) / direct < 0.03, (pred, direct)


def test_prop5_single_edge_regression():
    T, p = 20000, 3
    Zu = rng.normal(size=(T, p))
    M = rng.normal(size=(p, p))
    Zv = Zu @ M + 0.8 * rng.normal(size=(T, p))
    Zu = (Zu - Zu.mean(0)) / Zu.std(0); Zv = (Zv - Zv.mean(0)) / Zv.std(0)
    Z = np.stack([Zu, Zv], axis=1)
    sh = MaskedSheaf(Z, [(0, 1)], p, "reg", ridge=1e-9)
    x = Z[5]
    z = sh.reconstruct(Z[5:6], [1], 1.0, prior="station_cov")[0, 0, :]
    Suu = np.cov(Zu.T); Svu = np.cov(Zv.T, Zu.T)[:p, p:]
    reg = Svu @ np.linalg.solve(Suu, x[0])
    assert np.allclose(z, reg, atol=2e-2), (z, reg)


def test_prop6_vertex_prior_is_conditional_mean():
    T, p = 20000, 4
    A = rng.normal(size=(p, p)); X = rng.normal(size=(T, p)) @ A
    X = (X - X.mean(0)) / X.std(0)
    Z = X[:, None, :]                                                
    sh = MaskedSheaf(Z, [], 1, "reg")
    S = np.cov(X.T); Pm = np.linalg.inv(S)
    x = X[3]
    Zc = Z[3:4].copy(); Zc[0, 0, [1, 3]] = np.nan
    z = sh.reconstruct_mask(Zc, np.zeros((1, 1), bool), 1.0)[0, 0, [1, 3]]
    O, U = [0, 2], [1, 3]
    cond = S[np.ix_(U, O)] @ np.linalg.solve(S[np.ix_(O, O)], x[O])
    assert np.allclose(z, cond, atol=1e-2), (z, cond)


def test_real_data_bound_and_identity():
    from sheafcode.cohort import select_cohort
    from sheafcode.graph import knn_edges
    from sheafcode.data import haversine_km
    ds = select_cohort("bengaluru")
    Z = ds.Z(); D = haversine_km(ds.coords)
    sh = MaskedSheaf(Z[ds.split == 0], knn_edges(D, 4), ds.p, "reg")
    Zt = Z[ds.split == 2]
    full = np.where(np.isfinite(Zt).all(axis=(1, 2)))[0][:200]
    assert len(full)>0, 'Real-data bound check requires observed test vectors'
    viol = 0
    for t in full:
        for v in range(ds.n):
            z = sh.reconstruct(Zt[t:t + 1], [v], 1.0, prior="identity")[0, 0, :]
            if np.linalg.norm(z - Zt[t, v, :]) > sh.deterministic_bound([v], 1.0, Zt[t]) + 1e-9:
                viol += 1
    assert viol == 0, viol


def test_prop3prime_general_P_bound():
                                                                                             
    Z = random_data()
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if rng.random() < 0.5]
    sh = MaskedSheaf(Z[:3000], edges, 3, "reg")
    viol = 0
    for E in ([0], [2, 5], [1, 3, 6]):
        for t in range(3000, 3100):
            z = sh.reconstruct(Z[t:t + 1], E, 1.0)[0].reshape(-1)
            err = np.linalg.norm(z - Z[t].reshape(-1)[sh.cols(E)])
            if err > sh.general_bound(E, 1.0, Z[t]) + 1e-9:
                viol += 1
    assert viol == 0, viol


def test_prop5prime_resilience_monotone_and_dblock():
                                                                           
                                                                              
    Z = random_data(n=8)
    edges = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)]                          
    sh = MaskedSheaf(Z[:3000], edges, 1, "reg")                                                                          
    for _ in range(50):
        k = int(rng.integers(1, 4)); S = sorted(rng.choice(8, k, replace=False).tolist())
        u = int(rng.choice([v for v in range(8) if v not in S])); S2 = sorted(S + [u])
        assert sh.certificate(S)["sigma_min"] >= sh.certificate(S2)["sigma_min"] - 1e-12
                                                        
    S1, S2 = [1, 2], [5, 6]
    r1 = sh.certificate(S1)["rank"]; r2 = sh.certificate(S2)["rank"]; r12 = sh.certificate(S1 + S2)["rank"]
    assert r12 == r1 + r2
                                                                                                                        
    sh3 = MaskedSheaf(Z[:3000], edges, 3, "reg")
    assert sh3.certificate([1])["identifiable"] and sh3.certificate([2])["identifiable"]
    assert not sh3.certificate([0, 1, 2, 3])["identifiable"], "a whole component is never identifiable"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
