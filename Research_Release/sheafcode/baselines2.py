                                                  
                                                                        
                                                                         
                                              
from __future__ import annotations

import numpy as np

from .graph import distance_weights


def _hide(Zblock, erased):
    Z = Zblock.copy()
    Z[:, list(erased), :] = np.nan
    return Z


class IDW2:
    name = "idw"

    def __init__(self, D, power=1.0):
        self.D, self.power = D, power

    def reconstruct(self, Zblock, erased):
        Z = _hide(Zblock, erased); W, n, p = Z.shape
        out = np.zeros((W, len(erased), p))
        for i, v in enumerate(erased):
            w = 1.0 / np.power(np.maximum(self.D[v], 1e-3), self.power); w[v] = 0
            for q in range(p):
                X = Z[:, :, q]; m = np.isfinite(X)
                num = np.nansum(np.where(m, X, 0) * w, axis=1); den = (m * w).sum(axis=1)
                out[:, i, q] = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
        return out


class ScalarLaplacian2:
    name = "scalar_laplacian"

    def __init__(self, D, edges):
        n = D.shape[0]
        w = distance_weights(D, edges)
        L = np.zeros((n, n))
        for (u, v), we in zip(edges, w):
            L[u, u] += we; L[v, v] += we; L[u, v] -= we; L[v, u] -= we
        self.L = L

    def reconstruct(self, Zblock, erased):
        Z = _hide(Zblock, erased); W, n, p = Z.shape
        out = np.zeros((W, len(erased), p))
        for t in range(W):
            for q in range(p):
                x = Z[t, :, q]; U = np.where(~np.isfinite(x))[0]; O = np.where(np.isfinite(x))[0]
                if len(O) == 0:
                    continue
                LUU = self.L[np.ix_(U, U)] + 1e-9 * np.eye(len(U)); LUO = self.L[np.ix_(U, O)]
                z = -np.linalg.solve(LUU, LUO @ x[O])
                pos = {c: i for i, c in enumerate(U)}
                out[t, :, q] = z[[pos[v] for v in erased]]
        return out


class Kriging2:
    name = "kriging"

    def __init__(self, Ztrain, D):
        T, n, p = Ztrain.shape
        self.D, self.n, self.p = D, n, p
        iu = np.triu_indices(n, 1); h = D[iu]; self.params = []
        for q in range(p):
            X = Ztrain[:, :, q]
            g = np.array([0.5 * np.nanmean((X[:, i] - X[:, j]) ** 2) for i, j in zip(*iu)])
            ok = np.isfinite(g); self.params.append(self._fit(h[ok], g[ok]))

    @staticmethod
    def _fit(h, g):
        best, bestv = (0.1, 1.0, 10.0), np.inf
        for rng_ in np.linspace(0.5, 60, 100):
            f = 1 - np.exp(-h / rng_); A = np.vstack([np.ones_like(f), f]).T
            coef, *_ = np.linalg.lstsq(A, g, rcond=None)
            nug, sill = max(coef[0], 1e-4), max(coef[1], 1e-4)
            v = np.sum((A @ np.array([nug, sill]) - g) ** 2)
            if v < bestv:
                bestv, best = v, (nug, sill, rng_)
        return best

    def reconstruct(self, Zblock, erased):
        Z = _hide(Zblock, erased); W, n, p = Z.shape
        out = np.zeros((W, len(erased), p))
        for q in range(p):
            nug, sill, rng_ = self.params[q]
            C = sill * np.exp(-self.D / rng_)
            keys = {}
            for t in range(W):
                keys.setdefault(np.isfinite(Z[t, :, q]).tobytes(), []).append(t)
            for key, ts in keys.items():
                O = np.where(np.isfinite(Z[ts[0], :, q]))[0]
                if len(O) == 0:
                    continue
                m = len(O); K = np.zeros((m + 1, m + 1)); K[:m, :m] = C[np.ix_(O, O)] + nug * np.eye(m); K[m, :m] = 1; K[:m, m] = 1
                for i, v in enumerate(erased):
                    c = np.append(C[v, O], 1.0)
                    try:
                        lam = np.linalg.solve(K, c)[:m]
                    except np.linalg.LinAlgError:
                        lam = np.ones(m) / m
                    out[ts, i, q] = Z[np.ix_(ts, O)][:, :, q] @ lam
        return out


class LowRank2:
                                                                          
                                                                         
    name = "lowrank"

    def __init__(self, Ztrain, k, iters=60):
        T, n, p = Ztrain.shape
        M = Ztrain.reshape(T, -1); mask = np.isfinite(M)
        self.mu = np.nanmean(M, axis=0)
        X = np.where(mask, M - self.mu, 0.0)
        for _ in range(iters):
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            L = (U[:, :k] * s[:k]) @ Vt[:k]
            X = np.where(mask, M - self.mu, L)
        self.V = Vt[:k].T; self.n, self.p, self.k = n, p, k

    def reconstruct(self, Zblock, erased):
        Z = _hide(Zblock, erased); W, n, p = Z.shape
        M = Z.reshape(W, -1) - self.mu
        ecols = np.concatenate([np.arange(v * p, (v + 1) * p) for v in erased])
        out = np.zeros((W, len(ecols)))
        keys = {}
        for t in range(W):
            keys.setdefault(np.isfinite(M[t]).tobytes(), []).append(t)
        for key, ts in keys.items():
            O = np.where(np.isfinite(M[ts[0]]))[0]
            C = M[np.ix_(ts, O)] @ np.linalg.pinv(self.V[O]).T
            out[ts] = C @ self.V[ecols].T + self.mu[ecols]
        return out.reshape(W, len(erased), p)


class GaussianConditional:
                                                                          
                                                                        
                                                           
    name = "gaussian_cond"

    def __init__(self, Ztrain, shrink=0.05):
        T, n, p = Ztrain.shape
        M = Ztrain.reshape(T, -1)
        self.mu = np.nanmean(M, axis=0)
        C = np.ma.cov(np.ma.masked_invalid(M), rowvar=False).filled(0.0)
        C = 0.5 * (C + C.T)
        w, V = np.linalg.eigh(C); w = np.clip(w, 1e-4, None)
        C = (V * w) @ V.T
        self.S = (1 - shrink) * C + shrink * np.diag(np.diag(C))
        self.Lam = np.linalg.inv(self.S)                                                                                 
        self.n, self.p = n, p

    def conditional_mean(self, M):
                                                                             
                                                                              
                                                                                
        W = M.shape[0]
        out = M.copy()
        keys = {}
        for t in range(W):
            keys.setdefault(np.isfinite(M[t]).tobytes(), []).append(t)
        for key, ts in keys.items():
            fin = np.isfinite(M[ts[0]]); O = np.where(fin)[0]; Mi = np.where(~fin)[0]
            if len(Mi) == 0:
                continue
            B = -np.linalg.solve(self.Lam[np.ix_(Mi, Mi)], self.Lam[np.ix_(Mi, O)])               
            out[np.ix_(ts, Mi)] = M[np.ix_(ts, O)] @ B.T
        return out

    def reconstruct(self, Zblock, erased):
        Z = _hide(Zblock, erased); W, n, p = Z.shape
        M = Z.reshape(W, -1) - self.mu
        ecols = np.concatenate([np.arange(v * p, (v + 1) * p) for v in erased])
        full = self.conditional_mean(M) + self.mu
        return full[:, ecols].reshape(W, len(erased), p)


class Persistence2:
                                                                     
    name = "persistence"

    def reconstruct_with_history(self, last, W):
        return np.repeat(last[None, :, :], W, axis=0)
