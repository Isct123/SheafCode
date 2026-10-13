                                                                        
                                                                

                                                                       
                                                                      
                                                           
                                                                       
                                                                      
                                                              
                                                                        
                                                              
                                                                        
                                                   
                                                                         
   
from __future__ import annotations

import numpy as np

from .graph import distance_weights


def _split(n, erased):
    erased = list(erased)
    observed = [v for v in range(n) if v not in erased]
    return observed, erased


def idw(Z, D, erased, power=2.0):
    T, n, p = Z.shape
    O, E = _split(n, erased)
    out = np.zeros((T, len(E), p))
    for i, v in enumerate(E):
        w = 1.0 / np.power(np.maximum(D[v, O], 1e-3), power)
        w = w / w.sum()
        out[:, i, :] = np.tensordot(Z[:, O, :], w, axes=([1], [0]))
    return out


def scalar_laplacian(Z, D, edges, erased, length_scale=None):
                                                                       
                                                             
    T, n, p = Z.shape
    O, E = _split(n, erased)
    w = distance_weights(D, edges, length_scale)
    L = np.zeros((n, n))
    for (u, v), we in zip(edges, w):
        L[u, u] += we; L[v, v] += we; L[u, v] -= we; L[v, u] -= we
    LEE = L[np.ix_(E, E)]
    LEO = L[np.ix_(E, O)]
    out = np.zeros((T, len(E), p))
    pinv = np.linalg.pinv(LEE)
    for q in range(p):
        out[:, :, q] = -(Z[:, O, q] @ LEO.T) @ pinv.T
    return out


class LowRank:
    def __init__(self, Ztrain, k):
        T, n, p = Ztrain.shape
        M = Ztrain.reshape(T, -1)
        self.mu = M.mean(axis=0)
        U, s, Vt = np.linalg.svd(M - self.mu, full_matrices=False)
        self.k = k
        self.V = Vt[:k].T                     
        self.n, self.p = n, p

    def reconstruct(self, Z, erased):
        T = Z.shape[0]
        O, E = _split(self.n, erased)
        colsO = np.concatenate([np.arange(v * self.p, (v + 1) * self.p) for v in O])
        colsE = np.concatenate([np.arange(v * self.p, (v + 1) * self.p) for v in E])
        M = Z.reshape(T, -1) - self.mu
        VO = self.V[colsO]
        C = M[:, colsO] @ np.linalg.pinv(VO).T                              
        out = C @ self.V[colsE].T + self.mu[colsE]
        return out.reshape(T, len(E), self.p)


class Kriging:
                                                            
                                                                          
                                                                     
                                  

    def __init__(self, Ztrain, D):
        T, n, p = Ztrain.shape
        self.D = D
        self.n, self.p = n, p
        self.params = []
        iu = np.triu_indices(n, 1)
        h = D[iu]
        for q in range(p):
            X = Ztrain[:, :, q]
            g = np.array([0.5 * np.mean((X[:, i] - X[:, j]) ** 2) for i, j in zip(*iu)])
            self.params.append(self._fit(h, g))

    @staticmethod
    def _fit(h, g):
        best, bestv = None, np.inf
        for rng_ in np.linspace(0.5, 40, 80):
            f = 1 - np.exp(-h / rng_)
            A = np.vstack([np.ones_like(f), f]).T
            coef, *_ = np.linalg.lstsq(A, g, rcond=None)
            nug, sill = max(coef[0], 1e-4), max(coef[1], 1e-4)
            v = np.sum((A @ np.array([nug, sill]) - g) ** 2)
            if v < bestv:
                bestv, best = v, (nug, sill, rng_)
        return best

    def _cov(self, h, q):
        nug, sill, rng_ = self.params[q]
        return sill * np.exp(-h / rng_) + (nug if np.isscalar(h) and h == 0 else 0) * 0

    def reconstruct(self, Z, erased):
        T = Z.shape[0]
        O, E = _split(self.n, erased)
        out = np.zeros((T, len(E), self.p))
        for q in range(self.p):
            nug, sill, rng_ = self.params[q]
            C = sill * np.exp(-self.D[np.ix_(O, O)] / rng_)
            C[np.diag_indices(len(O))] += nug
            m = len(O)
            K = np.zeros((m + 1, m + 1)); K[:m, :m] = C; K[m, :m] = 1; K[:m, m] = 1
            for i, v in enumerate(E):
                c = np.append(sill * np.exp(-self.D[v, O] / rng_), 1.0)
                lam = np.linalg.solve(K, c)[:m]
                out[:, i, q] = Z[:, O, q] @ lam
        return out


def persistence(Zfull_before, T):
                                                                           
    return np.repeat(Zfull_before[None, :, :], T, axis=0)
