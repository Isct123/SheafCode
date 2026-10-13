                                                                          
                                                                         
                                                               
             

                                                                      
                                  
                                    
                                                                         
                                                                         
                                                                         
                                                                          
                                                                           

                                                             
                                           
                                         
                                                                     
                                                           
                                                                                     
                                                                         
                                                                 
                                                                                                
                                                                        
                                                                          
                                         
   
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .sheaf import cca_maps


def _weights(rho, weighting):
    rho_c = np.clip(rho, 1e-3, 1 - 1e-3)
    if weighting == "invvar":
        return 1.0 / (2.0 * (1.0 - rho_c))
    if weighting == "reg":
        return rho_c / (1.0 - rho_c)
    return np.ones_like(rho)


class MaskedSheaf:
    def __init__(self, Ztrain, edges, r, weighting="reg", temporal=False, r_t=None, ridge=1e-3, min_pairs=200, t_scale=1.0, lags=(1, 24), stalk="cca"):
        T, n, p = Ztrain.shape
        self.n, self.p, self.edges = n, p, list(edges)
        self.weighting, self.temporal = weighting, temporal
        full = np.isfinite(Ztrain).all(axis=2)                                          
                           
        A, rhos, W = [], [], []
        self.stalk = stalk
        for (u, v) in self.edges:
            ok = full[:, u] & full[:, v]
            if ok.sum() < min_pairs:
                Au, Av, rho = np.eye(p)[:r], np.eye(p)[:r], np.full(r, 0.3)
            else:
                Au, Av, rho = cca_maps(Ztrain[ok, u, :], Ztrain[ok, v, :], r, ridge)
            if stalk == "cca+id":
                                                                                           
                rq = np.array([np.corrcoef(Ztrain[ok, u, q], Ztrain[ok, v, q])[0, 1] if ok.sum() >= min_pairs else 0.3 for q in range(p)])
                rq = np.nan_to_num(rq, nan=0.3)
                Au = np.vstack([Au, np.eye(p)]); Av = np.vstack([Av, np.eye(p)]); rho = np.concatenate([rho, np.abs(rq)])
            A.append((Au, Av)); rhos.append(rho); W.append(_weights(rho, weighting))
        self.A, self.rho, self.W = A, rhos, W
        blocks, self.edge_rows, start = [], [], 0
        for (u, v), (Au, Av), w in zip(self.edges, A, W):
            rr = Au.shape[0]
            B = np.zeros((rr, n * p)); sw = np.sqrt(w)[:, None]
            B[:, u * p:(u + 1) * p] = sw * Au; B[:, v * p:(v + 1) * p] = -sw * Av
            blocks.append(B); self.edge_rows.append(slice(start, start + rr)); start += rr
        self.delta = np.vstack(blocks) if blocks else np.zeros((0, n * p))
        self.L = self.delta.T @ self.delta                                                               
                                                                                   
        P = np.zeros((n * p, n * p))
        for v in range(n):
            ok = full[:, v]
            S = np.cov(Ztrain[ok, v, :].T) + ridge * np.eye(p) if ok.sum() > p + 2 else np.eye(p)
            P[v * p:(v + 1) * p, v * p:(v + 1) * p] = np.linalg.inv(S)
        self.P = P
                            
        self.Tmaps = None
        self.lags = tuple(lags)
        if temporal:
            r_t = r_t or p
            Tm = {}
            for lag in self.lags:
                Tm[lag] = []
                for v in range(n):
                    ok = full[:-lag, v] & full[lag:, v]
                    if ok.sum() < min_pairs:
                        Bv, Cv, rho = np.eye(p)[:r_t], np.eye(p)[:r_t], np.full(r_t, 0.5)
                    else:
                        Bv, Cv, rho = cca_maps(Ztrain[:-lag][ok, v, :], Ztrain[lag:][ok, v, :], r_t, ridge)
                    Tm[lag].append((Bv, Cv, t_scale * _weights(rho, weighting), rho))
            self.Tmaps = Tm

                                                                                
    def cols(self, stations):
        p = self.p
        return np.concatenate([np.arange(v * p, (v + 1) * p) for v in stations]) if len(stations) else np.zeros(0, int)

    def kernel_dimension(self, tol=1e-8):
        s = np.linalg.svd(self.delta, compute_uv=False)
        return int(self.n * self.p - np.sum(s > tol * s.max()))

                                                                               
    def certificate(self, erased, lam=1.0, Ztrain=None, tol=1e-8):
        cols = self.cols(erased)
        dE = self.delta[:, cols]
        s = np.linalg.svd(dE, compute_uv=False)
        needed = len(cols)
        rank = int(np.sum(s > tol * max(s.max(), 1e-300)))
        smin = float(s[needed - 1]) if len(s) >= needed else 0.0
        out = dict(rank=rank, needed=needed, identifiable=(rank == needed), sigma_min=smin,
                   cond=float(s[0] / smin) if smin > 0 else np.inf,
                   amp_factor=float(np.max(s / (s ** 2 + lam))) if lam > 0 else (1 / smin if smin > 0 else np.inf))
        if Ztrain is not None:
            out["pred_rmse"] = self.predicted_rmse(erased, lam, Ztrain)
        return out

    def predicted_rmse(self, erased, lam, Ztrain):
                                                                        
                                                                              
        erased = list(erased)
        cols = self.cols(erased)
        full = np.isfinite(Ztrain).all(axis=(1, 2))
        if full.sum() >= 200:
            X = Ztrain[full].reshape(full.sum(), -1)
        else:
                                                                            
                                                                
            if getattr(self, "_train_filled", None) is None or getattr(self, "_train_filled_src", None) != Ztrain.shape[0]:
                sub = Ztrain[::max(1, Ztrain.shape[0] // 3000)]
                self._train_filled = np.vstack([self.reconstruct_mask(sub[i:i + 200], np.zeros((min(200, len(sub) - i), self.n), bool), lam)
                                                for i in range(0, len(sub), 200)]).reshape(-1, self.n * self.p)
                self._train_filled_src = Ztrain.shape[0]
            X = self._train_filled
        if len(X) < 10:
            return np.nan
        dE = self.delta[:, cols]
        G = dE.T @ dE + lam * self.P[np.ix_(cols, cols)]
        R = X @ self.delta.T                                        
        PX = X @ self.P[:, cols]                                         
        E = -np.linalg.solve(G, (R @ dE + lam * PX).T).T                      
        return float(np.sqrt(np.mean(E ** 2)))

    def c2star(self, erased, lam, prior="station_cov"):
        cols = self.cols(erased)
        P = self.P if prior == "station_cov" else np.eye(self.n * self.p)
        dU = self.delta[:, cols]
        G = dU.T @ dU + lam * P[np.ix_(cols, cols)]
        M = np.linalg.solve(G, dU.T)
        return float(np.linalg.svd(M, compute_uv=False)[0]), float(1.0 / np.linalg.eigvalsh(G)[0])

    def general_bound(self, erased, lam, Zsnap, prior="station_cov"):
                                                                               
                                                                         
        cols = self.cols(erased)
        P = self.P if prior == "station_cov" else np.eye(self.n * self.p)
        c2s, ginv = self.c2star(erased, lam, prior)
        x = np.nan_to_num(Zsnap.reshape(-1))
        r = self.delta @ x
        px = (P @ x)[cols]
        return float(c2s * np.linalg.norm(r) + lam * ginv * np.linalg.norm(px))

    def predicted_rmse_empirical(self, erased, lam, Ztrain, max_rows=1500, prior="station_cov"):
                                                                        
                                                                           
        erased = list(erased)
        obs = np.isfinite(Ztrain[:, erased, :]).all(axis=(1, 2))
        rows = np.where(obs)[0]
        if len(rows) == 0:
            return np.nan
        rows = rows[:: max(1, len(rows) // max_rows)]
        sub = Ztrain[rows]
        hide = np.zeros((len(rows), self.n), bool); hide[:, erased] = True
        pred = self.reconstruct_mask(sub, hide, lam, prior)[:, erased, :]
        return float(np.sqrt(np.mean((pred - sub[:, erased, :]) ** 2)))

                                                                                 
    def sensitivities(self, unknown_cols, lam, prior="station_cov"):
                                                                             
                                                                          
                                                         
                                                                          
                                                                             
                                                                              
        N = self.n * self.p
        U = np.asarray(unknown_cols, int)
        O = np.setdiff1d(np.arange(N), U)
        P = self.P if prior == "station_cov" else np.eye(N)
        if len(U) == 0:
            return dict(c2_star=0.0, obs_sensitivity=0.0, cond=1.0, isolated=0, n_unknown=0)
        G = self.L[np.ix_(U, U)] + lam * P[np.ix_(U, U)]
        H = self.L[np.ix_(U, O)] + lam * P[np.ix_(U, O)]
        ev = np.linalg.eigvalsh(G)
        Ginv = np.linalg.inv(G)
        M = Ginv @ self.L[np.ix_(U, U)] @ Ginv                                       
        c2 = float(np.sqrt(max(np.linalg.eigvalsh(M)[-1], 0.0)))
        obs = float(np.linalg.norm(Ginv @ H, 2)) if len(O) else np.inf
        coupled = (np.abs(H) > 1e-12).any(axis=1)
        return dict(c2_star=c2, obs_sensitivity=obs, cond=float(ev[-1] / max(ev[0], 1e-300)), isolated=int((~coupled).sum()), n_unknown=int(len(U)))

    def rank_profile(self, erased, tols=(1e-6, 1e-8, 1e-10, 1e-12)):
                                                                            
        cols = self.cols(erased)
        s = np.linalg.svd(self.delta[:, cols], compute_uv=False)
        smax = max(s.max(), 1e-300)
        return {str(t): int(np.sum(s > t * smax)) for t in tols} | {"needed": len(cols), "sigma_min": float(s[len(cols) - 1]) if len(s) >= len(cols) else 0.0, "sigma_max": float(smax)}

    def bias_and_rmse_empirical(self, erased, lam, Ztrain, max_rows=1500, prior="station_cov"):
                                                                       
                                                                      
                                                               
        erased = list(erased)
        obs = np.isfinite(Ztrain[:, erased, :]).all(axis=(1, 2))
        rows = np.where(obs)[0]
        if len(rows) == 0:
            return np.nan, np.nan
        rows = rows[:: max(1, len(rows) // max_rows)]
        sub = Ztrain[rows]
        hide = np.zeros((len(rows), self.n), bool); hide[:, erased] = True
        err = self.reconstruct_mask(sub, hide, lam, prior)[:, erased, :] - sub[:, erased, :]
        return float(np.sqrt(np.mean(err ** 2))), float(np.linalg.norm(err.mean(axis=0)) / np.sqrt(err.shape[1] * err.shape[2]))

    def deterministic_bound(self, erased, lam, Zsnap):
                                                                                 
        cols = self.cols(erased)
        s = np.linalg.svd(self.delta[:, cols], compute_uv=False)
        r = self.delta @ np.nan_to_num(Zsnap.reshape(-1))
        xE = Zsnap.reshape(-1)[cols]
        return float(np.linalg.norm(r) * np.max(s / (s ** 2 + lam)) + lam * np.linalg.norm(xE) / (s[-1] ** 2 + lam))

    def syndrome(self, Zsnap):
        x = np.nan_to_num(Zsnap.reshape(-1))
        s = self.delta @ x
        score = np.zeros(self.n); count = np.zeros(self.n)
        for (u, v), sl in zip(self.edges, self.edge_rows):
            e = float(np.sum(s[sl] ** 2)); score[u] += e; score[v] += e
            count[u] += sl.stop - sl.start; count[v] += sl.stop - sl.start
        return s, score / np.maximum(count, 1)

                                                                               
    def reconstruct(self, Zblock, erased, lam=1.0, prior="station_cov"):
                                                                              
                                                         
        W, n, p = Zblock.shape
        hide = np.zeros((W, n), bool); hide[:, list(erased)] = True
        full = self.reconstruct_mask(Zblock, hide, lam, prior)
        return full[:, list(erased), :]

    def reconstruct_mask(self, Zblock, hide, lam=1.0, prior="station_cov"):
                                                                             
                                                                               
                                                                   
        W, n, p = Zblock.shape
        P = self.P if prior == "station_cov" else np.eye(n * p)
        Z = Zblock.reshape(W, -1).copy()
        unknown = ~np.isfinite(Z)
        unknown |= np.repeat(hide, p, axis=1)
        Z = np.nan_to_num(Z)
        if not self.temporal:
            keys = {}
            for t in range(W):
                keys.setdefault(unknown[t].tobytes(), []).append(t)
            L = self.L
            for key, ts in keys.items():
                U = np.where(unknown[ts[0]])[0]; O = np.where(~unknown[ts[0]])[0]
                if len(U) == 0:
                    continue
                G = L[np.ix_(U, U)] + lam * P[np.ix_(U, U)]
                H = L[np.ix_(U, O)] + lam * P[np.ix_(U, O)]
                Z[np.ix_(ts, U)] = -(Z[np.ix_(ts, O)] @ np.linalg.solve(G, H).T)
            return Z.reshape(W, n, p)
        N = n * p
        Ds = sp.csr_matrix(self.delta)
        blocks = [sp.kron(sp.eye(W), Ds, format="csr")]
        if W > 1:
            rows, cols, vals = [], [], []
            r0 = 0
            for lag, maps in self.Tmaps.items():
                if lag >= W:
                    continue
                for v, (Bv, Cv, w, rho) in enumerate(maps):
                    rr = Bv.shape[0]; sw = np.sqrt(w)[:, None]
                    SB, SC = sw * Bv, -sw * Cv
                    for t in range(W - lag):
                        for i in range(rr):
                            for j in range(p):
                                rows += [r0 + i, r0 + i]; cols += [t * N + v * p + j, (t + lag) * N + v * p + j]; vals += [SB[i, j], SC[i, j]]
                        r0 += rr
            if r0:
                blocks.append(sp.csr_matrix((vals, (rows, cols)), shape=(r0, W * N)))
        D = sp.vstack(blocks, format="csr")
        Pbig = sp.kron(sp.eye(W), sp.csr_matrix(P), format="csr")
        u = unknown.reshape(-1); U = np.where(u)[0]; O = np.where(~u)[0]
        if len(U) == 0:
            return Z.reshape(W, n, p)
        DU, DO = D[:, U], D[:, O]
        G = (DU.T @ DU + lam * Pbig[U][:, U]).tocsc()
        H = (DU.T @ DO + lam * Pbig[U][:, O]).tocsc()
        rhs = -(H @ Z.reshape(-1)[O])
        z = spla.splu(G).solve(rhs)
        flat = Z.reshape(-1).copy(); flat[U] = z
        return flat.reshape(W, n, p)
