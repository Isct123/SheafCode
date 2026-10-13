                                                                       
                                                      

                                                                   
                                                                       
                                                                        
                                                                         
                                                                         
                    

                                                    
                                                      
                                                              
                                                                             
   
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


                                                                             
                          
                                                                             

def _inv_sqrt(S: np.ndarray, ridge: float) -> np.ndarray:
    S = S + ridge * np.eye(S.shape[0])
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-12, None)
    return (V / np.sqrt(w)) @ V.T


def cca_maps(Zu: np.ndarray, Zv: np.ndarray, r: int, ridge: float = 1e-3):
                                                               

                                                                       
                                                                        
                                                                     
       
    T = Zu.shape[0]
    Zu = Zu - Zu.mean(axis=0)
    Zv = Zv - Zv.mean(axis=0)
    Suu = Zu.T @ Zu / (T - 1)
    Svv = Zv.T @ Zv / (T - 1)
    Suv = Zu.T @ Zv / (T - 1)
    Wu = _inv_sqrt(Suu, ridge)
    Wv = _inv_sqrt(Svv, ridge)
    U, s, Vt = np.linalg.svd(Wu @ Suv @ Wv)
    r = min(r, len(s))
    Au = (Wu @ U[:, :r]).T                                             
    Av = (Wv @ Vt[:r, :].T).T
    rho = np.clip(s[:r], -1, 1)
    return Au, Av, rho


                                                                             
                  
                                                                             

@dataclass
class Sheaf:
    n: int
    p: int
    edges: list                                      
    A: list                                                                     
    rho: list                                                         
    row_weights: list                                                     
    delta: np.ndarray = field(default=None, repr=False)
    edge_rows: list = field(default_factory=list, repr=False)                       

    def __post_init__(self):
        self.delta, self.edge_rows = self.assemble()

    def assemble(self):
        blocks, rows, start = [], [], 0
        for (u, v), (Au, Av), w in zip(self.edges, self.A, self.row_weights):
            r = Au.shape[0]
            B = np.zeros((r, self.n * self.p))
            sw = np.sqrt(w)[:, None]
            B[:, u * self.p:(u + 1) * self.p] = sw * Au
            B[:, v * self.p:(v + 1) * self.p] = -sw * Av
            blocks.append(B)
            rows.append(slice(start, start + r))
            start += r
        return np.vstack(blocks), rows

    @property
    def laplacian(self):
        return self.delta.T @ self.delta

    def cols(self, stations):
        return np.concatenate([np.arange(v * self.p, (v + 1) * self.p) for v in stations]) if len(stations) else np.array([], int)

    def kernel_dimension(self, tol=1e-8):
        s = np.linalg.svd(self.delta, compute_uv=False)
        return int(self.n * self.p - np.sum(s > tol * s.max()))

                                                                             
    def certificate(self, erased, tol=1e-8):
                                                                              
        dE = self.delta[:, self.cols(erased)]
        if dE.shape[1] == 0:
            return dict(rank=0, needed=0, identifiable=True, sigma_min=np.inf, cond=1.0)
        s = np.linalg.svd(dE, compute_uv=False)
        needed = dE.shape[1]
        rank = int(np.sum(s > tol * max(s.max(), 1e-300)))
        smin = float(s[needed - 1]) if len(s) >= needed else 0.0
        return dict(rank=rank, needed=needed, identifiable=(rank == needed),
                    sigma_min=smin, cond=float(s[0] / smin) if smin > 0 else np.inf)

                                                                             
    def reconstruct(self, Zsnap: np.ndarray, erased, rcond=1e-10):
                                                                            
                                                                            
                                                           
        erased = list(erased)
        observed = [v for v in range(self.n) if v not in erased]
        dO = self.delta[:, self.cols(observed)]
        dE = self.delta[:, self.cols(erased)]
        xO = Zsnap[observed].reshape(-1)
        rhs = -dO @ xO
        z, *_ = np.linalg.lstsq(dE, rhs, rcond=rcond)
        resid = np.linalg.norm(dE @ z - rhs)
        return z.reshape(len(erased), self.p), resid

    def reconstruct_batch(self, Z: np.ndarray, erased, ridge: float = 0.0, rcond=1e-10):
                                             

                                                                        
                                                                 
                                                                        
                                                                          
                                                                       
           
        erased = list(erased)
        observed = [v for v in range(self.n) if v not in erased]
        dO = self.delta[:, self.cols(observed)]
        dE = self.delta[:, self.cols(erased)]
        XO = Z[:, observed, :].reshape(Z.shape[0], -1)
        rhs = -(XO @ dO.T)                                 
        if ridge > 0:
            G = dE.T @ dE + ridge * np.eye(dE.shape[1])
            solver = np.linalg.solve(G, dE.T)                 
        else:
            solver = np.linalg.pinv(dE, rcond=rcond)
        Zhat = rhs @ solver.T
        return Zhat.reshape(Z.shape[0], len(erased), self.p)

                                                                            
    def certificate_cols(self, cols, tol=1e-8):
        cols = np.asarray(cols, int)
        dE = self.delta[:, cols]
        s = np.linalg.svd(dE, compute_uv=False)
        needed = len(cols)
        rank = int(np.sum(s > tol * max(s.max(), 1e-300)))
        smin = float(s[needed - 1]) if len(s) >= needed else 0.0
        return dict(rank=rank, needed=needed, identifiable=(rank == needed), sigma_min=smin,
                    cond=float(s[0] / smin) if smin > 0 else np.inf)

    def reconstruct_cols(self, Z: np.ndarray, cols, ridge: float = 0.0, P: np.ndarray = None, rcond=1e-10):
                                                                           
                                                                   

                                                                          
                                                               
                                                                        
                                                                  
                                                                      
                                        
                                                                                      
           
        cols = np.asarray(cols, int)
        allc = np.arange(self.n * self.p)
        obs = np.setdiff1d(allc, cols)
        dO = self.delta[:, obs]
        dE = self.delta[:, cols]
        XO = Z.reshape(Z.shape[0], -1)[:, obs]
        if ridge > 0:
            if P is None:
                P = np.eye(self.n * self.p)
            G = dE.T @ dE + ridge * P[np.ix_(cols, cols)]
            H = dE.T @ dO + ridge * P[np.ix_(cols, obs)]
            solver = np.linalg.solve(G, H)                        
            return -(XO @ solver.T)
        rhs = -(XO @ dO.T)
        return rhs @ np.linalg.pinv(dE, rcond=rcond).T

    def syndrome(self, Zsnap: np.ndarray):
                                                                      
                                                                    
                                                        
        s = self.delta @ Zsnap.reshape(-1)
        score = np.zeros(self.n)
        count = np.zeros(self.n)
        for (u, v), sl in zip(self.edges, self.edge_rows):
            e = float(np.sum(s[sl] ** 2))
            score[u] += e
            score[v] += e
            count[u] += sl.stop - sl.start
            count[v] += sl.stop - sl.start
        return s, score / np.maximum(count, 1)


def learn_sheaf(Ztrain: np.ndarray, edges, r: int, weighting: str = "invvar", ridge: float = 1e-3) -> Sheaf:
                                                                    
                                                              
                      
                   
                                                                  
                                                                   
                                                                         
                                                             
                                                                          
                                                                         
    T, n, p = Ztrain.shape
    A, rhos, W = [], [], []
    for (u, v) in edges:
        Au, Av, rho = cca_maps(Ztrain[:, u, :], Ztrain[:, v, :], r, ridge)
        A.append((Au, Av))
        rhos.append(rho)
        rho_c = np.clip(rho, 1e-3, 1 - 1e-3)
        if weighting == "invvar":
            W.append(1.0 / (2.0 * (1.0 - rho_c)))
        elif weighting == "reg":
            W.append(rho_c / (1.0 - rho_c))
        else:
            W.append(np.ones_like(rho))
    return Sheaf(n, p, list(edges), A, rhos, W)


def learn_scalar_sheaves(Ztrain: np.ndarray, edges, weighting="invvar", ridge=1e-3) -> list:
                                                                  
                                                                           
                                                         
    T, n, p = Ztrain.shape
    return [learn_sheaf(Ztrain[:, :, [q]], edges, 1, weighting, ridge) for q in range(p)]
