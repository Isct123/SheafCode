                                                                  

                                                          
                                                                          
                                                                     
                                                                   
                                                                       
                                                                      
                                                               
                                      
from __future__ import annotations

import numpy as np

from .baselines2 import LowRank2


class KalmanVAR:
    name = "kalman_var"

    def __init__(self, Ztrain, ridge=10.0, r_obs=0.05, k_complete=8, seed=0):
        T, n, p = Ztrain.shape
        self.n, self.p, self.d = n, p, n * p
        lr = LowRank2(Ztrain, min(k_complete, n * p // 2))
        M = Ztrain.reshape(T, -1)
        mask = np.isfinite(M)
                                                                           
        X = M.copy()
        C = np.where(mask, M - lr.mu, 0.0) @ lr.V @ lr.V.T + lr.mu
        X[~mask] = C[~mask]
        X = np.nan_to_num(X)                                                                                                              
        X0, X1 = X[:-1], X[1:]
        G = X0.T @ X0 + ridge * np.eye(self.d)
        self.A = np.linalg.solve(G, X0.T @ X1).T                 
        resid = X1 - X0 @ self.A.T
        self.Q = np.diag(np.maximum(resid.var(axis=0), 1e-3))
        self.R = r_obs
        self.mu = np.nan_to_num(lr.mu)

    def filter(self, Zwin):
                                                                         
                                                                      
        return self.smooth(Zwin, causal=True)

    def smooth(self, Zwin, causal=False):
                                                                              
                                                                        
        W = Zwin.shape[0]
        Y = Zwin.reshape(W, -1)
        obs = np.isfinite(Y)
        d = self.d
        x_pred = np.zeros((W, d)); P_pred = np.zeros((W, d, d))
        x_filt = np.zeros((W, d)); P_filt = np.zeros((W, d, d))
        x = np.zeros(d); P = np.eye(d) * 1.0
        for t in range(W):
            if t > 0:
                x = self.A @ x; P = self.A @ P @ self.A.T + self.Q
            x_pred[t], P_pred[t] = x, P
            o = np.where(obs[t])[0]
            if len(o):
                S = P[np.ix_(o, o)] + self.R * np.eye(len(o))
                K = np.linalg.solve(S, P[o, :]).T                    
                x = x + K @ (Y[t, o] - x[o])
                P = P - K @ P[o, :]
            x_filt[t], P_filt[t] = x, P
        xs = x_filt.copy()
        for t in ([] if causal else range(W - 2, -1, -1)):
            J = np.linalg.solve(P_pred[t + 1] + 1e-9 * np.eye(d), self.A @ P_filt[t]).T
            xs[t] = x_filt[t] + J @ (xs[t + 1] - x_pred[t + 1])
        out = np.where(obs, Y, xs)
        return out.reshape(W, self.n, self.p)
