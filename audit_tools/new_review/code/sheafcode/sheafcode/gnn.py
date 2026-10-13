                                                                      

                                                                          
                                                                   
                                                                      
                                                                        
                                                                     
                                                       
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH = True
except Exception:                    
    TORCH = False


def norm_adj(n, edges):
    A = np.zeros((n, n))
    for u, v in edges:
        A[u, v] = A[v, u] = 1.0
    A = A + np.eye(n)
    d = A.sum(1)
    return A / np.sqrt(np.outer(d, d))


if TORCH:
    class GCNImputer(nn.Module):
        def __init__(self, p, hidden=64):
            super().__init__()
            self.l1 = nn.Linear(2 * p, hidden); self.l2 = nn.Linear(hidden, hidden); self.l3 = nn.Linear(hidden, hidden)
            self.head = nn.Sequential(nn.Linear(hidden + 2 * p, hidden), nn.ReLU(), nn.Linear(hidden, p))

        def forward(self, x, A):                                 
            h = torch.relu(A @ self.l1(x))
            h = torch.relu(A @ self.l2(h)) + h
            h = torch.relu(A @ self.l3(h)) + h
            return self.head(torch.cat([h, x], dim=-1))


class GNNModel:
    name = "gnn"

    def __init__(self, Ztrain, Zval, edges, hidden=64, epochs=60, batch=128, lr=2e-3, seed=0):
        assert TORCH
        torch.manual_seed(seed); np.random.seed(seed)
        T, n, p = Ztrain.shape
        self.n, self.p = n, p
        self.A = torch.tensor(norm_adj(n, edges), dtype=torch.float32)
        self.model = GCNImputer(p, hidden)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        Xtr = torch.tensor(np.nan_to_num(Ztrain), dtype=torch.float32)
        Mtr = torch.tensor(np.isfinite(Ztrain), dtype=torch.float32)
        Xva = torch.tensor(np.nan_to_num(Zval), dtype=torch.float32)
        Mva = torch.tensor(np.isfinite(Zval), dtype=torch.float32)
        best, best_state, patience = np.inf, None, 0
        rng = np.random.default_rng(seed)
        for ep in range(epochs):
            self.model.train(); perm = rng.permutation(T)
            for i in range(0, T, batch):
                idx = perm[i:i + batch]; x, m = Xtr[idx], Mtr[idx]
                frac = rng.uniform(0.1, 0.4)
                drop = torch.tensor(rng.random((len(idx), n, 1)) < frac, dtype=torch.float32)
                m_in = m * (1 - drop); x_in = x * m_in
                pred = self.model(torch.cat([x_in, m_in], -1), self.A)
                target_mask = m * drop
                loss = ((pred - x) ** 2 * target_mask).sum() / target_mask.sum().clamp(min=1)
                opt.zero_grad(); loss.backward(); opt.step()
                                                                  
            self.model.eval()
            with torch.no_grad():
                sub = torch.arange(0, len(Xva), max(1, len(Xva) // 400))
                errs = []
                for v in range(n):
                    m_in = Mva[sub].clone(); m_in[:, v, :] = 0
                    pred = self.model(torch.cat([Xva[sub] * m_in, m_in], -1), self.A)
                    tm = Mva[sub][:, v, :]
                    errs.append(float((((pred[:, v, :] - Xva[sub][:, v, :]) ** 2 * tm).sum() / tm.sum().clamp(min=1))))
                val = float(np.sqrt(np.mean(errs)))
            if val < best - 1e-4:
                best, best_state, patience = val, {k: v.clone() for k, v in self.model.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= 8:
                    break
        self.model.load_state_dict(best_state); self.model.eval(); self.val_rmse = best

    def reconstruct(self, Zblock, erased):
        Z = Zblock.copy(); Z[:, list(erased), :] = np.nan
        x = torch.tensor(np.nan_to_num(Z), dtype=torch.float32); m = torch.tensor(np.isfinite(Z), dtype=torch.float32)
        with torch.no_grad():
            pred = self.model(torch.cat([x * m, m], -1), self.A).numpy()
        return pred[:, list(erased), :]
