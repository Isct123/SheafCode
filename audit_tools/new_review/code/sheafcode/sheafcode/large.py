                                    

                                                                    
                                                                        
                                                                    
                                                                         
                                                                          
                                                                           
                                                                         
                                                                        
                             
   
from __future__ import annotations

import glob
import itertools
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cpcb_index import index_to_concentration, is_censored
from .data import haversine_km
from .graph import knn_edges
from .sheaf import Sheaf, cca_maps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAWN = os.path.join(ROOT, "data", "raw", "india_cpcb_subindex")
POLS = ["PM2.5", "PM10", "NO2", "O3"]


def _parse_times(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    m = t.isna()
    if m.any():
        t[m] = pd.to_datetime(s[m], format="%A, %d %b %Y, %I:%M %p", errors="coerce")
    return t - pd.Timedelta(hours=1)


def load_national(cities=None, pollutants=POLS, min_cov=0.5, min_hours=4000):
                                                                       
    frames = []
    for f in sorted(glob.glob(os.path.join(RAWN, "20*_air_pollution_data_india.csv.gz"))):
        d = pd.read_csv(f, usecols=["datetime", "name", "cityid", "PM2.5", "PM10", "NO2", "OZONE"])
        if cities is not None:
            d = d[d["cityid"].isin(cities)]
        frames.append(d)
    d = pd.concat(frames, ignore_index=True).rename(columns={"OZONE": "O3"})
    d["t"] = _parse_times(d["datetime"])
    d = d[d["t"].notna()]
    for p in pollutants:
        idx = d[p].to_numpy(float)
        c = index_to_concentration(p, idx)
        c[is_censored(idx)] = np.nan
        d[p] = c
    meta = pd.read_csv(os.path.join(RAWN, "air_data_stations.csv")).set_index("name")
    long = d.groupby(["t", "name"])[pollutants].mean()
    wide = long.unstack("name")                                                   
                                              
    keep = []
    for s in wide.columns.get_level_values(1).unique():
        blk = wide.xs(s, axis=1, level=1)
        act = blk.notna().any(axis=1)
        if act.sum() < min_hours:
            continue
        span = blk.loc[act.idxmax(): act[::-1].idxmax()]
        cov = span.notna().mean()
        if (cov[pollutants] >= min_cov).all() and s in meta.index:
            keep.append(s)
    keep = sorted(keep)
    T = len(wide)
    X = np.full((T, len(keep), len(pollutants)), np.nan)
    for j, s in enumerate(keep):
        for q, p in enumerate(pollutants):
            X[:, j, q] = wide[(p, s)].to_numpy(float)
    sm = meta.loc[keep, ["latitude", "longitude", "cityID"]].rename(columns={"cityID": "city"})
    sm.index.name = "name"
    return X, wide.index, sm


@dataclass
class LargeDataset:
    X: np.ndarray
    times: pd.DatetimeIndex
    meta: pd.DataFrame
    pollutants: list
    split: np.ndarray
    mean: np.ndarray = None
    std: np.ndarray = None

    @property
    def n(self):
        return self.X.shape[1]

    @property
    def p(self):
        return self.X.shape[2]

    def standardise(self):
        tr = self.X[self.split == 0]
        self.mean = np.nanmean(tr, axis=0)
        self.std = np.nanstd(tr, axis=0, ddof=1)
        self.std[~np.isfinite(self.std) | (self.std == 0)] = 1.0
        return self

    def Z(self, part=None):
        Z = (self.X - self.mean) / self.std
        return Z if part is None else Z[self.split == part]

    def rows(self, part):
        return np.where(self.split == part)[0]


def build_large(X, times, meta, pollutants, fractions=(0.6, 0.2, 0.2), min_frac_observed=0.3):
                                                                            
                                                                              
    ok = np.isfinite(X).mean(axis=(1, 2)) >= min_frac_observed
    X, times = X[ok], times[ok]
    T = len(times)
    n_tr = int(round(fractions[0] * T)); n_va = int(round(fractions[1] * T))
    split = np.zeros(T, int); split[n_tr:n_tr + n_va] = 1; split[n_tr + n_va:] = 2
    return LargeDataset(X, times, meta, list(pollutants), split).standardise()


                                                                             
                            
                                                                             

def learn_sheaf_nan(Ztr, edges, r, weighting="reg", min_rows=300, ridge=1e-3):
    T, n, p = Ztr.shape
    A, rhos, W, kept = [], [], [], []
    for (u, v) in edges:
        m = np.isfinite(Ztr[:, u, :]).all(axis=1) & np.isfinite(Ztr[:, v, :]).all(axis=1)
        if m.sum() < min_rows:
            continue
        Au, Av, rho = cca_maps(Ztr[m, u, :], Ztr[m, v, :], r, ridge)
        rho_c = np.clip(rho, 1e-3, 1 - 1e-3)
        if weighting == "invvar":
            w = 1.0 / (2.0 * (1.0 - rho_c))
        elif weighting == "reg":
            w = rho_c / (1.0 - rho_c)
        else:
            w = np.ones_like(rho)
        A.append((Au, Av)); rhos.append(rho); W.append(w); kept.append((u, v))
    return Sheaf(n, p, kept, A, rhos, W)


def station_precision_nan(Ztr, ridge=1e-3):
    T, n, p = Ztr.shape
    P = np.zeros((n * p, n * p))
    for v in range(n):
        m = np.isfinite(Ztr[:, v, :]).all(axis=1)
        S = np.cov(Ztr[m, v, :].T) + ridge * np.eye(p) if m.sum() > p + 2 else np.eye(p)
        P[v * p:(v + 1) * p, v * p:(v + 1) * p] = np.linalg.inv(S)
    return P


class NanReconstructor:
                                                                            
                                                 

    def __init__(self, sheaf: Sheaf, lam: float, P: np.ndarray = None):
        self.sh = sheaf
        self.G = sheaf.delta.T @ sheaf.delta
        self.lam = lam
        self.P = P if P is not None else np.eye(self.G.shape[0])
        self.M = self.G + lam * self.P

    def reconstruct_hour(self, z: np.ndarray, erased_cols: np.ndarray):
                                                                             
                                                                                
        allc = np.arange(len(z))
        obs = np.setdiff1d(allc, erased_cols)
        obs = obs[np.isfinite(z[obs])]
        if len(erased_cols) == 0:
            return np.zeros(0)
        A = self.M[np.ix_(erased_cols, erased_cols)]
        B = self.M[np.ix_(erased_cols, obs)]
        try:
            return -np.linalg.solve(A, B @ z[obs])
        except np.linalg.LinAlgError:
            return -np.linalg.lstsq(A, B @ z[obs], rcond=None)[0]

    def reconstruct_block(self, Zb: np.ndarray, erased_stations):
                                                                             
                                                   
        T, n, p = Zb.shape
        E = list(erased_stations)
        ecols = self.sh.cols(E)
        out = np.full((T, len(E) * p), np.nan)
        flat = Zb.reshape(T, -1)
        for t in range(T):
            z = flat[t]
            nan_cols = np.where(~np.isfinite(z))[0]
            cols = np.union1d(ecols, nan_cols)
            vals = self.reconstruct_hour(z, cols)
            pos = np.searchsorted(cols, ecols)
            out[t] = vals[pos]
        return out.reshape(T, len(E), p)

    def realised_sigma_min(self, z: np.ndarray, erased_cols: np.ndarray):
        nan_cols = np.where(~np.isfinite(z))[0]
        cols = np.union1d(erased_cols, nan_cols)
        dE = self.sh.delta[:, cols]
        s = np.linalg.svd(dE, compute_uv=False)
        return float(s[len(cols) - 1]) if len(s) >= len(cols) else 0.0


class ScalarNan:
                                                         

    def __init__(self, Ztr, edges, weighting, lam):
        self.recs = []
        for q in range(Ztr.shape[2]):
            sh = learn_sheaf_nan(Ztr[:, :, [q]], edges, 1, weighting)
            self.recs.append(NanReconstructor(sh, lam))
        self.p = Ztr.shape[2]

    def reconstruct_block(self, Zb, E):
        outs = [rec.reconstruct_block(Zb[:, :, [q]], E)[:, :, 0] for q, rec in enumerate(self.recs)]
        return np.stack(outs, axis=-1)


def idw_nan(Zb, D, E, power=1.0):
    T, n, p = Zb.shape
    out = np.full((T, len(E), p), np.nan)
    for i, v in enumerate(E):
        w = 1.0 / np.power(np.maximum(D[v], 1e-3), power); w[v] = 0
        for q in range(p):
            X = Zb[:, :, q]
            m = np.isfinite(X); m[:, list(E)] = False
            ww = m * w[None, :]
            num = np.nansum(np.where(m, X, 0) * ww, axis=1); den = ww.sum(axis=1)
            out[:, i, q] = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    return out


class LowRankNan:
    def __init__(self, Ztr, k):
        T, n, p = Ztr.shape
        M = Ztr.reshape(T, -1)
        self.mu = np.nanmean(M, axis=0)
        Mf = np.where(np.isfinite(M), M, self.mu)
        U, s, Vt = np.linalg.svd(Mf - self.mu, full_matrices=False)
        self.V = Vt[:k].T; self.n, self.p = n, p

    def reconstruct_block(self, Zb, E):
        T = Zb.shape[0]
        ecols = np.concatenate([np.arange(v * self.p, (v + 1) * self.p) for v in E])
        out = np.full((T, len(ecols)), np.nan)
        flat = Zb.reshape(T, -1) - self.mu
        for t in range(T):
            obs = np.where(np.isfinite(flat[t]))[0]
            obs = np.setdiff1d(obs, ecols)
            c, *_ = np.linalg.lstsq(self.V[obs], flat[t, obs], rcond=None)
            out[t] = self.V[ecols] @ c + self.mu[ecols]
        return out.reshape(T, len(E), self.p)


def sample_patterns(n, sizes, per_size, rng, D=None, clusters=(3, 5)):
    pats = [[v] for v in range(n)]
    for s in sizes:
        if s <= 1:
            continue
        seen = set()
        tries = 0
        while len(seen) < min(per_size, int(min(1e9, __import__("math").comb(n, s)))) and tries < 50 * per_size:
            E = tuple(sorted(rng.choice(n, s, replace=False).tolist())); seen.add(E); tries += 1
        pats += [list(E) for E in sorted(seen)]
    if D is not None:
        for v in range(n):
            order = np.argsort(D[v])
            for c in clusters:
                pats.append(sorted(int(x) for x in order[:c]))
            
    uniq = []; seen = set()
    for E in pats:
        t = tuple(E)
        if t not in seen:
            seen.add(t); uniq.append(list(E))
    return uniq


def day_blocks_large(times, rows, min_hours=6):
    days = times[rows].floor("D")
    return [(pd.Timestamp(d), rows[days == d]) for d in pd.unique(days) if (days == d).sum() >= min_hours]


def rmse_masked(pred, truth):
    m = np.isfinite(truth) & np.isfinite(pred)
    return float(np.sqrt(np.mean((pred[m] - truth[m]) ** 2))) if m.sum() else np.nan, int(m.sum())
