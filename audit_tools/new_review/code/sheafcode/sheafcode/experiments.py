                                                                 
                                                                        
                                                                
from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import baselines as bl
from .data import Dataset, build_dataset, load, select_subset, haversine_km
from .graph import geographic_graph, correlation_graph
from .sheaf import learn_sheaf, learn_scalar_sheaves

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


                                                                             
                                                                             
                                                                             

def station_precision(Ztrain, ridge=1e-3):
                                                                          
                                                        
    T, n, p = Ztrain.shape
    P = np.zeros((n * p, n * p))
    for v in range(n):
        S = np.cov(Ztrain[:, v, :].T) + ridge * np.eye(p)
        P[v * p:(v + 1) * p, v * p:(v + 1) * p] = np.linalg.inv(S)
    return P


class VectorSheafModel:
    name = "vector_sheaf"

    def __init__(self, Ztrain, edges, r, weighting, lam=0.0, prior="identity", ridge=1e-3):
        self.sheaf = learn_sheaf(Ztrain, edges, r, weighting, ridge)
        self.lam = lam
        self.prior = prior
        self.P = station_precision(Ztrain) if prior == "station_cov" else None

    def cols(self, erased):
        return self.sheaf.cols(list(erased))

    def reconstruct(self, Z, erased, lam=None):
        lam = self.lam if lam is None else lam
        if self.P is None:
            return self.sheaf.reconstruct_batch(Z, erased, lam)
        out = self.sheaf.reconstruct_cols(Z, self.cols(erased), lam, self.P)
        return out.reshape(Z.shape[0], len(erased), self.sheaf.p)

    def reconstruct_cols(self, Z, cols, lam=None):
        lam = self.lam if lam is None else lam
        return self.sheaf.reconstruct_cols(Z, cols, lam, self.P)

    def certificate(self, erased):
        return self.sheaf.certificate(erased)


class ScalarSheafModel:
    name = "scalar_sheaf"

    def __init__(self, Ztrain, edges, weighting, lam=0.0, ridge=1e-3):
        self.sheaves = learn_scalar_sheaves(Ztrain, edges, weighting, ridge)
        self.lam = lam

    def reconstruct(self, Z, erased, lam=None):
        lam = self.lam if lam is None else lam
        outs = [sh.reconstruct_batch(Z[:, :, [q]], erased, lam)[:, :, 0] for q, sh in enumerate(self.sheaves)]
        return np.stack(outs, axis=-1)

    def certificate(self, erased):
                                                                                 
        return self.sheaves[0].certificate(erased)


class ScalarLaplacianModel:
    name = "scalar_laplacian"

    def __init__(self, D, edges, length_scale=None):
        self.D, self.edges, self.ls = D, edges, length_scale

    def reconstruct(self, Z, erased):
        return bl.scalar_laplacian(Z, self.D, self.edges, erased, self.ls)


class IDWModel:
    name = "idw"

    def __init__(self, D, power=2.0):
        self.D, self.power = D, power

    def reconstruct(self, Z, erased):
        return bl.idw(Z, self.D, erased, self.power)


class LowRankModel:
    name = "lowrank"

    def __init__(self, Ztrain, k):
        self.m = bl.LowRank(Ztrain, k)

    def reconstruct(self, Z, erased):
        return self.m.reconstruct(Z, erased)


class KrigingModel:
    name = "kriging"

    def __init__(self, Ztrain, D):
        self.m = bl.Kriging(Ztrain, D)

    def reconstruct(self, Z, erased):
        return self.m.reconstruct(Z, erased)


                                                                             
                               
                                                                             

def all_patterns(n, kmax):
    pats = []
    for k in range(1, kmax + 1):
        pats += [list(c) for c in itertools.combinations(range(n), k)]
    return pats


def clustered_patterns(D, sizes=(2, 3)):
                                                                 
                                                    
    n = D.shape[0]
    pats = set()
    for v in range(n):
        order = np.argsort(D[v])
        for s in sizes:
            pats.add(tuple(sorted(int(x) for x in order[:s])))
    return [list(p) for p in sorted(pats)]


def day_blocks(times: pd.DatetimeIndex, rows: np.ndarray, min_hours=12):
                                                                         
    days = times[rows].floor("D")
    out = []
    for d in pd.unique(days):
        idx = rows[days == d]
        if len(idx) >= min_hours:
            out.append((pd.Timestamp(d), idx))
    return out


def evaluate(model, ds: Dataset, Z, patterns, blocks, method_name, cert_model=None):
                                                                    
    rows = []
    for E in patterns:
        cert = cert_model.certificate(E) if cert_model is not None else None
        for day, idx in blocks:
            Zb = Z[idx]
            Zhat = model.reconstruct(Zb, E)
            truth = Zb[:, E, :]
            err = Zhat - truth
            rec = dict(method=method_name, pattern="+".join(ds.stations[v] for v in E), size=len(E),
                       day=day, n_hours=len(idx), rmse_z=float(np.sqrt(np.mean(err ** 2))),
                       mae_z=float(np.mean(np.abs(err))))
                                           
            raw_hat = ds.unstandardise(Zhat, E)
            raw_true = ds.unstandardise(truth, E)
            for q, pol in enumerate(ds.pollutants):
                rec[f"rmse_{pol}"] = float(np.sqrt(np.mean((raw_hat[:, :, q] - raw_true[:, :, q]) ** 2)))
                rec[f"mae_{pol}"] = float(np.mean(np.abs(raw_hat[:, :, q] - raw_true[:, :, q])))
            if cert is not None:
                rec.update(rank=cert["rank"], needed=cert["needed"], identifiable=cert["identifiable"],
                           sigma_min=cert["sigma_min"], cond=cert["cond"])
            rows.append(rec)
    return pd.DataFrame(rows)


                                                                             
               
                                                                             

@dataclass
class Config:
    dataset: str = "b10"
    pollutants: tuple = None                                                                    
    kmax_val: int = 2                                                             
    kmax_test: int = 4
    graph: str = "geo"                               
    ks: tuple = (2, 3, 4)
    rs: tuple = (1, 2, 3)
    weightings: tuple = ("none", "invvar", "reg")
    lams: tuple = (0.0, 0.1, 0.3, 1.0, 3.0)
    priors: tuple = ("identity", "station_cov")
    pca_ks: tuple = (1, 2, 3, 4, 6, 8, 12)
    min_station_coverage: float = 0.5
    min_complete_hours: int = 3000


def prepare(cfg: Config):
    wide = load(cfg.dataset)
    if cfg.pollutants is None:
        sel = select_subset(wide, cfg.min_station_coverage, cfg.min_complete_hours)
    else:
        sel = select_subset(wide, cfg.min_station_coverage, cfg.min_complete_hours, list(cfg.pollutants))
                                                                           
        t = sel.table
        row = t[t["pollutants"].apply(lambda P: list(P) == list(cfg.pollutants))].iloc[0]
        from .data import Selection
        sel = Selection(list(row["stations"]), list(row["pollutants"]), int(row["n_complete_hours"]),
                        row["start"], row["end"], t)
    ds = build_dataset(cfg.dataset, wide, sel)
    return ds, sel


def graph_for(ds: Dataset, kind: str, k: int):
    if kind == "geo":
        edges, D = geographic_graph(ds.coords, k)
        return edges, haversine_km(ds.coords)
    edges, _ = correlation_graph(ds.Z(0), k)
    return edges, haversine_km(ds.coords)


def fit_all(ds: Dataset, cfg: Config, choice: dict):
                                                           
    Ztr = ds.Z(0)
    models = {}
    edges, D = graph_for(ds, cfg.graph, choice["vector_sheaf"]["k"])
    models["vector_sheaf"] = VectorSheafModel(Ztr, edges, choice["vector_sheaf"]["r"], choice["vector_sheaf"]["weighting"],
                                              choice["vector_sheaf"]["lam"], choice["vector_sheaf"].get("prior", "identity"))
    e2, _ = graph_for(ds, cfg.graph, choice["scalar_sheaf"]["k"])
    models["scalar_sheaf"] = ScalarSheafModel(Ztr, e2, choice["scalar_sheaf"]["weighting"], choice["scalar_sheaf"]["lam"])
    e3, _ = graph_for(ds, cfg.graph, choice["scalar_laplacian"]["k"])
    models["scalar_laplacian"] = ScalarLaplacianModel(D, e3)
    models["idw"] = IDWModel(D, choice["idw"]["power"])
    models["lowrank"] = LowRankModel(Ztr, choice["lowrank"]["k"])
    models["kriging"] = KrigingModel(Ztr, D)
    return models


def validate(ds: Dataset, cfg: Config):
                                                                  
                                                                      
    Ztr, Zva = ds.Z(0), ds.Z(1)
    pats = all_patterns(ds.n, cfg.kmax_val)
    blocks = [("all", np.arange(len(Zva)))]
    log = []

    def score(model):
        df = evaluate(model, ds, Zva, pats, blocks, "x")
        return float(df["rmse_z"].mean())

    choice = {}
    best = (np.inf, None)
    for k in cfg.ks:
        edges, D = graph_for(ds, cfg.graph, k)
        for r in cfg.rs:
            if r > ds.p:
                continue
            for w in cfg.weightings:
                for prior in cfg.priors:
                    m = VectorSheafModel(Ztr, edges, r, w, prior=prior)
                    for lam in cfg.lams:
                        if lam == 0 and prior != "identity":
                            continue
                        m.lam = lam
                        s = score(m)
                        log.append(dict(method="vector_sheaf", k=k, r=r, weighting=w, lam=lam, prior=prior, val_rmse_z=s))
                        if s < best[0]:
                            best = (s, dict(k=k, r=r, weighting=w, lam=lam, prior=prior))
    choice["vector_sheaf"] = best[1]
    best = (np.inf, None)
    for k in cfg.ks:
        edges, D = graph_for(ds, cfg.graph, k)
        for w in cfg.weightings:
            m = ScalarSheafModel(Ztr, edges, w)
            for lam in cfg.lams:
                m.lam = lam
                s = score(m)
                log.append(dict(method="scalar_sheaf", k=k, weighting=w, lam=lam, val_rmse_z=s))
                if s < best[0]:
                    best = (s, dict(k=k, weighting=w, lam=lam))
    choice["scalar_sheaf"] = best[1]
    best = (np.inf, None)
    for k in cfg.ks:
        edges, D = graph_for(ds, cfg.graph, k)
        s = score(ScalarLaplacianModel(D, edges))
        log.append(dict(method="scalar_laplacian", k=k, val_rmse_z=s))
        if s < best[0]:
            best = (s, dict(k=k))
    choice["scalar_laplacian"] = best[1]
    _, D = graph_for(ds, "geo", 2)
    best = (np.inf, None)
    for pw in (1.0, 2.0, 3.0):
        s = score(IDWModel(D, pw))
        log.append(dict(method="idw", power=pw, val_rmse_z=s))
        if s < best[0]:
            best = (s, dict(power=pw))
    choice["idw"] = best[1]
    best = (np.inf, None)
    for kk in cfg.pca_ks:
        if kk >= ds.n * ds.p:
            continue
        s = score(LowRankModel(Ztr, kk))
        log.append(dict(method="lowrank", k=kk, val_rmse_z=s))
        if s < best[0]:
            best = (s, dict(k=kk))
    choice["lowrank"] = best[1]
    s = score(KrigingModel(Ztr, D))
    log.append(dict(method="kriging", val_rmse_z=s))
    choice["kriging"] = {}
    return choice, pd.DataFrame(log)


def run_test(ds: Dataset, cfg: Config, models: dict, kmax=None):
                                                                 
    kmax = kmax or cfg.kmax_test
    Zte = ds.Z(2)
    te_rows = ds.rows(2)
    blocks_abs = day_blocks(ds.times, te_rows)
                                                              
    pos = {r: i for i, r in enumerate(te_rows)}
    blocks = [(d, np.array([pos[r] for r in idx])) for d, idx in blocks_abs]
    pats = all_patterns(ds.n, min(kmax, ds.n - 2))
    _, D = graph_for(ds, "geo", 2)
    for cp in clustered_patterns(D):
        if cp not in pats and len(cp) <= ds.n - 2:
            pats.append(cp)
    frames = []
    cert_model = models["vector_sheaf"]
    for name, m in models.items():
        frames.append(evaluate(m, ds, Zte, pats, blocks, name, cert_model))
    return pd.concat(frames, ignore_index=True)


def save_json(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return str(o)
        return str(o)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=conv)
