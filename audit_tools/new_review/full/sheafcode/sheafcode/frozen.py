                                                                    
                                                                        
                                                                       
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .india import select_region, ROOT
from .masked import MaskedSheaf
from .graph import knn_edges
from .data import haversine_km


def corr_distance(Ztr):
    T, n, p = Ztr.shape
    C = np.zeros((n, n))
    for q in range(p):
        X = np.ma.masked_invalid(Ztr[:, :, q])
        C += np.abs(np.ma.corrcoef(X, rowvar=False).filled(0.0))
    C /= p
    Dc = 1.0 - C
    np.fill_diagonal(Dc, 0.0)
    return Dc


def rmse_masked(pred, truth):
    m = np.isfinite(truth)
    return float(np.sqrt(np.mean((pred[m] - truth[m]) ** 2))) if m.sum() else np.nan


class Frozen:
    def __init__(self, region, tag=None, temporal=False):
        self.region = region
        self.out = os.path.join(ROOT, "results", tag or f"v3_{region}")
        self.diag = json.load(open(os.path.join(self.out, "diagnostics.json")))
        self.choice = self.diag["choice"]
        c = self.c = self.choice["vector_sheaf"]
        ds = self.ds = select_region(region)
        self.n, self.p = ds.n, ds.p
        self.D = haversine_km(ds.coords)
        ds.set_transform("none")
        self.DC = corr_distance(ds.Z()[ds.split == 0])                                                                              
        ds.set_transform(c["transform"])
        self.Z = ds.Z()
        self.Ztr = self.Z[ds.split == 0]
        self.Dg = self.D if c["graph"] == "geo" else self.DC
        self.edges = knn_edges(self.Dg, c["k"])
        assert [(ds.stations[u], ds.stations[v]) for u, v in self.edges] == [tuple(e) for e in self.diag["edges"]], "edge list differs from the frozen run"
        self.sheaf = self.build(self.edges)
        self.lam = c["lam"]
        if temporal:
            self.sheaf_t = self.build(self.edges, temporal=True)

    def build(self, edges, temporal=False, Ztr=None):
        c = self.c
        kw = dict(temporal=True, t_scale=self.choice["st_sheaf"]["t_scale"]) if temporal else {}
        return MaskedSheaf(self.Ztr if Ztr is None else Ztr, edges, c["r"], c["weighting"], stalk=c["stalk"], **kw)

    def test_blocks(self, max_days=40, min_hours=12):
        te = self.ds.rows(2); days = self.ds.times[te].floor("D")
        blocks = [(pd.Timestamp(d), te[days == d]) for d in pd.unique(days) if (days == d).sum() >= min_hours]
        if len(blocks) > max_days:
            sel = np.linspace(0, len(blocks) - 1, max_days).round().astype(int); blocks = [blocks[i] for i in sel]
        return blocks

    def val_blocks(self, max_days=40, min_hours=12):
        va = self.ds.rows(1); days = self.ds.times[va].floor("D")
        blocks = [(pd.Timestamp(d), va[days == d]) for d in pd.unique(days) if (days == d).sum() >= min_hours]
        if len(blocks) > max_days:
            sel = np.linspace(0, len(blocks) - 1, max_days).round().astype(int); blocks = [blocks[i] for i in sel]
        return blocks

    def pattern_error(self, sheaf, E, blocks, lam=None):
                                                                               
        lam = self.lam if lam is None else lam
        errs = []
        for _, idx in blocks:
            Zb = self.Z[idx]
            if np.isfinite(Zb[:, E, :]).all(axis=(1, 2)).sum() < 6:
                continue
            errs.append(rmse_masked(sheaf.reconstruct(Zb, E, lam), Zb[:, E, :]))
        return float(np.nanmean(errs)) if errs else np.nan

    def neighbours(self, edges=None):
        edges = self.edges if edges is None else edges
        nb = {v: set() for v in range(self.n)}
        for u, v in edges:
            nb[u].add(v); nb[v].add(u)
        return nb
