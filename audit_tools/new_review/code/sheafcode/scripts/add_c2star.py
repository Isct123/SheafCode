                                                                         
                                                                          
                                                                        
                                                                        
                                          
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
from sheafcode.india import select_region, ROOT              
from sheafcode.masked import MaskedSheaf              
from sheafcode.graph import knn_edges              
from sheafcode.data import haversine_km              

ap = argparse.ArgumentParser()
ap.add_argument("--regions", default="delhi,ncr,mumbai,kolkata,hyderabad,bengaluru,india")
args = ap.parse_args()
for reg in args.regions.split(","):
    out = os.path.join(ROOT, "results", f"v2_{reg}")
    diag = json.load(open(os.path.join(out, "diagnostics.json")))
    c = diag["choice"]["vector_sheaf"]
    ds = select_region(reg); ds.set_transform(c["transform"]); Z = ds.Z(); Ztr = Z[ds.split == 0]
    D = haversine_km(ds.coords)
    if c["graph"] == "corr":
        from run_v2 import corr_distance
        Dg = corr_distance(Ztr)
    else:
        Dg = D
    sheaf = MaskedSheaf(Ztr, knn_edges(Dg, c["k"]), c["r"], c["weighting"], stalk=c.get("stalk", "cca"))
    assert list(sheaf.delta.shape) == diag["delta_shape"], (sheaf.delta.shape, diag["delta_shape"])
    res = pd.read_csv(os.path.join(out, "test_results.csv"))
    name2idx = {s: i for i, s in enumerate(ds.stations)}
    lam = c["lam"]
    c2s = {}
    for pat in res.pattern.unique():
        E = [name2idx[s] for s in pat.split("+")]
        cols = sheaf.cols(E)
        dU = sheaf.delta[:, cols]
        G = dU.T @ dU + lam * sheaf.P[np.ix_(cols, cols)]
        M = np.linalg.solve(G, dU.T)
        c2s[pat] = float(np.linalg.svd(M, compute_uv=False)[0])
    res["c2_star"] = res.pattern.map(c2s)
    res.to_csv(os.path.join(out, "test_results.csv"), index=False)
    v = res[res.method == "vector_sheaf"].groupby("pattern").agg(rmse=("rmse_z", "mean"), s=("sigma_min", "first"), c2=("c2_star", "first")).dropna()
    from scipy.stats import spearmanr
    print(f"{reg}: patterns={len(v)} spearman(1/sigma_min, rmse)={spearmanr(1 / v.s, v.rmse).statistic:.3f}  spearman(C2*, rmse)={spearmanr(v.c2, v.rmse).statistic:.3f}  "
          f"C2* range {v.c2.min():.3f}..{v.c2.max():.3f}  lambda={lam}", flush=True)
