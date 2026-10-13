                                                                    
                                                                          
                                                                        
                                                                        
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from sheafcode.india import select_region, ROOT              
from sheafcode.masked import MaskedSheaf              
from sheafcode.graph import knn_edges              
from sheafcode.data import haversine_km              

ap = argparse.ArgumentParser()
ap.add_argument("--region", default="delhi")
ap.add_argument("--max_days", type=int, default=40)
args = ap.parse_args()
out = os.path.join(ROOT, "results", f"v2_{args.region}")
diag = json.load(open(os.path.join(out, "diagnostics.json")))
c = diag["choice"]["vector_sheaf"]
ds = select_region(args.region); ds.set_transform(c["transform"]); Z = ds.Z()
n, p = ds.n, ds.p
D = haversine_km(ds.coords)
if c["graph"] == "corr":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_v2 import corr_distance        
    Dg = corr_distance(Z[ds.split == 0])
else:
    Dg = D
edges = knn_edges(Dg, c["k"])
sheaf = MaskedSheaf(Z[ds.split == 0], edges, c["r"], c["weighting"], stalk=c.get("stalk", "cca"))
nbrs = {v: [u for (a, b) in edges for u in ((b,) if a == v else (a,) if b == v else ())] for v in range(n)}
te = ds.rows(2); days = ds.times[te].floor("D")
blocks = [(pd.Timestamp(d), te[days == d]) for d in pd.unique(days) if (days == d).sum() >= 12]
sel = np.linspace(0, len(blocks) - 1, min(args.max_days, len(blocks))).round().astype(int)
blocks = [blocks[i] for i in sel]
rng = np.random.default_rng(0)
stations = list(range(n)) if n <= 12 else sorted(rng.choice(n, 12, replace=False).tolist())


def scores(Zb):
    T = Zb.shape[0]
    syn = np.zeros(n)
    for t in range(T):
        syn += sheaf.syndrome(Zb[t])[1]
    syn /= T
    mag = np.mean(Zb ** 2, axis=(0, 2))
    nbd = np.array([np.mean((Zb[:, v, :] - Zb[:, nbrs[v], :].mean(axis=1)) ** 2) if nbrs[v] else 0 for v in range(n)])
    loo = np.array([np.mean((sheaf.reconstruct(Zb, [v], c["lam"])[:, 0, :] - Zb[:, v, :]) ** 2) for v in range(n)])
    return dict(syndrome=syn, magnitude=mag, neighbour_diff=nbd, loo_residual=loo)


def inject(Zb, v, kind, mag, q=None):
    Y = Zb.copy(); T = Y.shape[0]
    if kind == "bias":
        Y[:, v, :] += mag
    elif kind == "gain":
        Y[:, v, :] *= (1 + mag)
    elif kind == "drift":
        Y[:, v, :] += mag * np.linspace(0, 1, T)[:, None]
    elif kind == "stuck":
        Y[:, v, :] = Y[0, v, :]
    elif kind == "channel":
        Y[:, v, q] += mag
    return Y


rows = []
for day, idx in blocks:
    Zb = sheaf.reconstruct_mask(Z[idx], np.zeros((len(idx), n), bool), c["lam"])
    base = scores(Zb); clean = {k: float(v.max()) for k, v in base.items()}
    for v in stations:
        for kind in ["bias", "gain", "drift", "stuck", "channel"]:
            for m in ([0.0] if kind == "stuck" else [0.5, 1.0, 2.0]):
                q = int(rng.integers(p)) if kind == "channel" else None
                for det, sc in scores(inject(Zb, v, kind, m, q)).items():
                    rows.append(dict(day=day, station=ds.stations[v], kind=kind, magnitude=m, detector=det,
                                     top1=int(np.argmax(sc) == v), score_max=float(sc.max()), clean_max=clean[det]))
res = pd.DataFrame(rows); res.to_csv(os.path.join(out, "fault_results.csv"), index=False)
print(res.groupby(["kind", "magnitude", "detector"]).top1.mean().unstack().round(3).to_string())
print(res.groupby("detector").top1.mean().round(3).to_dict(), "chance", round(1 / n, 3))
