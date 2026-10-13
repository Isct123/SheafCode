                                                                         

                                                                        
                                                                     
                                                                     
                                                                 
                                                                    
                                                                
                                                                      
                                                               
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from sheafcode.experiments import Config, prepare, fit_all, day_blocks, graph_for, ROOT              

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="b10")
ap.add_argument("--pollutants", default=None)
ap.add_argument("--tag", default="b10_geo")
args = ap.parse_args()
out = os.path.join(ROOT, "results", args.tag)
diag = json.load(open(os.path.join(out, "sheaf_diagnostics.json")))
cfg = Config(dataset=args.dataset, pollutants=tuple(args.pollutants.split(",")) if args.pollutants else None)
ds, sel = prepare(cfg)
models = fit_all(ds, cfg, diag["choice"])
sh = models["vector_sheaf"].sheaf
n, p = ds.n, ds.p
Zte = ds.Z(2)
te_rows = ds.rows(2)
pos = {r: i for i, r in enumerate(te_rows)}
blocks = [(d, np.array([pos[r] for r in idx])) for d, idx in day_blocks(ds.times, te_rows)]
edges, D = graph_for(ds, "geo", diag["choice"]["vector_sheaf"]["k"])
nbrs = {v: [u for (a, b) in edges for u in ((b,) if a == v else (a,) if b == v else ())] for v in range(n)}
rng = np.random.default_rng(0)


def scores(Zb):
                                                                         
    T = Zb.shape[0]
    syn = np.zeros(n); mag = np.zeros(n); nbd = np.zeros(n); loo = np.zeros(n)
    for t in range(T):
        _, sc = sh.syndrome(Zb[t])
        syn += sc
    syn /= T
    mag = np.mean(Zb ** 2, axis=(0, 2))
    for v in range(n):
        nb = nbrs[v]
        nbd[v] = np.mean((Zb[:, v, :] - Zb[:, nb, :].mean(axis=1)) ** 2) if nb else 0.0
        loo[v] = np.mean((models["vector_sheaf"].reconstruct(Zb, [v])[:, 0, :] - Zb[:, v, :]) ** 2)
    return dict(syndrome=syn, magnitude=mag, neighbour_diff=nbd, loo_residual=loo)


def inject(Zb, v, kind, mag, q=None):
    Y = Zb.copy()
    T = Y.shape[0]
    if kind == "bias":
        Y[:, v, :] += mag
    elif kind == "gain":
        Y[:, v, :] *= (1.0 + mag)
    elif kind == "drift":
        Y[:, v, :] += mag * np.linspace(0, 1, T)[:, None]
    elif kind == "stuck":
        Y[:, v, :] = Y[0, v, :] + mag * 0                                        
    elif kind == "channel":
        Y[:, v, q] += mag
    return Y


rows = []
clean_max = {}
for day, idx in blocks:
    Zb = Zte[idx]
    base = scores(Zb)
    for det, sc in base.items():
        clean_max[(det, day)] = float(np.max(sc))
    for v in range(n):
        for kind in ["bias", "gain", "drift", "stuck", "channel"]:
            mags = [0.0] if kind == "stuck" else [0.5, 1.0, 2.0]
            for m in mags:
                q = int(rng.integers(p)) if kind == "channel" else None
                Y = inject(Zb, v, kind, m, q)
                for det, sc in scores(Y).items():
                    rows.append(dict(day=day, station=ds.stations[v], kind=kind, magnitude=m, detector=det,
                                     top1=int(np.argmax(sc) == v), score_faulty=float(sc[v]),
                                     score_max=float(np.max(sc)), clean_max=clean_max[(det, day)]))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(out, "fault_results.csv"), index=False)
acc = res.groupby(["kind", "magnitude", "detector"])["top1"].mean().unstack("detector").round(3)
print(acc.to_string())


def auc(pos_, neg_):
    pos_, neg_ = np.asarray(pos_), np.asarray(neg_)
    return float(np.mean(pos_[:, None] > neg_[None, :]) + 0.5 * np.mean(pos_[:, None] == neg_[None, :]))


aucs = []
for (kind, m, det), g in res.groupby(["kind", "magnitude", "detector"]):
    aucs.append(dict(kind=kind, magnitude=m, detector=det, auc=auc(g.score_max, g.clean_max)))
aucs = pd.DataFrame(aucs)
aucs.to_csv(os.path.join(out, "fault_auc.csv"), index=False)
print(aucs.pivot_table(index=["kind", "magnitude"], columns="detector", values="auc").round(3).to_string())
