                                                                     
                                                                  
                                                                     
                                                                   
                                                                       
                                                                
import argparse
import itertools
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from sheafcode.experiments import Config, prepare, fit_all, graph_for, all_patterns, VectorSheafModel, ScalarSheafModel, ROOT              
from sheafcode.data import Dataset              

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="b10")
ap.add_argument("--pollutants", default=None)
ap.add_argument("--tag", default="b10_geo")
args = ap.parse_args()
out = os.path.join(ROOT, "results", args.tag)
diag = json.load(open(os.path.join(out, "sheaf_diagnostics.json")))
choice = diag["choice"]
cfg = Config(dataset=args.dataset, pollutants=tuple(args.pollutants.split(",")) if args.pollutants else None)
ds, sel = prepare(cfg)
models = fit_all(ds, cfg, choice)
Ztr, Zte = ds.Z(0), ds.Z(2)
n, p = ds.n, ds.p
te_rows = ds.rows(2)
times = ds.times[te_rows]

                                                                             
runs, start = [], 0
for i in range(1, len(times) + 1):
    if i == len(times) or (times[i] - times[i - 1]) != pd.Timedelta(hours=1):
        if i - start >= 25:
            runs.append((start, i))
        start = i
rows = []
for (a, b) in runs:
    L = min(b - a - 1, 72)
    for v in range(n):
        last = Zte[a, v, :]                                                                 
        seg = Zte[a + 1:a + 1 + L]
        truth = seg[:, v, :]
        preds = {"vector_sheaf": models["vector_sheaf"].reconstruct(seg, [v])[:, 0, :],
                 "scalar_sheaf": models["scalar_sheaf"].reconstruct(seg, [v])[:, 0, :],
                 "idw": models["idw"].reconstruct(seg, [v])[:, 0, :],
                 "persistence": np.repeat(last[None, :], L, axis=0)}
        for h in range(L):
            for name, ph in preds.items():
                rows.append(dict(run_start=str(times[a]), station=ds.stations[v], hours_elapsed=h + 1, method=name,
                                 sq_err=float(np.mean((ph[h] - truth[h]) ** 2))))
pro = pd.DataFrame(rows)
pro.to_csv(os.path.join(out, "prolonged_results.csv"), index=False)
print("contiguous runs:", len(runs), [(str(times[a]), b - a) for a, b in runs][:6])
print(pro.groupby(["method", pd.cut(pro.hours_elapsed, [0, 6, 12, 24, 48, 72])])["sq_err"].mean().pow(0.5).unstack().round(3))

                                                                                
w, lam, prior = choice["vector_sheaf"]["weighting"], choice["vector_sheaf"]["lam"], choice["vector_sheaf"].get("prior", "identity")
pats1 = [[v] for v in range(n)]
abl = []


def eval_station(model, pats):
    return float(np.mean([np.sqrt(np.mean((model.reconstruct(Zte, E)[:, 0, :] - Zte[:, E[0], :]) ** 2)) for E in pats]))


def eval_channel(model, plist):
    errs = []
    for v in range(n):
        for q in plist:
            col = v * len(plist) + q
            errs.append(np.sqrt(np.mean((model.reconstruct_cols(Zte_sub, [col])[:, 0] - Zte_sub[:, v, q]) ** 2)))
    return float(np.mean(errs))


for k in (1, 2, 3, 4, 5, 6):
    if k > n - 1:
        continue
    edges, _ = graph_for(ds, "geo", k)
    for r in range(1, p + 1):
        m = VectorSheafModel(Ztr, edges, r, w, lam, prior)
        abl.append(dict(ablation="k_r", k=k, r=r, n_edges=len(edges), rmse_station=eval_station(m, pats1),
                        kernel_dim=m.sheaf.kernel_dimension()))
                                                                                                         
edges, _ = graph_for(ds, "geo", choice["vector_sheaf"]["k"])
for size in range(1, p + 1):
    for sub in itertools.combinations(range(p), size):
        sub = list(sub)
        Ztr_sub, Zte_sub = Ztr[:, :, sub], Zte[:, :, sub]
        m = VectorSheafModel(Ztr_sub, edges, min(choice["vector_sheaf"]["r"], size), w, lam, prior)
                                            
        for qi, q in enumerate(sub):
            e_st = float(np.mean([np.sqrt(np.mean((m.reconstruct(Zte_sub, [v])[:, 0, qi] - Zte_sub[:, v, qi]) ** 2)) for v in range(n)]))
            e_ch = float(np.mean([np.sqrt(np.mean((m.reconstruct_cols(Zte_sub, [v * size + qi])[:, 0] - Zte_sub[:, v, qi]) ** 2)) for v in range(n)]))
            abl.append(dict(ablation="pollutant_subset", subset="+".join(ds.pollutants[j] for j in sub), n_pollutants=size,
                            target=ds.pollutants[q], rmse_station=e_st, rmse_channel=e_ch))
abl = pd.DataFrame(abl)
abl.to_csv(os.path.join(out, "ablation_results.csv"), index=False)
print(abl[abl.ablation == "k_r"].pivot(index="k", columns="r", values="rmse_station").round(3))
ps = abl[abl.ablation == "pollutant_subset"]
print(ps.groupby(["target", "n_pollutants"])[["rmse_station", "rmse_channel"]].mean().round(3))
