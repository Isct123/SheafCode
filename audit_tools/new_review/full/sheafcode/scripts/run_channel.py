                                                                    

                                                                       
                                                                     
                                                                    
                                                                    
                                                                       
                                                                       
                                                                         
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from sheafcode.experiments import Config, prepare, fit_all, day_blocks, graph_for, VectorSheafModel, ScalarSheafModel, ROOT              

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="b10")
ap.add_argument("--pollutants", default=None)
ap.add_argument("--tag", default="b10_geo")
args = ap.parse_args()

out = os.path.join(ROOT, "results", args.tag)
diag = json.load(open(os.path.join(out, "sheaf_diagnostics.json")))
cfg = Config(dataset=args.dataset, pollutants=tuple(args.pollutants.split(",")) if args.pollutants else None)
ds, sel = prepare(cfg)
assert ds.stations == diag["stations"] and ds.pollutants == diag["pollutants"]
choice = diag["choice"]
models = fit_all(ds, cfg, choice)
Ztr, Zva, Zte = ds.Z(0), ds.Z(1), ds.Z(2)
n, p = ds.n, ds.p


def channel_preds(models, vec, sca, Zb, v, q):
    col = v * p + q
    preds = {"vector_sheaf": vec.reconstruct_cols(Zb, [col])[:, 0],
             "scalar_sheaf": sca.reconstruct(Zb, [v])[:, 0, q]}
    lr = models["lowrank"].m
    M = Zb.reshape(len(Zb), -1) - lr.mu
    obs = np.setdiff1d(np.arange(n * p), [col])
    C = M[:, obs] @ np.linalg.pinv(lr.V[obs]).T
    preds["lowrank"] = (C @ lr.V[[col]].T + lr.mu[[col]])[:, 0]
    for name in ["scalar_laplacian", "idw", "kriging"]:
        preds[name] = models[name].reconstruct(Zb, [v])[:, 0, q]
    return preds


                                                                               
edges_v, _ = graph_for(ds, cfg.graph, choice["vector_sheaf"]["k"])
edges_s, _ = graph_for(ds, cfg.graph, choice["scalar_sheaf"]["k"])
vlog = []
best_v, best_s = (np.inf, None), (np.inf, None)
for lam in cfg.lams:
    for prior in cfg.priors:
        if lam == 0 and prior != "identity":
            continue
        vec = VectorSheafModel(Ztr, edges_v, choice["vector_sheaf"]["r"], choice["vector_sheaf"]["weighting"], lam, prior)
        errs = [np.sqrt(np.mean((vec.reconstruct_cols(Zva, [v * p + q])[:, 0] - Zva[:, v, q]) ** 2)) for v in range(n) for q in range(p)]
        vlog.append(dict(method="vector_sheaf", lam=lam, prior=prior, val_rmse_z=float(np.mean(errs))))
        if np.mean(errs) < best_v[0]:
            best_v = (float(np.mean(errs)), dict(lam=lam, prior=prior))
    sca = ScalarSheafModel(Ztr, edges_s, choice["scalar_sheaf"]["weighting"], lam)
    errs = [np.sqrt(np.mean((sca.reconstruct(Zva, [v])[:, 0, q] - Zva[:, v, q]) ** 2)) for v in range(n) for q in range(p)]
    vlog.append(dict(method="scalar_sheaf", lam=lam, val_rmse_z=float(np.mean(errs))))
    if np.mean(errs) < best_s[0]:
        best_s = (float(np.mean(errs)), dict(lam=lam))
pd.DataFrame(vlog).to_csv(os.path.join(out, "channel_validation_log.csv"), index=False)
print("channel-regime choice:", best_v[1], best_s[1])
vec = VectorSheafModel(Ztr, edges_v, choice["vector_sheaf"]["r"], choice["vector_sheaf"]["weighting"], best_v[1]["lam"], best_v[1]["prior"])
sca = ScalarSheafModel(Ztr, edges_s, choice["scalar_sheaf"]["weighting"], best_s[1]["lam"])

                                                                                 
te_rows = ds.rows(2)
pos = {r: i for i, r in enumerate(te_rows)}
blocks = [(d, np.array([pos[r] for r in idx])) for d, idx in day_blocks(ds.times, te_rows)]
rows = []
for v in range(n):
    for q in range(p):
        cert = vec.sheaf.certificate_cols([v * p + q])
        for day, idx in blocks:
            Zb = Zte[idx]
            truth = Zb[:, v, q]
            for name, ph in channel_preds(models, vec, sca, Zb, v, q).items():
                err = ph - truth
                rows.append(dict(method=name, station=ds.stations[v], pollutant=ds.pollutants[q], day=day,
                                 n_hours=len(idx), rmse_z=float(np.sqrt(np.mean(err ** 2))),
                                 rmse_raw=float(np.sqrt(np.mean((err * ds.std[v, q]) ** 2))),
                                 sigma_min=cert["sigma_min"], identifiable=cert["identifiable"]))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(out, "channel_results.csv"), index=False)
json.dump(dict(vector_sheaf=best_v[1], scalar_sheaf=best_s[1]), open(os.path.join(out, "channel_choice.json"), "w"))
print(res.groupby(["method", "pollutant"])["rmse_z"].mean().unstack().round(3))
print(res.groupby("method")["rmse_z"].mean().round(3))
