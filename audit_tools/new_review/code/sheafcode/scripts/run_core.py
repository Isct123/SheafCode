                                       
                                                                         
                                                                     
                                                                
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from sheafcode.experiments import Config, prepare, validate, fit_all, run_test, save_json, ROOT              

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="b10")
ap.add_argument("--pollutants", default=None, help="comma list to force a pollutant set")
ap.add_argument("--tag", default=None)
ap.add_argument("--graph", default="geo")
ap.add_argument("--kmax_test", type=int, default=4)
args = ap.parse_args()

cfg = Config(dataset=args.dataset, graph=args.graph, kmax_test=args.kmax_test,
             pollutants=tuple(args.pollutants.split(",")) if args.pollutants else None)
tag = args.tag or f"{args.dataset}_{args.graph}" + (f"_{args.pollutants.replace(',', '-')}" if args.pollutants else "")
out = os.path.join(ROOT, "results", tag)
os.makedirs(out, exist_ok=True)

t0 = time.time()
ds, sel = prepare(cfg)
print(f"[{tag}] stations={ds.stations} pollutants={ds.pollutants} T={len(ds.times)} "
      f"train/val/test={np.bincount(ds.split).tolist()} span={ds.times.min()}..{ds.times.max()}")
sel.table.to_csv(os.path.join(out, "coverage_candidates.csv"), index=False)

choice, vlog = validate(ds, cfg)
vlog.to_csv(os.path.join(out, "validation_log.csv"), index=False)
print(f"[{tag}] chosen: {choice}")

models = fit_all(ds, cfg, choice)
sh = models["vector_sheaf"].sheaf
diag = dict(stations=ds.stations, pollutants=ds.pollutants, n_train=int((ds.split == 0).sum()),
            n_val=int((ds.split == 1).sum()), n_test=int((ds.split == 2).sum()),
            train_span=[str(ds.times[ds.split == 0].min()), str(ds.times[ds.split == 0].max())],
            test_span=[str(ds.times[ds.split == 2].min()), str(ds.times[ds.split == 2].max())],
            edges=[(ds.stations[u], ds.stations[v]) for u, v in sh.edges],
            canonical_correlations=[list(map(float, r)) for r in sh.rho],
            delta_shape=list(sh.delta.shape), kernel_dim=sh.kernel_dimension(),
            choice=choice, train_mean=ds.mean.tolist(), train_std=ds.std.tolist(),
            coords=ds.coords.reset_index().to_dict(orient="records"))
save_json(diag, os.path.join(out, "sheaf_diagnostics.json"))
np.save(os.path.join(out, "delta.npy"), sh.delta)

res = run_test(ds, cfg, models)
res.to_csv(os.path.join(out, "test_results.csv"), index=False)
summ = res.groupby(["method", "size"])["rmse_z"].mean().unstack("size").round(3)
print(summ.to_string())
print(f"[{tag}] kernel dim = {sh.kernel_dimension()}  delta {sh.delta.shape}  done in {time.time() - t0:.0f}s")
