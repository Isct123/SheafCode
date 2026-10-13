                                                                        
                                                                       
                                                                    
                                                                     
                                                               
                                                                     
                                                                      
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
from sheafcode.experiments import Config, prepare, graph_for, all_patterns, ROOT              
from sheafcode.sheaf import learn_sheaf              

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="b10")
ap.add_argument("--pollutants", default=None)
ap.add_argument("--tag", default="b10_geo")
ap.add_argument("--weighting", default="reg")
args = ap.parse_args()

out = os.path.join(ROOT, "results", args.tag)
cfg = Config(dataset=args.dataset, pollutants=tuple(args.pollutants.split(",")) if args.pollutants else None)
ds, sel = prepare(cfg)
Ztr, Zte = ds.Z(0), ds.Z(2)
n, p = ds.n, ds.p
pats = all_patterns(n, n - 2)
rows = []
for k in range(1, min(4, n - 1) + 1):
    edges, D = graph_for(ds, "geo", k)
    for r in range(1, p + 1):
        sh = learn_sheaf(Ztr, edges, r, args.weighting)
        kdim = sh.kernel_dimension()
        for E in pats:
            cert = sh.certificate(E)
            cols = sh.cols(E)
            dE = sh.delta[:, cols]
                                                 
            U, s, Vt = np.linalg.svd(dE, full_matrices=True)
            null = Vt[cert["rank"]:]
            undetermined = []
            if len(null):
                sup = np.abs(null).max(axis=0) > 1e-8
                undetermined = sorted({ds.stations[cols[i] // p] for i in np.where(sup)[0]})
            truth = Zte[:, E, :]
            for lam, label in [(0.0, "ls"), (1.0, "ridge")]:
                zhat = sh.reconstruct_batch(Zte, E, lam)
                err = zhat - truth
                rows.append(dict(k=k, r=r, n_edges=len(edges), kernel_dim=kdim,
                                 pattern="+".join(ds.stations[v] for v in E), size=len(E),
                                 rank=cert["rank"], needed=cert["needed"], identifiable=cert["identifiable"],
                                 sigma_min=cert["sigma_min"], cond=cert["cond"], solver=label,
                                 rmse_z=float(np.sqrt(np.mean(err ** 2))),
                                 undetermined_stations="+".join(undetermined)))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(out, "identifiability_results.csv"), index=False)
summ = res[res.solver == "ls"].groupby(["k", "r", "identifiable"]).agg(n=("pattern", "size"), rmse=("rmse_z", "mean"),
                                                                       med=("rmse_z", "median")).round(3)
print(summ.to_string())
