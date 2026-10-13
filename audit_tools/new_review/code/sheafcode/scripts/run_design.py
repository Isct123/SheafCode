                                               

                                                                         
                                                                          
                                                                          
                                                                      
                                                                 
                                                                      
                                                                      
                                                                       
                                                                       
                                                                   
                                                    
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
from sheafcode.experiments import Config, prepare, all_patterns, ROOT              
from sheafcode.graph import knn_edges              
from sheafcode.sheaf import learn_sheaf              
from sheafcode.data import haversine_km              

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="b10")
ap.add_argument("--pollutants", default=None)
ap.add_argument("--tag", default="b10_geo")
args = ap.parse_args()
out = os.path.join(ROOT, "results", args.tag)
diag = json.load(open(os.path.join(out, "sheaf_diagnostics.json")))
ch = diag["choice"]["vector_sheaf"]
cfg = Config(dataset=args.dataset, pollutants=tuple(args.pollutants.split(",")) if args.pollutants else None)
ds, sel = prepare(cfg)
Ztr, Zte = ds.Z(0), ds.Z(2)
n, p = ds.n, ds.p
Dfull = haversine_km(ds.coords)


def network_metrics(stations, k, r, kmax=2):
                                                                          
                                                                          
    idx = list(stations)
    D = Dfull[np.ix_(idx, idx)]
    edges = knn_edges(D, min(k, len(idx) - 1))
    sh = learn_sheaf(Ztr[:, idx, :], edges, r, ch["weighting"])
    pats = all_patterns(len(idx), min(kmax, len(idx) - 2))
    certs = [sh.certificate(E) for E in pats]
    smin = np.array([c["sigma_min"] for c in certs])
    ident = np.array([c["identifiable"] for c in certs])
    Zt = Zte[:, idx, :]
    err = np.mean([np.sqrt(np.mean((sh.reconstruct_batch(Zt, [v], ch["lam"])[:, 0, :] - Zt[:, v, :]) ** 2)) for v in range(len(idx))])
    return dict(mean_sigma_min=float(smin.mean()), min_sigma_min=float(smin.min()), frac_identifiable=float(ident.mean()),
                test_rmse_single=float(err), n_edges=len(edges))


rows = []
full = network_metrics(range(n), ch["k"], ch["r"])
rows.append(dict(removed="none", **full))
for v in range(n):
    rest = [u for u in range(n) if u != v]
    m = network_metrics(rest, ch["k"], ch["r"])
    rows.append(dict(removed=ds.stations[v], **m))
crit = pd.DataFrame(rows)
crit.to_csv(os.path.join(out, "design_criticality.csv"), index=False)
print(crit.round(3).to_string())

                                                                       
rows = []
for v in range(n):
    rest = [u for u in range(n) if u != v]
    m = network_metrics(rest, 2, ch["r"], kmax=3)
    rows.append(dict(removed=ds.stations[v], **m))
pd.DataFrame(rows).to_csv(os.path.join(out, "design_criticality_sparse.csv"), index=False)

                        
rows = []
for a, b in itertools.combinations(range(n), 2):
    base = [u for u in range(n) if u not in (a, b)]
    ma = network_metrics(base + [a], ch["k"], ch["r"])
    mb = network_metrics(base + [b], ch["k"], ch["r"])
    cert_pick = a if ma["mean_sigma_min"] > mb["mean_sigma_min"] else b
    err_pick = a if ma["test_rmse_single"] < mb["test_rmse_single"] else b
    geo_pick = a if Dfull[a, base].min() > Dfull[b, base].min() else b
    rows.append(dict(removed=f"{ds.stations[a]}+{ds.stations[b]}", cert_pick=ds.stations[cert_pick],
                     err_pick=ds.stations[err_pick], geo_pick=ds.stations[geo_pick],
                     cert_agrees=int(cert_pick == err_pick), geo_agrees=int(geo_pick == err_pick),
                     rmse_a=ma["test_rmse_single"], rmse_b=mb["test_rmse_single"],
                     smin_a=ma["mean_sigma_min"], smin_b=mb["mean_sigma_min"],
                     regret_cert=float(abs(ma["test_rmse_single"] - mb["test_rmse_single"]) * (cert_pick != err_pick)),
                     regret_geo=float(abs(ma["test_rmse_single"] - mb["test_rmse_single"]) * (geo_pick != err_pick))))
add = pd.DataFrame(rows)
add.to_csv(os.path.join(out, "design_readdition.csv"), index=False)
print("re-addition: certificate agrees with test-optimal choice in", add.cert_agrees.mean().round(3),
      "| geographic rule agrees in", add.geo_agrees.mean().round(3),
      "| mean regret cert", add.regret_cert.mean().round(4), "geo", add.regret_geo.mean().round(4), "n", len(add))
