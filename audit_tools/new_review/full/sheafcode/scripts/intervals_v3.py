                                                                          

                                                                        
                                           
                                                                       
                                                                       
                                                                        
                                                                         
                                                                      
                                                                  
                                                                        
                            
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from sheafcode.frozen import Frozen              
from sheafcode.experiments import save_json              

ap = argparse.ArgumentParser()
ap.add_argument("--region", default="delhi")
ap.add_argument("--max_days", type=int, default=40)
ap.add_argument("--n_stations", type=int, default=30)
args = ap.parse_args()
rng = np.random.default_rng(0)
F = Frozen(args.region); n = F.n
stations = list(range(n)) if n <= args.n_stations else sorted(rng.choice(n, args.n_stations, replace=False).tolist())
c3 = {v: F.sheaf.predicted_rmse_empirical([v], F.lam, F.Ztr, max_rows=1000) for v in stations}


def scaled_residuals(blocks):
    rows = []
    for day, idx in blocks:
        Zb = F.Z[idx]
        for v in stations:
            if not np.isfinite(c3[v]) or c3[v] <= 0:
                continue
            m = np.isfinite(Zb[:, v, :])
            if m.sum() < 6:
                continue
            pred = F.sheaf.reconstruct(Zb, [v], F.lam)[:, 0, :]
            r = np.abs(pred - Zb[:, v, :])[m] / c3[v]
            rows.append(dict(day=day, station=v, mean_scaled=float(r.mean()), scaled=r))
    return rows


val = scaled_residuals(F.val_blocks(args.max_days)); test = scaled_residuals(F.test_blocks(args.max_days))
out = {}
for level in (0.5, 0.9):
    q_day = float(np.quantile([r["mean_scaled"] for r in val], level))                                      
    q_cell = float(np.quantile(np.concatenate([r["scaled"] for r in val]), level))                           
    cov_cell = float(np.mean(np.concatenate([r["scaled"] for r in test]) <= q_cell))
    cov_day = float(np.mean([r["mean_scaled"] <= q_day for r in test]))
    out[f"level_{int(level * 100)}"] = dict(q_cell=q_cell, q_day=q_day, test_coverage_cells=cov_cell, test_coverage_day_means=cov_day,
                                           n_val_station_days=len(val), n_test_station_days=len(test))
out["note"] = "interval = prediction +/- q * C3_empirical; q from the validation block; coverage on the test block; empirical check, not a conformal guarantee"
out["c3_range"] = [float(np.nanmin(list(c3.values()))), float(np.nanmax(list(c3.values())))]
save_json(out, os.path.join(F.out, "intervals.json"))
print(args.region, {k: (round(v["test_coverage_cells"], 3), round(v["test_coverage_day_means"], 3)) for k, v in out.items() if k.startswith("level")})
