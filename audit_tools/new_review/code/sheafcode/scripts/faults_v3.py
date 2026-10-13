                                                              
                                                                        
                                                                    
                                                                   
                                                                      
                                                                     
                                                                 
                                                                  
                              
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from sheafcode.frozen import Frozen              
from sheafcode.experiments import save_json              

ap = argparse.ArgumentParser()
ap.add_argument("--region", default="delhi")
ap.add_argument("--max_days", type=int, default=30)
ap.add_argument("--n_stations", type=int, default=12)
ap.add_argument("--no_loo", action="store_true", help="skip the leave-one-out detector (n reconstructions per injection; used for the national graph)")
args = ap.parse_args()
t0 = time.time()
rng = np.random.default_rng(0)
F = Frozen(args.region); ds, n, p, sheaf, lam = F.ds, F.n, F.p, F.sheaf, F.lam
blocks = F.test_blocks(args.max_days)
stations = list(range(n)) if n <= args.n_stations else sorted(rng.choice(n, args.n_stations, replace=False).tolist())
nb = F.neighbours()
SEV = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
KINDS = ["bias", "gain", "drift", "channel", "stuck"]


def scores(Zraw):
                                                                    
    hide = np.zeros((Zraw.shape[0], n), bool)
    Zb = sheaf.reconstruct_mask(Zraw, hide, lam)
    T = Zb.shape[0]
    syn = np.zeros(n)
    for t in range(T):
        syn += sheaf.syndrome(Zb[t])[1]
    syn /= T
    mag = np.mean(Zb ** 2, axis=(0, 2))
    nbd = np.array([np.mean((Zb[:, v, :] - Zb[:, list(nb[v]), :].mean(axis=1)) ** 2) if nb[v] else 0 for v in range(n)])
    out = dict(syndrome=syn, magnitude=mag, neighbour_diff=nbd)
    if not args.no_loo:
        out["loo_residual"] = np.array([np.mean((sheaf.reconstruct(Zraw, [v], lam)[:, 0, :] - np.nan_to_num(Zraw[:, v, :], nan=0.0)) ** 2 * np.isfinite(Zraw[:, v, :])) for v in range(n)])
    return out


def inject(Z, v, kind, mag, q):
    Y = Z.copy(); T = Y.shape[0]
    if kind == "bias":
        Y[:, v, :] += mag
    elif kind == "gain":
        Y[:, v, :] *= (1 + mag)
    elif kind == "drift":
        Y[:, v, :] += mag * np.linspace(0, 2, T)[:, None]
    elif kind == "stuck":
        first = np.where(np.isfinite(Y[:, v, :]).all(axis=1))[0]
        if len(first):
            Y[:, v, :] = np.where(np.isfinite(Y[:, v, :]), Y[first[0], v, :], np.nan)
    elif kind == "channel":
        Y[:, v, q] += mag
    return Y


rows = []
for day, idx in blocks:
    Zraw = F.Z[idx]
    for v in stations:
        if np.isfinite(Zraw[:, v, :]).mean() < 0.5:
            continue
        for kind in KINDS:
            for m in ([0.0] if kind == "stuck" else SEV):
                q = int(rng.integers(p))
                sc = scores(inject(Zraw, v, kind, m, q))
                for det, s in sc.items():
                    s = np.nan_to_num(s, nan=-np.inf)
                    order = np.argsort(-s)
                    y = np.zeros(n); y[v] = 1
                    auc = roc_auc_score(y, np.nan_to_num(s, neginf=-1e9)) if n > 1 else np.nan
                    rows.append(dict(day=day, station=ds.stations[v], kind=kind, severity=m, detector=det,
                                     top1=int(order[0] == v), top3=int(v in order[:3]), auc=float(auc)))
    print(f"  {day.date()} ({time.time() - t0:.0f}s)", flush=True)
res = pd.DataFrame(rows); res.to_csv(os.path.join(F.out, "fault_results.csv"), index=False)
tab = res.groupby(["kind", "severity", "detector"])[["top1", "top3", "auc"]].mean().reset_index()
summ = dict(region=args.region, n_stations=len(stations), test_days=len(blocks), chance_top1=1 / n, chance_top3=min(1, 3 / n),
            by_kind_severity=tab.round(4).to_dict(orient="records"),
            overall_by_detector=res.groupby("detector")[["top1", "top3", "auc"]].mean().round(4).to_dict(orient="index"),
            severity_curve_syndrome=res[res.detector == "syndrome"].groupby("severity")[["top1", "top3", "auc"]].mean().round(4).to_dict(orient="index"),
            severity_curve_loo=res[res.detector == "loo_residual"].groupby("severity")[["top1", "top3", "auc"]].mean().round(4).to_dict(orient="index") if "loo_residual" in set(res.detector) else None,
            no_loo=bool(args.no_loo),
            seconds=round(time.time() - t0))
save_json(summ, os.path.join(F.out, "faults.json"))
print(tab.pivot_table(index=["kind", "severity"], columns="detector", values="top1").round(3).to_string())
print("overall", summ["overall_by_detector"])
