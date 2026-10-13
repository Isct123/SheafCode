                                                                           
                                                                   
                                                                    
                                                                  
                                                                    
                                                                
                                                                      
                                                                      
                                                              
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from sheafcode.frozen import Frozen, rmse_masked              
from sheafcode.masked import MaskedSheaf              
from sheafcode.baselines2 import ScalarLaplacian2              
from sheafcode.experiments import save_json              

ap = argparse.ArgumentParser()
ap.add_argument("--region", default="delhi")
ap.add_argument("--max_days", type=int, default=30)
ap.add_argument("--n_stations", type=int, default=40)
args = ap.parse_args()
t0 = time.time()
rng = np.random.default_rng(0)
F = Frozen(args.region); ds, n, p = F.ds, F.n, F.p
blocks = F.test_blocks(args.max_days); vblocks = F.val_blocks(args.max_days)
stations = list(range(n)) if n <= args.n_stations else sorted(rng.choice(n, args.n_stations, replace=False).tolist())
c = F.c


def score(fn, blk):
    per = {}
    for v in stations:
        errs = []
        for _, idx in blk:
            Zb = F.Z[idx]
            if np.isfinite(Zb[:, v, :]).all(axis=1).sum() < 6:
                continue
            errs.append(rmse_masked(fn(Zb, [v])[:, 0, :], Zb[:, v, :]))
        per[v] = float(np.nanmean(errs)) if errs else np.nan
    return per


variants = []
base = dict(weighting=c["weighting"], stalk=c["stalk"], r=c["r"], prior="station_cov", lam=c["lam"])
variants.append(("frozen", dict(base)))
for w in ("none", "invvar", "reg"):
    if w != c["weighting"]:
        variants.append((f"weighting={w}", {**base, "weighting": w}))
for st in ("cca", "cca+id"):
    if st != c["stalk"]:
        variants.append((f"stalk={st}", {**base, "stalk": st}))
for r in sorted({1, 2, p}):
    if r != c["r"]:
        variants.append((f"r={r}", {**base, "r": r}))
variants.append(("prior=identity", {**base, "prior": "identity"}))
for lam in (0.1, 0.3, 1.0, 3.0, 10.0):
    if lam != c["lam"]:
        variants.append((f"lambda={lam}", {**base, "lam": lam}))
variants.append(("prior=identity,weighting=none (bare consistency)", {**base, "prior": "identity", "weighting": "none"}))
rows = []
cache = {}
for name, cfg in variants:
    key = (cfg["weighting"], cfg["stalk"], cfg["r"])
    if key not in cache:
        cache[key] = MaskedSheaf(F.Ztr, F.edges, cfg["r"], cfg["weighting"], stalk=cfg["stalk"])
    sh = cache[key]
    fn = lambda Zb, E, sh=sh, cfg=cfg: sh.reconstruct(Zb, E, cfg["lam"], prior=cfg["prior"])
    te = score(fn, blocks); va = score(fn, vblocks)
    rows.append(dict(variant=name, **cfg, val_rmse=float(np.nanmean(list(va.values()))), test_rmse=float(np.nanmean(list(te.values()))),
                     test_per_station=te))
    print(f"  {name}: val={rows[-1]['val_rmse']:.4f} test={rows[-1]['test_rmse']:.4f} ({time.time() - t0:.0f}s)", flush=True)
            
shs = [MaskedSheaf(F.Ztr[:, :, [q]], F.edges, 1, "reg") for q in range(p)]
fn = lambda Zb, E: np.stack([shs[q].reconstruct(Zb[:, :, [q]], E, c["lam"])[:, :, 0] for q in range(p)], -1)
te = score(fn, blocks); va = score(fn, vblocks)
rows.append(dict(variant="scalar sheaf (per-pollutant, same graph)", val_rmse=float(np.nanmean(list(va.values()))), test_rmse=float(np.nanmean(list(te.values()))), test_per_station=te))
sl = ScalarLaplacian2(F.D, F.edges)
te = score(sl.reconstruct, blocks); va = score(sl.reconstruct, vblocks)
rows.append(dict(variant="scalar Laplacian (same graph)", val_rmse=float(np.nanmean(list(va.values()))), test_rmse=float(np.nanmean(list(te.values()))), test_per_station=te))
print(f"  reductions: {rows[-2]['test_rmse']:.4f} {rows[-1]['test_rmse']:.4f}", flush=True)
                                               
ref = rows[0]["test_per_station"]
for r in rows:
    d = np.array([r["test_per_station"][v] - ref[v] for v in stations])
    d = d[np.isfinite(d)]
    r["mean_diff_vs_frozen"] = float(d.mean()); r["frac_stations_worse"] = float((d > 0).mean())
    idx = rng.integers(len(d), size=(2000, len(d))); r["diff_ci"] = [float(x) for x in np.percentile(d[idx].mean(1), [2.5, 97.5])]
    del r["test_per_station"]
tab = pd.DataFrame(rows); tab.to_csv(os.path.join(F.out, "ablation.csv"), index=False)
save_json(dict(region=args.region, n_stations_scored=len(stations), test_days=len(blocks), frozen=c, table=tab.round(4).to_dict(orient="records"),
               seconds=round(time.time() - t0)), os.path.join(F.out, "ablation.json"))
print(tab[["variant", "val_rmse", "test_rmse", "mean_diff_vs_frozen"]].round(4).to_string(index=False))
