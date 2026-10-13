                                                                         
                                                                    
                                                                      
                                                                   
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from sheafcode.large import (ROOT, POLS, load_national, build_large, learn_sheaf_nan, station_precision_nan,              
                             NanReconstructor, ScalarNan, idw_nan, LowRankNan, sample_patterns, day_blocks_large, rmse_masked)
from sheafcode.data import haversine_km              
from sheafcode.graph import knn_edges              

ap = argparse.ArgumentParser()
ap.add_argument("--cities", default="Delhi")                                
ap.add_argument("--tag", default="delhi")
ap.add_argument("--per_size", type=int, default=40)
ap.add_argument("--sizes", default="2,3,5,8")
ap.add_argument("--max_test_days", type=int, default=60)
ap.add_argument("--val_days", type=int, default=20)
args = ap.parse_args()
out = os.path.join(ROOT, "results", args.tag); os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
t0 = time.time()

cities = None if args.cities == "ALL" else args.cities.split(",")
X, times, meta = load_national(cities)
ds = build_large(X, times, meta, POLS)
n, p = ds.n, ds.p
D = haversine_km(ds.meta)
print(f"[{args.tag}] stations={n} pollutants={p} hours={len(ds.times)} span={ds.times.min()}..{ds.times.max()} "
      f"cells observed={np.isfinite(ds.X).mean():.2f} train/val/test={np.bincount(ds.split).tolist()}  ({time.time()-t0:.0f}s)")
Ztr, Zva, Zte = ds.Z(0), ds.Z(1), ds.Z(2)


def mean_rmse(model, Zb, pats):
    errs = []
    for E in pats:
        pred = model.reconstruct_block(Zb, E)
        e, m = rmse_masked(pred, Zb[:, E, :])
        if m > 0:
            errs.append(e)
    return float(np.mean(errs))


                                                                                   
va_rows = ds.rows(1)
va_blocks = day_blocks_large(ds.times, va_rows)
va_sel = [va_blocks[i] for i in rng.choice(len(va_blocks), min(args.val_days, len(va_blocks)), replace=False)]
va_idx = np.concatenate([np.searchsorted(va_rows, idx) for _, idx in va_sel])
Zv = Zva[va_idx]
va_pats = [[v] for v in rng.choice(n, min(n, 25), replace=False)]
vlog = []; best = (np.inf, None)
for k in (3, 4, 6):
    edges = knn_edges(D, k)
    for r in (2, 4):
        for w in ("invvar", "reg"):
            sh = learn_sheaf_nan(Ztr, edges, r, w)
            P = station_precision_nan(Ztr)
            for lam in (0.3, 1.0, 3.0):
                for prior in ("identity", "station_cov"):
                    rec = NanReconstructor(sh, lam, P if prior == "station_cov" else None)
                    s = mean_rmse(rec, Zv, va_pats)
                    vlog.append(dict(method="vector_sheaf", k=k, r=r, weighting=w, lam=lam, prior=prior, val_rmse_z=s))
                    if s < best[0]:
                        best = (s, dict(k=k, r=r, weighting=w, lam=lam, prior=prior))
choice = {"vector_sheaf": best[1]}
best = (np.inf, None)
for k in (3, 4, 6):
    edges = knn_edges(D, k)
    for w in ("invvar", "reg"):
        for lam in (0.3, 1.0, 3.0):
            s = mean_rmse(ScalarNan(Ztr, edges, w, lam), Zv, va_pats)
            vlog.append(dict(method="scalar_sheaf", k=k, weighting=w, lam=lam, val_rmse_z=s))
            if s < best[0]:
                best = (s, dict(k=k, weighting=w, lam=lam))
choice["scalar_sheaf"] = best[1]
best = (np.inf, None)
for kk in (2, 4, 8, 16, 32):
    if kk >= n * p:
        continue
    s = mean_rmse(LowRankNan(Ztr, kk), Zv, va_pats)
    vlog.append(dict(method="lowrank", k=kk, val_rmse_z=s))
    if s < best[0]:
        best = (s, dict(k=kk))
choice["lowrank"] = best[1]


class IDW:
    def __init__(self, pw): self.pw = pw

    def reconstruct_block(self, Zb, E): return idw_nan(Zb, D, E, self.pw)


best = (np.inf, None)
for pw in (1.0, 2.0):
    s = mean_rmse(IDW(pw), Zv, va_pats)
    vlog.append(dict(method="idw", power=pw, val_rmse_z=s))
    if s < best[0]:
        best = (s, dict(power=pw))
choice["idw"] = best[1]
pd.DataFrame(vlog).to_csv(os.path.join(out, "validation_log.csv"), index=False)
print(f"[{args.tag}] chosen {choice}  ({time.time()-t0:.0f}s)")

                                                                                    
cv = choice["vector_sheaf"]
edges = knn_edges(D, cv["k"])
sh = learn_sheaf_nan(Ztr, edges, cv["r"], cv["weighting"])
P = station_precision_nan(Ztr)
vec = NanReconstructor(sh, cv["lam"], P if cv["prior"] == "station_cov" else None)
cs = choice["scalar_sheaf"]
sca = ScalarNan(Ztr, knn_edges(D, cs["k"]), cs["weighting"], cs["lam"])
models = {"vector_sheaf": vec, "scalar_sheaf": sca, "lowrank": LowRankNan(Ztr, choice["lowrank"]["k"]), "idw": IDW(choice["idw"]["power"])}
diag = dict(stations=list(ds.meta.index), cities=ds.meta.city.tolist(), n=n, p=p, choice=choice, n_edges=len(sh.edges),
            delta_shape=list(sh.delta.shape), kernel_dim=sh.kernel_dimension(), hours=len(ds.times),
            train_span=[str(ds.times[ds.split == 0].min()), str(ds.times[ds.split == 0].max())],
            test_span=[str(ds.times[ds.split == 2].min()), str(ds.times[ds.split == 2].max())],
            cells_observed=float(np.isfinite(ds.X).mean()), coords=ds.meta.reset_index().to_dict(orient="records"))
json.dump(diag, open(os.path.join(out, "diagnostics.json"), "w"), indent=1, default=str)

                                                                                    
sizes = [int(s) for s in args.sizes.split(",")]
pats = sample_patterns(n, sizes, args.per_size, rng, D)
city_pats = {}
if ds.meta.city.nunique() > 1:
    for c, g in ds.meta.reset_index().groupby("city"):
        if len(g) >= 3:
            city_pats[c] = [i for i, s in enumerate(ds.meta.index) if s in set(g["name"])]
te_rows = ds.rows(2)
te_blocks = day_blocks_large(ds.times, te_rows)
te_sel = [te_blocks[i] for i in sorted(rng.choice(len(te_blocks), min(args.max_test_days, len(te_blocks)), replace=False))]
print(f"[{args.tag}] patterns={len(pats)} (+{len(city_pats)} city-wide), test days={len(te_sel)}  ({time.time()-t0:.0f}s)")

rows = []
for pi, E in enumerate(pats + list(city_pats.values())):
    name = "+".join(ds.meta.index[v] for v in E) if E not in city_pats.values() else "CITY:" + [c for c, v in city_pats.items() if v == E][0]
    cert = sh.certificate(E)
    ecols = sh.cols(E)
    for day, idx in te_sel:
        Zb = Zte[np.searchsorted(te_rows, idx)]
        truth = Zb[:, E, :]
        if not np.isfinite(truth).any():
            continue
        real_smin = float(np.mean([vec.realised_sigma_min(Zb[t].reshape(-1), ecols) for t in range(0, len(Zb), max(1, len(Zb) // 4))]))
        for mname, m in models.items():
            pred = m.reconstruct_block(Zb, E)
            e, cnt = rmse_masked(pred, truth)
            rec = dict(method=mname, pattern=name, size=len(E), city_wide=int(name.startswith("CITY:")), day=day, n_cells=cnt, rmse_z=e,
                       rank=cert["rank"], needed=cert["needed"], identifiable=cert["identifiable"], sigma_min=cert["sigma_min"],
                       realised_sigma_min=real_smin)
            for q, pol in enumerate(POLS):
                eq, cq = rmse_masked(pred[:, :, q], truth[:, :, q])
                rec[f"rmse_z_{pol}"] = eq
            rows.append(rec)
    if pi % 25 == 0:
        print(f"  pattern {pi}/{len(pats)} ({time.time()-t0:.0f}s)")
res = pd.DataFrame(rows)
res.to_csv(os.path.join(out, "test_results.csv"), index=False)

                                                                                  
pl = res.groupby(["method", "pattern"]).agg(rmse_z=("rmse_z", "mean"), size=("size", "first"), identifiable=("identifiable", "first"),
                                              sigma_min=("sigma_min", "first"), realised=("realised_sigma_min", "mean"), city=("city_wide", "first")).reset_index()
summ = pl[pl.city == 0].groupby(["method", "size"]).rmse_z.mean().unstack("size").round(3)
print(summ.to_string())
pv = pl[(pl.method == "vector_sheaf") & (pl.city == 0)].set_index("pattern")
stats = {"choice": choice, "n_patterns": int(pv.shape[0]), "n_test_days": len(te_sel), "kernel_dim": diag["kernel_dim"],
         "table": summ.to_dict(), "frac_identifiable": float(pv.identifiable.mean())}
for m in ("scalar_sheaf", "lowrank", "idw"):
    pm = pl[(pl.method == m) & (pl.city == 0)].set_index("pattern").loc[pv.index]
    d = pm.rmse_z - pv.rmse_z
    stats[f"paired_{m}"] = dict(mean_diff=float(d.mean()), frac_vector_better=float((d > 0).mean()), wilcoxon_p=float(wilcoxon(d).pvalue))
g = pv[pv.identifiable]
stats["h3_structural_all"] = dict(spearman=float(spearmanr(1 / g.sigma_min, g.rmse_z).statistic), n=len(g))
stats["h3_realised_all"] = dict(spearman=float(spearmanr(1 / g.realised, g.rmse_z).statistic), n=len(g))
stats["h3_by_size"] = {int(s): dict(structural=float(spearmanr(1 / gg.sigma_min, gg.rmse_z).statistic), realised=float(spearmanr(1 / gg.realised, gg.rmse_z).statistic), n=len(gg))
                       for s, gg in g.groupby("size") if len(gg) >= 8}
if city_pats:
    cw = pl[pl.city == 1]
    stats["city_wide"] = cw.pivot(index="pattern", columns="method", values="rmse_z").round(3).to_dict(orient="index")
    stats["city_wide_cert"] = pv_all = pl[(pl.method == "vector_sheaf") & (pl.city == 1)][["pattern", "size", "identifiable", "sigma_min"]].to_dict(orient="records")
    stats["single_station_by_city"] = res[(res.method == "vector_sheaf") & (res["size"] == 1)].assign(city=lambda d: d.pattern.map(dict(zip(ds.meta.index, ds.meta.city)))).groupby("city").rmse_z.mean().round(3).to_dict()
                                     
ps = pl[(pl.method == "scalar_sheaf") & (pl.city == 0)].set_index("pattern").loc[pv.index]
per_pol = res.groupby(["method", "pattern"])[[f"rmse_z_{q}" for q in POLS]].mean()
stats["h1_per_pollutant"] = {q: dict(vector=float(per_pol.loc["vector_sheaf"][f"rmse_z_{q}"].mean()), scalar=float(per_pol.loc["scalar_sheaf"][f"rmse_z_{q}"].mean())) for q in POLS}

                                                                                  
fl = []
nbrs = {v: [] for v in range(n)}
for (a, b) in sh.edges:
    nbrs[a].append(b); nbrs[b].append(a)
fault_days = te_sel[:min(12, len(te_sel))]
fault_stations = rng.choice(n, min(n, 15), replace=False)
for day, idx in fault_days:
    Zb = Zte[np.searchsorted(te_rows, idx)]
    for v in fault_stations:
        if not np.isfinite(Zb[:, v, :]).all(axis=1).any():
            continue
        for mag in (0.5, 1.0, 2.0):
            Y = Zb.copy(); Y[:, v, :] += mag
            syn = np.zeros(n); mag_s = np.nanmean(Y ** 2, axis=(0, 2)); nb = np.zeros(n)
            for t in range(Y.shape[0]):
                z = Y[t].reshape(-1)
                zf = np.where(np.isfinite(z), z, 0.0)
                s = sh.delta @ zf
                                                                          
                bad = ~np.isfinite(z)
                for (a, b), sl in zip(sh.edges, sh.edge_rows):
                    if bad[a * p:(a + 1) * p].any() or bad[b * p:(b + 1) * p].any():
                        s[sl] = 0
                    else:
                        e = float(np.sum(s[sl] ** 2)); syn[a] += e; syn[b] += e
            for u in range(n):
                if nbrs[u]:
                    nb[u] = np.nanmean((Y[:, u, :] - np.nanmean(Y[:, nbrs[u], :], axis=1)) ** 2)
            for det, sc in (("syndrome", syn), ("magnitude", np.nan_to_num(mag_s)), ("neighbour_diff", np.nan_to_num(nb))):
                fl.append(dict(day=day, station=ds.meta.index[v], magnitude=mag, detector=det, top1=int(np.argmax(sc) == v)))
fl = pd.DataFrame(fl); fl.to_csv(os.path.join(out, "fault_results.csv"), index=False)
stats["fault_top1"] = fl.groupby(["magnitude", "detector"]).top1.mean().unstack().round(3).to_dict(orient="index")
stats["fault_chance"] = 1.0 / n
json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in stats.items() if k not in ("table", "city_wide_cert")}, indent=1, default=str))
print(f"[{args.tag}] done in {time.time()-t0:.0f}s")
