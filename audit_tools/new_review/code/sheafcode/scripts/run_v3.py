                                                                   
                                                                        
                                                           

                                                                  
                                                                   
                                                                       
                                                            
                   
import argparse
import hashlib
import itertools
import json
import os
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from sheafcode.india import select_region, ROOT              
from sheafcode.masked import MaskedSheaf              
from sheafcode.baselines2 import IDW2, ScalarLaplacian2, Kriging2, LowRank2, GaussianConditional              
from sheafcode.kalman import KalmanVAR              
from sheafcode.graph import knn_edges              
from sheafcode.data import haversine_km              
from sheafcode.experiments import save_json              

ap = argparse.ArgumentParser()
ap.add_argument("--region", default="delhi")
ap.add_argument("--max_days", type=int, default=80)
ap.add_argument("--n_random", type=int, default=40)
ap.add_argument("--ks", default="3,5,8,12")
ap.add_argument("--no_gnn", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default=None)
ap.add_argument("--fast_grid", action="store_true", help="reduced validation grid (automatic for n > 40 stations)")
args = ap.parse_args()
t0 = time.time()
tag = args.tag or f"v3_{args.region}"
out = os.path.join(ROOT, "results", tag)
if os.path.exists(os.path.join(out, "test_results.csv")):
    i = 2
    while os.path.exists(os.path.join(ROOT, "results", f"{tag}_run{i}")):
        i += 1
    out = os.path.join(ROOT, "results", f"{tag}_run{i}")
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(args.seed)
LOCKDOWN = pd.Timestamp("2020-03-25")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def corr_distance(Ztr):
    T, n, p = Ztr.shape
    C = np.zeros((n, n))
    for q in range(p):
        X = np.ma.masked_invalid(Ztr[:, :, q])
        C += np.abs(np.ma.corrcoef(X, rowvar=False).filled(0.0))
    C /= p
    Dc = 1.0 - C
    np.fill_diagonal(Dc, 0.0)
    return Dc


def rmse_masked(pred, truth):
    m = np.isfinite(truth)
    return float(np.sqrt(np.mean((pred[m] - truth[m]) ** 2))) if m.sum() else np.nan


def contiguous_runs(times_, min_len):
    runs_, st_ = [], 0
    for i in range(1, len(times_) + 1):
        if i == len(times_) or (times_[i] - times_[i - 1]) != pd.Timedelta(hours=1):
            if i - st_ >= min_len:
                runs_.append((st_, i))
            st_ = i
    return runs_


                                                                   
ds = select_region(args.region)
n, p = ds.n, ds.p
D = haversine_km(ds.coords)
KS = [int(k) for k in args.ks.split(",") if int(k) < n]
transforms = ["none", "log", "diurnal"]
Zs, MEANSTD = {}, {}
for tr in transforms:
    ds.set_transform(tr); Zs[tr] = ds.Z(); MEANSTD[tr] = (ds.mean.copy(), ds.std.copy(), None if ds.hod is None else ds.hod.copy())
print(f"[{args.region}] n={n} p={ds.pollutants} T={len(ds.times)} split={np.bincount(ds.split).tolist()} obs={np.isfinite(ds.X).mean():.3f}", flush=True)


def to_raw(tr, Zsub, rows, stations):
    mean, std, hod = MEANSTD[tr]
    Y = Zsub * std[stations] + mean[stations]
    if hod is not None:
        Y = Y + hod[ds.times[rows].hour.to_numpy()][:, stations, :]
    return np.expm1(Y) if tr in ("log", "diurnal") else Y


def to_common(tr, Zsub, rows, stations):
                                                                          
                                                                         
                                                                         
                                                                      
    if tr == "none":
        return Zsub
    mean0, std0, _ = MEANSTD["none"]
    return (to_raw(tr, Zsub, rows, stations) - mean0[stations]) / std0[stations]


                                                                                        
FAST = args.fast_grid or n > 40
args.fast_grid = FAST
val_rows = ds.rows(1); val_sub = val_rows[::(6 if FAST else 3)]
val_stations = list(range(n)) if n <= 40 else sorted(rng.choice(n, 40 if not FAST else 25, replace=False).tolist())
GRID = dict(transforms=["none", "log"] if FAST else ["none", "log", "diurnal"], rs=lambda p: [p] if FAST else sorted({min(2, p), p}),
            weightings=["reg"] if FAST else ["reg", "invvar"], stalks=["cca+id"] if FAST else ["cca", "cca+id"], lams=[1.0, 3.0] if FAST else [0.3, 1.0, 3.0])
if FAST:
    KS = [k for k in KS if k in (5, 8)] or KS
    print(f"[{args.region}] reduced validation grid (n={n} > 40): {GRID}", flush=True)
DC = corr_distance(Zs["none"][ds.split == 0])


def val_score(fn, Z, tr):
    Zv = Z[val_sub]; Z0 = Zs["none"][val_sub]
    errs = [rmse_masked(to_common(tr, fn(Zv, [v]), val_sub, [v])[:, 0, :], Z0[:, v, :]) for v in val_stations]
    return float(np.nanmean(errs))


choice, vlog = {}, []
best = (np.inf, None)
for tr in GRID["transforms"]:
    Ztr = Zs[tr][ds.split == 0]
    for graph in ("geo", "corr"):
        for k in KS:
            edges = knn_edges(D if graph == "geo" else DC, k)
            for r in GRID["rs"](p):
                for w in GRID["weightings"]:
                    for stalk in GRID["stalks"]:
                        sh = MaskedSheaf(Ztr, edges, r, w, stalk=stalk)
                        for lam in GRID["lams"]:
                            s = val_score(lambda Zb, E: sh.reconstruct(Zb, E, lam), Zs[tr], tr)
                            vlog.append(dict(method="vector_sheaf", transform=tr, graph=graph, k=k, r=r, weighting=w, stalk=stalk, lam=lam, val=s))
                            if s < best[0]:
                                best = (s, dict(transform=tr, graph=graph, k=k, r=r, weighting=w, stalk=stalk, lam=lam))
choice["vector_sheaf"] = best[1]
print(f"[{args.region}] vector sheaf {best} ({time.time() - t0:.0f}s)", flush=True)
best = (np.inf, None)
for tr in GRID["transforms"]:
    Ztr = Zs[tr][ds.split == 0]
    for k in KS:
        edges = knn_edges(D, k)
        shs = [MaskedSheaf(Ztr[:, :, [q]], edges, 1, "reg") for q in range(p)]
        for lam in GRID["lams"]:
            fn = lambda Zb, E, shs=shs, lam=lam: np.stack([shs[q].reconstruct(Zb[:, :, [q]], E, lam)[:, :, 0] for q in range(p)], -1)
            s = val_score(fn, Zs[tr], tr); vlog.append(dict(method="scalar_sheaf", transform=tr, k=k, lam=lam, val=s))
            if s < best[0]:
                best = (s, dict(transform=tr, k=k, lam=lam))
choice["scalar_sheaf"] = best[1]
for name in ["idw", "scalar_laplacian", "kriging", "lowrank", "gaussian_cond"]:
    best = (np.inf, None)
    for tr in GRID["transforms"]:
        Z = Zs[tr]; Ztr = Z[ds.split == 0]
        grid = {"idw": [dict(power=pw) for pw in (1.0, 2.0)], "scalar_laplacian": [dict(k=k) for k in KS], "kriging": [dict()],
                "lowrank": [dict(k=kk) for kk in (2, 4, 8, 16, 32) if kk < n * p // 2], "gaussian_cond": [dict(shrink=sr) for sr in (0.05, 0.2, 0.5)]}[name]
        for g in grid:
            m = {"idw": lambda: IDW2(D, g["power"]), "scalar_laplacian": lambda: ScalarLaplacian2(D, knn_edges(D, g["k"])),
                 "kriging": lambda: Kriging2(Ztr, D), "lowrank": lambda: LowRank2(Ztr, g["k"]), "gaussian_cond": lambda: GaussianConditional(Ztr, g["shrink"])}[name]()
            s = val_score(m.reconstruct, Z, tr); vlog.append(dict(method=name, transform=tr, **g, val=s))
            if s < best[0]:
                best = (s, dict(transform=tr, **g))
    choice[name] = best[1]
print(f"[{args.region}] choices {choice} ({time.time() - t0:.0f}s)", flush=True)

             
c = choice["vector_sheaf"]; Ztr_v = Zs[c["transform"]][ds.split == 0]
edges_v = knn_edges(D if c["graph"] == "geo" else DC, c["k"])
sheaf = MaskedSheaf(Ztr_v, edges_v, c["r"], c["weighting"], stalk=c["stalk"])
sheaf_id = MaskedSheaf(Ztr_v, edges_v, c["r"], c["weighting"], stalk=c["stalk"])                                     
cs = choice["scalar_sheaf"]; Ztr_s = Zs[cs["transform"]][ds.split == 0]
shs = [MaskedSheaf(Ztr_s[:, :, [q]], knn_edges(D, cs["k"]), 1, "reg") for q in range(p)]
models = {
    "vector_sheaf": (c["transform"], lambda Zb, E: sheaf.reconstruct(Zb, E, c["lam"])),
    "scalar_sheaf": (cs["transform"], lambda Zb, E: np.stack([shs[q].reconstruct(Zb[:, :, [q]], E, cs["lam"])[:, :, 0] for q in range(p)], -1)),
    "idw": (choice["idw"]["transform"], IDW2(D, choice["idw"]["power"]).reconstruct),
    "scalar_laplacian": (choice["scalar_laplacian"]["transform"], ScalarLaplacian2(D, knn_edges(D, choice["scalar_laplacian"]["k"])).reconstruct),
    "kriging": (choice["kriging"]["transform"], Kriging2(Zs[choice["kriging"]["transform"]][ds.split == 0], D).reconstruct),
    "lowrank": (choice["lowrank"]["transform"], LowRank2(Zs[choice["lowrank"]["transform"]][ds.split == 0], choice["lowrank"]["k"]).reconstruct),
    "gaussian_cond": (choice["gaussian_cond"]["transform"], GaussianConditional(Zs[choice["gaussian_cond"]["transform"]][ds.split == 0], choice["gaussian_cond"]["shrink"]).reconstruct),
}
                                                                                
Zv_ = Zs[c["transform"]][ds.split == 1]; times_v = ds.times[ds.split == 1]
vruns = contiguous_runs(times_v, 49)[:(12 if n > 100 else 60)]
vstations = list(range(n)) if n <= 20 else sorted(rng.choice(n, 8 if n > 100 else 20, replace=False).tolist())
HIST = 24; HB = [0, 3, 6, 12, 24, 1000]
best_t = (np.inf, None); hyb_num = np.zeros(len(HB) - 1); hyb_den = np.zeros(len(HB) - 1)
for ts in ((2.0, 8.0) if n > 100 else (1.0, 2.0, 4.0, 8.0)):
    sh_t = MaskedSheaf(Ztr_v, edges_v, c["r"], c["weighting"], temporal=True, t_scale=ts, stalk=c["stalk"])
    errs = []
    for (a, b) in vruns:
        L = min(b - a - HIST, 24)
        for v in vstations:
            blk = Zv_[a:a + HIST + L]
            if not np.isfinite(blk[:, v, :]).all():
                continue
            hide = np.zeros((HIST + L, n), bool); hide[HIST:, v] = True
            st = sh_t.reconstruct_mask(blk, hide, c["lam"])[HIST:, v, :]
            errs.append(np.mean((st - blk[HIST:, v, :]) ** 2))
            if ts == (2.0 if n > 100 else 1.0):
                sp_ = sheaf.reconstruct(blk[HIST:], [v], c["lam"])[:, 0, :]; pers = np.repeat(blk[HIST - 1:HIST, v, :], L, axis=0)
                for h in range(L):
                    bi = np.searchsorted(HB, h + 1, side="right") - 1
                    d_ = pers[h] - sp_[h]; y_ = blk[HIST + h, v, :] - sp_[h]
                    hyb_num[bi] += float(d_ @ y_); hyb_den[bi] += float(d_ @ d_)
    m_ = float(np.sqrt(np.mean(errs))) if errs else np.inf
    vlog.append(dict(method="st_sheaf", t_scale=ts, val=m_))
    if m_ < best_t[0]:
        best_t = (m_, ts)
hyb_w = np.clip(hyb_num / np.maximum(hyb_den, 1e-9), 0, 1)
choice["st_sheaf"] = dict(t_scale=best_t[1], val=best_t[0]); choice["hybrid_weights"] = hyb_w.tolist()
sheaf_t = MaskedSheaf(Ztr_v, edges_v, c["r"], c["weighting"], temporal=True, t_scale=best_t[1], stalk=c["stalk"])
models["st_sheaf"] = (c["transform"], lambda Zb, E: sheaf_t.reconstruct(Zb, E, c["lam"]))
kalman = KalmanVAR(Ztr_v)
if not args.no_gnn:
    from sheafcode.gnn import GNNModel
    tg = c["transform"]; Zg = Zs[tg]
    gnn = GNNModel(Zg[ds.split == 0], Zg[ds.split == 1], edges_v, seed=args.seed)
    models["gnn"] = (tg, gnn.reconstruct); choice["gnn"] = dict(transform=tg, val=gnn.val_rmse, edges_k=c["k"])
pd.DataFrame(vlog).to_csv(os.path.join(out, "validation_log.csv"), index=False)
print(f"[{args.region}] frozen; st t_scale={best_t[1]} hybrid={np.round(hyb_w, 2)} ({time.time() - t0:.0f}s)", flush=True)

               
code_files = sorted([os.path.join(ROOT, "sheafcode", f) for f in os.listdir(os.path.join(ROOT, "sheafcode")) if f.endswith(".py")]) + [os.path.abspath(__file__)]
data_dir = {"delhi2025": "delhi_2025_openaq_cpcb", "bengaluru_raw": "bengaluru_4station_cpcb_raw"}.get(args.region, "india_cpcb_subindex")
data_files = sorted(os.path.join(ROOT, "data", "raw", data_dir, f) for f in os.listdir(os.path.join(ROOT, "data", "raw", data_dir)))
import scipy, sklearn, matplotlib        
manifest = dict(region=args.region, protocol="FINAL_PROTOCOL.md 2026-09-30", seed=args.seed, started=time.strftime("%Y-%m-%dT%H:%M:%S"),
                code_sha256={os.path.relpath(f, ROOT): sha(f) for f in code_files}, data_sha256={os.path.relpath(f, ROOT): sha(f) for f in data_files},
                python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                stations=ds.stations, names=ds.meta.loc[ds.stations, "name"].tolist(), pollutants=ds.pollutants,
                split_rows=np.bincount(ds.split).tolist(), train_span=[str(ds.times[ds.split == 0].min()), str(ds.times[ds.split == 0].max())],
                val_span=[str(ds.times[ds.split == 1].min()), str(ds.times[ds.split == 1].max())],
                test_span=[str(ds.times[ds.split == 2].min()), str(ds.times[ds.split == 2].max())],
                edges=[(ds.stations[u], ds.stations[v]) for u, v in edges_v], choice=choice, delta_shape=list(sheaf.delta.shape),
                kernel_dim=sheaf.kernel_dimension(), args=vars(args))
save_json(manifest, os.path.join(out, "manifest.json"))
save_json(dict(stations=ds.stations, names=manifest["names"], pollutants=ds.pollutants, n_rows=len(ds.times), split=manifest["split_rows"],
               obs_fraction=float(np.isfinite(ds.X).mean()), span=[str(ds.times.min()), str(ds.times.max())], test_span=manifest["test_span"],
               choice=choice, edges=manifest["edges"], delta_shape=list(sheaf.delta.shape), kernel_dim=sheaf.kernel_dimension(),
               canonical_correlations=[list(map(float, r)) for r in sheaf.rho],
               temporal_correlations={str(l): [list(map(float, t[3])) for t in m] for l, m in sheaf_t.Tmaps.items()},
               coords=ds.coords.reset_index().to_dict(orient="records")), os.path.join(out, "diagnostics.json"))

                                                                                                 
single_stations = list(range(n)) if not FAST else sorted(rng.choice(n, min(80, n), replace=False).tolist())
if FAST:
    args.max_days = min(args.max_days, 30); args.n_random = min(args.n_random, 20)
pats = [[v] for v in single_stations]
for size in (2, 3, 5, 8, 12):
    if size > n - 3:
        continue
    seen = set(); tries = 0
    while len(seen) < min(args.n_random, 200) and tries < 5000:
        tries += 1; seen.add(tuple(sorted(rng.choice(n, size, replace=False).tolist())))
    pats += [list(s) for s in sorted(seen)]
for v in single_stations:
    cp = sorted(int(x) for x in np.argsort(D[v])[:3])
    if cp not in pats and len(cp) <= n - 3:
        pats.append(cp)
te_rows = ds.rows(2); days = ds.times[te_rows].floor("D")
blocks = [(pd.Timestamp(d), te_rows[days == d]) for d in pd.unique(days) if (days == d).sum() >= 12]
if len(blocks) > args.max_days:
    sel = np.linspace(0, len(blocks) - 1, args.max_days).round().astype(int); blocks = [blocks[i] for i in sel]
Ztr_c = Zs[c["transform"]][ds.split == 0]
full_train = np.isfinite(Ztr_c).all(axis=(1, 2)).sum()
certs = {}
for E in pats:
    key = "+".join(ds.stations[v] for v in E)
    ce = sheaf.certificate(E, c["lam"])
    c2s, ginv = sheaf.c2star(E, c["lam"])
    c3c = sheaf.predicted_rmse(E, c["lam"], Ztr_c) if full_train >= 200 else np.nan
    c3e = sheaf.predicted_rmse_empirical(E, c["lam"], Ztr_c)
    certs[key] = dict(rank=ce["rank"], needed=ce["needed"], identifiable=ce["identifiable"], sigma_min=ce["sigma_min"], c2_star=c2s, g_inv_norm=ginv,
                      c3_complete=c3c, c3_empirical=c3e, size=len(E))
ranked = sorted(certs, key=lambda k: certs[k]["c2_star"])
dec = max(1, len(ranked) // 10)
save_json(dict(written_before_test=time.strftime("%Y-%m-%dT%H:%M:%S"), ranking_metric="c2_star (training model only)",
               safest_decile=ranked[:dec], riskiest_decile=ranked[-dec:], all_ranked=ranked,
               c2_star={k: certs[k]["c2_star"] for k in ranked}), os.path.join(out, "prospective_deciles.json"))
print(f"[{args.region}] {len(pats)} patterns x {len(blocks)} days; deciles pre-registered; full training rows={full_train} ({time.time() - t0:.0f}s)", flush=True)

                                                                                                      
rows = []
for pi, E in enumerate(pats):
    key = "+".join(ds.stations[v] for v in E)
    for day, idx in blocks:
        truth_raw = ds.X[idx][:, E, :]
        if np.isfinite(truth_raw).all(axis=(1, 2)).sum() < 6:
            continue
        for name, (tr, fn) in models.items():
            Zb = Zs[tr][idx]
            pred = fn(Zb, E)                                                                                            
            truth0 = Zs["none"][idx][:, E, :]
            rec = dict(method=name, pattern=key, size=len(E), day=day, lockdown=int(day >= LOCKDOWN), n_hours=len(idx),
                       rmse_z=rmse_masked(to_common(tr, pred, idx, E), truth0), rmse_own_transform=rmse_masked(pred, Zb[:, E, :]))
            pr = to_raw(tr, pred, idx, E)
            for q, pol in enumerate(ds.pollutants):
                rec[f"rmse_{pol}"] = rmse_masked(pr[:, :, q], truth_raw[:, :, q])
            rec.update(certs[key])
            rows.append(rec)
    if pi % 25 == 0:
        print(f"[{args.region}] pattern {pi}/{len(pats)} ({time.time() - t0:.0f}s)", flush=True)
res = pd.DataFrame(rows); res.to_csv(os.path.join(out, "test_results.csv"), index=False)
print(res.groupby(["method", "size"]).rmse_z.mean().unstack().round(3).to_string(), flush=True)

                                              
Zt = Zs[c["transform"]][ds.split == 2]
fullrows = np.where(np.isfinite(Zt).all(axis=(1, 2)))[0][:300]
bounds = []
for v in range(min(n, 40)):
    for t in fullrows[::10]:
        pred = sheaf.reconstruct(Zt[t:t + 1], [v], c["lam"])[0, 0, :]
        bounds.append(dict(station=ds.stations[v], row=int(t), error=float(np.linalg.norm(pred - Zt[t, v, :])), bound=sheaf.general_bound([v], c["lam"], Zt[t])))
pd.DataFrame(bounds).to_csv(os.path.join(out, "bound_check.csv"), index=False)

                                                                                                          
chan = []
stations_c = list(range(n)) if n <= 40 else sorted(rng.choice(n, 40, replace=False).tolist())
pm10_idx = ds.pollutants.index("PM10") if "PM10" in ds.pollutants else None
lr_m = models["lowrank"][1].__self__; gc_m = models["gaussian_cond"][1].__self__
for v in stations_c:
    for q in range(p):
        for day, idx in blocks[::max(1, len(blocks) // 40)]:
            variants = {}
                                            
            Zb = Zs[c["transform"]][idx]
            truth = Zb[:, v, q].copy()
            if np.isfinite(truth).sum() < 6:
                continue
            Zc = Zb.copy(); Zc[:, v, q] = np.nan
            hide0 = np.zeros((len(idx), n), bool)
            variants["vector_sheaf"] = sheaf.reconstruct_mask(Zc, hide0, c["lam"])[:, v, q]
            variants["vector_sheaf_Pidentity"] = sheaf.reconstruct_mask(Zc, hide0, c["lam"], prior="identity")[:, v, q]
            if pm10_idx is not None and q != pm10_idx:
                Z2 = Zc.copy(); Z2[:, v, pm10_idx] = np.nan
                variants["vector_sheaf_noPM10"] = sheaf.reconstruct_mask(Z2, hide0, c["lam"])[:, v, q]
            variants["vector_sheaf_ownhidden"] = sheaf.reconstruct(Zb, [v], c["lam"])[:, 0, q]                         
            variants["scalar_sheaf"] = shs[q].reconstruct(Zs[cs["transform"]][idx][:, :, [q]], [v], cs["lam"])[:, 0, 0]
            variants["kriging"] = models["kriging"][1](Zs[choice["kriging"]["transform"]][idx], [v])[:, 0, q]
            for name, m in (("gaussian_cond", gc_m), ("lowrank", lr_m)):
                Zx = Zs[choice[name]["transform"]][idx].copy(); Zx[:, v, q] = np.nan
                W_ = len(idx); M = Zx.reshape(W_, -1) - m.mu; col = v * p + q
                if name == "gaussian_cond":
                    preds = m.conditional_mean(M)[:, col] + m.mu[col]
                else:
                    preds = np.zeros(W_); keys = {}
                    for t in range(W_):
                        keys.setdefault(np.isfinite(M[t]).tobytes(), []).append(t)
                    for kk, ts in keys.items():
                        O = np.where(np.isfinite(M[ts[0]]))[0]
                        C_ = M[np.ix_(ts, O)] @ np.linalg.pinv(m.V[O]).T; preds[ts] = (C_ @ m.V[[col]].T)[:, 0] + m.mu[col]
                variants[name] = preds
            if "gnn" in models:
                import torch
                Zx = Zs[models["gnn"][0]][idx].copy(); Zx[:, v, q] = np.nan
                x = torch.tensor(np.nan_to_num(Zx), dtype=torch.float32); mm = torch.tensor(np.isfinite(Zx), dtype=torch.float32)
                with torch.no_grad():
                    variants["gnn"] = gnn.model(torch.cat([x * mm, mm], -1), gnn.A).numpy()[:, v, q]
            for name, ph in variants.items():
                tr_ = {"scalar_sheaf": cs["transform"], "kriging": choice["kriging"]["transform"], "gaussian_cond": choice["gaussian_cond"]["transform"],
                       "lowrank": choice["lowrank"]["transform"]}.get(name, c["transform"])
                full = np.full((len(idx), 1, p), np.nan); full[:, 0, q] = ph
                phc = to_common(tr_, full, idx, [v])[:, 0, q]
                chan.append(dict(method=name, station=ds.stations[v], pollutant=ds.pollutants[q], day=day, lockdown=int(day >= LOCKDOWN),
                                 rmse_z=rmse_masked(phc, Zs["none"][idx][:, v, q]), rmse_own_transform=rmse_masked(ph, Zs[tr_][idx][:, v, q])))
pd.DataFrame(chan).to_csv(os.path.join(out, "channel_results.csv"), index=False)
print(pd.DataFrame(chan).groupby(["method", "pollutant"]).rmse_z.mean().unstack().round(3).to_string(), flush=True)

                                                                                                  
pro = []
times_te = ds.times[te_rows]
runs = contiguous_runs(times_te, 73)[:(12 if n > 100 else 30)]
Ztp = Zs[c["transform"]][ds.split == 2]
abl_stations = stations_c if n <= 12 else sorted(rng.choice(stations_c, 12, replace=False).tolist())
lag_variants = {"st_lag1": MaskedSheaf(Ztr_v, edges_v, c["r"], c["weighting"], temporal=True, t_scale=best_t[1], stalk=c["stalk"], lags=(1,)),
                "st_lag24": MaskedSheaf(Ztr_v, edges_v, c["r"], c["weighting"], temporal=True, t_scale=best_t[1], stalk=c["stalk"], lags=(24,))}
for (a, b) in runs:
    L = min(b - a - HIST - 24, 72)
    if L < 24:
        continue
    for v in stations_c:
        blk = Ztp[a:a + 48 + L]                                                                                          
        if not np.isfinite(blk[24:, v, :]).all():
            continue
        truth = blk[48:, v, :]
        hide = np.zeros((48 + L, n), bool); hide[48:, v] = True
        win = blk[24:]; hide24 = hide[24:]
        st = sheaf_t.reconstruct_mask(win, hide24, c["lam"])[HIST:, v, :]
        sp_ = sheaf.reconstruct(win[HIST:], [v], c["lam"])[:, 0, :]
        pers = np.repeat(win[HIST - 1:HIST, v, :], L, axis=0)
        seas = np.stack([win[HIST + h - 24, v, :] if h < 24 else win[HIST + (h % 24) - 24, v, :] for h in range(L)])
        wk = win.copy(); wk[HIST:, v, :] = np.nan
        kal = kalman.smooth(wk)[HIST:, v, :]
        wh = np.array([hyb_w[np.searchsorted(HB, h + 1, side="right") - 1] for h in range(L)])[:, None]
        preds = {"st_sheaf": st, "vector_sheaf": sp_, "persistence": pers, "seasonal_persistence": seas, "kalman_var": kal, "hybrid": wh * pers + (1 - wh) * sp_}
        if "gnn" in models:
            preds["gnn"] = models["gnn"][1](win[HIST:], [v])[:, 0, :]
        if v in abl_stations:
            for nm, shv in lag_variants.items():
                preds[nm] = shv.reconstruct_mask(win, hide24, c["lam"])[HIST:, v, :]
            for H in (6, 12, 48):
                start = 48 - H
                w2 = blk[start:]; h2 = hide[start:]
                preds[f"st_hist{H}"] = sheaf_t.reconstruct_mask(w2, h2, c["lam"])[H:, v, :]
        rows_abs = te_rows[a + 48:a + 48 + L]
        truth0 = Zs["none"][rows_abs][:, v, :]
        for name, ph in preds.items():
            phc = to_common(c["transform"], ph[:, None, :], rows_abs, [v])[:, 0, :]
            for h in range(L):
                pro.append(dict(run=str(times_te[a + 48]), station=ds.stations[v], hours_elapsed=h + 1, method=name,
                                lockdown=int(times_te[a + 48] >= LOCKDOWN), sq_err=float(np.mean((phc[h] - truth0[h]) ** 2)),
                                sq_err_own_transform=float(np.mean((ph[h] - truth[h]) ** 2))))
pro = pd.DataFrame(pro); pro.to_csv(os.path.join(out, "prolonged_results.csv"), index=False)
if len(pro):
    print(pro.groupby(["method", pd.cut(pro.hours_elapsed, [0, 3, 6, 12, 24, 48, 72])]).sq_err.mean().pow(0.5).unstack().round(3).to_string(), flush=True)
manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S"); manifest["seconds"] = round(time.time() - t0)
save_json(manifest, os.path.join(out, "manifest.json"))
print(f"[{args.region}] done in {time.time() - t0:.0f}s -> {out}", flush=True)
