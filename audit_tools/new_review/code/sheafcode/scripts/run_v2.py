                                                                  

                                                                     
                                                                      
                                                                         
                                                                         
                                                                         
                                                 
import argparse
import itertools
import json
import os
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
from sheafcode.graph import knn_edges              


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
from sheafcode.data import haversine_km              
from sheafcode.experiments import save_json              

ap = argparse.ArgumentParser()
ap.add_argument("--region", default="delhi")
ap.add_argument("--max_days", type=int, default=120)
ap.add_argument("--n_random", type=int, default=60)
ap.add_argument("--no_gnn", action="store_true")
ap.add_argument("--ks", default="3,5,8")
args = ap.parse_args()
t0 = time.time()
out = os.path.join(ROOT, "results", f"v2_{args.region}")
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)

ds = select_region(args.region)
n, p = ds.n, ds.p
D = haversine_km(ds.coords)
print(f"[{args.region}] n={n} p={ds.pollutants} T={len(ds.times)} split={np.bincount(ds.split).tolist()}", flush=True)
KS = [int(k) for k in args.ks.split(",") if int(k) < n]


def rmse_masked(pred, truth):
    m = np.isfinite(truth)
    return float(np.sqrt(np.mean((pred[m] - truth[m]) ** 2))) if m.sum() else np.nan


                                                                             
val_rows = ds.rows(1)
val_sub = val_rows[::3]
val_stations = list(range(n)) if n <= 40 else sorted(rng.choice(n, 40, replace=False).tolist())


def val_score(model_fn, Z, tr=None):
                                                                        
                                                                    
                      
    Zv = VAL_FILLED[tr]; Zraw = Z[val_sub]
    errs = []
    for v in val_stations:
        pred = model_fn(Zv, [v])[:, 0, :]
        errs.append(rmse_masked(pred, Zraw[:, v, :]))
    return float(np.nanmean(errs))


choice, vlog = {}, []
transforms = ["none", "log", "diurnal"]
Zs, MEANSTD, VAL_FILLED = {}, {}, {}
edges_default = knn_edges(D, 5 if n > 5 else n - 1)
for tr in transforms:
    ds.set_transform(tr); Zs[tr] = ds.Z(); MEANSTD[tr] = (ds.mean.copy(), ds.std.copy(), None if ds.hod is None else ds.hod.copy())
    sh0 = MaskedSheaf(Zs[tr][ds.split == 0], edges_default, p, "reg")
    VAL_FILLED[tr] = sh0.reconstruct_mask(Zs[tr][val_sub], np.zeros((len(val_sub), n), bool), 1.0)
print(f"[{args.region}] validation rows filled ({time.time() - t0:.0f}s)", flush=True)


def to_raw(tr, Zsub, rows, stations):
    mean, std, hod = MEANSTD[tr]
    Y = Zsub * std[stations] + mean[stations]
    if hod is not None:
        Y = Y + hod[ds.times[rows].hour.to_numpy()][:, stations, :]
    return np.expm1(Y) if tr in ("log", "diurnal") else Y

best = (np.inf, None)
DC = corr_distance(Zs["none"][ds.split == 0])
for tr in transforms:
    Z = Zs[tr]; Ztr = Z[ds.split == 0]
    for graph in ("geo", "corr"):
        for k in KS:
            edges = knn_edges(D if graph == "geo" else DC, k)
            for r in sorted({min(2, p), p}):
                for w in ("reg", "invvar"):
                    for stalk in ("cca", "cca+id"):
                        sh = MaskedSheaf(Ztr, edges, r, w, stalk=stalk)
                        for lam in (0.3, 1.0, 3.0):
                            s = val_score(lambda Zb, E: sh.reconstruct(Zb, E, lam), Z, tr)
                            vlog.append(dict(method="vector_sheaf", transform=tr, graph=graph, k=k, r=r, weighting=w, stalk=stalk, lam=lam, val=s))
                            if s < best[0]:
                                best = (s, dict(transform=tr, graph=graph, k=k, r=r, weighting=w, stalk=stalk, lam=lam))
choice["vector_sheaf"] = best[1]
print(f"[{args.region}] vector sheaf: {best}", flush=True)
                                   
best = (np.inf, None)
for tr in transforms:
    Z = Zs[tr]; Ztr = Z[ds.split == 0]
    for k in KS:
        edges = knn_edges(D, k)
        shs = [MaskedSheaf(Ztr[:, :, [q]], edges, 1, "reg") for q in range(p)]
        for lam in (0.3, 1.0, 3.0):
            def fn(Zb, E, shs=shs, lam=lam):
                return np.stack([shs[q].reconstruct(Zb[:, :, [q]], E, lam)[:, :, 0] for q in range(p)], -1)
            s = val_score(fn, Z, tr)
            vlog.append(dict(method="scalar_sheaf", transform=tr, k=k, lam=lam, val=s))
            if s < best[0]:
                best = (s, dict(transform=tr, k=k, lam=lam))
choice["scalar_sheaf"] = best[1]
           
for name in ["idw", "scalar_laplacian", "kriging", "lowrank", "gaussian_cond"]:
    best = (np.inf, None)
    for tr in transforms:
        Z = Zs[tr]; Ztr = Z[ds.split == 0]
        if name == "idw":
            for pw in (1.0, 2.0):
                s = val_score(IDW2(D, pw).reconstruct, Z, tr); vlog.append(dict(method=name, transform=tr, power=pw, val=s))
                if s < best[0]:
                    best = (s, dict(transform=tr, power=pw))
        elif name == "scalar_laplacian":
            for k in KS:
                s = val_score(ScalarLaplacian2(D, knn_edges(D, k)).reconstruct, Z, tr); vlog.append(dict(method=name, transform=tr, k=k, val=s))
                if s < best[0]:
                    best = (s, dict(transform=tr, k=k))
        elif name == "kriging":
            s = val_score(Kriging2(Ztr, D).reconstruct, Z, tr); vlog.append(dict(method=name, transform=tr, val=s))
            if s < best[0]:
                best = (s, dict(transform=tr))
        elif name == "lowrank":
            for kk in (2, 4, 8, 16, 32):
                if kk >= n * p // 2:
                    continue
                s = val_score(LowRank2(Ztr, kk).reconstruct, Z, tr); vlog.append(dict(method=name, transform=tr, k=kk, val=s))
                if s < best[0]:
                    best = (s, dict(transform=tr, k=kk))
        elif name == "gaussian_cond":
            for shr in (0.05, 0.2, 0.5):
                s = val_score(GaussianConditional(Ztr, shr).reconstruct, Z, tr); vlog.append(dict(method=name, transform=tr, shrink=shr, val=s))
                if s < best[0]:
                    best = (s, dict(transform=tr, shrink=shr))
    choice[name] = best[1]
print(f"[{args.region}] choices: {choice}  ({time.time() - t0:.0f}s)", flush=True)

                                                                         
models = {}
c = choice["vector_sheaf"]; Ztr_v = Zs[c["transform"]][ds.split == 0]
edges_v = knn_edges(D if c["graph"] == "geo" else DC, c["k"])
sheaf = MaskedSheaf(Ztr_v, edges_v, c["r"], c["weighting"], stalk=c["stalk"])
models["vector_sheaf"] = (c["transform"], lambda Zb, E: sheaf.reconstruct(Zb, E, c["lam"]))
                                                                                               
def contiguous_runs(times_, min_len):
    runs_, st_ = [], 0
    for i in range(1, len(times_) + 1):
        if i == len(times_) or (times_[i] - times_[i - 1]) != pd.Timedelta(hours=1):
            if i - st_ >= min_len:
                runs_.append((st_, i))
            st_ = i
    return runs_


Zv_ = Zs[c["transform"]][ds.split == 1]; times_v = ds.times[ds.split == 1]
vruns = contiguous_runs(times_v, 49)[:60]
vstations = list(range(n)) if n <= 20 else sorted(rng.choice(n, 20, replace=False).tolist())
best_t = (np.inf, None); HB = [0, 3, 6, 12, 24, 1000]
hyb_num = np.zeros(len(HB) - 1); hyb_den = np.zeros(len(HB) - 1)
HIST = 24
for ts in (1.0, 2.0, 4.0, 8.0):
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
            if ts == 1.0:                                                                                      
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
print(f"[{args.region}] st_sheaf t_scale={best_t[1]} val={best_t[0]:.3f}; hybrid weights {np.round(hyb_w, 2)}", flush=True)
cs = choice["scalar_sheaf"]; Ztr_s = Zs[cs["transform"]][ds.split == 0]
shs = [MaskedSheaf(Ztr_s[:, :, [q]], knn_edges(D, cs["k"]), 1, "reg") for q in range(p)]
models["scalar_sheaf"] = (cs["transform"], lambda Zb, E: np.stack([shs[q].reconstruct(Zb[:, :, [q]], E, cs["lam"])[:, :, 0] for q in range(p)], -1))
ci = choice["idw"]; models["idw"] = (ci["transform"], IDW2(D, ci["power"]).reconstruct)
cl = choice["scalar_laplacian"]; models["scalar_laplacian"] = (cl["transform"], ScalarLaplacian2(D, knn_edges(D, cl["k"])).reconstruct)
ck = choice["kriging"]; models["kriging"] = (ck["transform"], Kriging2(Zs[ck["transform"]][ds.split == 0], D).reconstruct)
cr = choice["lowrank"]; models["lowrank"] = (cr["transform"], LowRank2(Zs[cr["transform"]][ds.split == 0], cr["k"]).reconstruct)
cg = choice["gaussian_cond"]; models["gaussian_cond"] = (cg["transform"], GaussianConditional(Zs[cg["transform"]][ds.split == 0], cg["shrink"]).reconstruct)
if not args.no_gnn:
    from sheafcode.gnn import GNNModel
    tg = c["transform"]; Zg = Zs[tg]
    gnn = GNNModel(Zg[ds.split == 0], Zg[ds.split == 1], edges_v)
    models["gnn"] = (tg, gnn.reconstruct)
    choice["gnn"] = dict(transform=tg, val=gnn.val_rmse, edges_k=c["k"])
    print(f"[{args.region}] gnn val {gnn.val_rmse:.3f} ({time.time() - t0:.0f}s)", flush=True)

pd.DataFrame(vlog).to_csv(os.path.join(out, "validation_log.csv"), index=False)
diag = dict(region=args.region, stations=ds.stations, names=ds.meta.loc[ds.stations, "name"].tolist(), pollutants=ds.pollutants,
            n_rows=len(ds.times), split=np.bincount(ds.split).tolist(), obs_fraction=float(np.isfinite(ds.X).mean()),
            span=[str(ds.times.min()), str(ds.times.max())], test_span=[str(ds.times[ds.split == 2].min()), str(ds.times[ds.split == 2].max())],
            choice=choice, edges=[(ds.stations[u], ds.stations[v]) for u, v in edges_v], delta_shape=list(sheaf.delta.shape),
            kernel_dim=sheaf.kernel_dimension(), canonical_correlations=[list(map(float, r)) for r in sheaf.rho],
            temporal_correlations={str(l): [list(map(float, t[3])) for t in m] for l, m in sheaf_t.Tmaps.items()})
save_json(diag, os.path.join(out, "diagnostics.json"))

                                                                                      
pats = [[v] for v in range(n)]
for size in (2, 3, 5, 8, 12):
    if size > n - 3:
        continue
    seen = set(); tries = 0
    while len([q for q in seen]) < min(args.n_random, 200) and tries < 5000:
        tries += 1
        seen.add(tuple(sorted(rng.choice(n, size, replace=False).tolist())))
    pats += [list(s) for s in sorted(seen)]
for v in range(n):
    cp = sorted(int(x) for x in np.argsort(D[v])[:3])
    if cp not in pats:
        pats.append(cp)
te_rows = ds.rows(2)
days = ds.times[te_rows].floor("D")
blocks = []
for d in pd.unique(days):
    idx = te_rows[days == d]
    if len(idx) >= 12:
        blocks.append((pd.Timestamp(d), idx))
if len(blocks) > args.max_days:
    sel = np.linspace(0, len(blocks) - 1, args.max_days).round().astype(int)
    blocks = [blocks[i] for i in sel]
print(f"[{args.region}] {len(pats)} patterns x {len(blocks)} test days", flush=True)

                                                                                      
                                                                           
                                                                          
                                                                        
                                        
def fill_with(name, Zb):
    tr, fn = models[name]
    if not np.isnan(Zb).any():
        return Zb
    if name in ("vector_sheaf", "st_sheaf"):
        return (sheaf_t if name == "st_sheaf" else sheaf).reconstruct_mask(Zb, np.zeros(Zb.shape[:2], bool), c["lam"])
    if name == "scalar_sheaf":
        return np.stack([shs[q].reconstruct_mask(Zb[:, :, [q]], np.zeros(Zb.shape[:2], bool), cs["lam"])[:, :, 0] for q in range(p)], -1)
    if name == "gnn":
        import torch
        x = torch.tensor(np.nan_to_num(Zb), dtype=torch.float32); m = torch.tensor(np.isfinite(Zb), dtype=torch.float32)
        with torch.no_grad():
            pred = gnn.model(torch.cat([x * m, m], -1), gnn.A).numpy()
        return np.where(np.isfinite(Zb), Zb, pred)
                                                                            
    Zf = Zb.copy()
    miss = np.where(np.isnan(Zb).any(axis=(0, 2)))[0]
    for v in miss:
        pred = fn(Zb, [v])[:, 0, :]
        Zf[:, v, :] = np.where(np.isfinite(Zb[:, v, :]), Zb[:, v, :], pred)
    return Zf


FILLED = {name: {} for name in models}
for name in models:
    for day, idx in blocks:
        FILLED[name][day] = fill_with(name, Zs[models[name][0]][idx])
    print(f"[{args.region}] pass 1 filled for {name} ({time.time() - t0:.0f}s)", flush=True)

rows = []
Zfull = {tr: Zs[tr] for tr in transforms}
Ztr_cert = Zs[c["transform"]][ds.split == 0]
for pi, E in enumerate(pats):
    cert = sheaf.certificate(E, c["lam"], Ztr_cert)
    for day, idx in blocks:
        truth_raw = ds.X[idx][:, E, :]
        if np.isfinite(truth_raw).all(axis=(1, 2)).sum() < 6:
            continue
        for name, (tr, fn) in models.items():
            Zb = FILLED[name][day]
            pred = fn(Zb, E)
            truth = Zfull[tr][idx][:, E, :]
            rec = dict(method=name, pattern="+".join(ds.stations[v] for v in E), size=len(E), day=day, n_hours=len(idx),
                       rmse_z=rmse_masked(pred, truth))
                       
            pr = to_raw(tr, pred, idx, E)
            for q, pol in enumerate(ds.pollutants):
                rec[f"rmse_{pol}"] = rmse_masked(pr[:, :, q], truth_raw[:, :, q])
            rec.update({k: cert[k] for k in ("rank", "needed", "identifiable", "sigma_min", "cond", "amp_factor", "pred_rmse")})
            rows.append(rec)
    if pi % 25 == 0:
        print(f"[{args.region}] pattern {pi}/{len(pats)} ({time.time() - t0:.0f}s)", flush=True)
res = pd.DataFrame(rows)
res.to_csv(os.path.join(out, "test_results.csv"), index=False)
print(res.groupby(["method", "size"]).rmse_z.mean().unstack().round(3).to_string(), flush=True)

                                                                                  
bounds = []
Zt = Zs[c["transform"]][ds.split == 2]
fullrows = np.where(np.isfinite(Zt).all(axis=(1, 2)))[0][:300]
for v in range(min(n, 40)):
    for t in fullrows[::10]:
        pred = sheaf.reconstruct(Zt[t:t + 1], [v], c["lam"])[0, 0, :]
        err = float(np.linalg.norm(pred - Zt[t, v, :]))
        bounds.append(dict(station=ds.stations[v], row=int(t), error=err, bound=sheaf.deterministic_bound([v], c["lam"], Zt[t])))
pd.DataFrame(bounds).to_csv(os.path.join(out, "bound_check.csv"), index=False)

                                                                                         
chan = []
stations_c = list(range(n)) if n <= 40 else sorted(rng.choice(n, 40, replace=False).tolist())
for v in stations_c:
    for q in range(p):
        for day, idx in blocks[::max(1, len(blocks) // 40)]:
            for name in ["vector_sheaf", "scalar_sheaf", "gaussian_cond", "lowrank", "kriging", "gnn"]:
                if name not in models:
                    continue
                tr, fn = models[name]
                Zb = FILLED[name][day].copy()
                truth = Zfull[tr][idx][:, v, q].copy()
                if np.isfinite(truth).sum() < 6:
                    continue
                if name == "vector_sheaf":
                    hide = np.zeros((len(idx), n), bool)
                    Zc = Zb.copy(); Zc[:, v, q] = np.nan
                    pred = sheaf.reconstruct_mask(Zc, hide, c["lam"])[:, v, q]
                elif name == "scalar_sheaf":
                    pred = shs[q].reconstruct(Zb[:, :, [q]], [v], cs["lam"])[:, 0, 0]
                elif name in ("gaussian_cond", "lowrank", "kriging"):
                                                                                                
                    Zc = Zb.copy(); Zc[:, v, q] = np.nan
                    if name == "kriging":
                        pred = fn(Zb, [v])[:, 0, q]
                    else:
                        m = models[name][1].__self__
                        W_ = len(idx); M = Zc.reshape(W_, -1) - m.mu; col = v * p + q
                        preds = np.zeros(W_)
                        keys = {}
                        for t in range(W_):
                            keys.setdefault(np.isfinite(M[t]).tobytes(), []).append(t)
                        for key, ts in keys.items():
                            O = np.where(np.isfinite(M[ts[0]]))[0]
                            if name == "gaussian_cond":
                                B = np.linalg.solve(m.S[np.ix_(O, O)], m.S[np.ix_(O, [col])])
                                preds[ts] = (M[np.ix_(ts, O)] @ B)[:, 0] + m.mu[col]
                            else:
                                C = M[np.ix_(ts, O)] @ np.linalg.pinv(m.V[O]).T
                                preds[ts] = (C @ m.V[[col]].T)[:, 0] + m.mu[col]
                        pred = preds
                else:                             
                    Zc = Zb.copy(); Zc[:, v, q] = np.nan
                    import torch
                    x = torch.tensor(np.nan_to_num(Zc), dtype=torch.float32); mm = torch.tensor(np.isfinite(Zc), dtype=torch.float32)
                    with torch.no_grad():
                        pred = gnn.model(torch.cat([x * mm, mm], -1), gnn.A).numpy()[:, v, q]
                chan.append(dict(method=name, station=ds.stations[v], pollutant=ds.pollutants[q], day=day, rmse_z=rmse_masked(pred, truth)))
pd.DataFrame(chan).to_csv(os.path.join(out, "channel_results.csv"), index=False)
print(pd.DataFrame(chan).groupby(["method", "pollutant"]).rmse_z.mean().unstack().round(3).to_string(), flush=True)

                                                                                                                                          
pro = []
times_te = ds.times[te_rows]
runs, start = [], 0
for i in range(1, len(times_te) + 1):
    if i == len(times_te) or (times_te[i] - times_te[i - 1]) != pd.Timedelta(hours=1):
        if i - start >= 73:
            runs.append((start, i))
        start = i
runs = runs[:30]
tr = c["transform"]; Zt = Zs[tr][ds.split == 2]
for (a, b) in runs:
    L = min(b - a - HIST, 72)
    for v in (range(n) if n <= 40 else stations_c):
        blk = Zt[a:a + HIST + L]
        if not np.isfinite(blk[:, v, :]).all():
            continue
        truth = blk[HIST:, v, :]
        hide = np.zeros((HIST + L, n), bool); hide[HIST:, v] = True
        st = sheaf_t.reconstruct_mask(blk, hide, c["lam"])[HIST:, v, :]
        sp_ = sheaf.reconstruct(blk[HIST:], [v], c["lam"])[:, 0, :]
        pers = np.repeat(blk[HIST - 1:HIST, v, :], L, axis=0)
        seas = np.stack([blk[HIST + h - 24, v, :] if h < 24 else blk[HIST + (h % 24) - 24 + 0, v, :] for h in range(L)])
        wh = np.array([hyb_w[np.searchsorted(HB, h + 1, side="right") - 1] for h in range(L)])[:, None]
        preds = {"st_sheaf": st, "vector_sheaf": sp_, "persistence": pers, "hybrid": wh * pers + (1 - wh) * sp_, "seasonal_persistence": seas}
        if "gnn" in models:
            preds["gnn"] = models["gnn"][1](blk[HIST:], [v])[:, 0, :]
        for h in range(L):
            for name, ph in preds.items():
                pro.append(dict(run=str(times_te[a]), station=ds.stations[v], hours_elapsed=h + 1, method=name,
                                sq_err=float(np.mean((ph[h] - truth[h]) ** 2))))
pro = pd.DataFrame(pro)
pro.to_csv(os.path.join(out, "prolonged_results.csv"), index=False)
if len(pro):
    print(pro.groupby(["method", pd.cut(pro.hours_elapsed, [0, 3, 6, 12, 24, 48, 72])]).sq_err.mean().pow(0.5).unstack().round(3).to_string(), flush=True)
print(f"[{args.region}] done in {time.time() - t0:.0f}s", flush=True)
