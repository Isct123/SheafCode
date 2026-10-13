                                                                        
                                                                     
                                                                 
                                                            
                                                                     
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from sheafcode.india import select_region, ROOT              
from sheafcode.data import haversine_km              
from sheafcode.experiments import save_json              

ap = argparse.ArgumentParser()
ap.add_argument("--region", default="delhi")
args = ap.parse_args()
out = os.path.join(ROOT, "results", f"v3_{args.region}")
res = pd.read_csv(os.path.join(out, "test_results.csv"), parse_dates=["day"])
diag = json.load(open(os.path.join(out, "diagnostics.json")))
ds = select_region(args.region); n, p = ds.n, ds.p
D = haversine_km(ds.coords)
X = ds.X; times = ds.times
te = ds.rows(2); days_all = times[te].floor("D")
rng = np.random.default_rng(0)
E = {}


def boot(x, B=2000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return [np.nan, np.nan]
    return [float(v) for v in np.percentile(x[rng.integers(len(x), size=(B, len(x)))].mean(1), [2.5, 97.5])]


                                                                      
cov = {}; level = {}
for d in pd.unique(days_all):
    idx = te[days_all == d]
    cov[pd.Timestamp(d)] = float(np.isfinite(X[idx]).mean())
    q = ds.pollutants.index("PM2.5") if "PM2.5" in ds.pollutants else 0
    level[pd.Timestamp(d)] = float(np.nanmean(X[idx][:, :, q]))
res["coverage"] = res.day.map(cov); res["pm_level"] = res.day.map(level)
                                                                         
bins = pd.qcut(res.coverage, 3, labels=["low", "mid", "high"], duplicates="drop") if res.coverage.nunique() > 3 else pd.Series(["all"] * len(res))
res["cov_bin"] = bins
tab = {}
for b, g in res.groupby("cov_bin"):
    pl = g.groupby(["method", "pattern"]).rmse_z.mean().reset_index()
    w = pl.pivot(index="pattern", columns="method", values="rmse_z")
    row = dict(n_days=int(g.day.nunique()), coverage_range=[float(g.coverage.min()), float(g.coverage.max())], rmse=w.mean().round(4).to_dict())
    if "gaussian_cond" in w and "vector_sheaf" in w:
        d = (w["gaussian_cond"] - w["vector_sheaf"]).dropna(); row["sheaf_improvement_over_gaussian_rel"] = float(d.mean() / w["gaussian_cond"].mean()); row["ci_diff"] = boot(d)
    tab[str(b)] = row
E["by_coverage"] = tab
                                                                 
pols = ds.pollutants
sing = res[res["size"] == 1]
pl = sing.groupby(["method", "pattern"])[[f"rmse_{q}" for q in pols]].mean().reset_index()
E["by_pollutant_raw_units"] = {m: {q: float(g[f"rmse_{q}"].mean()) for q in pols} for m, g in pl.groupby("method")}
tr_std = {q: float(np.nanstd(X[ds.split == 0][:, :, i])) for i, q in enumerate(pols)}
E["train_std_raw_units"] = tr_std
E["by_pollutant_relative_to_train_std"] = {m: {q: float(g[f"rmse_{q}"].mean() / tr_std[q]) for q in pols} for m, g in pl.groupby("method")}
                                                                                      
thr = res.drop_duplicates("day").pm_level.quantile(0.9)
res["extreme"] = res.pm_level >= thr
tab = {}
for flag, g in res.groupby("extreme"):
    pl = g.groupby(["method", "pattern"]).agg(rmse_z=("rmse_z", "mean"), **{f"rmse_{q}": (f"rmse_{q}", "mean") for q in pols}).reset_index()
    w = pl.pivot(index="pattern", columns="method", values="rmse_z")
    row = dict(n_days=int(g.day.nunique()), pm25_threshold=float(thr), rmse_z=w.mean().round(4).to_dict(),
               rmse_raw={m: {q: float(gg[f"rmse_{q}"].mean()) for q in pols} for m, gg in pl.groupby("method")})
    if "gaussian_cond" in w and "vector_sheaf" in w:
        d = (w["gaussian_cond"] - w["vector_sheaf"]).dropna(); row["sheaf_improvement_over_gaussian_rel"] = float(d.mean() / w["gaussian_cond"].mean()); row["ci_diff"] = boot(d)
    tab["extreme" if flag else "normal"] = row
E["extreme_days"] = tab
                                                      
idx_of = {s: i for i, s in enumerate(ds.stations)}
clusters = set()
for v in range(n):
    clusters.add("+".join(ds.stations[i] for i in sorted(int(x) for x in np.argsort(D[v])[:3])))
pl = res[(res["size"] == 3) & (res.method == "vector_sheaf")].groupby("pattern").agg(rmse_z=("rmse_z", "mean"), c2=("c2_star", "first"), c3=("c3_empirical", "first"))
pl["cluster"] = pl.index.isin(clusters)
if pl.cluster.any() and (~pl.cluster).any():
    E["clusters_vs_random_size3"] = dict(n_cluster=int(pl.cluster.sum()), n_random=int((~pl.cluster).sum()),
                                         rmse_cluster=float(pl[pl.cluster].rmse_z.mean()), rmse_random=float(pl[~pl.cluster].rmse_z.mean()),
                                         ci_cluster=boot(pl[pl.cluster].rmse_z), ci_random=boot(pl[~pl.cluster].rmse_z),
                                         c2_cluster=float(pl[pl.cluster].c2.mean()), c2_random=float(pl[~pl.cluster].c2.mean()),
                                         c3_cluster=float(np.nanmean(pl[pl.cluster].c3)), c3_random=float(np.nanmean(pl[~pl.cluster].c3)))
                                                                             
edges = diag["edges"]; rho = diag["canonical_correlations"]
lens = [D[idx_of[u], idx_of[v]] for u, v in edges]; r1 = [r[0] if len(r) else np.nan for r in rho]
ok = np.isfinite(r1)
E["edges"] = dict(n_edges=len(edges), mean_length_km=float(np.mean(lens)), median_length_km=float(np.median(lens)),
                  mean_leading_cca=float(np.nanmean(r1)), spearman_cca_vs_length=float(spearmanr(np.array(lens)[ok], np.array(r1)[ok])[0]) if ok.sum() > 4 else np.nan,
                  strongest=[dict(edge=list(edges[i]), km=float(lens[i]), rho1=float(r1[i])) for i in np.argsort(-np.nan_to_num(r1))[:5]],
                  weakest=[dict(edge=list(edges[i]), km=float(lens[i]), rho1=float(r1[i])) for i in np.argsort(np.nan_to_num(r1, nan=9))[:5]])
                                                                                                             
tr_ = X[ds.split == 0]; te_ = X[ds.split == 2]
std_tr = np.nanstd(tr_, axis=0); shift = np.abs(np.nanmean(te_, axis=0) - np.nanmean(tr_, axis=0)) / std_tr
n_tr = np.isfinite(tr_).sum(axis=0)
worst = np.argsort(np.nan_to_num(shift).ravel())[::-1][:5]
E["train_test_shift"] = dict(n_cells=int(np.isfinite(shift).sum()), cells_shift_over_3sd=int(np.nansum(shift > 3)), cells_shift_over_10sd=int(np.nansum(shift > 10)),
                             cells_train_std_under_5=int(np.nansum(std_tr < 5)), median_shift=float(np.nanmedian(shift)),
                             worst=[dict(station=ds.stations[i // p], pollutant=pols[i % p], shift_sd=float(shift.ravel()[i]), train_std=float(std_tr.ravel()[i]), n_train=int(n_tr.ravel()[i])) for i in worst])
                                                                          
sv = res[(res.method == "vector_sheaf")].groupby("pattern").agg(rmse_z=("rmse_z", "mean"), c2=("c2_star", "first"), size=("size", "first"))
sv = sv[np.isfinite(sv.rmse_z) & np.isfinite(sv.c2)]
E["c2star_spearman_by_size"] = {int(s): dict(n=int(len(g)), spearman=float(spearmanr(g.c2, g.rmse_z)[0])) for s, g in sv.groupby("size") if len(g) >= 6}
save_json(E, os.path.join(out, "extras.json"))
print(json.dumps({k: E[k] for k in ("by_coverage", "extreme_days", "c2star_spearman_by_size")}, indent=None, default=float)[:1500])
