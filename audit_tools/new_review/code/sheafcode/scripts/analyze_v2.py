                                                                         
                                                                      
                                                
import argparse
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon, pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from sheafcode.experiments import ROOT              

PAL = {"vector_sheaf": "#2a78d6", "st_sheaf": "#1c4f8e", "scalar_sheaf": "#eb6834", "gaussian_cond": "#4a3aa7", "gnn": "#008300",
       "lowrank": "#1baf7a", "kriging": "#eda100", "idw": "#e87ba4", "scalar_laplacian": "#9085e9", "persistence": "#e34948",
       "hybrid": "#c98500", "seasonal_persistence": "#d55181"}
LABEL = {"vector_sheaf": "Sheaf (spatial)", "st_sheaf": "Sheaf (space-time)", "scalar_sheaf": "Scalar sheaf", "gaussian_cond": "Dense Gaussian conditional",
         "gnn": "GNN imputer", "lowrank": "Low-rank (SoftImpute)", "kriging": "Kriging", "idw": "IDW", "scalar_laplacian": "Scalar Laplacian",
         "persistence": "Persistence", "hybrid": "Persistence + spatial hybrid", "seasonal_persistence": "Value 24 h earlier"}
ORDER = ["vector_sheaf", "scalar_sheaf", "gaussian_cond", "gnn", "lowrank", "kriging", "idw", "scalar_laplacian"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e4df", "grid.linewidth": 0.6, "axes.edgecolor": "#b5b2a9",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 160})
ap = argparse.ArgumentParser()
ap.add_argument("--regions", default="delhi,ncr,mumbai,kolkata,hyderabad,bengaluru,india")
args = ap.parse_args()
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(0)
summary = {}


def boot_ci(x, B=2000):
    x = np.asarray(x); x = x[np.isfinite(x)]
    if len(x) < 3:
        return (np.nan, np.nan)
    s = [np.mean(x[rng.integers(len(x), size=len(x))]) for _ in range(B)]
    return tuple(np.percentile(s, [2.5, 97.5]))


for reg in args.regions.split(","):
    out = os.path.join(ROOT, "results", f"v2_{reg}")
    if not os.path.exists(os.path.join(out, "test_results.csv")):
        continue
    R = {}
    diag = json.load(open(os.path.join(out, "diagnostics.json")))
    res = pd.read_csv(os.path.join(out, "test_results.csv"))
    pols = diag["pollutants"]
    R.update(n_stations=len(diag["stations"]), pollutants=pols, n_rows=diag["n_rows"], split=diag["split"], obs_fraction=diag["obs_fraction"],
             span=diag["span"], test_span=diag["test_span"], choice=diag["choice"], kernel_dim=diag["kernel_dim"], delta_shape=diag["delta_shape"],
             n_patterns=int(res.pattern.nunique()), n_test_days=int(res.day.nunique()))
    cols = [c for c in res.columns if c.startswith("rmse_")]
    pl = res.groupby(["method", "pattern"]).agg({**{c: "mean" for c in cols}, "size": "first", "identifiable": "first",
                                                  "sigma_min": "first", "pred_rmse": "first", "amp_factor": "first",
                                                  **({"c2_star": "first"} if "c2_star" in res else {})}).reset_index()
    pl = pl[np.isfinite(pl.rmse_z)]
    t1 = []
    for (m, s), g in pl.groupby(["method", "size"]):
        lo, hi = boot_ci(g.rmse_z)
        row = dict(method=m, size=int(s), n=len(g), rmse_z=g.rmse_z.mean(), ci_lo=lo, ci_hi=hi)
        for pol in pols:
            row[f"rmse_{pol}"] = g[f"rmse_{pol}"].mean()
        t1.append(row)
    t1 = pd.DataFrame(t1); t1.to_csv(os.path.join(out, "table_methods_by_size.csv"), index=False)
    R["table1"] = t1.round(3).to_dict(orient="records")
    R["overall"] = pl.groupby("method").rmse_z.mean().round(4).to_dict()
    pv = pl[pl.method == "vector_sheaf"].set_index("pattern")
    comp = []
    for m in pl.method.unique():
        if m == "vector_sheaf":
            continue
        pm = pl[pl.method == m].set_index("pattern").reindex(pv.index)
        d = (pm.rmse_z - pv.rmse_z).dropna()
        lo, hi = boot_ci(d)
        comp.append(dict(method=m, mean_diff_vs_sheaf=d.mean(), ci_lo=lo, ci_hi=hi, wilcoxon_p=float(wilcoxon(d).pvalue) if len(d) > 5 else np.nan,
                         frac_sheaf_better=float((d > 0).mean()), n=len(d)))
    comp = pd.DataFrame(comp); comp.to_csv(os.path.join(out, "table_paired_vs_sheaf.csv"), index=False)
    R["paired_vs_sheaf"] = comp.round(4).to_dict(orient="records")
                  
    h3 = []
    for s, g in pv.groupby("size"):
        g = g[g.identifiable & np.isfinite(g.sigma_min)]
        if len(g) >= 8:
            h3.append(dict(size=int(s), n=len(g), spearman_sigma=spearmanr(1 / g.sigma_min, g.rmse_z).statistic,
                           spearman_pred=spearmanr(g.pred_rmse, g.rmse_z).statistic,
                           spearman_c2star=spearmanr(g.c2_star, g.rmse_z).statistic if "c2_star" in g else np.nan))
    g = pv[pv.identifiable]
    h3.append(dict(size=0, n=len(g), spearman_sigma=spearmanr(1 / g.sigma_min, g.rmse_z).statistic, spearman_pred=spearmanr(g.pred_rmse, g.rmse_z).statistic,
                   spearman_c2star=spearmanr(g.c2_star, g.rmse_z).statistic if "c2_star" in g else np.nan,
                   pearson_pred=pearsonr(g.pred_rmse, g.rmse_z).statistic, mean_pred=float(g.pred_rmse.mean()), mean_actual=float(g.rmse_z.mean()),
                   frac_within_20pct=float((np.abs(g.pred_rmse - g.rmse_z) / g.rmse_z < 0.2).mean())))
    R["certificates"] = pd.DataFrame(h3).round(4).to_dict(orient="records")
    R["frac_identifiable"] = float(pv.identifiable.mean())
    if os.path.exists(os.path.join(out, "bound_check.csv")) and os.path.getsize(os.path.join(out, "bound_check.csv")) > 5:
        bc = pd.read_csv(os.path.join(out, "bound_check.csv"))
        if len(bc):
            R["bound"] = dict(n=len(bc), violations=int((bc.error > bc.bound + 1e-9).sum()), median_ratio=float((bc.error / bc.bound).median()))
    if os.path.exists(os.path.join(out, "channel_results.csv")):
        ch = pd.read_csv(os.path.join(out, "channel_results.csv"))
        cl = ch.groupby(["method", "station", "pollutant"]).rmse_z.mean().reset_index()
        tab = cl.groupby(["method", "pollutant"]).rmse_z.mean().unstack(); tab["all"] = cl.groupby("method").rmse_z.mean()
        R["channel"] = tab.round(3).to_dict(orient="index")
        cv = cl[cl.method == "vector_sheaf"].set_index(["station", "pollutant"])
        R["channel_paired"] = {}
        for m in tab.index:
            if m == "vector_sheaf":
                continue
            cm = cl[cl.method == m].set_index(["station", "pollutant"]).reindex(cv.index)
            d = (cm.rmse_z - cv.rmse_z).dropna()
            R["channel_paired"][m] = dict(mean_diff=float(d.mean()), frac_sheaf_better=float((d > 0).mean()), wilcoxon_p=float(wilcoxon(d).pvalue) if len(d) > 5 else np.nan, n=len(d))
    if os.path.exists(os.path.join(out, "prolonged_results.csv")):
        pro = pd.read_csv(os.path.join(out, "prolonged_results.csv"))
        if len(pro):
            bins = pd.cut(pro.hours_elapsed, [0, 3, 6, 12, 24, 48, 72])
            tab = pro.groupby(["method", bins]).sq_err.mean().pow(0.5).round(3)
            R["prolonged"] = {m: {str(k): float(v) for k, v in tab[m].items()} for m in tab.index.get_level_values(0).unique()}
            R["prolonged_overall"] = pro.groupby("method").sq_err.mean().pow(0.5).round(4).to_dict()
            R["prolonged_n"] = dict(runs=int(pro.run.nunique()), stations=int(pro.station.nunique()))
                                                         
            per = pro.groupby(["method", "run", "station"]).sq_err.mean().unstack(0)
            R["prolonged_paired"] = {m: dict(frac_st_better=float((per[m] > per["st_sheaf"]).mean()), wilcoxon_p=float(wilcoxon(per[m] - per["st_sheaf"]).pvalue), n=len(per))
                                     for m in per.columns if m != "st_sheaf"}
    summary[reg] = R

             
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for m in ORDER:
        g = t1[t1.method == m]
        if g.empty:
            continue
        ax.plot(g["size"], g.rmse_z, marker="o", ms=4, lw=2, color=PAL[m], label=LABEL[m])
    ax.set_xlabel("Stations erased simultaneously"); ax.set_ylabel("Held-out RMSE (standardised)")
    ax.set_title(f"{reg}: {R['n_stations']} stations, error by outage size", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=7, ncol=2); ax.set_xscale("log"); ax.set_xticks(sorted(t1["size"].unique())); ax.set_xticklabels(sorted(t1["size"].unique()))
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"v2_{reg}_error_by_size.png")); plt.close(fig)
    g = pv[pv.identifiable]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    axes[0].scatter(1 / g.sigma_min, g.rmse_z, s=14, color="#2a78d6", alpha=0.7)
    axes[0].set_xlabel("1 / sigma_min (C2)"); axes[0].set_ylabel("Actual held-out RMSE"); axes[0].set_title(f"C2 ranks error: Spearman {spearmanr(1 / g.sigma_min, g.rmse_z).statistic:.2f}", loc="left", fontsize=10)
    axes[1].scatter(g.pred_rmse, g.rmse_z, s=14, color="#eb6834", alpha=0.7)
    lim = [min(g.pred_rmse.min(), g.rmse_z.min()) * 0.95, max(g.pred_rmse.max(), g.rmse_z.max()) * 1.05]
    axes[1].plot(lim, lim, color="#b5b2a9", lw=1, ls="--"); axes[1].set_xlabel("Predicted RMSE from training residuals (C3)"); axes[1].set_ylabel("Actual held-out RMSE")
    axes[1].set_title(f"C3 predicts error: Pearson {pearsonr(g.pred_rmse, g.rmse_z).statistic:.2f}", loc="left", fontsize=10)
    fig.suptitle(f"{reg}: certificates vs actual error over {len(g)} outage patterns", x=0.02, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"v2_{reg}_certificates.png")); plt.close(fig)
    if "prolonged" in R:
        g = pro.groupby(["method", "hours_elapsed"]).sq_err.mean().pow(0.5).reset_index()
        fig, ax = plt.subplots(figsize=(6.6, 3.6))
        for m in ["st_sheaf", "hybrid", "vector_sheaf", "gnn", "persistence", "seasonal_persistence"]:
            gg = g[g.method == m]
            if gg.empty:
                continue
            ax.plot(gg.hours_elapsed, gg.sq_err.rolling(3, center=True, min_periods=1).mean(), lw=2, color=PAL[m], label=LABEL[m])
        ax.set_xlabel("Hours since the station went offline"); ax.set_ylabel("RMSE (standardised)")
        ax.set_title(f"{reg}: prolonged outage with 24 h of history", loc="left", fontsize=11); ax.legend(frameon=False, fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"v2_{reg}_prolonged.png")); plt.close(fig)

                                                     
regs = [r for r in args.regions.split(",") if r in summary]
if regs:
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    x = np.arange(len(regs)); w = 0.1
    ms = [m for m in ORDER if all(m in summary[r]["overall"] for r in regs)]
    for i, m in enumerate(ms):
        vals = [next(t["rmse_z"] for t in summary[r]["table1"] if t["method"] == m and t["size"] == 1) for r in regs]
        ax.bar(x + (i - len(ms) / 2 + 0.5) * w, vals, width=w * 0.92, color=PAL[m], label=LABEL[m])
    ax.set_xticks(x); ax.set_xticklabels([f"{r}\n({summary[r]['n_stations']} st.)" for r in regs]); ax.set_ylabel("Single-station outage RMSE (standardised)")
    ax.set_title("Single-station reconstruction error across regions", loc="left", fontsize=11); ax.legend(frameon=False, fontsize=7, ncol=4); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "v2_cross_region.png")); plt.close(fig)
json.dump(summary, open(os.path.join(ROOT, "results", "v2_summary.json"), "w"), indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
for r in regs:
    R = summary[r]
    print(f"== {r}: n={R['n_stations']} patterns={R['n_patterns']} days={R['n_test_days']} overall={ {k: round(v, 3) for k, v in R['overall'].items()} }")
    print("   certs:", R["certificates"][-1], "bound:", R.get("bound"))
    if "prolonged_overall" in R:
        print("   prolonged:", {k: round(v, 3) for k, v in R["prolonged_overall"].items()})
