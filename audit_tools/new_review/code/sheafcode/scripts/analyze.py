                                                                   

                                                                         
                                                                        
                                                                      
                                                                    
                                                                     
                                         
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
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from sheafcode.experiments import ROOT              

PAL = {"vector_sheaf": "#2a78d6", "scalar_sheaf": "#eb6834", "lowrank": "#1baf7a", "kriging": "#eda100",
       "idw": "#e87ba4", "scalar_laplacian": "#4a3aa7", "persistence": "#e34948",
       "syndrome": "#2a78d6", "loo_residual": "#eb6834", "neighbour_diff": "#1baf7a", "magnitude": "#eda100"}
LABEL = {"vector_sheaf": "Vector sheaf", "scalar_sheaf": "Scalar sheaf (per pollutant)", "lowrank": "Low-rank PCA",
         "kriging": "Kriging", "idw": "IDW", "scalar_laplacian": "Scalar Laplacian", "persistence": "Persistence",
         "syndrome": "Sheaf syndrome", "loo_residual": "Leave-one-out residual", "neighbour_diff": "Neighbour difference",
         "magnitude": "Reading magnitude"}
ORDER = ["vector_sheaf", "scalar_sheaf", "lowrank", "kriging", "idw", "scalar_laplacian"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e4df", "grid.linewidth": 0.6, "axes.edgecolor": "#b5b2a9",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 160})

ap = argparse.ArgumentParser()
ap.add_argument("--tags", default="b10_geo,b10_geo_5x4,b4_geo")
ap.add_argument("--primary", default="b10_geo")
args = ap.parse_args()
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(0)
report = {}


def boot_ci(x, stat=np.mean, B=2000):
    x = np.asarray(x)
    if len(x) < 2:
        return (np.nan, np.nan)
    s = [stat(x[rng.integers(len(x), size=len(x))]) for _ in range(B)]
    return tuple(np.percentile(s, [2.5, 97.5]))


def pattern_level(res):
    cols = [c for c in res.columns if c.startswith("rmse_") or c.startswith("mae_")]
    keep = ["method", "pattern", "size"] + [c for c in ["rank", "needed", "identifiable", "sigma_min", "cond"] if c in res]
    g = res.groupby(["method", "pattern"]).agg({**{c: "mean" for c in cols}, **{c: "first" for c in keep if c not in ("method", "pattern")}}).reset_index()
    return g


for tag in args.tags.split(","):
    out = os.path.join(ROOT, "results", tag)
    if not os.path.exists(os.path.join(out, "test_results.csv")):
        continue
    R = {}
    res = pd.read_csv(os.path.join(out, "test_results.csv"))
    diag = json.load(open(os.path.join(out, "sheaf_diagnostics.json")))
    pols = diag["pollutants"]
    pl = pattern_level(res)
    R["n_patterns"] = int(pl.pattern.nunique()); R["n_test_days"] = int(res.day.nunique())
    R["n_test_hours"] = int(res[res.method == "vector_sheaf"].groupby("day").n_hours.first().sum())
    R["stations"] = diag["stations"]; R["pollutants"] = pols; R["choice"] = diag["choice"]
    R["kernel_dim"] = diag["kernel_dim"]; R["delta_shape"] = diag["delta_shape"]
    R["train_span"] = diag["train_span"]; R["test_span"] = diag["test_span"]

                                                                                 
    t1 = []
    for (m, s), g in pl.groupby(["method", "size"]):
        lo, hi = boot_ci(g.rmse_z)
        row = dict(method=m, size=int(s), n_patterns=len(g), rmse_z=g.rmse_z.mean(), ci_lo=lo, ci_hi=hi)
        for p in pols:
            row[f"rmse_{p}"] = g[f"rmse_{p}"].mean()
        t1.append(row)
    t1 = pd.DataFrame(t1)
    t1.to_csv(os.path.join(out, "table_methods_by_size.csv"), index=False)
    R["table1"] = t1.round(3).to_dict(orient="records")

                                                                                
    pv = pl[pl.method == "vector_sheaf"].set_index("pattern")
    comp = []
    for m in ORDER:
        if m == "vector_sheaf" or m not in set(pl.method):
            continue
        pm = pl[pl.method == m].set_index("pattern").loc[pv.index]
        d = pm.rmse_z - pv.rmse_z
        lo, hi = boot_ci(d)
        try:
            pw = wilcoxon(d).pvalue
        except ValueError:
            pw = np.nan
        comp.append(dict(method=m, mean_diff_vs_vector=d.mean(), ci_lo=lo, ci_hi=hi, wilcoxon_p=pw,
                         frac_patterns_vector_better=float((d > 0).mean()), n=len(d)))
    comp = pd.DataFrame(comp)
    comp.to_csv(os.path.join(out, "table_paired_vs_vector.csv"), index=False)
    R["paired_vs_vector"] = comp.round(4).to_dict(orient="records")
                                    
    ps = pl[pl.method == "scalar_sheaf"].set_index("pattern").loc[pv.index]
    R["h1_station_per_pollutant"] = {p: dict(vector=float(pv[f"rmse_{p}"].mean()), scalar=float(ps[f"rmse_{p}"].mean()),
                                            wilcoxon_p=float(wilcoxon(ps[f"rmse_{p}"] - pv[f"rmse_{p}"]).pvalue)) for p in pols}

                                                                                 
    h3 = []
    for s, g in pv.groupby("size"):
        g = g[g.identifiable]
        if len(g) >= 5:
            r_, p_ = spearmanr(1 / g.sigma_min, g.rmse_z)
            bs = [spearmanr(1 / g.sigma_min.iloc[i], g.rmse_z.iloc[i]).statistic for i in [rng.integers(len(g), size=len(g)) for _ in range(1000)]]
            h3.append(dict(size=int(s), n=len(g), spearman=r_, p=p_, ci_lo=np.nanpercentile(bs, 2.5), ci_hi=np.nanpercentile(bs, 97.5)))
    allg = pv[pv.identifiable]
    r_, p_ = spearmanr(1 / allg.sigma_min, allg.rmse_z)
    h3.append(dict(size=0, n=len(allg), spearman=r_, p=p_, ci_lo=np.nan, ci_hi=np.nan))
    h3 = pd.DataFrame(h3); h3.to_csv(os.path.join(out, "table_h3_sigma_min.csv"), index=False)
    R["h3"] = h3.round(4).to_dict(orient="records")

                                                                                 
    if os.path.exists(os.path.join(out, "identifiability_results.csv")):
        idr = pd.read_csv(os.path.join(out, "identifiability_results.csv"))
        h2 = []
        for (k, r, s), g in idr[idr.solver == "ridge"].groupby(["k", "r", "size"]):
            if g.identifiable.nunique() < 2 or (~g.identifiable).sum() < 2 or g.identifiable.sum() < 2:
                continue
            a, b = g[g.identifiable].rmse_z, g[~g.identifiable].rmse_z
            h2.append(dict(k=int(k), r=int(r), size=int(s), n_ident=len(a), n_nonident=len(b), rmse_ident=a.mean(),
                           rmse_nonident=b.mean(), mwu_p=mannwhitneyu(b, a, alternative="greater").pvalue))
        h2 = pd.DataFrame(h2); h2.to_csv(os.path.join(out, "table_h2_identifiability.csv"), index=False)
        R["h2"] = h2.round(4).to_dict(orient="records")
        frac = idr[idr.solver == "ridge"].groupby(["k", "r"]).identifiable.mean().unstack()
        R["frac_identifiable_by_k_r"] = frac.round(3).to_dict()
        h3ls = []
        for (k, r), g in idr[(idr.solver == "ls") & idr.identifiable].groupby(["k", "r"]):
            if len(g) > 10:
                h3ls.append(dict(k=int(k), r=int(r), n=len(g), spearman=spearmanr(1 / g.sigma_min, g.rmse_z).statistic))
        R["h3_ls_by_k_r"] = pd.DataFrame(h3ls).round(3).to_dict(orient="records")

                                                                                  
    if os.path.exists(os.path.join(out, "channel_results.csv")):
        chn = pd.read_csv(os.path.join(out, "channel_results.csv"))
        cl = chn.groupby(["method", "station", "pollutant"]).rmse_z.mean().reset_index()
        tab = cl.groupby(["method", "pollutant"]).rmse_z.mean().unstack().round(3)
        tab["all"] = cl.groupby("method").rmse_z.mean().round(3)
        R["channel_table"] = tab.to_dict(orient="index")
        cv = cl[cl.method == "vector_sheaf"].set_index(["station", "pollutant"])
        cs = cl[cl.method == "scalar_sheaf"].set_index(["station", "pollutant"]).loc[cv.index]
        d = cs.rmse_z - cv.rmse_z
        R["h1_channel"] = dict(mean_diff_scalar_minus_vector=float(d.mean()), ci=boot_ci(d), wilcoxon_p=float(wilcoxon(d).pvalue),
                               per_pollutant={p: dict(vector=float(cv.xs(p, level=1).rmse_z.mean()), scalar=float(cs.xs(p, level=1).rmse_z.mean()),
                                                      wilcoxon_p=float(wilcoxon(cs.xs(p, level=1).rmse_z - cv.xs(p, level=1).rmse_z).pvalue))
                                              for p in pols})
        R["channel_choice"] = json.load(open(os.path.join(out, "channel_choice.json")))

    if os.path.exists(os.path.join(out, "ablation_results.csv")):
        abl = pd.read_csv(os.path.join(out, "ablation_results.csv"))
        R["ablation_k_r"] = abl[abl.ablation == "k_r"].pivot(index="k", columns="r", values="rmse_station").round(3).to_dict()
        ps_ = abl[abl.ablation == "pollutant_subset"].groupby(["target", "n_pollutants"])[["rmse_station", "rmse_channel"]].mean().round(3)
        R["ablation_pollutant_subset"] = {f"{a}|{int(b)}": v for (a, b), v in ps_.to_dict(orient="index").items()}
    if os.path.exists(os.path.join(out, "prolonged_results.csv")):
        pro = pd.read_csv(os.path.join(out, "prolonged_results.csv"))
        bins = pd.cut(pro.hours_elapsed, [0, 6, 12, 24, 48, 72])
        R["prolonged"] = pro.groupby(["method", bins]).sq_err.mean().pow(0.5).unstack().round(3).astype(object).to_dict(orient="index")
        R["prolonged"] = {m: {str(k): v for k, v in d.items()} for m, d in R["prolonged"].items()}
    if os.path.exists(os.path.join(out, "fault_results.csv")):
        fr = pd.read_csv(os.path.join(out, "fault_results.csv"))
        fa = pd.read_csv(os.path.join(out, "fault_auc.csv"))
        ft = fr.groupby(["kind", "magnitude", "detector"]).top1.mean().unstack().round(3); ft.columns = [f"top1_{c}" for c in ft.columns]
        R["fault_top1"] = ft.reset_index().to_dict(orient="records")
        fap = fa.pivot_table(index=["kind", "magnitude"], columns="detector", values="auc").round(3); fap.columns = [f"auc_{c}" for c in fap.columns]
        R["fault_auc"] = fap.reset_index().to_dict(orient="records")
        R["fault_top1_overall"] = fr.groupby("detector").top1.mean().round(3).to_dict()
    if os.path.exists(os.path.join(out, "design_criticality.csv")):
        cr = pd.read_csv(os.path.join(out, "design_criticality.csv"))
        R["criticality"] = cr.round(3).to_dict(orient="records")
        c2 = cr[cr.removed != "none"]
        R["criticality_spearman_sigma_vs_error"] = float(spearmanr(-c2.mean_sigma_min, c2.test_rmse_single).statistic)
        ad = pd.read_csv(os.path.join(out, "design_readdition.csv"))
        R["readdition"] = dict(n=len(ad), cert_agrees=float(ad.cert_agrees.mean()), geo_agrees=float(ad.geo_agrees.mean()),
                               regret_cert=float(ad.regret_cert.mean()), regret_geo=float(ad.regret_geo.mean()))
    report[tag] = R

                                                                                   
    sizes = sorted(t1["size"].unique())
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for m in ORDER:
        g = t1[t1.method == m]
        if g.empty:
            continue
        ax.plot(g["size"], g.rmse_z, marker="o", ms=5, lw=2, color=PAL[m], label=LABEL[m])
        ax.fill_between(g["size"], g.ci_lo, g.ci_hi, color=PAL[m], alpha=0.12, lw=0)
    ax.set_xticks(sizes); ax.set_xlabel("Stations erased simultaneously"); ax.set_ylabel("Held-out RMSE (standardised units)")
    ax.set_title(f"Reconstruction error by outage size ({tag})", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"{tag}_fig_error_by_size.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    cols = {1: "#c6d9f2", 2: "#8db4e8", 3: "#4f8ddd", 4: "#2a78d6", 5: "#1c4f8e"}
    for s, g in pv[pv.identifiable].groupby("size"):
        ax.scatter(1 / g.sigma_min, g.rmse_z, s=22, color=cols.get(int(s), "#2a78d6"), edgecolor="#fcfcfb", lw=0.6, label=f"{int(s)} station{'s' if s > 1 else ''}")
    ax.set_xlabel("1 / sigma_min(delta_E)   (robustness certificate, larger = more fragile)")
    ax.set_ylabel("Held-out RMSE (standardised)")
    rho_all = spearmanr(1 / allg.sigma_min, allg.rmse_z).statistic
    ax.set_title(f"Certificate vs actual error, Spearman {rho_all:.2f} ({tag})", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, title="Outage size", title_fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"{tag}_fig_sigma_min_vs_error.png")); plt.close(fig)

    if "channel_table" in R:
        tab = pd.DataFrame(R["channel_table"]).T.drop(columns="all")
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        x = np.arange(len(tab.columns)); w = 0.13
        for i, m in enumerate([m for m in ORDER if m in tab.index]):
            ax.bar(x + (i - 2.5) * w, tab.loc[m].values, width=w * 0.92, color=PAL[m], label=LABEL[m])
        ax.set_xticks(x); ax.set_xticklabels(tab.columns); ax.set_ylabel("Held-out RMSE (standardised)")
        ax.set_title(f"Single-channel outage: one pollutant lost at one station ({tag})", loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=7, ncol=3); ax.grid(axis="x", visible=False)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"{tag}_fig_channel.png")); plt.close(fig)

    if "ablation_pollutant_subset" in R:
        ps_ = abl[abl.ablation == "pollutant_subset"].groupby(["target", "n_pollutants"])[["rmse_station", "rmse_channel"]].mean().reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
        for ax, col, ttl in zip(axes, ["rmse_station", "rmse_channel"], ["Whole-station outage", "Single-channel outage"]):
            for j, (tgt, g) in enumerate(ps_.groupby("target")):
                ax.plot(g.n_pollutants, g[col], marker="o", ms=4, lw=2, color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"][j], label=tgt)
            ax.set_xticks(sorted(ps_.n_pollutants.unique())); ax.set_xlabel("Pollutants in the sheaf"); ax.set_title(ttl, loc="left", fontsize=10)
        axes[0].set_ylabel("RMSE of target pollutant (standardised)"); axes[1].legend(frameon=False, fontsize=8, title="Target")
        fig.suptitle(f"Does adding pollutants help? ({tag})", x=0.02, ha="left", fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"{tag}_fig_pollutant_ablation.png")); plt.close(fig)

    if os.path.exists(os.path.join(out, "identifiability_results.csv")):
        idr = pd.read_csv(os.path.join(out, "identifiability_results.csv"))
        g = idr[(idr.solver == "ridge") & (idr.k == 2) & (idr.r == 1)] if ((idr.k == 2) & (idr.r == 1)).any() else idr[idr.solver == "ridge"]
        if g.identifiable.nunique() == 2:
            fig, ax = plt.subplots(figsize=(6.4, 3.4))
            for s in sorted(g["size"].unique()):
                a = g[(g["size"] == s) & g.identifiable].rmse_z; b = g[(g["size"] == s) & ~g.identifiable].rmse_z
                ax.scatter(np.full(len(a), s - 0.12) + rng.normal(0, 0.03, len(a)), a, s=16, color="#2a78d6", label="identifiable" if s == 1 else None)
                ax.scatter(np.full(len(b), s + 0.12) + rng.normal(0, 0.03, len(b)), b, s=16, color="#e34948", label="not identifiable" if s == 1 else None)
            ax.set_xlabel("Stations erased"); ax.set_ylabel("Held-out RMSE (standardised)")
            ax.set_title(f"Rank certificate on a sparse graph (k=2, r=1): identifiable vs not ({tag})", loc="left", fontsize=10)
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout(); fig.savefig(os.path.join(FIG, f"{tag}_fig_identifiability.png")); plt.close(fig)

    if "prolonged" in R:
        pro = pd.read_csv(os.path.join(out, "prolonged_results.csv"))
        g = pro.groupby(["method", "hours_elapsed"]).sq_err.mean().pow(0.5).reset_index()
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        for m in ["vector_sheaf", "scalar_sheaf", "idw", "persistence"]:
            gg = g[g.method == m]
            ax.plot(gg.hours_elapsed, gg.sq_err.rolling(3, center=True, min_periods=1).mean(), lw=2, color=PAL[m], label=LABEL[m])
        ax.set_xlabel("Hours since the station went offline"); ax.set_ylabel("RMSE (standardised)")
        ax.set_title(f"Prolonged single-station outage ({tag})", loc="left", fontsize=11); ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"{tag}_fig_prolonged.png")); plt.close(fig)

    if "fault_top1" in R:
        fr = pd.read_csv(os.path.join(out, "fault_results.csv"))
        g = fr[fr.kind.isin(["bias", "drift", "gain", "channel"])].groupby(["kind", "magnitude", "detector"]).top1.mean().reset_index()
        fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.9), sharey=True)
        for ax, kind in zip(axes, ["bias", "gain", "drift", "channel"]):
            for det in ["syndrome", "loo_residual", "neighbour_diff", "magnitude"]:
                gg = g[(g.kind == kind) & (g.detector == det)]
                ax.plot(gg.magnitude, gg.top1, marker="o", ms=4, lw=2, color=PAL[det], label=LABEL[det])
            ax.set_title(kind, loc="left", fontsize=10); ax.set_xlabel("Fault size (sd units)"); ax.set_ylim(0, 1)
        axes[0].set_ylabel("Top-1 localisation accuracy"); axes[3].legend(frameon=False, fontsize=7)
        fig.suptitle(f"Fault localisation ({tag})", x=0.02, ha="left", fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"{tag}_fig_faults.png")); plt.close(fig)

    if "criticality" in R:
        cr = pd.read_csv(os.path.join(out, "design_criticality.csv"))
        c2 = cr[cr.removed != "none"].sort_values("mean_sigma_min")
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        ax.scatter(c2.mean_sigma_min, c2.test_rmse_single, s=40, color="#2a78d6")
        for _, r_ in c2.iterrows():
            ax.annotate(r_.removed, (r_.mean_sigma_min, r_.test_rmse_single), xytext=(4, 4), textcoords="offset points", fontsize=8)
        base = cr[cr.removed == "none"].iloc[0]
        ax.scatter([base.mean_sigma_min], [base.test_rmse_single], s=60, marker="s", color="#eb6834")
        ax.annotate("full network", (base.mean_sigma_min, base.test_rmse_single), xytext=(4, -10), textcoords="offset points", fontsize=8)
        ax.set_xlabel("Mean sigma_min of the remaining network (training data only)"); ax.set_ylabel("Held-out single-outage RMSE")
        ax.set_title(f"Which station's loss hurts most? certificate vs actual ({tag})", loc="left", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"{tag}_fig_criticality.png")); plt.close(fig)

json.dump(report, open(os.path.join(ROOT, "results", "summary.json"), "w"), indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
print(json.dumps({t: {k: v for k, v in r.items() if k in ("n_patterns", "n_test_days", "n_test_hours", "kernel_dim", "paired_vs_vector", "h1_channel", "h3", "readdition", "fault_top1_overall", "criticality_spearman_sigma_vs_error")} for t, r in report.items()}, indent=1, default=str)[:6000])
