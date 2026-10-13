                                                                          

                                                                      
                                                                    
                                                                        
                                                                     
                                                                   
                                                                       
                                       

                                                      
                                                                 
                                                                       
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon, mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from sheafcode.experiments import ROOT              

ap = argparse.ArgumentParser()
ap.add_argument("--regions", default="delhi,ncr,mumbai,kolkata,hyderabad,bengaluru,india,delhi2025,bengaluru_raw")
ap.add_argument("--B", type=int, default=2000)
args = ap.parse_args()
rng = np.random.default_rng(0)
P1_BASE = ["gaussian_cond", "gnn", "kriging", "scalar_sheaf"]
P2_BASE = ["gaussian_cond", "scalar_sheaf", "kriging", "gnn"]
P3_BASE = ["kalman_var", "hybrid", "gnn", "persistence"]


def boot_ci(x, B=None):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    B = B or args.B
    if len(x) < 3:
        return [float("nan"), float("nan")]
    idx = rng.integers(len(x), size=(B, len(x)))
    return [float(v) for v in np.percentile(x[idx].mean(axis=1), [2.5, 97.5])]


def block_boot_ci(df, value, block, B=None):
                                                                          
    B = B or args.B
    groups = [g[value].to_numpy() for _, g in df.groupby(block)]
    groups = [g[np.isfinite(g)] for g in groups if np.isfinite(g).any()]
    if len(groups) < 3:
        return [float("nan"), float("nan")]
    stats = []
    for _ in range(B):
        pick = rng.integers(len(groups), size=len(groups))
        stats.append(np.concatenate([groups[i] for i in pick]).mean())
    return [float(v) for v in np.percentile(stats, [2.5, 97.5])]


def cliffs_delta(a, b):
                                                                   
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    if len(a) * len(b) > 4_000_000:
        a = rng.choice(a, 2000); b = rng.choice(b, 2000)
    return float(((a[:, None] > b[None, :]).mean() - (a[:, None] < b[None, :]).mean()))


def holm(pvals):
                                                            
    names = [k for k, v in pvals.items() if np.isfinite(v)]
    ps = np.array([pvals[k] for k in names])
    order = np.argsort(ps); m = len(ps); adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        adj[i] = min(1.0, running)
    out = {k: float("nan") for k in pvals}
    out.update({names[i]: float(adj[i]) for i in range(m)})
    return out


def paired_family(wide, ref, comps, unit_name):
                                                                          
                                                                            
                           
    fam, raw_p = {}, {}
    for m in comps:
        if m not in wide.columns:
            continue
        d = (wide[m] - wide[ref]).dropna()
        if len(d) < 5:
            continue
        try:
            p = float(wilcoxon(d, alternative="greater").pvalue)                                       
        except ValueError:
            p = float("nan")
        raw_p[m] = p
        fam[m] = dict(competitor=m, n_units=int(len(d)), unit=unit_name,
                      ref_rmse=float(wide[ref].loc[d.index].mean()), comp_rmse=float(wide[m].loc[d.index].mean()),
                      mean_diff=float(d.mean()), diff_ci=boot_ci(d),
                      rel_improvement=float(d.mean() / wide[m].loc[d.index].mean()),
                      frac_units_sheaf_better=float((d > 0).mean()),
                      cliffs_delta=cliffs_delta(wide[m].loc[d.index], wide[ref].loc[d.index]),
                      p_wilcoxon_one_sided=p)
    adj = holm(raw_p)
    for m in fam:
        fam[m]["p_holm"] = adj[m]
        fam[m]["significant_005_holm"] = bool(adj[m] < 0.05)
    return fam


final = {"protocol": "FINAL_PROTOCOL.md 2026-09-30", "regions": {}}
for reg in args.regions.split(","):
    out = os.path.join(ROOT, "results", f"v3_{reg}")
    if not os.path.exists(os.path.join(out, "prolonged_results.csv")):
        print(f"[{reg}] not finished, skipped"); continue
    R = {}
    diag = json.load(open(os.path.join(out, "diagnostics.json")))
    man = json.load(open(os.path.join(out, "manifest.json")))
    res = pd.read_csv(os.path.join(out, "test_results.csv"), parse_dates=["day"])
    pols = diag["pollutants"]
    R.update(n_stations=len(diag["stations"]), pollutants=pols, n_rows=diag["n_rows"], split=diag["split"], obs_fraction=diag["obs_fraction"],
             span=diag["span"], test_span=diag["test_span"], choice=diag["choice"], kernel_dim=diag["kernel_dim"], delta_shape=diag["delta_shape"],
             n_patterns=int(res.pattern.nunique()), n_test_days=int(res.day.nunique()), n_edges=len(diag["edges"]),
             code_sha256=man["code_sha256"], data_sha256=man["data_sha256"], run_seconds=man.get("seconds"))
    cols = [c for c in res.columns if c.startswith("rmse_")]
    certcols = ["size", "rank", "needed", "identifiable", "sigma_min", "c2_star", "g_inv_norm", "c3_complete", "c3_empirical"]
                                                  
    pl = res.groupby(["method", "pattern"]).agg({**{c: "mean" for c in cols}, **{c: "first" for c in certcols}}).reset_index()
    pl = pl[np.isfinite(pl.rmse_z)]
    t1 = []
    for (m, s), g in pl.groupby(["method", "size"]):
        row = dict(method=m, size=int(s), n=len(g), rmse_z=float(g.rmse_z.mean()), ci=boot_ci(g.rmse_z))
        for pol in pols:
            row[f"rmse_{pol}"] = float(g[f"rmse_{pol}"].mean())
        t1.append(row)
    R["snapshot_by_size"] = t1
    R["snapshot_overall"] = pl.groupby("method").rmse_z.mean().round(4).to_dict()
    R["snapshot_overall_ci"] = {m: boot_ci(g.rmse_z) for m, g in pl.groupby("method")}
    R["snapshot_overall_median"] = pl.groupby("method").rmse_z.median().round(4).to_dict()
    R["snapshot_singles_median"] = pl[pl["size"] == 1].groupby("method").rmse_z.median().round(4).to_dict()
    wide = pl.pivot(index="pattern", columns="method", values="rmse_z")
    R["P1"] = paired_family(wide, "vector_sheaf", P1_BASE, "pattern")
    R["snapshot_all_vs_sheaf"] = paired_family(wide, "vector_sheaf", [m for m in wide.columns if m != "vector_sheaf"], "pattern")
                                                  
    R["P1_singles"] = paired_family(pl[pl["size"] == 1].pivot(index="pattern", columns="method", values="rmse_z"), "vector_sheaf", P1_BASE, "pattern(size1)")
                                                                                                  
    R["snapshot_sheaf_block_ci_days"] = block_boot_ci(res[res.method == "vector_sheaf"], "rmse_z", "day")
                                                         
    if res.lockdown.nunique() > 1:
        ld = {}
        for flag, g in res.groupby("lockdown"):
            gl = g.groupby(["method", "pattern"]).rmse_z.mean().reset_index()
            w = gl.pivot(index="pattern", columns="method", values="rmse_z")
            ld["lockdown" if flag else "pre_lockdown"] = dict(n_days=int(g.day.nunique()), overall=gl.groupby("method").rmse_z.mean().round(4).to_dict(),
                                                              vs_sheaf=paired_family(w, "vector_sheaf", P1_BASE, "pattern"))
        R["lockdown_split"] = ld
                                      
    sv = pl[pl.method == "vector_sheaf"].copy()
    cert = {}
    for name, col in (("c2_star", "c2_star"), ("inv_sigma_min", None), ("c3_complete", "c3_complete"), ("c3_empirical", "c3_empirical"), ("g_inv_norm", "g_inv_norm")):
        x = 1.0 / sv.sigma_min.replace(0, np.nan) if col is None else sv[col]
        ok = np.isfinite(x) & np.isfinite(sv.rmse_z)
        if ok.sum() >= 5:
            rho, pv = spearmanr(x[ok], sv.rmse_z[ok])
            cert[name] = dict(spearman=float(rho), p=float(pv), n=int(ok.sum()))
            if col in ("c3_complete", "c3_empirical"):
                d = sv.rmse_z[ok] - x[ok]
                cert[name].update(mean_actual=float(sv.rmse_z[ok].mean()), mean_predicted=float(x[ok].mean()),
                                  mean_actual_minus_pred=float(d.mean()), ci=boot_ci(d), mae_of_prediction=float(np.abs(d).mean()))
                                                                                 
    adj = holm({k: cert[k]["p"] for k in ("c2_star", "inv_sigma_min") if k in cert})
    for k, v in adj.items():
        cert[k]["p_holm"] = v
    R["P4_certificates"] = cert
    R["identifiable_fraction"] = float(sv.identifiable.mean())
    R["identifiable_by_size"] = sv.groupby("size").identifiable.mean().round(3).to_dict()
                             
    pro_file = os.path.join(out, "prospective_deciles.json")
    if os.path.exists(pro_file):
        pdx = json.load(open(pro_file))
        e = sv.set_index("pattern").rmse_z
        safe = e.reindex(pdx["safest_decile"]).dropna(); risky = e.reindex(pdx["riskiest_decile"]).dropna()
        if len(safe) >= 2 and len(risky) >= 2:
            R["prospective_deciles"] = dict(written_before_test=pdx["written_before_test"], n_safe=int(len(safe)), n_risky=int(len(risky)),
                                            rmse_safe=float(safe.mean()), rmse_risky=float(risky.mean()), ratio=float(risky.mean() / safe.mean()),
                                            safe_ci=boot_ci(safe), risky_ci=boot_ci(risky),
                                            p_mannwhitney_one_sided=float(mannwhitneyu(risky, safe, alternative="greater").pvalue),
                                            cliffs_delta=cliffs_delta(risky, safe))
                                      
    bf = os.path.join(out, "bound_check.csv")
    if os.path.exists(bf) and os.path.getsize(bf) > 10:
        bc = pd.read_csv(bf)
        if len(bc):
            R["bound_check"] = dict(n=int(len(bc)), violations=int((bc.error > bc.bound * (1 + 1e-8)).sum()),
                                    median_ratio_error_over_bound=float((bc.error / bc.bound).median()),
                                    p90_ratio=float((bc.error / bc.bound).quantile(0.9)))
                                                             
    chan = pd.read_csv(os.path.join(out, "channel_results.csv"), parse_dates=["day"])
    chan["cell"] = chan.station + ":" + chan.pollutant
    cl = chan.groupby(["method", "cell", "pollutant"]).rmse_z.mean().reset_index()
    R["channel_by_pollutant"] = cl.groupby(["method", "pollutant"]).rmse_z.mean().round(4).unstack().to_dict(orient="index")
    R["channel_overall"] = cl.groupby("method").rmse_z.mean().round(4).to_dict()
    R["channel_overall_ci"] = {m: boot_ci(g.rmse_z) for m, g in cl.groupby("method")}
    wc = cl.pivot(index="cell", columns="method", values="rmse_z")
    R["P2"] = paired_family(wc, "vector_sheaf", P2_BASE, "station x pollutant")
                                                                                      
    mech = {}
    for var in ("vector_sheaf_Pidentity", "vector_sheaf_noPM10", "vector_sheaf_ownhidden"):
        if var in wc.columns:
            d = (wc[var] - wc["vector_sheaf"]).dropna()
            mech[var] = dict(n_cells=int(len(d)), rmse_variant=float(wc[var].loc[d.index].mean()), rmse_full=float(wc["vector_sheaf"].loc[d.index].mean()),
                             mean_increase=float(d.mean()), ci=boot_ci(d), frac_cells_worse=float((d > 0).mean()))
            if var == "vector_sheaf_noPM10":
                sub = cl[cl.method.isin([var, "vector_sheaf"])].pivot_table(index="cell", columns="method", values="rmse_z")
                sub["pollutant"] = [c.split(":")[1] for c in sub.index]
                mech[var]["increase_by_pollutant"] = (sub[var] - sub["vector_sheaf"]).groupby(sub.pollutant).mean().round(4).to_dict()
    R["mechanism"] = mech
                                                                     
    R["channel_by_pollutant_vs_gaussian"] = {}
    for pol, g in cl.groupby("pollutant"):
        w = g.pivot(index="cell", columns="method", values="rmse_z")
        if "gaussian_cond" in w and "vector_sheaf" in w:
            d = (w["gaussian_cond"] - w["vector_sheaf"]).dropna()
            R["channel_by_pollutant_vs_gaussian"][pol] = dict(n=int(len(d)), mean_diff=float(d.mean()), ci=boot_ci(d), sheaf=float(w["vector_sheaf"].loc[d.index].mean()), gaussian=float(w["gaussian_cond"].loc[d.index].mean()))
                                                         
    pro = pd.read_csv(os.path.join(out, "prolonged_results.csv"))
    if len(pro):
        pro["unit"] = pro.run + "|" + pro.station
        bins = pd.cut(pro.hours_elapsed, [0, 3, 6, 12, 24, 48, 72], labels=["1-3", "4-6", "7-12", "13-24", "25-48", "49-72"])
        R["prolonged_by_horizon"] = pro.groupby(["method", bins]).sq_err.mean().pow(0.5).round(4).unstack().to_dict(orient="index")
        R["prolonged_by_horizon"] = {m: {str(k): v for k, v in d.items()} for m, d in R["prolonged_by_horizon"].items()}
        R["prolonged_n_runs"] = int(pro.run.nunique()); R["prolonged_n_units"] = int(pro.unit.nunique())
        for span, hi in (("1_24h", 24), ("1_72h", 72)):
            sub = pro[pro.hours_elapsed <= hi]
            ul = sub.groupby(["method", "unit"]).sq_err.mean().pow(0.5).reset_index()
            w = ul.pivot(index="unit", columns="method", values="sq_err")
            R[f"prolonged_overall_{span}"] = ul.groupby("method").sq_err.mean().round(4).to_dict()
            R[f"prolonged_overall_{span}_ci"] = {m: boot_ci(g.sq_err) for m, g in ul.groupby("method")}
            R[f"P3_{span}"] = paired_family(w, "st_sheaf", P3_BASE, "run x station")
            R[f"prolonged_all_vs_st_{span}"] = paired_family(w, "st_sheaf", [m for m in w.columns if m != "st_sheaf"], "run x station")
                                                                     
            R[f"prolonged_st_block_ci_runs_{span}"] = block_boot_ci(sub[sub.method == "st_sheaf"].assign(r=lambda d: np.sqrt(d.sq_err)), "r", "run")
                                                                                         
        abl = [m for m in ("st_sheaf", "st_lag1", "st_lag24", "st_hist6", "st_hist12", "st_hist48", "vector_sheaf") if m in pro.method.unique()]
        ul = pro[pro.method.isin(abl)].groupby(["method", "unit"]).sq_err.mean().pow(0.5).reset_index()
        w = ul.pivot(index="unit", columns="method", values="sq_err").dropna()
        R["temporal_ablation"] = dict(n_units=int(len(w)), rmse=w.mean().round(4).to_dict(),
                                      note="st_hist48 equals st_sheaf by construction: with lags {1,24} an unknown cell is adjacent only to cells within 24 h, so observed history beyond 24 h cannot enter the solution.")
        if res.lockdown.nunique() > 1 and "lockdown" in pro:
            R["prolonged_lockdown_split"] = {("lockdown" if f else "pre_lockdown"): g.groupby("method").sq_err.mean().pow(0.5).round(4).to_dict() for f, g in pro[pro.hours_elapsed <= 24].groupby("lockdown")}
    json.dump(R, open(os.path.join(out, "summary.json"), "w"), indent=1, default=float)
    final["regions"][reg] = R
    p1 = {k: (round(v["rel_improvement"], 3), round(v["p_holm"], 4)) for k, v in R["P1"].items()}
    p3 = {k: (round(v["rel_improvement"], 3), round(v["p_holm"], 4)) for k, v in R.get("P3_1_24h", {}).items()}
    print(f"[{reg}] n={R['n_stations']} snapshot sheaf={R['snapshot_overall'].get('vector_sheaf'):.3f} P1={p1} | P3={p3} | C2* rho={R['P4_certificates'].get('c2_star', {}).get('spearman', float('nan')):.2f}", flush=True)

                                                                                                          
fr = os.path.join(ROOT, "results", "FINAL_RESULTS.json")
prev = json.load(open(fr)) if os.path.exists(fr) else {}
regs = prev.get("regions", {}); regs.update(final["regions"])
if regs:
    roll = {}
    for fam, key in (("P1", "P1"), ("P2", "P2"), ("P3", "P3_1_24h")):
        tab = {}
        for reg, R in regs.items():
            for comp, v in R.get(key, {}).items():
                tab.setdefault(comp, {})[reg] = dict(rel=v["rel_improvement"], p_holm=v["p_holm"], sig=v["significant_005_holm"])
        roll[fam] = {comp: dict(per_region=d, n_regions=len(d), n_significant=sum(x["sig"] for x in d.values()),
                                n_sheaf_better=sum(x["rel"] > 0 for x in d.values()),
                                median_rel_improvement=float(np.median([x["rel"] for x in d.values()]))) for comp, d in tab.items()}
    roll["P4"] = {reg: {k: dict(spearman=v["spearman"], p_holm=v.get("p_holm", v["p"])) for k, v in R["P4_certificates"].items() if k in ("c2_star", "inv_sigma_min")} for reg, R in regs.items()}
    roll["prospective"] = {reg: R["prospective_deciles"] for reg, R in regs.items() if "prospective_deciles" in R}
    roll["bound"] = {reg: R["bound_check"] for reg, R in regs.items() if "bound_check" in R}
    roll["mechanism"] = {reg: R["mechanism"] for reg, R in regs.items()}
    roll["temporal_ablation"] = {reg: R.get("temporal_ablation", {}).get("rmse") for reg, R in regs.items()}
    prev.update(dict(protocol=final["protocol"], regions=regs, rollup=roll))
    json.dump(prev, open(fr, "w"), indent=1, default=float)
    print("wrote", fr, "regions:", sorted(regs))
