                                                                            
                                                      

                                                                                                             
                                                                                   
                                                                                           
                                                                                                      
                                                                   
   
import argparse
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from sheafcode.experiments import ROOT              

ap = argparse.ArgumentParser()
ap.add_argument("--regions", default="delhi,ncr,mumbai,kolkata,hyderabad,bengaluru,india,delhi2025,bengaluru_raw")
args = ap.parse_args()
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
FR = json.load(open(os.path.join(ROOT, "results", "FINAL_RESULTS.json")))
regs = [r for r in args.regions.split(",") if r in FR["regions"]]
PAL = {"vector_sheaf": "#2a78d6", "st_sheaf": "#1c4f8e", "scalar_sheaf": "#eb6834", "gaussian_cond": "#4a3aa7", "gnn": "#008300",
       "lowrank": "#1baf7a", "kriging": "#eda100", "idw": "#e87ba4", "scalar_laplacian": "#9085e9", "persistence": "#e34948",
       "hybrid": "#c98500", "seasonal_persistence": "#d55181", "kalman_var": "#6b6b6b"}
LABEL = {"vector_sheaf": "Sheaf (spatial)", "st_sheaf": "Sheaf (space-time)", "scalar_sheaf": "Scalar sheaf", "gaussian_cond": "Dense Gaussian conditional",
         "gnn": "GNN imputer", "lowrank": "Low-rank (SoftImpute)", "kriging": "Kriging", "idw": "IDW", "scalar_laplacian": "Scalar Laplacian",
         "persistence": "Persistence", "hybrid": "Persistence + spatial hybrid", "seasonal_persistence": "Value 24 h earlier", "kalman_var": "Kalman / VAR smoother"}
RL = {"delhi": "Delhi", "ncr": "NCR", "mumbai": "Mumbai", "kolkata": "Kolkata", "hyderabad": "Hyderabad", "bengaluru": "Bengaluru", "india": "India (national)",
      "delhi2025": "Delhi 25-26\n(raw)", "bengaluru_raw": "Bengaluru 19-25\n(raw)"}
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e4df",
                     "grid.linewidth": 0.6, "axes.edgecolor": "#b5b2a9", "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 170})

                                                                                   
meths = ["vector_sheaf", "gaussian_cond", "scalar_sheaf", "gnn", "kriging", "lowrank", "idw"]
fig, ax = plt.subplots(figsize=(11, 4.2))
w = 0.11
for j, m in enumerate(meths):
    ys, lo, hi = [], [], []
    for r in regs:
        R = FR["regions"][r]; v = R["snapshot_overall"].get(m, np.nan); ci = R["snapshot_overall_ci"].get(m, [np.nan, np.nan])
        ys.append(v); lo.append(v - ci[0]); hi.append(ci[1] - v)
    x = np.arange(len(regs)) + (j - len(meths) / 2 + 0.5) * w
    ax.bar(x, ys, w, yerr=[np.nan_to_num(lo), np.nan_to_num(hi)], color=PAL[m], label=LABEL[m], error_kw=dict(lw=0.7, capsize=1.5))
ax.set_xticks(np.arange(len(regs))); ax.set_xticklabels([f"{RL[r]}\n(n={FR['regions'][r]['n_stations']})" for r in regs], fontsize=8.5)
ax.set_ylabel("standardised RMSE on erased stations\n(unit = outage pattern; 95% bootstrap CI)")
ax.set_title("Single- and multi-station outages, leakage-free joint reconstruction (v3)", loc="left")
ax.legend(ncol=4, fontsize=8, loc="upper left", frameon=False)
ax.set_ylim(0, max(1.05, ax.get_ylim()[1]))
plt.tight_layout(); plt.savefig(os.path.join(FIG, "v3_hero1_snapshot.png")); plt.close()

                                                                                    
cols = min(3, len(regs)); rows_ = int(np.ceil(len(regs) / cols))
fig, axes = plt.subplots(rows_, cols, figsize=(4.0 * cols, 3.1 * rows_), squeeze=False)
bins = ["1-3", "4-6", "7-12", "13-24", "25-48", "49-72"]
for i, r in enumerate(regs):
    ax = axes[i // cols][i % cols]; H = FR["regions"][r].get("prolonged_by_horizon", {})
    for m in ["st_sheaf", "kalman_var", "hybrid", "gnn", "persistence", "vector_sheaf"]:
        if m in H:
            ax.plot(range(len(bins)), [H[m].get(b, np.nan) for b in bins], "-o", ms=3, color=PAL[m], label=LABEL[m], lw=1.4 if m == "st_sheaf" else 1)
    ax.set_xticks(range(len(bins))); ax.set_xticklabels(bins, fontsize=8); ax.set_title(RL[r], loc="left", fontsize=10)
    ax.set_ylim(0, None)
    if i % cols == 0:
        ax.set_ylabel("RMSE (z)")
    if i // cols == rows_ - 1:
        ax.set_xlabel("hours since the station went dark")
for k in range(len(regs), rows_ * cols):
    axes[k // cols][k % cols].axis("off")
axes[0][0].legend(fontsize=7.5, frameon=False, loc="lower right")
fig.suptitle("Prolonged outage with 24 h of history: space-time sheaf vs state-space and learned baselines", x=0.01, ha="left", fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "v3_hero2_prolonged.png")); plt.close()

                                                                                       
fig, axes = plt.subplots(2, max(3, len(regs)) if len(regs) <= 5 else 5, figsize=(3.2 * min(5, max(3, len(regs))), 6.2), squeeze=False)
shown = regs[:5]
for i, r in enumerate(shown):
    out = os.path.join(ROOT, "results", f"v3_{r}")
    res = pd.read_csv(os.path.join(out, "test_results.csv"))
    pl = res[res.method == "vector_sheaf"].groupby("pattern").agg(rmse_z=("rmse_z", "mean"), c2=("c2_star", "first"), size=("size", "first"))
    ax = axes[0][i]
    sc = ax.scatter(pl.c2, pl.rmse_z, c=pl["size"], cmap="viridis", s=14, alpha=0.8)
    P4 = FR["regions"][r]["P4_certificates"].get("c2_star", {})
    ax.set_title(f"{RL[r]}: Spearman {P4.get('spearman', np.nan):.2f}", loc="left", fontsize=9.5)
    ax.set_xlabel("C2* (from training model only)"); ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    if i == 0:
        ax.set_ylabel("held-out RMSE of the pattern")
    ax = axes[1][i]
    pdx = FR["regions"][r].get("prospective_deciles")
    if pdx:
        ax.bar([0, 1], [pdx["rmse_safe"], pdx["rmse_risky"]], color=["#2a78d6", "#e34948"],
               yerr=[[pdx["rmse_safe"] - pdx["safe_ci"][0], pdx["rmse_risky"] - pdx["risky_ci"][0]], [pdx["safe_ci"][1] - pdx["rmse_safe"], pdx["risky_ci"][1] - pdx["rmse_risky"]]], capsize=3)
        ax.set_xticks([0, 1]); ax.set_xticklabels([f"safest decile\n(n={pdx['n_safe']})", f"riskiest decile\n(n={pdx['n_risky']})"], fontsize=8)
        ax.set_title(f"pre-registered: ratio {pdx['ratio']:.2f}, p={pdx['p_mannwhitney_one_sided']:.3g}", loc="left", fontsize=9)
        if i == 0:
            ax.set_ylabel("held-out RMSE")
    else:
        ax.axis("off"); ax.text(0.5, 0.5, "too few patterns\nfor a decile test", ha="center", va="center", transform=ax.transAxes, fontsize=9)
for k in range(len(shown), axes.shape[1]):
    axes[0][k].axis("off"); axes[1][k].axis("off")
fig.suptitle("Certificate validity: patterns ranked before any test error was computed", x=0.01, ha="left", fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "v3_hero3_certificates.png")); plt.close()

                                                                                                            
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
ax = axes[0]
for r in regs:
    S = FR.get("studies", {}).get(r, {}).get("structure", {})
    cv = S.get("D_resilience", {}).get("curve")
    if cv:
        ax.plot([c["k"] for c in cv], [c["R_star_c2"] for c in cv], "-o", ms=3, label=RL[r])
ax.set_xlabel("stations failing simultaneously, k"); ax.set_ylabel("R*(k) = worst C2* over connected sets")
ax.set_title("Worst-case resilience curves", loc="left"); ax.legend(fontsize=6.5, frameon=False, ncol=2)
ax = axes[1]
for r in regs:
    S = FR.get("studies", {}).get(r, {}).get("structure", {}); B = S.get("B_thinning")
    if B:
        B = pd.DataFrame(B)
        ax.plot(B.frac_edges, B.frac_identifiable_all, "-o", ms=3, label=RL[r])
ax.set_xlabel("fraction of edges kept"); ax.set_ylabel("fraction of outage patterns identifiable (C1)"); ax.invert_xaxis()
ax.set_title("C1 becomes binding as the graph thins", loc="left"); ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="lower left")
ax = axes[2]
xs = []; names = []
for r in regs:
    R = FR["regions"][r]; g = R["P1"].get("gaussian_cond")
    if g:
        xs.append(100 * g["rel_improvement"]); names.append(RL[r])
ax.barh(range(len(xs)), xs, color=["#2a78d6" if x > 0 else "#e34948" for x in xs]); ax.set_yticks(range(len(xs))); ax.set_yticklabels(names, fontsize=8.5)
ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("relative improvement (%)\n(negative = Gaussian better)")
ax.set_title("Sheaf vs Gaussian conditional, snapshots", loc="left")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "v3_hero4_structure.png")); plt.close()

                                                                                 
fig = plt.figure(figsize=(13, 4.6)); fig.patch.set_facecolor("white")
gs = fig.add_gridspec(1, 4, wspace=0.25)
rng = np.random.default_rng(2)
                                           
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_title("1. Network as a sheaf", loc="left", fontsize=10.5, fontweight="bold")
pts = np.array([[0.2, 0.7], [0.55, 0.85], [0.85, 0.6], [0.35, 0.3], [0.7, 0.25]]); E = [(0, 1), (1, 2), (0, 3), (3, 4), (2, 4), (1, 3)]
for u, v in E:
    ax.plot(pts[[u, v], 0], pts[[u, v], 1], color="#b5b2a9", lw=1.4, zorder=1)
    mid = pts[[u, v]].mean(0); ax.text(mid[0], mid[1] + 0.03, "A", fontsize=7, color="#7a7a7a", ha="center")
for i, (x, y) in enumerate(pts):
    col = "#e34948" if i == 3 else "#2a78d6"
    ax.scatter(x, y, s=260, color=col, zorder=3, edgecolor="white", lw=1.5)
    ax.text(x, y - 0.11, "x = (PM2.5, PM10, NO2, O3)" if i == 1 else "", fontsize=6.5, ha="center", color="#333")
ax.text(pts[3, 0], pts[3, 1] - 0.12, "erased", fontsize=8, ha="center", color="#e34948")
ax.text(0.02, 0.02, "stalks R^p at stations,\nlearned maps A_u, A_v on edges,\nconsistency A_u x_u = A_v x_v", fontsize=8, transform=ax.transAxes, va="bottom")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                       
ax = fig.add_subplot(gs[1]); ax.set_title("2. Coboundary delta and the erased block", loc="left", fontsize=10.5, fontweight="bold")
M = np.zeros((len(E) * 2, 5 * 3))
for e, (u, v) in enumerate(E):
    M[2 * e:2 * e + 2, 3 * u:3 * u + 3] = rng.normal(size=(2, 3)); M[2 * e:2 * e + 2, 3 * v:3 * v + 3] = -rng.normal(size=(2, 3))
ax.imshow(M, cmap="RdBu", vmin=-2, vmax=2, aspect="auto"); ax.add_patch(plt.Rectangle((3 * 3 - 0.5, -0.5), 3, M.shape[0], fill=False, color="#e34948", lw=2))
ax.set_xticks([1, 4, 7, 10, 13]); ax.set_xticklabels(["s1", "s2", "s3", "s4", "s5"]); ax.set_yticks([]); ax.grid(False)
ax.text(0.0, -0.16, "delta_U = red columns. C1: rank(delta_U) = |U|.  C2*: ||G^-1 delta_U^T||.\nC3: predicted RMSE from training residuals. All computed before the outage.", transform=ax.transAxes, fontsize=8, va="top")
                         
ax = fig.add_subplot(gs[2]); ax.set_title("3. Reconstruct by harmonic extension", loc="left", fontsize=10.5, fontweight="bold")
t = np.arange(48); truth = 1.2 * np.sin(t / 24 * 2 * np.pi - 1) + 0.15 * rng.normal(size=48).cumsum() / 4
rec = truth + 0.25 * rng.normal(size=48)
ax.plot(t, truth, color="#333", lw=1.2, label="hidden truth"); ax.plot(t[24:], rec[24:], color="#e34948", lw=1.6, label="sheaf reconstruction")
ax.axvspan(24, 48, color="#e34948", alpha=0.07); ax.set_xlabel("hour"); ax.set_yticks([]); ax.legend(fontsize=7.5, frameon=False, loc="upper left")
ax.text(0.02, 0.04, "argmin ||delta_O x_O + delta_U z||^2 + lam z^T P z\n(all unknown cells solved jointly; nothing pre-filled)", transform=ax.transAxes, fontsize=7.5)
                                                        
ax = fig.add_subplot(gs[3]); ax.set_title("4. Certificates predict held-out error", loc="left", fontsize=10.5, fontweight="bold")
r0 = "delhi" if "delhi" in regs else regs[0]
res = pd.read_csv(os.path.join(ROOT, "results", f"v3_{r0}", "test_results.csv"))
pl = res[res.method == "vector_sheaf"].groupby("pattern").agg(rmse_z=("rmse_z", "mean"), c2=("c2_star", "first"))
ax.scatter(pl.c2, pl.rmse_z, s=10, color="#2a78d6", alpha=0.7); ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4)); ax.set_xlabel("C2* (training model only)"); ax.set_ylabel("held-out RMSE")
P4 = FR["regions"][r0]["P4_certificates"].get("c2_star", {})
ax.text(0.03, 0.93, f"{RL[r0]}: Spearman {P4.get('spearman', np.nan):.2f}\n(pre-registered decile test in report)", transform=ax.transAxes, fontsize=8, va="top")
fig.suptitle("SheafCode: certified reconstruction of dark stations in the CPCB monitoring network", x=0.01, ha="left", fontsize=12, fontweight="bold")
plt.savefig(os.path.join(FIG, "v3_killer_diagram.png"), bbox_inches="tight"); plt.close()
print("figures written to", FIG)
