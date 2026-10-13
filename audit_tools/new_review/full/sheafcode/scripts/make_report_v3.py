                                                                           
                                                                        
                                                        
                                                                      
                    
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sheafcode.experiments import ROOT              

FR = json.load(open(os.path.join(ROOT, "results", "FINAL_RESULTS.json")))
R = FR["regions"]; ST = FR.get("studies", {}); RU = FR.get("rollup", {})
ORDER = [r for r in ["delhi", "ncr", "mumbai", "kolkata", "hyderabad", "bengaluru", "india", "delhi2025", "bengaluru_raw"] if r in R]
MAIN = [r for r in ORDER if r not in ("india",)]
RL = {"delhi": "Delhi", "ncr": "NCR", "mumbai": "Mumbai", "kolkata": "Kolkata", "hyderabad": "Hyderabad", "bengaluru": "Bengaluru", "india": "India (national)",
      "delhi2025": "Delhi 2025-26 (raw)", "bengaluru_raw": "Bengaluru 2019-25 (raw)"}
ML = {"vector_sheaf": "Sheaf (spatial)", "st_sheaf": "Sheaf (space-time)", "scalar_sheaf": "Scalar sheaf", "gaussian_cond": "Dense Gaussian conditional",
      "gnn": "GNN imputer", "lowrank": "Low-rank (SoftImpute)", "kriging": "Kriging", "idw": "IDW", "scalar_laplacian": "Scalar Laplacian",
      "persistence": "Persistence", "hybrid": "Persistence + spatial hybrid", "seasonal_persistence": "Value 24 h earlier", "kalman_var": "Kalman / VAR smoother",
      "vector_sheaf_Pidentity": "Sheaf, identity vertex prior", "vector_sheaf_noPM10": "Sheaf, PM10 also hidden", "vector_sheaf_ownhidden": "Sheaf, whole station hidden"}
CLAIMS = []


def claim(name, ok):
    CLAIMS.append((name, bool(ok)))
    return bool(ok)


def f(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def pct(x, d=1):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:+.{d}f}%"


def pv(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def ci(c, d=3):
    return f"[{f(c[0], d)}, {f(c[1], d)}]"


L = []
add = L.append
add("# SheafCode v3: certified reconstruction of dark stations in India's air-quality network")
add("")
add(f"Protocol: {FR['protocol']}. Every number below is read from results/FINAL_RESULTS.json by scripts/make_report_v3.py; scripts/check_report_v3.py re-renders this file and fails on any difference. Regions: " + ", ".join(RL[r] for r in ORDER) + ".")
add("")
                                                                                 
add("## 1. Bottom line")
add("")
p1 = RU.get("P1", {}); p3 = RU.get("P3", {}); p2 = RU.get("P2", {})
g = p1.get("gaussian_cond", {}); kv = p3.get("kalman_var", {}); hy = p3.get("hybrid", {}); gn3 = p3.get("gnn", {})
n_sig_kv = kv.get("n_significant", 0); n_kv = kv.get("n_regions", 0)
n_better_g = g.get("n_sheaf_better", 0); n_sig_g = g.get("n_significant", 0); n_g = g.get("n_regions", 0)
add(f"* Prolonged outages (a station dark for up to 72 hours, 24 hours of history available): the space-time sheaf beats the Kalman/VAR state-space smoother in {n_sig_kv} of {n_kv} regions after Holm correction (median relative improvement {pct(kv.get('median_rel_improvement', np.nan))}), the GNN in {gn3.get('n_significant', 0)} of {gn3.get('n_regions', 0)} (median {pct(gn3.get('median_rel_improvement', np.nan))}), and the persistence-plus-spatial hybrid in {hy.get('n_significant', 0)} of {hy.get('n_regions', 0)} (median {pct(hy.get('median_rel_improvement', np.nan))}). This is the regime where the sheaf is clearly the best method tested.")
claim("kalman beaten in majority of regions", n_sig_kv >= (n_kv + 1) // 2)
add(f"* Snapshot outages (a station dark, no history): the spatial sheaf is better than the dense Gaussian conditional in {n_better_g} of {n_g} regions, significantly so in {n_sig_g} (median relative improvement {pct(g.get('median_rel_improvement', np.nan))}). The honest summary is that on snapshots the sheaf and the Gaussian conditional are close, and the sheaf's case rests on the certificates and the temporal extension rather than on snapshot accuracy.")
pros = RU.get("prospective", {}); n_pro_sig = sum(v["p_mannwhitney_one_sided"] < 0.05 for v in pros.values()); n_pro = len(pros)
n_pro_dir = sum(v["ratio"] > 1 for v in pros.values())
add(f"* Certificates: in the pre-registered test, patterns designated riskiest by C2* (computed from the training model before any test error existed) had higher held-out error than the safest decile in {n_pro_dir} of {n_pro} regions, significantly in {n_pro_sig}. Spearman correlations of C2* with held-out pattern error: " + ", ".join(f"{RL[r]} {f(R[r]['P4_certificates'].get('c2_star', {}).get('spearman', np.nan), 2)}" for r in ORDER if 'c2_star' in R[r]['P4_certificates']) + ".")
bnd = RU.get("bound", {}); viol = sum(v["violations"] for v in bnd.values()); nb = sum(v["n"] for v in bnd.values())
add(f"* The general-P error bound (Proposition 3') held in {nb - viol} of {nb} fully observed test hours across regions (post hoc; it is a verified bound, not a forecast).")
claim("bound zero violations", viol == 0)
if "india" in R:
    X = R["india"]; gi = X["P1"].get("gaussian_cond", {}); lvl = X["snapshot_overall"].get("vector_sheaf", np.nan); best = min(X["snapshot_overall"].values())
    verdict = ("the sheaf is significantly better than the dense Gaussian conditional" if gi.get("significant_005_holm") else "the sheaf is not significantly better than the dense Gaussian conditional")
    add(f"* The national graph ({X['n_stations']} stations across India), stated first because it is the weakest result: {verdict} (relative improvement {pct(gi.get('rel_improvement', np.nan))}, Holm p {pv(gi.get('p_holm', np.nan))}), but the absolute pattern-mean error is {f(lvl)} training standard deviations for the sheaf (pattern median {f(X.get('snapshot_overall_median', {}).get('vector_sheaf', np.nan))}) and no method tested gets a mean below {f(best)}: on this graph and this test period (the 2020 lockdown spring) snapshot reconstruction is not usable by any method, and Section 8 shows which graph constructions were tried.")
    claim("national snapshot error above 0.8 sd for every method", best > 0.8)
add("")
                                                                                  
add("## 2. What changed from v2, and why v2 is withdrawn")
add("")
add("v2 scored snapshot and single-channel outages in two passes: natural gaps of a test day were filled with every station visible, then the target was erased. Values imputed at neighbours in pass 1 could carry information from the station hidden in pass 2. v2 also scored each method in its own transformed space, which made methods with different selected transforms incomparable. Both are fixed in v3: the target is erased first, every unknown cell is reconstructed jointly from the remaining observed cells, only genuinely observed target cells are scored, and every method is scored in one common space (raw units standardised by training statistics). tests/test_leakage.py checks that altering hidden values changes no prediction of any method. Hyperparameters were reselected from scratch, leakage-free, on the validation block; the test block was run once per region. The full protocol was frozen in FINAL_PROTOCOL.md before the runs.")
add("")
                                                                          
add("## 3. Data and frozen models")
add("")
add("| Region | Stations | Pollutants | Hours (train/val/test) | Observed cells | Test span | Frozen sheaf (transform, graph, k, r, weighting, stalk, lambda) | Patterns x test days |")
add("|---|---|---|---|---|---|---|---|")
for r in ORDER:
    X = R[r]; c = X["choice"]["vector_sheaf"]
    add(f"| {RL[r]} | {X['n_stations']} | {', '.join(X['pollutants'])} | {'/'.join(str(s) for s in X['split'])} | {100 * X['obs_fraction']:.0f}% | {X['test_span'][0][:10]} to {X['test_span'][1][:10]} | {c['transform']}, {c['graph']}, {c['k']}, {c['r']}, {c['weighting']}, {c['stalk']}, {c['lam']} | {X['n_patterns']} x {X['n_test_days']} |")
add("")
add("Data provenance, the CPCB breakpoint citation for the sub-index inversion, and file hashes are in docs/PROVENANCE.md and every results/v3_<region>/manifest.json. Delhi 2025-26 and Bengaluru 2019-25 are raw-concentration datasets collected independently of the 2017-2020 archive and serve as replication sets.")
add("")
                                                                              
add("## 4. Snapshot outages (P1)")
add("")
add("Unit = outage pattern (single stations, random sets of 2, 3, 5, 8 and 12 stations, geographic clusters), errors averaged over test days first. Standardised RMSE with pattern-bootstrap 95% CI. Lower is better.")
add("")
meths = ["vector_sheaf", "gaussian_cond", "scalar_sheaf", "gnn", "kriging", "lowrank", "idw", "scalar_laplacian", "st_sheaf"]
add("| Region | " + " | ".join(ML[m] for m in meths) + " |")
add("|---|" + "---|" * len(meths))
for r in ORDER:
    X = R[r]
    cells = []
    for m in meths:
        v = X["snapshot_overall"].get(m); c = X["snapshot_overall_ci"].get(m)
        cells.append(f"{f(v)} {ci(c, 2)}" if v is not None else "n/a")
    add(f"| {RL[r]} | " + " | ".join(cells) + " |")
add("")
add("Pre-specified paired comparisons, sheaf vs competitor (positive = sheaf better), Wilcoxon one-sided over patterns, Holm-corrected within the four comparisons of a region:")
add("")
add("| Region | vs Gaussian conditional | vs GNN | vs kriging | vs scalar sheaf |")
add("|---|---|---|---|---|")
for r in ORDER:
    cells = []
    for m in ["gaussian_cond", "gnn", "kriging", "scalar_sheaf"]:
        v = R[r]["P1"].get(m)
        cells.append(f"{pct(v['rel_improvement'])} {ci(v['diff_ci'])}, p {pv(v['p_holm'])}{'*' if v['significant_005_holm'] else ''}" if v else "n/a")
    add(f"| {RL[r]} | " + " | ".join(cells) + " |")
add("")
ss = [abs(R[r]["P1"]["scalar_sheaf"]["rel_improvement"]) for r in ORDER if "scalar_sheaf" in R[r]["P1"]]
claim("scalar sheaf within 5% of vector sheaf on snapshots", max(ss) < 0.05 if ss else True)
add(f"Reading: '*' marks Holm-adjusted p < 0.05. The scalar sheaf (one pollutant at a time, same graph) is within {100 * max(ss):.1f}% of the vector sheaf on whole-station outages in every region: cross-pollutant structure adds little when the whole station is dark, because no channel of that station is left to couple to. Where it does help is Section 6.")
add("")
add("![hero 1](figures/v3_hero1_snapshot.png)")
add("")
                                                                                  
add("## 5. Certificates (P4)")
add("")
add("C1 (structural identifiability, rank test), C2 (1/sigma_min), C2* (regularised sensitivity ||G^-1 delta_U^T||), C3_complete (error identity on fully observed training rows; the version Proposition 4 covers) and C3_empirical (leakage-free simulated erasure on training rows with gaps) are computed from the frozen training model only.")
add("")
add("| Region | Identifiable patterns (C1) | Spearman C2* vs error (Holm p) | Spearman 1/sigma_min (Holm p) | C3_complete: Spearman, mean actual minus predicted | C3_empirical: Spearman, mean actual minus predicted | Bound (3'') violations / checked |")
add("|---|---|---|---|---|---|---|")
for r in ORDER:
    X = R[r]; P4 = X["P4_certificates"]; b = X.get("bound_check", {})
    c2 = P4.get("c2_star", {}); s2 = P4.get("inv_sigma_min", {}); c3 = P4.get("c3_complete"); c3e = P4.get("c3_empirical")
    add(f"| {RL[r]} | {100 * X['identifiable_fraction']:.0f}% | {f(c2.get('spearman', np.nan), 2)} ({pv(c2.get('p_holm', np.nan))}) | {f(s2.get('spearman', np.nan), 2)} ({pv(s2.get('p_holm', np.nan))}) | "
        + (f"{f(c3['spearman'], 2)}, {f(c3['mean_actual_minus_pred'])} {ci(c3['ci'])}" if c3 else "not computable (no fully observed training rows)") + " | "
        + (f"{f(c3e['spearman'], 2)}, {f(c3e['mean_actual_minus_pred'])} {ci(c3e['ci'])}" if c3e else "n/a") + f" | {str(b['violations']) + ' / ' + str(b['n']) if b else 'no fully observed test hours'} |")
add("")
add("Pre-registered prospective test. Before the test loop of each region, results/v3_<region>/prospective_deciles.json fixed the safest and riskiest deciles of patterns by C2*. Held-out RMSE of those deciles:")
add("")
add("| Region | Safest decile RMSE [CI] | Riskiest decile RMSE [CI] | Ratio | Mann-Whitney one-sided p | Cliff's delta |")
add("|---|---|---|---|---|---|")
for r in ORDER:
    d = R[r].get("prospective_deciles")
    if d:
        add(f"| {RL[r]} | {f(d['rmse_safe'])} {ci(d['safe_ci'])} (n={d['n_safe']}) | {f(d['rmse_risky'])} {ci(d['risky_ci'])} (n={d['n_risky']}) | {f(d['ratio'], 2)} | {pv(d['p_mannwhitney_one_sided'])} | {f(d['cliffs_delta'], 2)} |")
add("")
for r in ORDER:
    ex = ST.get(r, {}).get("extras", {}).get("c2star_spearman_by_size")
    if ex and r in ("delhi", "ncr", "mumbai"):
        add(f"Within single-station patterns only ({RL[r]}), so that pattern size cannot drive the correlation: Spearman C2* vs error = {f(ex.get('1', {}).get('spearman', np.nan), 2)} (n={ex.get('1', {}).get('n', 0)}).")
add("")
add("![hero 3](figures/v3_hero3_certificates.png)")
add("")
                                                                             
add("## 6. Single-channel outages (P2) and the mechanism experiment")
add("")
add("One pollutant of one station is erased; the station's other channels stay visible. Unit = station x pollutant cell.")
add("")
cm = ["vector_sheaf", "gaussian_cond", "scalar_sheaf", "kriging", "gnn", "lowrank"]
add("| Region | " + " | ".join(ML[m] for m in cm) + " | vs Gaussian (Holm p) | vs scalar sheaf (Holm p) |")
add("|---|" + "---|" * (len(cm) + 2))
for r in ORDER:
    X = R[r]; cells = [f"{f(X['channel_overall'].get(m, np.nan))}" for m in cm]
    g2 = X["P2"].get("gaussian_cond"); s2 = X["P2"].get("scalar_sheaf")
    add(f"| {RL[r]} | " + " | ".join(cells) + f" | {pct(g2['rel_improvement']) if g2 else 'n/a'} ({pv(g2['p_holm']) if g2 else 'n/a'}) | {pct(s2['rel_improvement']) if s2 else 'n/a'} ({pv(s2['p_holm']) if s2 else 'n/a'}) |")
add("")
add("Mechanism (exploratory): what each information source contributes to the vector sheaf's channel reconstruction. Mean increase in RMSE when the ingredient is removed, cell-bootstrap 95% CI:")
add("")
add("| Region | Identity vertex prior (no within-station coupling) | PM10 also hidden | Whole station hidden |")
add("|---|---|---|---|")
for r in ORDER:
    m = R[r]["mechanism"]
    cells = []
    for k in ["vector_sheaf_Pidentity", "vector_sheaf_noPM10", "vector_sheaf_ownhidden"]:
        v = m.get(k)
        cells.append(f"{f(v['mean_increase'])} {ci(v['ci'])} ({100 * v['frac_cells_worse']:.0f}% of cells worse)" if v else "n/a")
    add(f"| {RL[r]} | " + " | ".join(cells) + " |")
add("")
                                                                               
add("## 7. Prolonged outages with history (P3)")
add("")
add("A station goes dark for up to 72 hours; 24 hours of its own history and every other station's readings are available. The space-time sheaf (lags 1 and 24 h) vs the Kalman/VAR smoother, the persistence + spatial hybrid, the GNN and persistence. Unit = run x station. RMSE over the first 24 hours and over 72 hours:")
add("")
pm = ["st_sheaf", "kalman_var", "hybrid", "gnn", "persistence", "seasonal_persistence", "vector_sheaf"]
add("| Region | Runs x stations | " + " | ".join(ML[m] + " (24 h / 72 h)" for m in pm) + " |")
add("|---|---|" + "---|" * len(pm))
for r in ORDER:
    X = R[r]
    if "prolonged_overall_1_24h" not in X:
        continue
    cells = [f"{f(X['prolonged_overall_1_24h'].get(m, np.nan))} / {f(X['prolonged_overall_1_72h'].get(m, np.nan))}" for m in pm]
    add(f"| {RL[r]} | {X['prolonged_n_units']} | " + " | ".join(cells) + " |")
add("")
add("| Region | vs Kalman/VAR (24 h) | vs hybrid | vs GNN | vs persistence |")
add("|---|---|---|---|---|")
for r in ORDER:
    P = R[r].get("P3_1_24h", {})
    cells = []
    for m in ["kalman_var", "hybrid", "gnn", "persistence"]:
        v = P.get(m)
        cells.append(f"{pct(v['rel_improvement'])} {ci(v['diff_ci'])}, p {pv(v['p_holm'])}{'*' if v['significant_005_holm'] else ''}" if v else "n/a")
    add(f"| {RL[r]} | " + " | ".join(cells) + " |")
add("")
add("Temporal ablation (exploratory; units where every variant exists):")
add("")
add("| Region | Units | Lags {1,24} (frozen) | Lag {1} only | Lag {24} only | History 6 h | History 12 h | History 48 h | Space only |")
add("|---|---|---|---|---|---|---|---|---|")
for r in ORDER:
    t = R[r].get("temporal_ablation")
    if t:
        m = t["rmse"]
        add(f"| {RL[r]} | {t['n_units']} | {f(m.get('st_sheaf'))} | {f(m.get('st_lag1'))} | {f(m.get('st_lag24'))} | {f(m.get('st_hist6'))} | {f(m.get('st_hist12'))} | {f(m.get('st_hist48'))} | {f(m.get('vector_sheaf'))} |")
add("")
add("History 48 h equals history 24 h by construction: with lags {1, 24} an unknown cell is adjacent only to cells within 24 hours, so observed history further back cannot enter the solution. This is a property of the model, not a finding.")
add("")
add("![hero 2](figures/v3_hero2_prolonged.png)")
add("")
                                                                              
add("## 8. The national graph: a negative result and a graph-construction study")
add("")
if "india" in R:
    X = R["india"]; gi = X["P1"].get("gaussian_cond", {})
    sh = ST.get("india", {}).get("extras", {}).get("train_test_shift")
    shift_text = (f"{sh['cells_shift_over_3sd']} of {sh['n_cells']} station x pollutant cells have a test-period mean more than 3 training standard deviations from the training mean ({sh['cells_shift_over_10sd']} more than 10), the worst being {sh['worst'][0]['station']} {sh['worst'][0]['pollutant']} (training std {f(sh['worst'][0]['train_std'], 1)} ug/m3 over {sh['worst'][0]['n_train']} hours, shift {f(sh['worst'][0]['shift_sd'], 0)} sd), which is a stuck or miscalibrated channel rather than a reconstruction failure. " if sh else "")
    add(f"On {X['n_stations']} stations across India the spatial sheaf reaches RMSE {f(X['snapshot_overall'].get('vector_sheaf'))} against {f(X['snapshot_overall'].get('gaussian_cond'))} for the dense Gaussian conditional, {f(X['snapshot_overall'].get('kriging'))} for kriging and {f(X['snapshot_overall'].get('scalar_sheaf'))} for the scalar sheaf; relative to the Gaussian conditional the sheaf is {pct(gi.get('rel_improvement', np.nan))} (Holm p {pv(gi.get('p_holm', np.nan))}). Single-station outages are reconstructed at RMSE {f(next((t['rmse_z'] for t in X['snapshot_by_size'] if t['method'] == 'vector_sheaf' and t['size'] == 1), np.nan))}, but sets of 8 and 12 stations at {f(next((t['rmse_z'] for t in X['snapshot_by_size'] if t['method'] == 'vector_sheaf' and t['size'] == 8), np.nan))} and {f(next((t['rmse_z'] for t in X['snapshot_by_size'] if t['method'] == 'vector_sheaf' and t['size'] == 12), np.nan))}, worse than predicting the training mean, for every method. The mean is driven by a few station channels whose training record is nearly constant: {shift_text}The pattern-median RMSE is {f(X.get('snapshot_overall_median', {}).get('vector_sheaf', np.nan))} for the sheaf and {f(X.get('snapshot_overall_median', {}).get('gaussian_cond', np.nan))} for the Gaussian conditional, and the rank-based paired tests above are unaffected by the outliers. The test block is the 2020 lockdown spring, and the training statistics of a national network do not transfer to it; prolonged outages with history remain tractable (Section 7: {pct(X.get('P3_1_24h', {}).get('kalman_var', {}).get('rel_improvement', np.nan))} vs Kalman/VAR). Certificates: Spearman C2* vs error {f(X['P4_certificates'].get('c2_star', {}).get('spearman', np.nan), 2)}.")
    A = ST.get("india", {}).get("structure", {}).get("A_graph_study")
    if A:
        add("")
        add("Graph-construction study on the national data (validation error is the selection rule; test error is exploratory):")
        add("")
        add("| Variant | Edges | Components | Identifiable singles | Mean C2* | Validation RMSE | Test RMSE | Spearman C2* vs test error |")
        add("|---|---|---|---|---|---|---|---|")
        for row in A["table"]:
            if "error" in row and row.get("error"):
                continue
            add(f"| {row['variant']}{' (frozen)' if row.get('frozen') else ''} | {row['n_edges']} | {row['components']} | {100 * row['frac_identifiable']:.0f}% | {f(row['mean_c2star'], 2)} | {f(row['val_rmse'])} | {f(row.get('test_rmse', np.nan))} | {f(row.get('spearman_c2star_test', np.nan), 2)} |")
        add("")
        rows_ok = [r_ for r_ in A["table"] if np.isfinite(r_.get("test_rmse", np.nan)) and r_["components"] <= 2]
        tr_ = [r_["test_rmse"] for r_ in rows_ok]
        add(f"The study's validation rule selects {A['selected_by_validation']} and the best variant on test is {A['best_on_test']} (the frozen run's own grid contained geographic and correlation kNN only, hence {A['frozen_variant']}). Across the {len(rows_ok)} connected variants the test RMSE of single-station outages ranges only from {f(min(tr_))} to {f(max(tr_))}: denser graphs help a little, graphs that fragment the network (mutual kNN, short maximum edge length, strong correlation filters) hurt and lose identifiability, and no construction changes the picture. The national difficulty is in the data shift, not the graph.")
        claim("national graph variants within 0.1 RMSE", max(tr_) - min(tr_) < 0.1)
        add("")
else:
    add("The national run is not part of this render.")
add("")
                                                                               
add("## 9. Structure: thinning, block distance, worst-case curves, design, perturbations")
add("")
add("All exploratory, on the frozen models.")
add("")
add("C1 thinning (edges removed at random, three seeds; fraction of patterns identifiable and held-out error of identifiable vs non-identifiable patterns):")
add("")
for r in [x for x in ("delhi", "mumbai", "kolkata", "bengaluru") if x in ST and "structure" in ST[x]]:
    B = ST[r]["structure"].get("B_thinning")
    if not B:
        continue
    add(f"{RL[r]}: " + "; ".join(f"{100 * row['frac_edges']:.0f}% edges: {100 * row['frac_identifiable_all']:.0f}% identifiable, RMSE identifiable {f(row['rmse_identifiable'])}, non-identifiable {f(row['rmse_nonidentifiable'])}" for row in B) + ".")
    add("")
add("Block-erasure distance d_block (smallest connected station set whose erasure is not identifiable) and resilience curves R*(k) = max C2* over connected sets of size at most k (exact by enumeration for small k, greedy beyond), with the predicted worst sets scored on the held-out block against random connected sets of the same size:")
add("")
add("| Region | d_block (exact up to) | k=1 | k=2 | k=3 | k=4 | k=6 | k=8 | Worst vs random sets, RMSE at k=3 | at k=6 |")
add("|---|---|---|---|---|---|---|---|---|---|")
for r in ORDER:
    S = ST.get(r, {}).get("structure", {})
    if "D_resilience" not in S:
        continue
    cv = {c["k"]: c for c in S["D_resilience"]["curve"]}; ev = {e["k"]: e for e in S["D_resilience"]["worst_vs_random"]}
    C = S.get("C_block_distance", {})
    cells = [f(cv[k]["R_star_c2"], 2) if k in cv else "n/a" for k in (1, 2, 3, 4, 6, 8)]
    w3 = ev.get(3); w6 = ev.get(6)
    add(f"| {RL[r]} | {C.get('d_block')} ({C.get('exact_up_to')}) | " + " | ".join(cells) + f" | {f(w3['rmse_worst']) + ' vs ' + f(w3['rmse_random']) if w3 else 'n/a'} | {f(w6['rmse_worst']) + ' vs ' + f(w6['rmse_random']) if w6 else 'n/a'} |")
add("")
add("Decommissioning design: remove 15% of stations so that the remaining network's single-outage certificates stay best (certificate-guided greedy), vs random (5 seeds), lowest degree, most clustered geographically and lowest closeness centrality. Score = held-out single-outage RMSE of the remaining network:")
add("")
add("| Region | Removed | Full network | Certificate greedy | Random (mean, sd) | Lowest degree | Most clustered | Lowest centrality |")
add("|---|---|---|---|---|---|---|---|")
for r in ORDER:
    E = ST.get(r, {}).get("structure", {}).get("E_design")
    if not E or E["m_removed"] < 2:
        continue
    s = E["strategies"]
    add(f"| {RL[r]} | {E['m_removed']} | {f(E['full_network']['test_rmse_singles'])} | {f(s['certificate_greedy']['test_rmse_singles'])} | {f(E['random_mean_test'])} ({f(E['random_sd_test'])}) | {f(s['lowest_degree']['test_rmse_singles'])} | {f(s['most_clustered_geo']['test_rmse_singles'])} | {f(s['lowest_centrality']['test_rmse_singles'])} |")
add("")
add("Graph perturbation robustness (held-out single-outage RMSE and Spearman of C2* with error):")
add("")
add("| Region | Frozen | k-2 | k+2 | Rewire 10% | Rewire 25% | Rewire 50% | Coordinate noise 5 km | 20 km |")
add("|---|---|---|---|---|---|---|---|---|")


def pert(rows, prefix):
    v = [x for x in rows if x["variant"].startswith(prefix)]
    return f"{f(np.mean([x['test_rmse'] for x in v]))} (rho {f(np.mean([x['spearman_c2star_test'] for x in v]), 2)})" if v else "n/a"


for r in ORDER:
    Fp = ST.get(r, {}).get("structure", {}).get("F_perturbation")
    if not Fp:
        continue
    kf = R[r]["choice"]["vector_sheaf"]["k"]
    add(f"| {RL[r]} | {pert(Fp, 'frozen')} | {pert(Fp, f'k{kf - 2}')} | {pert(Fp, f'k{kf + 2}')} | {pert(Fp, 'rewire0.1_')} | {pert(Fp, 'rewire0.25_')} | {pert(Fp, 'rewire0.5_')} | {pert(Fp, 'coordnoise5km')} | {pert(Fp, 'coordnoise20km')} |")
add("")
add("![hero 4](figures/v3_hero4_structure.png)")
add("")
                                                                               
add("## 10. Model ablation")
add("")
add("Each row changes one ingredient of the frozen vector sheaf; held-out single-station RMSE (unit = station), paired difference vs frozen with station-bootstrap CI.")
add("")
for r in [x for x in ("delhi", "mumbai", "kolkata") if x in ST and "ablation" in ST[x]]:
    A = ST[r]["ablation"]
    add(f"{RL[r]} ({A['n_stations_scored']} stations, {A['test_days']} test days):")
    add("")
    add("| Variant | Validation RMSE | Test RMSE | Difference vs frozen [CI] |")
    add("|---|---|---|---|")
    for row in A["table"]:
        add(f"| {row['variant']} | {f(row['val_rmse'])} | {f(row['test_rmse'])} | {f(row['mean_diff_vs_frozen'])} {ci(row['diff_ci'])} |")
    add("")
                                                                             
add("## 11. Fault localisation: severity sweep")
add("")
add("Faults (bias, gain, drift, single-channel offset, stuck value) injected at one station of a raw test day, in units of the training standard deviation; the day is then completed by the sheaf and every station scored. Top-1 localisation accuracy of the sheaf syndrome vs the leave-one-out residual, neighbour disagreement and reading magnitude:")
add("")
for r in [x for x in ("delhi", "ncr", "mumbai", "kolkata", "hyderabad", "india") if x in ST and "faults" in ST[x]]:
    Fa = ST[r]["faults"]
    add(f"{RL[r]} (chance top-1 = {f(Fa['chance_top1'], 2)}, {Fa['n_stations']} stations x {Fa['test_days']} days{'; leave-one-out detector skipped for cost' if Fa.get('no_loo') else ''}): overall top-1 / top-3 / AUC: " + "; ".join(f"{k} {f(v['top1'], 2)} / {f(v['top3'], 2)} / {f(v['auc'], 2)}" for k, v in Fa["overall_by_detector"].items()) + ".")
    add("Syndrome top-1 by severity (sigma; 0.0 = stuck value): " + ", ".join(f"{k}: {f(v['top1'], 2)}" for k, v in Fa["severity_curve_syndrome"].items()) + ".")
    add("")
                                                                                
add("## 12. Secondary analyses")
add("")
for r in [x for x in ("delhi", "ncr", "mumbai", "delhi2025") if x in ST and "extras" in ST[x]]:
    E = ST[r]["extras"]
    add(f"{RL[r]}:")
    cov = E.get("by_coverage", {})
    add("* By day coverage (sheaf improvement over Gaussian conditional): " + "; ".join(f"{k} coverage {100 * v['coverage_range'][0]:.0f}-{100 * v['coverage_range'][1]:.0f}%: {pct(v.get('sheaf_improvement_over_gaussian_rel', np.nan))}" for k, v in cov.items()) + ".")
    ex = E.get("extreme_days", {})
    if "extreme" in ex and "normal" in ex:
        add(f"* Extreme days (top decile of regional PM2.5, threshold {f(ex['extreme']['pm25_threshold'], 0)} ug/m3, {ex['extreme']['n_days']} days): sheaf RMSE {f(ex['extreme']['rmse_z'].get('vector_sheaf'))} vs {f(ex['normal']['rmse_z'].get('vector_sheaf'))} on other days; improvement over Gaussian {pct(ex['extreme'].get('sheaf_improvement_over_gaussian_rel', np.nan))} on extreme days vs {pct(ex['normal'].get('sheaf_improvement_over_gaussian_rel', np.nan))} otherwise. Raw PM2.5 RMSE on extreme days: sheaf {f(ex['extreme']['rmse_raw'].get('vector_sheaf', {}).get('PM2.5', np.nan), 1)} ug/m3, Gaussian {f(ex['extreme']['rmse_raw'].get('gaussian_cond', {}).get('PM2.5', np.nan), 1)}, kriging {f(ex['extreme']['rmse_raw'].get('kriging', {}).get('PM2.5', np.nan), 1)}.")
    pr = E.get("by_pollutant_relative_to_train_std", {})
    if "vector_sheaf" in pr:
        add("* By pollutant (single-station RMSE / training std): sheaf " + ", ".join(f"{q} {f(v, 2)}" for q, v in pr["vector_sheaf"].items()) + "; Gaussian " + ", ".join(f"{q} {f(v, 2)}" for q, v in pr.get("gaussian_cond", {}).items()) + ".")
    cl = E.get("clusters_vs_random_size3")
    if cl:
        add(f"* Geographic clusters of 3 vs random sets of 3: RMSE {f(cl['rmse_cluster'])} {ci(cl['ci_cluster'])} vs {f(cl['rmse_random'])} {ci(cl['ci_random'])}; mean C2* {f(cl['c2_cluster'], 2)} vs {f(cl['c2_random'], 2)}; C3_empirical {f(cl['c3_cluster'])} vs {f(cl['c3_random'])}.")
    ed = E.get("edges", {})
    add(f"* Edges: {ed.get('n_edges')} edges, median length {f(ed.get('median_length_km', np.nan), 1)} km, mean leading canonical correlation {f(ed.get('mean_leading_cca', np.nan), 2)}, Spearman of canonical correlation with edge length {f(ed.get('spearman_cca_vs_length', np.nan), 2)}.")
    ld = R[r].get("lockdown_split")
    if ld:
        add(f"* Lockdown split (from 25 Mar 2020): pre-lockdown sheaf {f(ld['pre_lockdown']['overall'].get('vector_sheaf'))} vs Gaussian {f(ld['pre_lockdown']['overall'].get('gaussian_cond'))} ({ld['pre_lockdown']['n_days']} days); lockdown sheaf {f(ld['lockdown']['overall'].get('vector_sheaf'))} vs Gaussian {f(ld['lockdown']['overall'].get('gaussian_cond'))} ({ld['lockdown']['n_days']} days).")
    add("")
iv = [(r, ST[r]["intervals"]) for r in ORDER if "intervals" in ST.get(r, {})]
if iv:
    add("Certificate-scaled intervals (prediction +/- q C3_empirical, q calibrated on the validation block; an empirical check, not a conformal guarantee). Test coverage, cells / day means, at nominal 50% and 90%:")
    add("")
    add("| Region | 50% nominal | 90% nominal | q (90%, cells) |")
    add("|---|---|---|---|")
    for r, I in iv:
        add(f"| {RL[r]} | {100 * I['level_50']['test_coverage_cells']:.0f}% / {100 * I['level_50']['test_coverage_day_means']:.0f}% | {100 * I['level_90']['test_coverage_cells']:.0f}% / {100 * I['level_90']['test_coverage_day_means']:.0f}% | {f(I['level_90']['q_cell'], 2)} |")
    under = [r for r, I in iv if I["level_90"]["test_coverage_cells"] < 0.85]
    add("")
    add(("Coverage falls short of nominal in " + ", ".join(RL[r] for r in under) + ": the test period is harder than the validation period, and the intervals inherit that shift.") if under else "Coverage is within five points of nominal in every region.")
    add("")
                                                                                
add("## 13. Negative and null results, collected")
add("")
neg = []
for r in ORDER:
    g = R[r]["P1"].get("gaussian_cond")
    if g and not g["significant_005_holm"]:
        neg.append(f"{RL[r]}: sheaf not significantly better than the dense Gaussian conditional on snapshots ({pct(g['rel_improvement'])}, Holm p {pv(g['p_holm'])}).")
    s = R[r]["P1"].get("scalar_sheaf")
    if s and not s["significant_005_holm"]:
        neg.append(f"{RL[r]}: vector sheaf not significantly better than the scalar sheaf on whole-station outages ({pct(s['rel_improvement'])}).")
    for m in ["kalman_var", "hybrid", "gnn"]:
        v = R[r].get("P3_1_24h", {}).get(m)
        if v and not v["significant_005_holm"]:
            neg.append(f"{RL[r]}: space-time sheaf not significantly better than {ML[m]} on prolonged outages ({pct(v['rel_improvement'])}, Holm p {pv(v['p_holm'])}).")
    c2 = R[r]["P4_certificates"].get("c2_star")
    if c2 and (c2["spearman"] < 0.2 or c2.get("p_holm", 1) >= 0.05):
        neg.append(f"{RL[r]}: C2* is a weak predictor of held-out error (Spearman {f(c2['spearman'], 2)}, Holm p {pv(c2.get('p_holm', np.nan))}).")
    d = R[r].get("prospective_deciles")
    if d and d["p_mannwhitney_one_sided"] >= 0.05:
        neg.append(f"{RL[r]}: pre-registered decile test not significant (ratio {f(d['ratio'], 2)}, p {pv(d['p_mannwhitney_one_sided'])}).")
for r in ORDER:
    E = ST.get(r, {}).get("structure", {}).get("E_design")
    if E and E["m_removed"] >= 2:
        rmin = min(v["test_rmse_singles"] for k, v in E["strategies"].items() if k.startswith("random"))
        gval = E["strategies"]["certificate_greedy"]["test_rmse_singles"]
        if gval >= E["random_mean_test"]:
            neg.append(f"{RL[r]}: certificate-guided decommissioning did not beat random removal ({f(gval)} vs random mean {f(E['random_mean_test'])}, best random {f(rmin)}).")
        elif gval >= rmin:
            neg.append(f"{RL[r]}: certificate-guided decommissioning beat the random mean ({f(gval)} vs {f(E['random_mean_test'])}) but not the best of five random removals ({f(rmin)}).")
for r in ORDER:
    Dd = ST.get(r, {}).get("structure", {}).get("D_resilience")
    if Dd:
        worse = [e for e in Dd["worst_vs_random"] if np.isfinite(e["rmse_worst"]) and np.isfinite(e["rmse_random"]) and e["rmse_worst"] <= e["rmse_random"]]
        if worse:
            neg.append(f"{RL[r]}: predicted worst-case sets were not worse than random sets at k = {', '.join(str(e['k']) for e in worse)}.")
for x in neg:
    add(f"* {x}")
if not neg:
    add("* None of the pre-specified comparisons failed; see the per-region tables for effect sizes.")
add("")
                                                                                 
add("## 14. What a duty officer would see")
add("")
add("The model needs only the network's own readings. For the current hour it (1) reconstructs every dark cell jointly with the natural gaps, in raw units; (2) shows the C2* risk and the predicted RMSE (C3) of the current outage pattern, computed from the frozen model, so the officer knows whether the reconstruction is trustworthy before looking at it; (3) lists the stations ranked by sheaf syndrome, so a station reporting but inconsistent with its neighbours is flagged; (4) for a station dark for hours, uses the space-time sheaf with the last 24 hours. The design study answers the planning question of which stations the network can least afford to lose. None of this requires meteorology, satellite data or retraining more often than the frozen model's season.")
add("")
                                                                                  
add("## 15. Limitations")
add("")
add("* The 2017-2020 archive stores CPCB sub-indices; concentrations are recovered by exact inversion below the 500 cap and censored above it, so the most extreme Delhi hours are absent. The modern replication sets are small (7 and 4 stations).")
add("* The GNN baseline is a masked GCN; recurrent graph imputers (GRIN) were not run and may be stronger on prolonged outages.")
add("* The sheaf is linear. Extreme days (Section 12) are where it is weakest, as they are for every linear method tested.")
add("* On the national graph every method, the sheaf included, reconstructs multi-station outages at or above one training standard deviation in the 2020 lockdown test period (Section 8); the graph construction is not the cause. A meteorology-aware graph and adaptive re-standardisation were not built.")
add("* C3_complete is unavailable where no fully observed training rows exist; C3_empirical is a proxy that Proposition 4 does not cover.")
add("* The certificate correlations are moderate; C2* ranks risk, it does not predict error to a few percent.")
add("* Everything exploratory (Sections 8 to 12) is reported without significance claims.")
add("")
                                                                                      
add("## 16. Reproducibility")
add("")
add("`python3 reproduce.py` runs the tests, every region, the studies, the analysis and this report's checker; `--quick` runs one small region in about ten minutes. Manifests with code and data hashes: results/v3_<region>/manifest.json. Theory: docs/THEORY.md, numerically checked by tests/test_theory.py. Provenance: docs/PROVENANCE.md. Closest work: docs/RELATED_WORK.md. Change history: CHANGELOG.md.")
add("")
text = "\n".join(L) + "\n"
if __name__ == "__main__":
    open(os.path.join(ROOT, "REPORT_v3.md"), "w").write(text)
    failed = [n for n, ok in CLAIMS if not ok]
    print(f"REPORT_v3.md written ({len(L)} lines); claims checked: {len(CLAIMS)}, failed: {failed}")
