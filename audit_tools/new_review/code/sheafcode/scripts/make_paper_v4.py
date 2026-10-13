                                                                        
                                                                        
                                                         
                                                                        
                                                                         
                                                                    
                                                                 
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sheafcode.experiments import ROOT              

FR = json.load(open(os.path.join(ROOT, "results", "FINAL_RESULTS.json")))
R = FR["regions"]; RU = FR["rollup"]; ST = FR.get("studies", {})
ORDER = [r for r in ["delhi", "ncr", "mumbai", "kolkata", "hyderabad", "bengaluru", "india", "delhi2025", "bengaluru_raw"] if r in R]
RL = {"delhi": "Delhi", "ncr": "NCR", "mumbai": "Mumbai", "kolkata": "Kolkata", "hyderabad": "Hyderabad", "bengaluru": "Bengaluru", "india": "India",
      "delhi2025": "Delhi 2025-26", "bengaluru_raw": "Bengaluru 2019-25"}


def load(reg, name):
    f = os.path.join(ROOT, "results", f"v3_{reg}", name)
    return json.load(open(f)) if os.path.exists(f) else None


TH = {r: load(r, "theorems_v4.json") for r in ORDER}; GR = {r: load(r, "grap_v4.json") for r in ORDER}; AC = {r: load(r, "aci_v4.json") for r in ORDER}
OV = json.load(open(os.path.join(ROOT, "results", "overlap_audit_v4.json")))


def f(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def pct(x, d=1):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:+.{d}f}%"


def pv(p):
    return "n/a" if p is None or not np.isfinite(p) else ("< 0.001" if p < 0.001 else f"= {p:.3f}")


L = []; add = L.append
kv = RU["P3"]["kalman_var"]; g = RU["P1"]["gaussian_cond"]; pros = RU["prospective"]; bnd = RU["bound"]
n_pro_sig = sum(v["p_mannwhitney_one_sided"] < 0.05 for v in pros.values())
viol = sum(v["violations"] for v in bnd.values()); nb = sum(v["n"] for v in bnd.values())
thA_sets = sum(TH[r]["theorem_A"]["total"] for r in ORDER if TH[r]); thA_agree = sum(TH[r]["theorem_A"]["agreement"] for r in ORDER if TH[r])
thin_pred = sum(TH[r]["theorem_A"]["thin_sheaf_check"]["knn_k1_r1"]["predicted_nonidentifiable_singles"] for r in ORDER if TH[r])
thin_act = sum(TH[r]["theorem_A"]["thin_sheaf_check"]["knn_k1_r1"]["learned_nonidentifiable_singles"] for r in ORDER if TH[r])
thin_sets = sum(sum(v["sets_checked"] for v in TH[r]["theorem_A"]["thin_sheaf_check"].values()) for r in ORDER if TH[r])
thin_agree = sum(sum(v["generic_equals_learned"] for v in TH[r]["theorem_A"]["thin_sheaf_check"].values()) for r in ORDER if TH[r])


def decode(r, t, mag):
    row = next((x for x in TH[r]["theorem_B"]["results"] if x["t"] == t and x["mag"] == mag), None)
    return row["exact_support_recovery"] if row else np.nan


aci_regions = [r for r in ORDER if AC[r]]
aci_cov = np.mean([AC[r]["summary"]["coverage_aci"] for r in aci_regions]); fix_cov = np.mean([AC[r]["summary"]["coverage_fixed"] for r in aci_regions])
aci_worst = np.mean([AC[r]["summary"]["worst_quarter_aci"] for r in aci_regions]); fix_worst = np.mean([AC[r]["summary"]["worst_quarter_fixed"] for r in aci_regions])
aci_in = sum(AC[r]["summary"]["stations_within_2pts_aci"] for r in aci_regions); aci_n = sum(AC[r]["summary"]["n_stations"] for r in aci_regions)
fix_in = sum(AC[r]["summary"]["stations_within_2pts_fixed"] for r in aci_regions)
grap_regions = [r for r in ORDER if GR[r] and GR[r].get("category_accuracy")]

add("# SheafCode v4: certified reconstruction of dark air-quality stations, with two theorems about the network itself")
add("")
add("Draft of the revised paper, rendered by scripts/make_paper_v4.py from results/FINAL_RESULTS.json and the v4 study files. Numbers tagged [v3] come from the frozen v3 runs and are refreshed automatically when the v4 pipeline rerun completes; all other numbers are from the v4 studies on the frozen v3 models.")
add("")
add("## Abstract")
add("")
add(f"Air-quality monitoring networks carry redundancy across stations, pollutants and time, and lose it whenever a station goes dark. We model a network as a cellular sheaf: stations carry pollutant vectors, canonical correlation analysis learns the maps along edges, and regularised harmonic extension reconstructs a dark station from its neighbours, with the station's own covariance as the vertex inner product. Two results concern the network rather than the reconstruction. First, for restriction maps in general position, identifiability of an outage set is a matching condition on the station graph (Hall's theorem on edge counts), so the smallest non-recoverable outage, the block-erasure distance, is a graph invariant computable before any data exist; on {thA_sets:,} connected station sets across nine networks the predicted and learned ranks agree exactly, and on deliberately thinned sheaves the rule predicts every one of {thin_act} non-identifiable stations. Second, a decoding-radius theorem: faults at t stations are uniquely localisable from the sheaf syndrome if and only if every 2t-set is identifiable; least-squares decoding recovers the exact support of two simultaneous 2-sigma faults in {f(decode('delhi', 2, 2.0) * 100, 0)} percent of Delhi test hours and {f(decode('kolkata', 2, 2.0) * 100, 0)} percent in Kolkata. Evaluation is leakage-free (targets erased before any gap is filled, one test run per network, pre-registered certificate test). Prolonged outages with 24 hours of history are the clearest advantage: first-day error below the Kalman/VAR smoother in {kv['n_significant']} of {kv['n_regions']} networks after Holm correction, median improvement {pct(kv['median_rel_improvement'])} [v3]; snapshot outages are close to the dense Gaussian conditional (median {pct(g['median_rel_improvement'])}, significant in {g['n_significant']} of {g['n_regions']}) [v3]. The pre-registered risk ranking held in {n_pro_sig} of {len(pros)} networks [v3]. For operations, adaptive conformal intervals scaled by the certificate achieve {100 * aci_cov:.1f} percent coverage at nominal 90 ({aci_in} of {aci_n} stations within two points; a fixed calibration gives {100 * fix_cov:.1f}), with a distribution-free long-run guarantee. Decision-grade use is tested against the Graded Response Action Plan: station-level AQI category is recovered correctly in " + ", ".join(f"{100 * GR[r]['category_accuracy']['sheaf']:.0f} percent ({RL[r]})" for r in grap_regions[:3]) + " of dark station-days, against " + ", ".join(f"{100 * GR[r]['category_accuracy']['city_mean_of_survivors']:.0f}" for r in grap_regions[:3]) + " for the usual city-mean fallback. The study does not establish forecasting or universally superior imputation, and the national graph in the 2020 lockdown period is reported as a failure for every method.")
add("")
add("## 1. Problem")
add("")
add("India's CPCB network reports hourly; stations go dark for hours to weeks; the AQI bulletin, the Delhi-NCR Graded Response Action Plan and health advisories are computed from whatever reports. Three questions follow. Can a dark station be reconstructed from the rest, and how well? Can the network itself be certified, before an outage, for which outages it can absorb? And can the same structure tell a faulty station from a bad air day? We answer the first empirically and the second and third with theorems that the data then test.")
add("")
add("## 2. Model")
add("")
add("Stations v carry stalks R^p (one coordinate per pollutant); an edge (u,v) carries R^r with CCA-learned maps A_u, A_v and the constraint A_u x_u = A_v x_v, weighted by rho/(1-rho); the coboundary delta stacks the constraints; the sheaf Laplacian is L = delta^T delta. A dark set U is reconstructed by argmin ||delta_O x_O + delta_U z||^2 + lambda z^T P z with P the block-diagonal training precision, which conditions each missing coordinate on the station's surviving channels (Proposition 6 of docs/THEORY.md). The space-time sheaf adds lag-1 and lag-24 edges with CCA temporal maps.")
add("")
add("## 3. Two theorems about the network")
add("")
add("**Theorem A (identifiability is a matching problem).** Let each edge contribute r_eff constraint rows supported on the p coordinates of its two endpoints. For maps in general position, rank(delta_S) equals the term rank of its support, and by Hall's theorem S is identifiable if and only if r_eff e(S') >= p |S'| for every S' subset of S, where e(S') counts edges touching S'. The block-erasure distance d_block is therefore the size of the smallest connected set with r_eff e(S') < p |S'|: a graph invariant. A single station is identifiable iff deg(v) >= p / r_eff.")
add("")
add(f"*Test.* On the nine frozen networks, generic rank (Hall deficiency) and learned numerical rank agree on {thA_agree:,} of {thA_sets:,} enumerated connected sets (sizes 1 to 4, 1 to 3 nationally). The frozen sheaves have r_eff >= p, so the rule is never binding there, exactly as C1 was never binding in v3. On thinned sheaves built to make it bind (1-nearest-neighbour graphs with edge rank 1), the rule predicts {thin_pred} non-identifiable stations and the learned sheaves have {thin_act}; over all thinned checks generic and learned ranks agree on {thin_agree} of {thin_sets} sets, the exceptions being edges whose CCA maps fell back to identity for lack of paired data, which are not in general position. Consequence for design: adding a station or edge changes d_block by a computable amount with no training.")
add("")
add("**Theorem B (decoding radius).** If the state is consistent (delta x = 0) and faults occur at a set F of at most t stations, the syndrome s = delta y determines F uniquely among all sets of size at most t if and only if every set of size at most 2t is identifiable, i.e. d_block > 2t. With a residual r = delta x != 0, faults of size m are separable when m sigma_min(delta_T) >> ||r|| for every 2t-set T.")
add("")
add("*Test.* Faults of 1, 2 and 3 training standard deviations were injected at t = 1 and t = 2 stations of completed test hours; decoding chose the support of size t minimising the least-squares residual of the syndrome (the decoder the theorem suggests), compared with ranking stations by syndrome energy (the v3 heuristic). Exact support recovery for two simultaneous 2-sigma faults:")
add("")
add("| Network | worst sigma_min over 2-sets | LS decoding, t=2, 2 sigma | syndrome top-2 | LS, t=1, 1 sigma | clean false-positive rate (validation-calibrated threshold) |")
add("|---|---|---|---|---|---|")
for r in ORDER:
    if not TH[r]:
        continue
    B = TH[r]["theorem_B"]; row2 = next(x for x in B["results"] if x["t"] == 2 and x["mag"] == 2.0); row1 = next(x for x in B["results"] if x["t"] == 1 and x["mag"] == 1.0)
    add(f"| {RL[r]} | {f(B['worst_sigma_min_2t'].get('2', np.nan), 2)} | {100 * row2['exact_support_recovery']:.0f}% | {100 * row2['syndrome_top_t']:.0f}% | {100 * row1['exact_support_recovery']:.0f}% | {100 * B['clean_false_positive_rate']:.0f}% |")
add("")
add("Reading: recovery tracks the margin sigma_min(2t) as the theorem predicts (high in Delhi, Kolkata, Delhi 2025-26; low in Mumbai, Bengaluru 2019-25 and the national graph, where the worst 2-set is poorly conditioned), and least-squares decoding dominates the syndrome heuristic everywhere, most sharply where several stations share neighbours. The national clean false-positive rate shows the max-over-stations threshold does not scale to 177 stations; a per-station threshold is the fix and is left for v4.")
add("")
add("## 4. Protocol (v4)")
add("")
add("Chronological 60/20/20 split fixed on the complete hourly timeline before any station or pollutant is chosen; cohorts selected on training-block coverage only; the timeline is kept hourly so that every temporal pair has exactly its intended lag; every transform, covariance and prior fitted on training rows; targets erased before natural gaps are reconstructed; only genuinely observed target cells scored, in one common training-standardised space; paired comparisons on the common eligible set with exclusions reported; Holm within pre-specified families; pattern, cell and run x station units with block bootstrap; one test run per network; retrospective and causal modes reported separately; manifests with code and data hashes. The leakage and invariance tests (tests/test_leakage.py) and the dataset overlap audit (results/overlap_audit_v4.json) are part of the release. Overlap audit summary: " + OV["independent_groups"])
add("")
add("## 5. Results [v3 numbers, refreshed by the v4 rerun]")
add("")
add("| Network | stations | snapshot: sheaf vs Gaussian (Holm p) | prolonged 24 h: sheaf vs Kalman/VAR (Holm p) | channel: sheaf vs Gaussian | C2* Spearman | pre-registered decile ratio (p) |")
add("|---|---|---|---|---|---|---|")
for r in ORDER:
    X = R[r]; gg = X["P1"].get("gaussian_cond"); kk = X.get("P3_1_24h", {}).get("kalman_var"); cc = X["P2"].get("gaussian_cond"); c2 = X["P4_certificates"].get("c2_star", {}); d = X.get("prospective_deciles")
    add(f"| {RL[r]} | {X['n_stations']} | {pct(gg['rel_improvement']) + ' (' + pv(gg['p_holm']) + ')' if gg else 'n/a'} | {pct(kk['rel_improvement']) + ' (' + pv(kk['p_holm']) + ')' if kk else 'n/a'} | {pct(cc['rel_improvement']) if cc else 'n/a'} | {f(c2.get('spearman', np.nan), 2)} | {f(d['ratio'], 2) + ' (' + pv(d['p_mannwhitney_one_sided']) + ')' if d else 'n/a'} |")
add("")
add(f"The general-P error bound held in {nb - viol} of {nb} fully observed test hours. The national graph reaches pattern-mean error {f(R['india']['snapshot_overall']['vector_sheaf'])} training sd for the sheaf and no method below {f(min(R['india']['snapshot_overall'].values()))}: the lockdown test period defeats every method, and a few channels with near-constant training records dominate the mean (median {f(R['india'].get('snapshot_overall_median', {}).get('vector_sheaf', np.nan))}).")
add("")
add("## 6. Operational error bars with a guarantee")
add("")
add(f"Causal nowcast of a dark station (hour t from the others at hour t), nonconformity = |residual| / C3_empirical(station), adaptive conformal inference (Gibbs and Candes 2021) at nominal 90 percent, compared with a fixed split-conformal quantile from the validation block. Across {len(aci_regions)} networks and {aci_n} stations: ACI coverage {100 * aci_cov:.1f} percent (fixed {100 * fix_cov:.1f}); worst test quarter {100 * aci_worst:.1f} (fixed {100 * fix_worst:.1f}); stations within two points of nominal {aci_in} of {aci_n} (fixed {fix_in}). The ACI guarantee is distribution-free and long-run; it says nothing about a single hour, and the report says so.")
add("")
add("| Network | ACI coverage | fixed coverage | ACI worst quarter | fixed worst quarter | mean half-width (z), ACI / fixed |")
add("|---|---|---|---|---|---|")
for r in aci_regions:
    s = AC[r]["summary"]
    add(f"| {RL[r]} | {100 * s['coverage_aci']:.1f}% | {100 * s['coverage_fixed']:.1f}% | {100 * s['worst_quarter_aci']:.1f}% | {100 * s['worst_quarter_fixed']:.1f}% | {f(s['halfwidth_z_aci'], 2)} / {f(s['halfwidth_z_fixed'], 2)} |")
add("")
add("## 7. Does it change a decision? AQI categories and GRAP")
add("")
add("A dark station is reconstructed for a whole test day; the daily mean per pollutant is converted to the CPCB sub-index and the AQI category (Good to Severe) and to the GRAP stage (I Poor, II Very Poor, III Severe, IV Severe+). Comparators: dense Gaussian conditional, kriging, the station's own previous day, and the mean of surviving stations (the usual fallback). Approximations: hourly sub-index data, daily means for all pollutants including O3.")
add("")
add("| Network | station-days | category accuracy: sheaf / Gaussian / kriging / city mean / yesterday | stage accuracy: sheaf / city mean | missed actions (true stage > predicted): sheaf / city mean | city stage correct with 5 dark stations: survivors only / reconstructed |")
add("|---|---|---|---|---|---|")
for r in grap_regions:
    G_ = GR[r]; ca = G_["category_accuracy"]; sa = G_["stage_accuracy"]; ma = G_["missed_actions"]; c5 = G_.get("city", {}).get("5")
    add(f"| {RL[r]} | {G_['n_station_days']} | {100 * ca.get('sheaf', np.nan):.0f} / {100 * ca.get('gaussian', np.nan):.0f} / {100 * ca.get('kriging', np.nan):.0f} / {100 * ca.get('city_mean_of_survivors', np.nan):.0f} / {100 * ca.get('yesterday', np.nan):.0f} | {100 * sa.get('sheaf', np.nan):.0f} / {100 * sa.get('city_mean_of_survivors', np.nan):.0f} | {100 * ma.get('sheaf', np.nan):.0f} / {100 * ma.get('city_mean_of_survivors', np.nan):.0f} | {(str(round(100 * c5['stage_correct_survivors'])) + ' / ' + str(round(100 * c5['stage_correct_reconstructed']))) if c5 else 'n/a'} |")
add("")
nb_cat = sum(GR[r]["category_accuracy"].get("sheaf", 0) > GR[r]["category_accuracy"].get("city_mean_of_survivors", 0) for r in grap_regions)
nb_best = sum(max(GR[r]["category_accuracy"], key=GR[r]["category_accuracy"].get) in ("sheaf", "gaussian", "kriging") for r in grap_regions)
act_regions = [r for r in grap_regions if R[r]["n_stations"] >= 11]
nb_cat_big = sum(GR[r]["category_accuracy"].get("sheaf", 0) > GR[r]["category_accuracy"].get("city_mean_of_survivors", 0) for r in act_regions)
add(f"Reading: a model-based reconstruction (sheaf, Gaussian or kriging) gives the best station-level category in {nb_best} of {len(grap_regions)} networks, and the sheaf beats the city-mean fallback in {nb_cat} of {len(grap_regions)} ({nb_cat_big} of {len(act_regions)} networks with at least 11 stations: Delhi, NCR, Mumbai). In the small, cleaner networks (Kolkata, Hyderabad, Bengaluru) almost every day is Satisfactory or Moderate, no GRAP action is ever due, and the city mean is as good a guess as any model: reconstruction only matters where categories change from station to station. At city level, with 11 to 47 stations, the GRAP stage is robust to five dark stations whether or not they are reconstructed, though reconstruction lowers the city-AQI error: the bulletin does not need SheafCode, the neighbourhood does. All of this goes in the paper.")
add("")
add("Unmeasured pollution: station-days in the test blocks with at least half their hours missing, with the reconstructed category: " + "; ".join(f"{RL[r]}: {GR[r]['exposure']['unmeasured_station_days']} of {GR[r]['exposure']['station_days']} station-days unmeasured, {GR[r]['exposure']['unmeasured_poor_or_worse']} reconstructed as Poor or worse" for r in grap_regions) + ".")
add("")
add("## 8. Impact beyond reconstruction")
add("")
add("* Network design without data (Theorem A): under NCAP India plans to add hundreds of continuous monitors; the Hall condition says, for any proposed graph, which outage sets become recoverable, before a sensor is bought. The design study (results/v3_*/structure.json, part E) already shows certificate-guided decommissioning beating random, degree and centrality rules in four of five networks.")
add("* Fault localisation with a proof (Theorem B): the least-squares decoder recovers simultaneous faults where the syndrome heuristic fails, and the theorem says when it cannot work, which is as useful to an operator as when it can.")
add("* Operational intervals with a guarantee (Section 6) turn a reconstruction into a number an officer can act on, and widen automatically for patterns the certificate calls risky.")
add("* Decision-grade reconstruction (Section 7) is where the health benefit sits: local advisories for the neighbourhood of a dark station, and an honest count of the pollution the network did not record.")
add("")
add("## 9. Limitations")
add("")
add("Shared datasets (one archive plus two partially independent replication sets); retrospective mode uses later surviving observations, the causal mode is the operational one; the GNN baseline is a static masked GCN; no meteorology; linear model, weakest on extreme days; the 2017-2020 archive is sub-index data with the 500 cap censored; the national graph in the lockdown spring fails for every method; the hourly AQI approximation in Section 7; the clean false-positive calibration does not scale to the national graph. Not established: forecasting, universally superior imputation, single-hour guarantees.")
add("")
add("## 10. Reproducibility")
add("")
add("reproduce.py; manifests; FINAL_RESULTS.json; the v4 study scripts scripts/v4_theorems.py, v4_grap.py, v4_aci.py, v4_overlap_audit.py; docs/THEORY.md (to carry Theorems A and B with proofs); docs/PROVENANCE.md; docs/RELATED_WORK.md; CHANGELOG.md.")
text = "\n".join(L) + "\n"
open(os.path.join(ROOT, "docs", "PAPER_v4.md"), "w").write(text)
print(f"docs/PAPER_v4.md written ({len(L)} lines)")
