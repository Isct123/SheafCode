                                                                           
                                                                      
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sheafcode.experiments import ROOT              

S = json.load(open(os.path.join(ROOT, "results", "v2_summary.json")))


def t1(reg, method, size=1, col="rmse_z"):
    return next(r[col] for r in S[reg]["table1"] if r["method"] == method and r["size"] == size)


def paired(reg, method, col):
    return next(r[col] for r in S[reg]["paired_vs_sheaf"] if r["method"] == method)


def cert(reg, size, col):
    return next(r[col] for r in S[reg]["certificates"] if r["size"] == size)


def pro(reg, method, b):
    return S[reg]["prolonged"][method][b]


C = []
for reg, n in [("delhi", 33), ("ncr", 47), ("mumbai", 11), ("kolkata", 10), ("hyderabad", 5), ("bengaluru", 7)]:
    C.append((f"{reg} stations", S[reg]["n_stations"], n))
    C.append((f"{reg} kernel", S[reg]["kernel_dim"], 0))
C += [("delhi split", S["delhi"]["split"][0], 10111), ("ncr split", S["ncr"]["split"][2], 2390), ("hyd split", S["hyderabad"]["split"][0], 12470),
      ("delhi patterns", S["delhi"]["n_patterns"], 256), ("ncr patterns", S["ncr"]["n_patterns"], 230), ("mumbai patterns", S["mumbai"]["n_patterns"], 178),
      ("kolkata patterns", S["kolkata"]["n_patterns"], 87), ("hyd patterns", S["hyderabad"]["n_patterns"], 18), ("blr patterns", S["bengaluru"]["n_patterns"], 63)]
                
tab = {"delhi": (0.539, 0.481, 0.604, 0.544, 0.553, 0.555, 0.540, 0.547, 0.558),
       "ncr": (0.603, 0.555, 0.654, 0.607, 0.603, 0.636, 0.612, 0.619, 0.625),
       "mumbai": (0.693, 0.595, 0.821, 0.683, 0.700, 0.745, 0.734, 0.724, 0.790),
       "kolkata": (0.516, 0.408, 0.651, 0.515, 0.557, 0.572, 0.548, 0.546, 0.554),
       "hyderabad": (0.570, 0.488, 0.679, 0.560, 0.593, 0.546, 0.554, 0.555, 0.587),
       "bengaluru": (0.645, 0.563, 0.746, 0.640, 0.632, 0.667, 0.667, 0.677, 0.701)}
for reg, v in tab.items():
    C += [(f"{reg} sheaf", t1(reg, "vector_sheaf"), v[0]), (f"{reg} lo", t1(reg, "vector_sheaf", 1, "ci_lo"), v[1]), (f"{reg} hi", t1(reg, "vector_sheaf", 1, "ci_hi"), v[2]),
          (f"{reg} scalar", t1(reg, "scalar_sheaf"), v[3]), (f"{reg} dense", t1(reg, "gaussian_cond"), v[4]), (f"{reg} gnn", t1(reg, "gnn"), v[5]),
          (f"{reg} kriging", t1(reg, "kriging"), v[6]), (f"{reg} idw", t1(reg, "idw"), v[7]), (f"{reg} lowrank", t1(reg, "lowrank"), v[8])]
C += [("delhi kriging diff", paired("delhi", "kriging", "mean_diff_vs_sheaf"), 0.000), ("delhi kriging p", paired("delhi", "kriging", "wilcoxon_p"), 0.96),
      ("ncr dense diff", paired("ncr", "gaussian_cond", "mean_diff_vs_sheaf"), 0.000), ("ncr dense p", paired("ncr", "gaussian_cond", "wilcoxon_p"), 0.76),
      ("mumbai dense diff", paired("mumbai", "gaussian_cond", "mean_diff_vs_sheaf"), 0.012), ("mumbai dense frac", paired("mumbai", "gaussian_cond", "frac_sheaf_better"), 0.78),
      ("kolkata idw diff", paired("kolkata", "idw", "mean_diff_vs_sheaf"), 0.031), ("kolkata idw frac", paired("kolkata", "idw", "frac_sheaf_better"), 0.89),
      ("hyd gnn diff", paired("hyderabad", "gnn", "mean_diff_vs_sheaf"), -0.019), ("hyd gnn p", paired("hyderabad", "gnn", "wilcoxon_p"), 0.003),
      ("blr dense diff", paired("bengaluru", "gaussian_cond", "mean_diff_vs_sheaf"), -0.008), ("blr dense p", paired("bengaluru", "gaussian_cond", "wilcoxon_p"), 0.0005),
      ("delhi size12", t1("delhi", "vector_sheaf", 12), 0.583), ("ncr size12", t1("ncr", "vector_sheaf", 12), 0.606),
      ("delhi gnn diff", paired("delhi", "gnn", "mean_diff_vs_sheaf"), 0.010), ("mumbai gnn diff", paired("mumbai", "gnn", "mean_diff_vs_sheaf"), 0.056),
      ("kolkata dense diff", paired("kolkata", "gaussian_cond", "mean_diff_vs_sheaf"), 0.043)]
              
certs = {"delhi": (256, 0.52, 0.56, 0.52, 0.51, 0.57, 0.76), "ncr": (227, 0.33, 0.44, 0.41, 0.47, 0.61, 0.44), "mumbai": (178, 0.48, 0.90, 0.90, 0.55, 0.73, 0.10),
         "kolkata": (87, 0.51, 0.60, 0.61, 0.39, 0.54, 0.41), "hyderabad": (18, 0.82, 0.48, 0.55, 0.57, 0.58, 0.94), "bengaluru": (63, 0.71, 0.68, 0.76, 0.68, 0.69, 0.98)}
for reg, v in certs.items():
    C += [(f"{reg} cert n", cert(reg, 0, "n"), v[0]), (f"{reg} sp sigma", cert(reg, 0, "spearman_sigma"), v[1]), (f"{reg} sp pred", cert(reg, 0, "spearman_pred"), v[2]),
          (f"{reg} pearson", cert(reg, 0, "pearson_pred"), v[3]), (f"{reg} mean pred", cert(reg, 0, "mean_pred"), v[4]), (f"{reg} mean actual", cert(reg, 0, "mean_actual"), v[5]),
          (f"{reg} within20", cert(reg, 0, "frac_within_20pct"), v[6])]
c2 = {"delhi": 0.51, "ncr": 0.27, "mumbai": 0.60, "kolkata": 0.52, "hyderabad": 0.43, "bengaluru": 0.44, "india": 0.44}
for reg, v in c2.items():
    C.append((f"{reg} c2star", cert(reg, 0, "spearman_c2star"), v))
C += [("delhi s1 c2star", cert("delhi", 1, "spearman_c2star"), 0.64), ("ncr s1 c2star", cert("ncr", 1, "spearman_c2star"), 0.51),
      ("mumbai s1 c2star", cert("mumbai", 1, "spearman_c2star"), 0.79), ("kolkata s1 c2star", cert("kolkata", 1, "spearman_c2star"), 0.82),
      ("india c3 within20", cert("india", 0, "frac_within_20pct"), 0.25), ("india c3 sp", cert("india", 0, "spearman_pred"), 0.32)]
C += [("delhi s1 sigma", cert("delhi", 1, "spearman_sigma"), 0.68), ("ncr s1 sigma", cert("ncr", 1, "spearman_sigma"), 0.51), ("mumbai s1 sigma", cert("mumbai", 1, "spearman_sigma"), 0.70),
      ("kolkata s1 sigma", cert("kolkata", 1, "spearman_sigma"), 0.85)]
for reg, nn in [("delhi", 66), ("mumbai", 308), ("hyderabad", 150), ("bengaluru", 210)]:
    C += [(f"{reg} bound n", S[reg]["bound"]["n"], nn), (f"{reg} bound viol", S[reg]["bound"]["violations"], 0)]
C += [("delhi bound ratio", S["delhi"]["bound"]["median_ratio"], 0.08), ("hyd bound ratio", S["hyderabad"]["bound"]["median_ratio"], 0.36)]
         
ch = {"delhi": ("PM2.5", 0.262, "PM10", 0.266, 0.425), "ncr": ("PM2.5", 0.324, "PM10", 0.337, 0.511), "kolkata": ("PM2.5", 0.362, "PM10", 0.242, 0.421),
      "hyderabad": ("PM2.5", 0.324, "PM10", 0.322, 0.472), "mumbai": ("PM2.5", 0.465, "PM10", 0.407, 0.636), "bengaluru": ("PM2.5", 0.554, None, None, 0.590)}
for reg, v in ch.items():
    C += [(f"{reg} ch pm25", S[reg]["channel"]["vector_sheaf"][v[0]], v[1]), (f"{reg} ch all", S[reg]["channel"]["vector_sheaf"]["all"], v[4])]
    if v[2]:
        C.append((f"{reg} ch pm10", S[reg]["channel"]["vector_sheaf"][v[2]], v[3]))
C += [("delhi ch dense pm25", S["delhi"]["channel"]["gaussian_cond"]["PM2.5"], 0.309), ("delhi ch dense pm10", S["delhi"]["channel"]["gaussian_cond"]["PM10"], 0.323),
      ("delhi ch kriging all", S["delhi"]["channel"]["kriging"]["all"], 0.453), ("ncr ch dense pm25", S["ncr"]["channel"]["gaussian_cond"]["PM2.5"], 0.387),
      ("ncr ch kriging pm10", S["ncr"]["channel"]["kriging"]["PM10"], 0.407), ("ncr ch dense all", S["ncr"]["channel"]["gaussian_cond"]["all"], 0.535),
      ("kolkata ch lowrank pm25", S["kolkata"]["channel"]["lowrank"]["PM2.5"], 0.348), ("kolkata ch scalar pm10", S["kolkata"]["channel"]["scalar_sheaf"]["PM10"], 0.277),
      ("mumbai ch scalar pm25", S["mumbai"]["channel"]["scalar_sheaf"]["PM2.5"], 0.417), ("blr ch dense pm25", S["bengaluru"]["channel"]["gaussian_cond"]["PM2.5"], 0.516),
      ("blr ch dense all", S["bengaluru"]["channel"]["gaussian_cond"]["all"], 0.560), ("blr ch dense p", S["bengaluru"]["channel_paired"]["gaussian_cond"]["wilcoxon_p"], 0.03)]
           
prol = {"delhi": ("(0, 3]", 0.53, 0.57, 0.63, 0.62, 0.99, 0.73, 1.17), "ncr": ("(0, 3]", 0.40, 0.43, 0.52, 0.57, 0.58, 0.61, 0.96),
        "mumbai": ("(0, 3]", 0.49, 0.48, 0.53, 0.55, 0.54, 0.58, 0.91), "kolkata": ("(0, 3]", 0.32, 0.27, 0.31, 0.54, 0.49, 0.64, 0.68),
        "hyderabad": ("(0, 3]", 0.34, 0.39, 0.41, 0.54, 0.61, 0.66, 0.86), "bengaluru": ("(0, 3]", 0.61, 0.59, 0.56, 0.60, 0.60, 0.70, 0.86)}
for reg, v in prol.items():
    C += [(f"{reg} pro st 0-3", pro(reg, "st_sheaf", v[0]), v[1]), (f"{reg} pro hyb 0-3", pro(reg, "hybrid", v[0]), v[2]), (f"{reg} pro pers 0-3", pro(reg, "persistence", v[0]), v[3]),
          (f"{reg} pro st all", S[reg]["prolonged_overall"]["st_sheaf"], v[4]), (f"{reg} pro hyb all", S[reg]["prolonged_overall"]["hybrid"], v[5]),
          (f"{reg} pro gnn all", S[reg]["prolonged_overall"]["gnn"], v[6]), (f"{reg} pro pers all", S[reg]["prolonged_overall"]["persistence"], v[7])]
C += [("delhi pro hyb 24-48", pro("delhi", "hybrid", "(24, 48]"), 1.03), ("delhi pro hyb 48-72", pro("delhi", "hybrid", "(48, 72]"), 1.21),
      ("delhi pro st 24-48", pro("delhi", "st_sheaf", "(24, 48]"), 0.60), ("delhi pro st 48-72", pro("delhi", "st_sheaf", "(48, 72]"), 0.67),
      ("delhi pro st 12-24", pro("delhi", "st_sheaf", "(12, 24]"), 0.51), ("delhi pro hyb 12-24", pro("delhi", "hybrid", "(12, 24]"), 0.75)]
                
fp = os.path.join(ROOT, "results", "v2_delhi", "fault_results.csv")
if os.path.exists(fp):
    f = pd.read_csv(fp)
    g = f.groupby(["kind", "magnitude", "detector"]).top1.mean()
    C += [("delhi fault bias1 syn", g[("bias", 1.0, "syndrome")], 0.73), ("delhi fault bias1 nb", g[("bias", 1.0, "neighbour_diff")], 0.28),
          ("delhi fault bias1 mag", g[("bias", 1.0, "magnitude")], 0.18), ("delhi fault drift2 syn", g[("drift", 2.0, "syndrome")], 0.87),
          ("delhi fault overall syn", f.groupby("detector").top1.mean()["syndrome"], 0.47),
          ("delhi fault overall other", f.groupby("detector").top1.mean().drop("syndrome").max(), 0.30)]
    def auc(a, b):
        a = np.asarray(a); b = np.asarray(b); return float(np.mean(a[:, None] > b[None, :]) + 0.5 * np.mean(a[:, None] == b[None, :]))
    gg = f[(f.kind == "bias") & (f.magnitude == 1.0)]
    C += [("delhi fault auc syn", auc(gg[gg.detector == "syndrome"].score_max, gg[gg.detector == "syndrome"].clean_max), 0.79),
          ("delhi fault auc loo", auc(gg[gg.detector == "loo_residual"].score_max, gg[gg.detector == "loo_residual"].clean_max), 0.55)]
                  
if "india" in S:
    I = S["india"]
    C += [("india stations", I["n_stations"], 177), ("india patterns", I["n_patterns"], 385), ("india days", I["n_test_days"], 40), ("india split", I["split"][0], 4267),
          ("india kriging", t1("india", "kriging"), 0.868), ("india sheaf", t1("india", "vector_sheaf"), 0.877), ("india lo", t1("india", "vector_sheaf", 1, "ci_lo"), 0.793),
          ("india hi", t1("india", "vector_sheaf", 1, "ci_hi"), 0.987), ("india scalar", t1("india", "scalar_sheaf"), 0.883), ("india idw", t1("india", "idw"), 0.891),
          ("india dense", t1("india", "gaussian_cond"), 0.901), ("india gnn", t1("india", "gnn"), 0.920), ("india lowrank", t1("india", "lowrank"), 0.922),
          ("india kriging diff", paired("india", "kriging", "mean_diff_vs_sheaf"), -0.005), ("india kriging p", paired("india", "kriging", "wilcoxon_p"), 0.37),
          ("india idw diff", paired("india", "idw", "mean_diff_vs_sheaf"), 0.019 + 0.0025), ("india lowrank diff", paired("india", "lowrank", "mean_diff_vs_sheaf"), 0.047),
          ("india ch pm25", I["channel"]["vector_sheaf"]["PM2.5"], 0.476), ("india ch kr pm25", I["channel"]["kriging"]["PM2.5"], 0.528),
          ("india ch pm10", I["channel"]["vector_sheaf"]["PM10"], 0.491), ("india ch kr pm10", I["channel"]["kriging"]["PM10"], 0.517),
          ("india c2 s1", cert("india", 1, "spearman_sigma"), 0.54), ("india c2 n1", cert("india", 1, "n"), 160), ("india c3 pearson", cert("india", 0, "pearson_pred"), -0.07),
          ("india c3 pred", cert("india", 0, "mean_pred"), 0.52), ("india c3 actual", cert("india", 0, "mean_actual"), 0.90),
          ("india pro st", I["prolonged_overall"]["st_sheaf"], 1.21), ("india pro pers", I["prolonged_overall"]["persistence"], 0.92), ("india pro seas", I["prolonged_overall"]["seasonal_persistence"], 0.71),
          ("india pro runs", I["prolonged_n"]["runs"], 5), ("india pro stations", I["prolonged_n"]["stations"], 21)]

bad = 0
for name, actual, claimed in C:
    actual = float(actual); claimed = float(claimed)
    if name.endswith(" p"):
        ok = (actual <= claimed * 1.6 + 1e-4) if claimed < 0.01 else abs(actual - claimed) < 0.015
    elif claimed >= 100:
        ok = abs(actual - claimed) < 1
    elif claimed >= 1:
        ok = abs(actual - claimed) <= 0.0125
    else:
        ok = abs(actual - claimed) <= 0.0075
    if not ok:
        bad += 1; print(f"MISMATCH {name}: report {claimed} vs results {actual:.4f}")
print(f"checked {len(C)} claims, {bad} mismatches")
sys.exit(1 if bad else 0)
