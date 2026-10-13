                                                                             
                                                                    
                                                                             
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sheafcode.experiments import ROOT              

S = json.load(open(os.path.join(ROOT, "results", "summary.json")))
P = S["b10_geo"]; Q = S["b10_geo_5x4"]; B = S["b4_geo"]


def t1(R, method, size, col="rmse_z"):
    return next(r[col] for r in R["table1"] if r["method"] == method and r["size"] == size)


def paired(R, method, col):
    return next(r[col] for r in R["paired_vs_vector"] if r["method"] == method)


def h3(R, size, col):
    return next(r[col] for r in R["h3"] if r["size"] == size)


def h2(R, k, r, size, col):
    return next(x[col] for x in R["h2"] if x["k"] == k and x["r"] == r and x["size"] == size)


def ft(R, kind, mag, det):
    return next(x[f"top1_{det}"] for x in R["fault_top1"] if x["kind"] == kind and x["magnitude"] == mag)


def fa(R, kind, mag, det):
    return next(x[f"auc_{det}"] for x in R["fault_auc"] if x["kind"] == kind and x["magnitude"] == mag)


CLAIMS = [
    ("b10 patterns", P["n_patterns"], 98), ("b10 test days", P["n_test_days"], 30), ("b10 test hours", P["n_test_hours"], 474),
    ("5x4 days", Q["n_test_days"], 54), ("5x4 hours", Q["n_test_hours"], 934), ("b4 days", B["n_test_days"], 123), ("b4 hours", B["n_test_hours"], 2354),
    ("kernel b10", P["kernel_dim"], 0), ("kernel 5x4", Q["kernel_dim"], 0), ("kernel b4", B["kernel_dim"], 1),
    ("vec b10 s1", t1(P, "vector_sheaf", 1), 0.694), ("vec b10 lo", t1(P, "vector_sheaf", 1, "ci_lo"), 0.585), ("vec b10 hi", t1(P, "vector_sheaf", 1, "ci_hi"), 0.846),
    ("sca b10 s1", t1(P, "scalar_sheaf", 1), 0.692), ("lr b10", t1(P, "lowrank", 1), 0.742), ("kr b10", t1(P, "kriging", 1), 0.733),
    ("idw b10", t1(P, "idw", 1), 0.749), ("lap b10", t1(P, "scalar_laplacian", 1), 0.753),
    ("vec 5x4", t1(Q, "vector_sheaf", 1), 0.588), ("sca 5x4", t1(Q, "scalar_sheaf", 1), 0.603), ("lr 5x4", t1(Q, "lowrank", 1), 0.679),
    ("kr 5x4", t1(Q, "kriging", 1), 0.640), ("idw 5x4", t1(Q, "idw", 1), 0.664), ("lap 5x4", t1(Q, "scalar_laplacian", 1), 0.660),
    ("vec b4", t1(B, "vector_sheaf", 1), 0.865), ("sca b4", t1(B, "scalar_sheaf", 1), 0.861), ("lr b4", t1(B, "lowrank", 1), 0.817),
    ("kr b4", t1(B, "kriging", 1), 0.938), ("idw b4", t1(B, "idw", 1), 0.977), ("lap b4", t1(B, "scalar_laplacian", 1), 1.006),
    ("pm25 vec raw", t1(P, "vector_sheaf", 1, "rmse_PM2.5"), 16.3), ("pm25 kr raw", t1(P, "kriging", 1, "rmse_PM2.5"), 16.5),
    ("pm25 idw raw", t1(P, "idw", 1, "rmse_PM2.5"), 17.0), ("pm25 lr raw", t1(P, "lowrank", 1, "rmse_PM2.5"), 15.9),
    ("no2 vec raw", t1(P, "vector_sheaf", 1, "rmse_NO2"), 16.1), ("no2 kr raw", t1(P, "kriging", 1, "rmse_NO2"), 17.9), ("no2 idw raw", t1(P, "idw", 1, "rmse_NO2"), 18.0),
    ("o3 vec raw", t1(P, "vector_sheaf", 1, "rmse_O3"), 17.0), ("o3 kr raw", t1(P, "kriging", 1, "rmse_O3"), 16.7), ("o3 idw raw", t1(P, "idw", 1, "rmse_O3"), 17.4),
    ("paired lr", paired(P, "lowrank", "mean_diff_vs_vector"), 0.095), ("paired lr lo", paired(P, "lowrank", "ci_lo"), 0.079), ("paired lr hi", paired(P, "lowrank", "ci_hi"), 0.113),
    ("paired lr frac", paired(P, "lowrank", "frac_patterns_vector_better"), 0.93), ("paired kr", paired(P, "kriging", "mean_diff_vs_vector"), 0.047),
    ("paired kr lo", paired(P, "kriging", "ci_lo"), 0.036), ("paired kr hi", paired(P, "kriging", "ci_hi"), 0.058),
    ("paired idw", paired(P, "idw", "mean_diff_vs_vector"), 0.060), ("paired lap", paired(P, "scalar_laplacian", "mean_diff_vs_vector"), 0.064),
    ("paired sca", paired(P, "scalar_sheaf", "mean_diff_vs_vector"), -0.003), ("paired sca frac", paired(P, "scalar_sheaf", "frac_patterns_vector_better"), 0.40),
    ("5x4 sca diff", paired(Q, "scalar_sheaf", "mean_diff_vs_vector"), 0.014), ("5x4 sca frac", paired(Q, "scalar_sheaf", "frac_patterns_vector_better"), 0.72),
    ("b4 lr diff", paired(B, "lowrank", "mean_diff_vs_vector"), -0.034), ("b4 lr p", paired(B, "lowrank", "wilcoxon_p"), 0.037),
    ("h3 all", h3(P, 0, "spearman"), 0.82), ("h3 s2", h3(P, 2, "spearman"), 0.70), ("h3 s2 lo", h3(P, 2, "ci_lo"), 0.36), ("h3 s2 hi", h3(P, 2, "ci_hi"), 0.88),
    ("h3 s3", h3(P, 3, "spearman"), 0.79), ("h3 s3 lo", h3(P, 3, "ci_lo"), 0.63), ("h3 s3 hi", h3(P, 3, "ci_hi"), 0.88),
    ("h3 s4", h3(P, 4, "spearman"), 0.76), ("h3 s4 lo", h3(P, 4, "ci_lo"), 0.59), ("h3 s4 hi", h3(P, 4, "ci_hi"), 0.85), ("h3 s1", h3(P, 1, "spearman"), 0.64),
    ("h3 5x4", h3(Q, 0, "spearman"), 0.56), ("h3 b4", h3(B, 0, "spearman"), -0.18),
    ("frac k1r2", P["frac_identifiable_by_k_r"]["2"]["1"], 0.19), ("frac k3r1", P["frac_identifiable_by_k_r"]["1"]["3"], 0.87), ("frac k2r1", P["frac_identifiable_by_k_r"]["1"]["2"], 0.19), ("frac k1r3", P["frac_identifiable_by_k_r"]["3"]["1"], 1.0),
    ("h2 a", h2(P, 1, 2, 2, "rmse_ident"), 0.763), ("h2 b", h2(P, 1, 2, 2, "rmse_nonident"), 0.952), ("h2 c", h2(P, 1, 2, 3, "rmse_nonident"), 0.902),
    ("h2 d", h2(P, 2, 1, 2, "rmse_nonident"), 0.990), ("h2 e", h2(P, 2, 1, 3, "rmse_nonident"), 0.935), ("h2 p", h2(P, 2, 1, 3, "mwu_p"), 0.0003),
    ("h3ls k1r2", next(x["spearman"] for x in P["h3_ls_by_k_r"] if x["k"] == 1 and x["r"] == 2), 0.97),
    ("h3ls k1r3", next(x["spearman"] for x in P["h3_ls_by_k_r"] if x["k"] == 1 and x["r"] == 3), 0.84),
    ("h3ls k2r3", next(x["spearman"] for x in P["h3_ls_by_k_r"] if x["k"] == 2 and x["r"] == 3), 0.72),
    ("h3ls k4r3", next(x["spearman"] for x in P["h3_ls_by_k_r"] if x["k"] == 4 and x["r"] == 3), 0.82),
    ("chan 5x4 pm25 vec", Q["channel_table"]["vector_sheaf"]["PM2.5"], 0.396), ("chan 5x4 pm25 sca", Q["channel_table"]["scalar_sheaf"]["PM2.5"], 0.525),
    ("chan 5x4 pm10 vec", Q["channel_table"]["vector_sheaf"]["PM10"], 0.416), ("chan 5x4 pm10 sca", Q["channel_table"]["scalar_sheaf"]["PM10"], 0.517),
    ("chan 5x4 no2 vec", Q["channel_table"]["vector_sheaf"]["NO2"], 0.666), ("chan 5x4 no2 sca", Q["channel_table"]["scalar_sheaf"]["NO2"], 0.638),
    ("chan 5x4 o3 vec", Q["channel_table"]["vector_sheaf"]["O3"], 0.543), ("chan 5x4 o3 sca", Q["channel_table"]["scalar_sheaf"]["O3"], 0.528),
    ("chan b10 vec", P["channel_table"]["vector_sheaf"]["all"], 0.633), ("chan b10 sca", P["channel_table"]["scalar_sheaf"]["all"], 0.626), ("chan b10 p", P["h1_channel"]["wilcoxon_p"], 0.49),
    ("chan b4 pm10 vec", B["channel_table"]["vector_sheaf"]["PM10"], 0.664), ("chan b4 pm10 sca", B["channel_table"]["scalar_sheaf"]["PM10"], 0.779), ("chan b4 lr", B["channel_table"]["lowrank"]["all"], 0.705),
    ("abl 5x4 pm25 st1", Q["ablation_pollutant_subset"]["PM2.5|1"]["rmse_station"], 0.618), ("abl 5x4 pm25 st4", Q["ablation_pollutant_subset"]["PM2.5|4"]["rmse_station"], 0.611),
    ("abl 5x4 pm25 ch2", Q["ablation_pollutant_subset"]["PM2.5|2"]["rmse_channel"], 0.559), ("abl 5x4 pm25 ch3", Q["ablation_pollutant_subset"]["PM2.5|3"]["rmse_channel"], 0.504),
    ("abl 5x4 pm25 ch4", Q["ablation_pollutant_subset"]["PM2.5|4"]["rmse_channel"], 0.456),
    ("pro b10 pers 0-6", P["prolonged"]["persistence"]["(0, 6]"], 0.53), ("pro b10 vec 0-6", P["prolonged"]["vector_sheaf"]["(0, 6]"], 0.69),
    ("fault bias1 syn", ft(P, "bias", 1.0, "syndrome"), 0.81), ("fault bias1 loo", ft(P, "bias", 1.0, "loo_residual"), 0.54), ("fault bias1 nb", ft(P, "bias", 1.0, "neighbour_diff"), 0.56), ("fault bias1 mag", ft(P, "bias", 1.0, "magnitude"), 0.47),
    ("fault bias.5 syn", ft(P, "bias", 0.5, "syndrome"), 0.44), ("fault bias2 syn", ft(P, "bias", 2.0, "syndrome"), 0.97), ("fault gain1 syn", ft(P, "gain", 1.0, "syndrome"), 0.75),
    ("fault drift1 syn", ft(P, "drift", 1.0, "syndrome"), 0.56), ("fault chan1 syn", ft(P, "channel", 1.0, "syndrome"), 0.43), ("fault stuck syn", ft(P, "stuck", 0.0, "syndrome"), 0.29),
    ("auc bias1 syn", fa(P, "bias", 1.0, "syndrome"), 0.86), ("auc bias1 loo", fa(P, "bias", 1.0, "loo_residual"), 0.64), ("auc bias1 mag", fa(P, "bias", 1.0, "magnitude"), 0.60),
    ("fault overall syn", P["fault_top1_overall"]["syndrome"], 0.60), ("fault overall best other", max(v for k, v in P["fault_top1_overall"].items() if k != "syndrome"), 0.46),
    ("fault 5x4 syn", Q["fault_top1_overall"]["syndrome"], 0.69), ("fault 5x4 other", max(v for k, v in Q["fault_top1_overall"].items() if k != "syndrome"), 0.65),
    ("fault b4 syn", B["fault_top1_overall"]["syndrome"], 0.38), ("fault b4 other", max(v for k, v in B["fault_top1_overall"].items() if k != "syndrome"), 0.49),
    ("crit spearman", P["criticality_spearman_sigma_vs_error"], 0.75),
    ("crit full smin", next(r["mean_sigma_min"] for r in P["criticality"] if r["removed"] == "none"), 1.59),
    ("crit full err", next(r["test_rmse_single"] for r in P["criticality"] if r["removed"] == "none"), 0.730),
    ("crit bapuji smin", next(r["mean_sigma_min"] for r in P["criticality"] if r["removed"] == "Bapuji"), 1.31),
    ("crit hebbal smin", next(r["mean_sigma_min"] for r in P["criticality"] if r["removed"] == "Hebbal"), 1.39),
    ("crit bapuji err", next(r["test_rmse_single"] for r in P["criticality"] if r["removed"] == "Bapuji"), 0.783),
    ("crit hebbal err", next(r["test_rmse_single"] for r in P["criticality"] if r["removed"] == "Hebbal"), 0.788),
    ("crit bwssb smin", next(r["mean_sigma_min"] for r in P["criticality"] if r["removed"] == "BWSSB"), 1.70),
    ("crit bwssb err", next(r["test_rmse_single"] for r in P["criticality"] if r["removed"] == "BWSSB"), 0.648),
    ("readd n", P["readdition"]["n"], 21), ("readd cert", P["readdition"]["cert_agrees"], 0.81), ("readd geo", P["readdition"]["geo_agrees"], 0.67),
    ("readd regret c", P["readdition"]["regret_cert"], 0.004), ("readd regret g", P["readdition"]["regret_geo"], 0.027),
    ("abl kr k4 r1", P["ablation_k_r"]["1.0"]["4.0"], 0.869), ("abl kr k4 r2", P["ablation_k_r"]["2.0"]["4.0"], 0.755), ("abl kr k4 r3", P["ablation_k_r"]["3.0"]["4.0"], 0.730),
]
                 
pl = pd.read_csv(os.path.join(ROOT, "results", "b10_geo", "test_results.csv"))
g = pl.groupby(["method", "pattern"])["rmse_PM2.5"].mean().unstack(0)
CLAIMS += [("pm25 sca vs vec raw", g["scalar_sheaf"].mean(), 17.9), ("pm25 vec raw all", g["vector_sheaf"].mean(), 18.6),
           ("pm25 wilcoxon", wilcoxon(g["scalar_sheaf"] - g["vector_sheaf"]).pvalue, 8e-11)]
idr = pd.read_csv(os.path.join(ROOT, "results", "b10_geo", "identifiability_results.csv"))
CLAIMS += [("ls k3 r1 ident mean", idr[(idr.solver == "ls") & (idr.k == 3) & (idr.r == 1) & idr.identifiable].rmse_z.mean(), 13.7),
           ("ls k3 r3 mean", idr[(idr.solver == "ls") & (idr.k == 3) & (idr.r == 3)].rmse_z.mean(), 0.845)]
dg = pd.read_csv(os.path.join(ROOT, "results", "b10_corr", "table_methods_by_size.csv")) if os.path.exists(os.path.join(ROOT, "results", "b10_corr", "table_methods_by_size.csv")) else None

bad = 0
for name, actual, claimed in CLAIMS:
    actual = float(actual); claimed = float(claimed)
    tol = 0.5 if claimed >= 10 else 0.0075 if claimed >= 1 else 0.0075
    if name.endswith("wilcoxon") or name.endswith(" p"):
        ok = (actual <= claimed * 1.5 + 1e-4) if claimed < 0.01 else abs(actual - claimed) < 0.01
    elif claimed >= 10:
        ok = abs(actual - claimed) <= 0.06 * max(1, claimed / 10)
    else:
        ok = abs(actual - claimed) <= tol
    if not ok:
        bad += 1
        print(f"MISMATCH {name}: report says {claimed}, results say {actual:.4f}")
print(f"checked {len(CLAIMS)} claims, {bad} mismatches")
sys.exit(1 if bad else 0)
