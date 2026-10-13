                                                                      
                                                                        
                                      

                                                                         
                                                                         
                                                                      
                                                                    
                                                                       
                                                                       
                                                            

                                                                      
                                                                      
                                                                     
                                                                         
                                                                         
                                        

                                                                      
                                                                       
                                                                     
                                 

                                                                       
                                                                        
               

                                           
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from sheafcode.frozen import Frozen              
from sheafcode.cpcb_index import concentration_to_index              
from sheafcode.baselines2 import GaussianConditional, Kriging2              
from sheafcode.experiments import save_json              

ap = argparse.ArgumentParser()
ap.add_argument("--region", default="delhi")
ap.add_argument("--max_days", type=int, default=60)
ap.add_argument("--n_stations", type=int, default=40)
args = ap.parse_args()
rng = np.random.default_rng(0)
F = Frozen(args.region); ds, n, p, sheaf, lam = F.ds, F.n, F.p, F.sheaf, F.lam
pols = ds.pollutants
blocks = F.test_blocks(args.max_days, min_hours=18)
stations = list(range(n)) if n <= args.n_stations else sorted(rng.choice(n, args.n_stations, replace=False).tolist())
ds.set_transform("none"); Z0 = ds.Z(); mean0, std0 = ds.mean.copy(), ds.std.copy()
ds.set_transform(F.c["transform"])
gc = GaussianConditional(Z0[ds.split == 0], 0.2); kr = Kriging2(Z0[ds.split == 0], F.D)
CAT_EDGES = [50, 100, 200, 300, 400, 500]
CAT = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def aqi_of_daymean(conc):                                                            
    sub = np.stack([concentration_to_index(q, conc[:, i]) for i, q in enumerate(pols)], axis=1)
    return np.nanmax(sub, axis=1)


def category(aqi):
    return np.searchsorted(CAT_EDGES, aqi, side="left").clip(0, 5)


def stage(aqi):
    return np.where(aqi > 450, 4, np.where(aqi > 400, 3, np.where(aqi > 300, 2, np.where(aqi > 200, 1, 0))))


def to_raw(F_, Zsub, rows, sts):                                      
    return F_.ds.raw_from_Z(Zsub, rows, sts)


rows = []; city = []; exposure = dict(station_days=0, unmeasured_station_days=0, unmeasured_poor_or_worse=0, by_category={})
prev_day = {}
for bi, (day, idx) in enumerate(blocks):
    Xraw = ds.X[idx]                                                 
    truth_mean = np.nanmean(Xraw, axis=0)                    
    true_aqi = aqi_of_daymean(truth_mean)
    Zb = F.Z[idx]; Zb0 = Z0[idx]
                                                                                      
    full = sheaf.reconstruct_mask(Zb, np.zeros((len(idx), n), bool), lam)
    full_raw = to_raw(F, full, idx, list(range(n)))
    rec_aqi_all = aqi_of_daymean(np.nanmean(full_raw, axis=0))
    gap_frac = (~np.isfinite(Xraw)).mean(axis=(0, 2))
    for v in range(n):
        exposure["station_days"] += 1
        if gap_frac[v] >= 0.5 and np.isfinite(rec_aqi_all[v]):
            exposure["unmeasured_station_days"] += 1
            c = CAT[int(category(rec_aqi_all[v]))]
            exposure["by_category"][c] = exposure["by_category"].get(c, 0) + 1
            exposure["unmeasured_poor_or_worse"] += int(rec_aqi_all[v] > 200)
                   
    for v in stations:
        if np.isfinite(Xraw[:, v, :]).all(axis=1).sum() < 12 or not np.isfinite(true_aqi[v]):
            continue
        preds = {}
        preds["sheaf"] = to_raw(F, sheaf.reconstruct(Zb, [v], lam)[:, 0, :][:, None, :], idx, [v])[:, 0, :]
        preds["gaussian"] = (gc.reconstruct(Zb0, [v])[:, 0, :] * std0[v] + mean0[v])
        preds["kriging"] = (kr.reconstruct(Zb0, [v])[:, 0, :] * std0[v] + mean0[v])
        if v in prev_day and np.isfinite(prev_day[v]).all():
            preds["yesterday"] = np.repeat(prev_day[v][None], len(idx), axis=0)
        others = [u for u in range(n) if u != v]
        preds["city_mean_of_survivors"] = np.repeat(np.nanmean(truth_mean[others], axis=0)[None], len(idx), axis=0)
        tcat = int(category(true_aqi[v])); tst = int(stage(true_aqi[v]))
        for name, pr in preds.items():
            a = aqi_of_daymean(np.nanmean(pr, axis=0)[None])[0]
            if not np.isfinite(a):
                continue
            rows.append(dict(day=str(day.date()), station=ds.stations[v], method=name, true_aqi=float(true_aqi[v]), pred_aqi=float(a),
                             true_cat=tcat, pred_cat=int(category(a)), true_stage=tst, pred_stage=int(stage(a))))
    for v in range(n):
        prev_day[v] = truth_mean[v]
                
    ok = np.isfinite(true_aqi)
    if ok.sum() >= 4:
        true_city = float(np.nanmean(true_aqi)); ts = int(stage(true_city))
        for size in (3, 5):
            if size > n - 3:
                continue
            for rep in range(3):
                E = sorted(rng.choice(np.where(ok)[0], size, replace=False).tolist())
                surv = [u for u in range(n) if u not in E and ok[u]]
                surv_city = float(np.nanmean(true_aqi[surv]))
                rec = to_raw(F, sheaf.reconstruct(Zb, E, lam), idx, E)
                rec_aqi = aqi_of_daymean(np.nanmean(rec, axis=0))
                full_city = float(np.nanmean(np.concatenate([true_aqi[surv], rec_aqi])))
                city.append(dict(day=str(day.date()), size=size, true_city_aqi=true_city, true_stage=ts,
                                 survivors_stage=int(stage(surv_city)), reconstructed_stage=int(stage(full_city)),
                                 survivors_aqi=surv_city, reconstructed_aqi=full_city))
R = pd.DataFrame(rows); C = pd.DataFrame(city)
summ = dict(region=args.region, n_station_days=int(R.groupby(["day", "station"]).ngroups) if len(R) else 0, test_days=len(blocks),
            category_accuracy=R.groupby("method").apply(lambda g: float((g.true_cat == g.pred_cat).mean())).round(3).to_dict() if len(R) else {},
            stage_accuracy=R.groupby("method").apply(lambda g: float((g.true_stage == g.pred_stage).mean())).round(3).to_dict() if len(R) else {},
            aqi_mae=R.groupby("method").apply(lambda g: float((g.true_aqi - g.pred_aqi).abs().mean())).round(1).to_dict() if len(R) else {},
            true_category_distribution={CAT[k]: int(v) for k, v in R[R.method == "sheaf"].true_cat.value_counts().sort_index().items()} if len(R) else {},
            missed_actions=R.groupby("method").apply(lambda g: float(((g.true_stage > 0) & (g.pred_stage < g.true_stage)).sum() / max((g.true_stage > 0).sum(), 1))).round(3).to_dict() if len(R) else {},
            false_actions=R.groupby("method").apply(lambda g: float(((g.pred_stage > g.true_stage)).sum() / max((g.true_stage == 0).sum() + (g.true_stage > 0).sum(), 1))).round(3).to_dict() if len(R) else {},
            exposure=exposure)
if len(C):
    summ["city"] = {str(s): dict(n=int(len(g)), stage_correct_survivors=float((g.true_stage == g.survivors_stage).mean()),
                                stage_correct_reconstructed=float((g.true_stage == g.reconstructed_stage).mean()),
                                aqi_mae_survivors=float((g.true_city_aqi - g.survivors_aqi).abs().mean()),
                                aqi_mae_reconstructed=float((g.true_city_aqi - g.reconstructed_aqi).abs().mean()),
                                days_with_action=int((g.true_stage > 0).sum())) for s, g in C.groupby("size")}
summ["caveats"] = "hourly sub-index data; daily mean used for all pollutants including O3 (CPCB uses the 8-hour maximum); 2017-2020 archive values above the 500 cap are censored"
save_json(summ, os.path.join(F.out, "grap_v4.json"))
R.to_csv(os.path.join(F.out, "grap_station_v4.csv"), index=False); C.to_csv(os.path.join(F.out, "grap_city_v4.csv"), index=False)
print(args.region, "category acc", summ["category_accuracy"], "stage acc", summ["stage_accuracy"], "missed", summ["missed_actions"])
print("city", summ.get("city")); print("exposure", exposure)
