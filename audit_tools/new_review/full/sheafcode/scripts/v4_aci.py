                                                                        
                                                                         
                                            

                                                                  
                                                                         
                                                                    
                                                                  
                                                                        
                                                                                   
                                                                          
                                                               
                                                                     
                                                 

                                                                      
                                                                     
                                                
import argparse
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from sheafcode.frozen import Frozen              
from sheafcode.experiments import save_json              

ap = argparse.ArgumentParser()
ap.add_argument("--region", default="delhi")
ap.add_argument("--n_stations", type=int, default=12)
ap.add_argument("--alpha", type=float, default=0.1)
ap.add_argument("--gamma", type=float, default=0.005)
args = ap.parse_args()
rng = np.random.default_rng(0)
F = Frozen(args.region); ds, n, p, sheaf, lam = F.ds, F.n, F.p, F.sheaf, F.lam
stations = list(range(n)) if n <= args.n_stations else sorted(rng.choice(n, args.n_stations, replace=False).tolist())
va, te = ds.rows(1), ds.rows(2)
alpha, gamma = args.alpha, args.gamma


def residual_stream(rows, v):
    Zb = F.Z[rows]
    obs = np.isfinite(Zb[:, v, :])
    pred = sheaf.reconstruct(Zb, [v], lam)[:, 0, :]                                                         
    e = np.abs(pred - Zb[:, v, :])                                                             
    return e, obs, pred


out = dict(region=args.region, alpha=alpha, gamma=gamma, stations=[], summary={})
cov_aci, cov_fix, w_aci, w_fix, worst_aci, worst_fix = [], [], [], [], [], []
for v in stations:
    scale = sheaf.predicted_rmse_empirical([v], lam, F.Ztr, max_rows=800)
    if not np.isfinite(scale) or scale <= 0:
        continue
    ev, obs_v, _ = residual_stream(va, v); et, obs_t, _ = residual_stream(te, v)
    s_val = (ev / scale)[obs_v]
    if len(s_val) < 50:
        continue
    q_fix = float(np.quantile(s_val, min(1.0, np.ceil((len(s_val) + 1) * (1 - alpha)) / len(s_val))))
                                                                                                      
                                                                               
    hist = list(s_val); a_t = alpha; hits_aci = []; hits_fix = []; widths_aci = []; qs = []
    for t in range(len(te)):
        for i in range(p):
            if not obs_t[t, i]:
                continue
            level = min(max(1 - a_t, 0.0), 1.0)
            q = float(np.quantile(hist, level)) if level < 1.0 else float(max(hist)) * 1.5
            s = et[t, i] / scale
            err = float(s > q); hits_aci.append(1 - err); hits_fix.append(float(s <= q_fix)); widths_aci.append(q * scale); qs.append(q)
            a_t = a_t + gamma * (alpha - err)
            hist.append(s)
    hits_aci = np.array(hits_aci); hits_fix = np.array(hits_fix)
    Q = max(len(hits_aci) // 4, 1)
    worst_a = min(hits_aci[i:i + Q].mean() for i in range(0, len(hits_aci) - Q + 1, Q))
    worst_f = min(hits_fix[i:i + Q].mean() for i in range(0, len(hits_fix) - Q + 1, Q))
    rec = dict(station=ds.stations[v], scale_c3=float(scale), q_fixed=q_fix, n=int(len(hits_aci)), coverage_aci=float(hits_aci.mean()), coverage_fixed=float(hits_fix.mean()),
               worst_quarter_aci=float(worst_a), worst_quarter_fixed=float(worst_f), mean_q_aci=float(np.mean(qs)), final_alpha_t=float(a_t),
               mean_halfwidth_z_aci=float(np.mean(widths_aci)), mean_halfwidth_z_fixed=float(q_fix * scale))
    out["stations"].append(rec)
    cov_aci.append(rec["coverage_aci"]); cov_fix.append(rec["coverage_fixed"]); w_aci.append(rec["mean_halfwidth_z_aci"]); w_fix.append(rec["mean_halfwidth_z_fixed"])
    worst_aci.append(worst_a); worst_fix.append(worst_f)
out["summary"] = dict(n_stations=len(cov_aci), nominal=1 - alpha, coverage_aci=float(np.mean(cov_aci)), coverage_fixed=float(np.mean(cov_fix)),
                      worst_quarter_aci=float(np.mean(worst_aci)), worst_quarter_fixed=float(np.mean(worst_fix)),
                      halfwidth_z_aci=float(np.mean(w_aci)), halfwidth_z_fixed=float(np.mean(w_fix)),
                      stations_within_2pts_aci=int(sum(abs(c - (1 - alpha)) <= 0.02 for c in cov_aci)), stations_within_2pts_fixed=int(sum(abs(c - (1 - alpha)) <= 0.02 for c in cov_fix)))
save_json(out, os.path.join(F.out, "aci_v4.json"))
print(args.region, out["summary"])
