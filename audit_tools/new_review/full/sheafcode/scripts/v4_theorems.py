                                                                             

                                                                          
                                                                   
                                                                    
                                                                   
                                                                        
                                                                       
                                                                        
                                                                     
                                                                         
                                                                          
                                                                          
                    

                                                                 
                                                                    
                                                                  
                                                                        
                                                                        
                                                                        
                                                                       
                           

                                               
import argparse
import itertools
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from sheafcode.frozen import Frozen              
from sheafcode.experiments import save_json              

ap = argparse.ArgumentParser()
ap.add_argument("--region", default="delhi")
ap.add_argument("--max_days", type=int, default=20)
ap.add_argument("--n_stations", type=int, default=12)
ap.add_argument("--kmax", type=int, default=4)
args = ap.parse_args()
t0 = time.time()
rng = np.random.default_rng(0)
F = Frozen(args.region); ds, n, p, sheaf, lam = F.ds, F.n, F.p, F.sheaf, F.lam
r_eff = F.c["r"] + (p if F.c["stalk"] == "cca+id" else 0)
nb = F.neighbours(); deg = np.array([len(nb[v]) for v in range(n)])
edges = F.edges
out = dict(region=args.region, n=n, p=p, r=F.c["r"], stalk=F.c["stalk"], r_eff=r_eff, n_edges=len(edges))


def connected_sets(kmax, cap=60000):
    sets = {1: [frozenset([v]) for v in range(n)]}
    for k in range(2, kmax + 1):
        new = set()
        for s in sets[k - 1]:
            lo = min(s); fr = set().union(*(nb[v] for v in s)) - s
            for u in fr:
                if u > lo:
                    new.add(s | {u})
            if len(new) > cap:
                break
        sets[k] = list(new)
    return sets


def e_touching(S):
    S = set(S); return sum(1 for u, v in edges if u in S or v in S)


def generic_rank(S):
                                                                                   
    S = sorted(S); best = 0
    for m in range(1, len(S) + 1):
        for sub in itertools.combinations(S, m):
            d = p * m - r_eff * e_touching(sub)
            best = max(best, d)
    return p * len(S) - best


                                                                            
kmax = args.kmax if n <= 60 else 3
csets = connected_sets(kmax)
rowsA = []; agree = 0; total = 0; generic_nonid = {}; learned_nonid = {}
for k, ss in csets.items():
    gn = ln = 0
    for s in ss:
        S = sorted(s)
        g = generic_rank(S); l = sheaf.certificate(S, lam)["rank"]
        total += 1; agree += (g == l)
        gn += (g < p * len(S)); ln += (l < p * len(S))
        if g != l and len(rowsA) < 20:
            rowsA.append(dict(S=[ds.stations[v] for v in S], generic_rank=g, learned_rank=l, needed=p * len(S)))
    generic_nonid[k] = gn; learned_nonid[k] = ln
d_generic = next((k for k in sorted(generic_nonid) if generic_nonid[k]), None)
d_learned = next((k for k in sorted(learned_nonid) if learned_nonid[k]), None)
single_rule = dict(min_degree=int(deg.min()), threshold_p_over_reff=p / r_eff,
                   stations_violating=[ds.stations[v] for v in range(n) if deg[v] < p / r_eff])
out["theorem_A"] = dict(enumerated={k: len(v) for k, v in csets.items()}, agreement=agree, total=total,
                        generic_nonidentifiable_by_size=generic_nonid, learned_nonidentifiable_by_size=learned_nonid,
                        d_block_generic=d_generic if d_generic else f">{kmax}", d_block_learned=d_learned if d_learned else f">{kmax}",
                        disagreements=rowsA, single_station_rule=single_rule,
                        hall_margin_singles=sorted(float(r_eff * deg[v] - p) for v in range(n))[:5])
print(f"[{args.region}] A: generic rank = learned rank on {agree}/{total} connected sets; d_block generic {out['theorem_A']['d_block_generic']}, learned {out['theorem_A']['d_block_learned']} ({time.time() - t0:.0f}s)", flush=True)

                                                                                                            
from sheafcode.graph import knn_edges              
from sheafcode.masked import MaskedSheaf              
thin = {}
for k_thin, r_thin in ((1, 1), (2, 1), (2, 2), (3, 2)):
    e_thin = knn_edges(F.Dg, k_thin)
    try:
        sh_thin = MaskedSheaf(F.Ztr, e_thin, r_thin, F.c["weighting"], stalk="cca")
    except Exception:
        continue
    nb_t = {v: set() for v in range(n)}
    for u, v in e_thin:
        nb_t[u].add(v); nb_t[v].add(u)

    def e_touch_t(S):
        S = set(S); return sum(1 for u, v in e_thin if u in S or v in S)

    def grank_t(S):
        best = 0
        for m in range(1, len(S) + 1):
            for sub in itertools.combinations(S, m):
                best = max(best, p * m - r_thin * e_touch_t(sub))
        return p * len(S) - best
    sets_t = [[v] for v in range(n)] + [sorted(s) for s in {frozenset((u, v)) for u, v in e_thin}]
    agree_t = sum(grank_t(S) == sh_thin.certificate(S, lam)["rank"] for S in sets_t)
    pred_single = sum(1 for v in range(n) if r_thin * len(nb_t[v]) < p)
    act_single = sum(1 for v in range(n) if not sh_thin.certificate([v], lam)["identifiable"])
    thin[f"knn_k{k_thin}_r{r_thin}"] = dict(n_edges=len(e_thin), r_eff=r_thin, sets_checked=len(sets_t), generic_equals_learned=agree_t,
                                             predicted_nonidentifiable_singles=pred_single, learned_nonidentifiable_singles=act_single,
                                             min_degree=int(min(len(nb_t[v]) for v in range(n))))
out["theorem_A"]["thin_sheaf_check"] = thin
print(f"[{args.region}] A thin sheaves: {thin}", flush=True)

                                                                            
blocks = F.test_blocks(args.max_days); vblocks = F.val_blocks(args.max_days)
stations = list(range(n)) if n <= args.n_stations else sorted(rng.choice(n, args.n_stations, replace=False).tolist())
delta = sheaf.delta


def syndrome_vec(Zrow):
    return delta @ np.nan_to_num(Zrow.reshape(-1))


def complete(Zb):
    return sheaf.reconstruct_mask(Zb, np.zeros((Zb.shape[0], n), bool), lam)


def ls_decode(s, t, cand):
                                                                                       
    best = (np.inf, None)
    for S in itertools.combinations(cand, t):
        cols = sheaf.cols(list(S)); D = delta[:, cols]
        e, *_ = np.linalg.lstsq(D, s, rcond=None)
        res = np.linalg.norm(s - D @ e)
        if res < best[0]:
            best = (res, tuple(S))
    return best[1], best[0]


                                                                                           
clean_val = []
for _, idx in vblocks[:10]:
    Zc = complete(F.Z[idx])
    for t_ in range(0, Zc.shape[0], 6):
        clean_val.append(sheaf.syndrome(Zc[t_])[1].max())
thr = float(np.quantile(clean_val, 0.95)) if clean_val else np.inf
rowsB = []; fp = 0; fp_n = 0
cand = stations if n <= 20 else sorted(rng.choice(n, 20, replace=False).tolist())
for day, idx in blocks:
    Zc = complete(F.Z[idx])
    for t_ in range(0, Zc.shape[0], 8):
        x = Zc[t_]
        fp_n += 1; fp += int(sheaf.syndrome(x)[1].max() > thr)
        for t in (1, 2):
            for mag in (1.0, 2.0, 3.0):
                S = sorted(rng.choice(cand, t, replace=False).tolist())
                y = x.copy(); y[S, :] += mag
                s = syndrome_vec(y)
                Shat, res = ls_decode(s, t, cand)
                syn_rank = np.argsort(-sheaf.syndrome(y)[1])[:t]
                rowsB.append(dict(t=t, mag=mag, exact=int(tuple(S) == Shat), syndrome_topt=int(set(syn_rank.tolist()) == set(S)),
                                  detected=int(sheaf.syndrome(y)[1].max() > thr)))
import pandas as pd              
B = pd.DataFrame(rowsB)
summ = B.groupby(["t", "mag"]).agg(exact_support_recovery=("exact", "mean"), syndrome_top_t=("syndrome_topt", "mean"), detected=("detected", "mean"), n=("exact", "size")).reset_index()
                                                                  
sig = {}
for k in (2, 4):
    if k in csets and csets[k]:
        sig[k] = float(min(sheaf.certificate(sorted(s), lam)["sigma_min"] for s in csets[k][:5000]))
out["theorem_B"] = dict(candidates=len(cand), syndrome_threshold_val95=thr, clean_false_positive_rate=fp / max(fp_n, 1), clean_hours=fp_n,
                        results=summ.round(3).to_dict(orient="records"), worst_sigma_min_2t={str(k): v for k, v in sig.items()},
                        prediction="exact recovery for t faults requires every 2t-set identifiable (d_block > 2t); with residual r, faults separable when magnitude * sigma_min(2t) >> ||r||")
print(summ.round(3).to_string(index=False), flush=True)
print(f"[{args.region}] B: threshold {thr:.3f}, clean FP rate {fp / max(fp_n, 1):.3f}, worst sigma_min(2t) {sig} ({time.time() - t0:.0f}s)", flush=True)
save_json(out, os.path.join(F.out, "theorems_v4.json"))
