                                                                        
                                                                        
                                            

                                                                    
                                                                       
                                                                        
                                                                       
                                               
                                                                        
                                                                     
                                                           
                                                                         
                                                            
                                                      
                                                                         
                                                                           
                                                                         
                                                       
                                                                     
                                                                        
                                                                       
                                                                             
                                                                         
                                          
                                                                        
               

                                                              
import argparse
import itertools
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from sheafcode.frozen import Frozen, rmse_masked              
from sheafcode.graph import knn_edges              
from sheafcode.masked import MaskedSheaf              
from sheafcode.experiments import save_json              

ap = argparse.ArgumentParser()
ap.add_argument("--region", default="delhi")
ap.add_argument("--max_days", type=int, default=30)
ap.add_argument("--n_stations", type=int, default=40)
ap.add_argument("--parts", default="ABCDEF")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
t0 = time.time()
rng = np.random.default_rng(args.seed)
F = Frozen(args.region)
ds, n, p, D = F.ds, F.n, F.p, F.D
blocks = F.test_blocks(args.max_days); vblocks = F.val_blocks(args.max_days)
stations = list(range(n)) if n <= args.n_stations else sorted(rng.choice(n, args.n_stations, replace=False).tolist())
out = F.out
S = dict(region=args.region, frozen_choice=F.c, n=n, n_edges=len(F.edges), test_days=len(blocks), stations_scored=len(stations))
print(f"[{args.region}] n={n} edges={len(F.edges)} graph={F.c['graph']} k={F.c['k']}", flush=True)


def single_errors(sheaf, blk, sts=None):
    return {v: F.pattern_error(sheaf, [v], blk) for v in (stations if sts is None else sts)}


def cert_singles(sheaf, sts=None):
    rows = []
    for v in (stations if sts is None else sts):
        ce = sheaf.certificate([v], F.lam); c2s, _ = sheaf.c2star([v], F.lam)
        rows.append(dict(station=v, identifiable=ce["identifiable"], sigma_min=ce["sigma_min"], c2_star=c2s))
    return pd.DataFrame(rows)


def degree(edges):
    d = np.zeros(n, int)
    for u, v in edges:
        d[u] += 1; d[v] += 1
    return d


def n_components(edges, nodes=None):
    nodes = list(range(n)) if nodes is None else list(nodes)
    parent = {v: v for v in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for u, v in edges:
        if u in parent and v in parent:
            parent[find(u)] = find(v)
    return len({find(v) for v in nodes})


def summarise_variant(name, edges, sheaf, do_test=True, extra=None):
    cs = cert_singles(sheaf)
    ve = single_errors(sheaf, vblocks)
    row = dict(variant=name, n_edges=len(edges), components=n_components(edges), mean_degree=float(degree(edges).mean()),
               isolated=int((degree(edges) == 0).sum()), kernel_dim=sheaf.kernel_dimension(),
               frac_identifiable=float(cs.identifiable.mean()), mean_c2star=float(cs.c2_star.mean()), median_sigma_min=float(cs.sigma_min.median()),
               val_rmse=float(np.nanmean(list(ve.values()))))
    if do_test:
        te = single_errors(sheaf, blocks)
        row["test_rmse"] = float(np.nanmean(list(te.values())))
        x = cs.set_index("station").c2_star.reindex(list(te)); y = pd.Series(te)
        ok = np.isfinite(x) & np.isfinite(y)
        row["spearman_c2star_test"] = float(spearmanr(x[ok], y[ok])[0]) if ok.sum() > 4 else np.nan
    if extra:
        row.update(extra)
    return row


                                                                                    
if "A" in args.parts:
    rows = []
    DC = F.DC; Cabs = 1.0 - DC
    KS = [k for k in (3, 5, 8, 12) if k < n]
    variants = {}
    for k in KS:
        variants[f"geo_knn_k{k}"] = knn_edges(D, k)
        variants[f"corr_knn_k{k}"] = knn_edges(DC, k)
                    
        nb = {i: set(int(j) for j in np.argsort(D[i])[1:k + 1]) for i in range(n)}
        variants[f"mutual_knn_k{k}"] = sorted({(min(i, j), max(i, j)) for i in range(n) for j in nb[i] if i in nb[j]})
        variants[f"union_geo_corr_k{k}"] = sorted(set(knn_edges(D, k)) | set(knn_edges(DC, k)))
    kf = F.c["k"]
    for dmax in (25, 50, 100, 200):
        variants[f"geo_knn_k{kf}_dmax{dmax}km"] = [(u, v) for u, v in knn_edges(D, kf) if D[u, v] <= dmax]
    for tau in (0.2, 0.35, 0.5):
        variants[f"geo_knn_k{kf}_corrfilter{tau}"] = [(u, v) for u, v in knn_edges(D, kf) if Cabs[u, v] >= tau]
    for name, edges in variants.items():
        if len(edges) == 0:
            continue
        try:
            sh = F.build(edges)
        except Exception as ex:        
            rows.append(dict(variant=name, n_edges=len(edges), error=str(ex))); continue
        rows.append(summarise_variant(name, edges, sh, extra=dict(frozen=(edges == F.edges))))
        print(f"  A {name}: edges={len(edges)} val={rows[-1]['val_rmse']:.3f} test={rows[-1].get('test_rmse', np.nan):.3f} ident={rows[-1]['frac_identifiable']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    A = pd.DataFrame(rows); A.to_csv(os.path.join(out, "graph_study.csv"), index=False)
    ok = A[np.isfinite(A.get("val_rmse", np.nan))]
    S["A_graph_study"] = dict(table=A.round(4).to_dict(orient="records"), selected_by_validation=str(ok.sort_values("val_rmse").iloc[0].variant) if len(ok) else None,
                              best_on_test=str(ok.sort_values("test_rmse").iloc[0].variant) if len(ok) else None,
                              frozen_variant=f"{F.c['graph']}_knn_k{kf}")

                                                                             
if "B" in args.parts:
    res = pd.read_csv(os.path.join(out, "test_results.csv"))
    pats = [k.split("+") for k in res.pattern.unique()]
    idx_of = {s: i for i, s in enumerate(ds.stations)}
    pats = [[idx_of[s] for s in P] for P in pats]
    sub_pats = [P for P in pats if len(P) == 1][:20] + [P for P in pats if len(P) in (3, 8)][:20]
    rows = []
    for seed in range(3):
        r_ = np.random.default_rng(100 + seed)
        order = r_.permutation(len(F.edges))
        for frac in (1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.12):
            keep = sorted(order[:max(1, int(round(frac * len(F.edges))))])
            edges = [F.edges[i] for i in keep]
            try:
                sh = F.build(edges)
            except Exception:
                continue
            ident = []; c2 = []; errs = []
            for P in pats:
                ce = sh.certificate(P, F.lam); ident.append(ce["identifiable"]); c2.append(sh.c2star(P, F.lam)[0])
            for P in sub_pats:
                errs.append((len(P), sh.certificate(P, F.lam)["identifiable"], sh.c2star(P, F.lam)[0], F.pattern_error(sh, P, blocks[:15])))
            e = pd.DataFrame(errs, columns=["size", "identifiable", "c2_star", "rmse"])
            rows.append(dict(seed=seed, frac_edges=frac, n_edges=len(edges), components=n_components(edges), kernel_dim=sh.kernel_dimension(),
                             frac_identifiable_all=float(np.mean(ident)), frac_identifiable_singles=float(np.mean([i for i, P in zip(ident, pats) if len(P) == 1])),
                             mean_c2star_finite=float(np.nanmean([c for c in c2 if np.isfinite(c)])),
                             rmse_identifiable=float(e[e.identifiable].rmse.mean()) if e.identifiable.any() else np.nan,
                             rmse_nonidentifiable=float(e[~e.identifiable].rmse.mean()) if (~e.identifiable).any() else np.nan,
                             rmse_all=float(e.rmse.mean()), n_scored=len(e)))
            print(f"  B seed{seed} frac={frac}: ident={rows[-1]['frac_identifiable_all']:.2f} rmse_id={rows[-1]['rmse_identifiable']:.3f} rmse_nonid={rows[-1]['rmse_nonidentifiable']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    B = pd.DataFrame(rows); B.to_csv(os.path.join(out, "thinning.csv"), index=False)
    S["B_thinning"] = B.groupby("frac_edges").mean(numeric_only=True).reset_index().round(4).to_dict(orient="records")


                                                                                                         
def connected_sets(nb, kmax, cap=200000):
                                                                            
                                                                                
    sets = {1: [frozenset([v]) for v in range(n)]}
    for k in range(2, kmax + 1):
        new = set()
        for s in sets[k - 1]:
            lo = min(s)
            frontier = set().union(*(nb[v] for v in s)) - s
            for u in frontier:
                if u > lo:
                    new.add(s | {u})
            if len(new) > cap:
                break
        sets[k] = list(new)
    return sets


if "C" in args.parts or "D" in args.parts:
    nb = F.neighbours()
    kmax_exact = 4 if n <= 60 else 3
    csets = connected_sets(nb, kmax_exact)
    S["connected_sets_enumerated"] = {k: len(v) for k, v in csets.items()}
    sig = {}; c2 = {}
    for k, ss in csets.items():
        for s in ss:
            E = sorted(s); ce = F.sheaf.certificate(E, F.lam); sig[s] = ce["sigma_min"]; c2[s] = F.sheaf.c2star(E, F.lam)[0]
                
    nonid = {k: [sorted(s) for s in ss if sig[s] < 1e-8 * max(1.0, 1.0)] for k, ss in csets.items()}
    d_block = next((k for k in sorted(nonid) if nonid[k]), None)
    S["C_block_distance"] = dict(d_block=d_block if d_block else f">{kmax_exact}", exact_up_to=kmax_exact,
                                 smallest_nonidentifiable_sets=[[ds.stations[v] for v in s] for s in (nonid[d_block][:10] if d_block else [])],
                                 n_nonidentifiable_by_size={k: len(v) for k, v in nonid.items()})
    print(f"  C d_block={S['C_block_distance']['d_block']} (exact up to {kmax_exact}) enumerated={S['connected_sets_enumerated']} ({time.time() - t0:.0f}s)", flush=True)
                          
    if "D" in args.parts:
        curve = []; worst = {}
        for k in range(1, kmax_exact + 1):
            ss = csets[k]
            if not ss:
                continue
            smin = min(sig[s] for s in ss); worst_sig = sorted(ss, key=lambda s: sig[s])[:5]
            finite = [s for s in ss if np.isfinite(c2[s])]
            worst_c2 = sorted(finite, key=lambda s: -c2[s])[:5]
            R = min(smin, curve[-1]["R_sigma_min"]) if curve else smin                                               
            Rs = max(max(c2[s] for s in finite), curve[-1]["R_star_c2"]) if curve else max(c2[s] for s in finite)
            curve.append(dict(k=k, exact=True, n_sets=len(ss), R_sigma_min=float(R), R_star_c2=float(Rs),
                              median_c2=float(np.median([c2[s] for s in finite])), frac_nonidentifiable=float(np.mean([sig[s] < 1e-8 for s in ss]))))
            worst[k] = [sorted(s) for s in worst_c2]
                                                                                                     
        for k in range(kmax_exact + 1, 9):
            cand = []
            for s in worst[k - 1]:
                fr = set().union(*(nb[v] for v in s)) - set(s)
                for u in fr:
                    E = sorted(s + [u]); cand.append((F.sheaf.c2star(E, F.lam)[0], E))
            cand = [c for c in cand if np.isfinite(c[0])]
            if not cand:
                break
            cand.sort(key=lambda t: -t[0]); worst[k] = [E for _, E in cand[:5]]
            curve.append(dict(k=k, exact=False, n_sets=len(cand), R_sigma_min=float(min(F.sheaf.certificate(E, F.lam)["sigma_min"] for _, E in cand[:5])),
                              R_star_c2=float(max(cand[0][0], curve[-1]["R_star_c2"]))))
                                                                                         
        ev = []
        for k, ws in worst.items():
            pool = csets.get(k) if k in csets else None
            rand = [sorted(s) for s in rng.choice(pool, min(10, len(pool)), replace=False)] if pool else []
            if not rand:                                              
                for _ in range(10):
                    s = [int(rng.integers(n))]
                    while len(s) < k:
                        fr = list(set().union(*(nb[v] for v in s)) - set(s))
                        if not fr:
                            break
                        s.append(int(rng.choice(fr)))
                    if len(s) == k:
                        rand.append(sorted(s))
            ew = [F.pattern_error(F.sheaf, E, blocks[:15]) for E in ws]
            er = [F.pattern_error(F.sheaf, E, blocks[:15]) for E in rand]
            ev.append(dict(k=k, n_worst=len(ws), n_random=len(rand), rmse_worst=float(np.nanmean(ew)), rmse_random=float(np.nanmean(er)),
                           c2_worst=float(np.mean([F.sheaf.c2star(E, F.lam)[0] for E in ws])), c2_random=float(np.nanmean([F.sheaf.c2star(E, F.lam)[0] for E in rand])),
                           worst_sets=[[ds.stations[v] for v in E] for E in ws[:3]]))
            print(f"  D k={k}: worst rmse={ev[-1]['rmse_worst']:.3f} random={ev[-1]['rmse_random']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(curve).to_csv(os.path.join(out, "resilience_curve.csv"), index=False)
        S["D_resilience"] = dict(curve=curve, worst_vs_random=ev)

                                                                                                  
if "E" in args.parts:
    m = max(1, int(round(0.15 * n)))
    kf = F.c["k"]

    def remaining_model(keep):
        keep = sorted(keep)
        sub = F.Ztr[:, keep, :]
        Dk = F.Dg[np.ix_(keep, keep)]
        edges = knn_edges(Dk, min(kf, len(keep) - 1))
        c = F.c
        return MaskedSheaf(sub, edges, c["r"], c["weighting"], stalk=c["stalk"]), keep

    def objective_proxy(keep):
                                                                              
                                                                        
                                            
        keep_set = set(keep)
        rows = np.concatenate([np.arange(F.sheaf.edge_rows[i].start, F.sheaf.edge_rows[i].stop) for i, (u, v) in enumerate(F.edges) if u in keep_set and v in keep_set] or [np.zeros(0, int)])
        vals = []
        for v in keep:
            cols = F.sheaf.cols([v]); dU = F.sheaf.delta[np.ix_(rows, cols)]
            G = dU.T @ dU + F.lam * F.sheaf.P[np.ix_(cols, cols)]
            M = np.linalg.solve(G, dU.T)
            vals.append(float(np.linalg.svd(M, compute_uv=False)[0]) if dU.shape[0] else 1e6)
        return float(np.mean(vals))

    def objective(keep):
        sh, keep = remaining_model(keep)
        vals = [sh.c2star([i], F.lam)[0] for i in range(len(keep))]
        vals = [v if np.isfinite(v) else 1e6 for v in vals]
        return float(np.mean(vals))

    def evaluate(keep):
        sh, keep = remaining_model(keep)
        errs = []
        for _, idx in blocks[:20]:
            Zb = F.Z[idx][:, keep, :]
            for i in range(len(keep)):
                if np.isfinite(Zb[:, i, :]).all(axis=1).sum() < 6:
                    continue
                errs.append(rmse_masked(sh.reconstruct(Zb, [i], F.lam)[:, 0, :], Zb[:, i, :]))
        return float(np.nanmean(errs)), objective(keep)

    strategies = {}
                                                                                                      
    keep = list(range(n))
    for step in range(m):
        best = min(keep, key=lambda v: objective_proxy([u for u in keep if u != v]))
        keep = [u for u in keep if u != best]
    strategies["certificate_greedy"] = keep
    deg = degree(F.edges)
    strategies["lowest_degree"] = sorted(np.argsort(deg, kind="stable")[m:].tolist())
    nn = np.sort(D + np.eye(n) * 1e9, axis=1)[:, 0]
    strategies["most_clustered_geo"] = sorted(np.argsort(nn, kind="stable")[m:].tolist())                                                      
                                                               
    nb = F.neighbours()

    def closeness(v):
        dist = {v: 0}; q = [v]
        while q:
            x = q.pop(0)
            for y in nb[x]:
                if y not in dist:
                    dist[y] = dist[x] + 1; q.append(y)
        return (len(dist) - 1) / max(1, sum(dist.values())) * (len(dist) - 1) / (n - 1)
    cen = np.array([closeness(v) for v in range(n)])
    strategies["lowest_centrality"] = sorted(np.argsort(cen, kind="stable")[m:].tolist())
    for s in range(5):
        strategies[f"random_{s}"] = sorted(np.random.default_rng(200 + s).choice(n, n - m, replace=False).tolist())
    rows = []
    for name, keep in strategies.items():
        te, obj = evaluate(keep)
        rows.append(dict(strategy=name, n_kept=len(keep), test_rmse_singles=te, mean_c2star=obj, removed=[ds.stations[v] for v in range(n) if v not in keep]))
        print(f"  E {name}: test={te:.3f} objective={obj:.3f} ({time.time() - t0:.0f}s)", flush=True)
    full_te, full_obj = evaluate(list(range(n)))
    E_ = pd.DataFrame(rows); E_.to_csv(os.path.join(out, "design.csv"), index=False)
    rnd = E_[E_.strategy.str.startswith("random")]
    S["E_design"] = dict(m_removed=m, full_network=dict(test_rmse_singles=full_te, mean_c2star=full_obj),
                         strategies={r["strategy"]: {k: v for k, v in r.items() if k != "strategy"} for r in rows},
                         random_mean_test=float(rnd.test_rmse_singles.mean()), random_sd_test=float(rnd.test_rmse_singles.std()),
                         certificate_beats_random=bool(E_[E_.strategy == "certificate_greedy"].test_rmse_singles.iloc[0] < rnd.test_rmse_singles.min()))

                                                                                               
if "F" in args.parts:
    rows = []
    base = summarise_variant("frozen", F.edges, F.sheaf)
    rows.append(base)
    kf = F.c["k"]
    for dk in (-2, 2):
        if 1 <= kf + dk < n:
            e = knn_edges(F.Dg, kf + dk); rows.append(summarise_variant(f"k{kf + dk}", e, F.build(e)))
    for frac in (0.1, 0.25, 0.5):
        for seed in range(3):
            r_ = np.random.default_rng(300 + seed)
            E = list(F.edges); nrw = int(round(frac * len(E)))
            for i in r_.choice(len(E), nrw, replace=False):
                u = int(r_.integers(n)); v = int(r_.integers(n))
                while v == u:
                    v = int(r_.integers(n))
                E[i] = (min(u, v), max(u, v))
            E = sorted(set(E))
            rows.append(summarise_variant(f"rewire{frac}_s{seed}", E, F.build(E)))
    if F.c["graph"] == "geo":
        for km in (5, 20):
            for seed in range(2):
                r_ = np.random.default_rng(400 + seed)
                lat = ds.coords["latitude"].to_numpy() + r_.normal(0, km / 111, n); lon = ds.coords["longitude"].to_numpy() + r_.normal(0, km / 111, n)
                from sheafcode.data import haversine_km
                Dn = haversine_km(pd.DataFrame(dict(latitude=lat, longitude=lon)))
                e = knn_edges(Dn, kf); rows.append(summarise_variant(f"coordnoise{km}km_s{seed}", e, F.build(e)))
    for r in rows:
        print(f"  F {r['variant']}: test={r['test_rmse']:.3f} ident={r['frac_identifiable']:.2f} rho={r['spearman_c2star_test']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    Fd = pd.DataFrame(rows); Fd.to_csv(os.path.join(out, "perturbation.csv"), index=False)
    S["F_perturbation"] = Fd.round(4).to_dict(orient="records")

S["seconds"] = round(time.time() - t0)
save_json(S, os.path.join(out, "structure.json"))
print(f"[{args.region}] structure done in {time.time() - t0:.0f}s", flush=True)
