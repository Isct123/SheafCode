from pathlib import Path
import json, hashlib, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu, spearmanr

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / 'evidence/sheafcode_iris_2026_final/sheafcode_iris_2026_final'
TABLES = BASE / 'evidence/sheafcode_iris_2026_final_result_tables/sheafcode_iris_2026_final/results'
OUT = BASE / 'Final_Manuscript'
OUT.mkdir(exist_ok=True)
FR = json.loads((SOURCE / 'results/FINAL_RESULTS.json').read_text())
REGIONS = list(FR['regions'])
LABELS = dict(zip(REGIONS, ['Region A','Region B','Region C','Region D','Region E','Region F','National','Raw replication 1','Raw replication 2']))
checks, discrepancies, results = [], [], {}

def check(name, a, b, tol=1e-9):
    ok = bool(np.allclose(a, b, atol=tol, rtol=tol, equal_nan=True))
    checks.append(dict(check=name, passed=ok))
    if not ok: discrepancies.append(dict(check=name, recomputed=str(a), recorded=str(b)))

def holm(ps):
    keys = [k for k,v in ps.items() if np.isfinite(v)]
    order = sorted(keys, key=lambda k:ps[k]); ans={}; current=0
    for i,k in enumerate(order):
        current=max(current, (len(order)-i)*ps[k]); ans[k]=min(1.,current)
    return ans

def compare(wide, ref, recorded, label):
    ps={}
    for name,v in recorded.items():
        d=(wide[name]-wide[ref]).dropna()
        p=float(wilcoxon(d, alternative='greater').pvalue)
        ps[name]=p
        for key,actual in [('n_units',len(d)),('ref_rmse',wide.loc[d.index,ref].mean()),('comp_rmse',wide.loc[d.index,name].mean()),('mean_diff',d.mean()),('rel_improvement',d.mean()/wide.loc[d.index,name].mean()),('p_wilcoxon_one_sided',p)]:
            check(f'{label}/{name}/{key}',actual,v[key],tol=1e-8)
    adj=holm(ps)
    for name,p in adj.items():check(f'{label}/{name}/p_holm',p,recorded[name]['p_holm'],tol=1e-8)

def boot_mean(values, seed=927, B=10000):
    x=np.asarray(values); rng=np.random.default_rng(seed)
    means=x[rng.integers(0,len(x),(B,len(x)))].mean(axis=1)
    return np.quantile(means,[.025,.975]).tolist()

for reg,R in FR['regions'].items():
    folder=TABLES/f'v3_{reg}'
    snap=pd.read_csv(folder/'test_results.csv')
    pl=snap.groupby(['method','pattern']).agg(rmse_z=('rmse_z','mean'),size=('size','first'),c2_star=('c2_star','first')).reset_index()
    wide=pl.pivot(index='pattern',columns='method',values='rmse_z')
    compare(wide,'vector_sheaf',R['P1'],f'{reg}/mixed_snapshot')
    singles=pl[pl['size']==1].pivot(index='pattern',columns='method',values='rmse_z')
    compare(singles,'vector_sheaf',R['P1_singles'],f'{reg}/single_snapshot')
    for m,v in R['snapshot_overall'].items():check(f'{reg}/snapshot_overall/{m}',pl[pl.method==m].rmse_z.mean(),v,tol=5.1e-5)
    chan=pd.read_csv(folder/'channel_results.csv');chan['cell']=chan.station+':'+chan.pollutant
    cl=chan.groupby(['method','cell']).rmse_z.mean().unstack('method')
    compare(cl,'vector_sheaf',R['P2'],f'{reg}/channel')
    for m,v in R['channel_overall'].items():check(f'{reg}/channel_overall/{m}',cl[m].mean(),v,tol=5.1e-5)
    pro=pd.read_csv(folder/'prolonged_results.csv');pro['unit']=pro.run+'|'+pro.station
    first=pro[pro.hours_elapsed<=24].groupby(['method','unit']).sq_err.mean().pow(.5).unstack('method')
    entire=pro.groupby(['method','unit']).sq_err.mean().pow(.5).unstack('method')
    compare(first,'st_sheaf',R['P3_1_24h'],f'{reg}/prolonged_first24')
    compare(entire,'st_sheaf',R['P3_1_72h'],f'{reg}/prolonged_available_horizon')
    for span,mat in [('1_24h',first),('1_72h',entire)]:
        for m,v in R[f'prolonged_overall_{span}'].items():check(f'{reg}/prolonged/{span}/{m}',mat[m].mean(),v,tol=5.1e-5)
    pc=pl[pl.method=='vector_sheaf'].set_index('pattern')
    dfile=json.loads((SOURCE/'results'/f'v3_{reg}'/'prospective_deciles.json').read_text())
    safe=pc.rmse_z.reindex(dfile['safest_decile']).dropna();risky=pc.rmse_z.reindex(dfile['riskiest_decile']).dropna()
    dec=None
    if len(safe)>=2 and len(risky)>=2:
        dec=dict(n_safe=len(safe),n_risky=len(risky),rmse_safe=safe.mean(),rmse_risky=risky.mean(),ratio=risky.mean()/safe.mean(),p=float(mannwhitneyu(risky,safe,alternative='greater').pvalue),safe_ci=boot_mean(safe),risky_ci=boot_mean(risky))
        for key in ['n_safe','n_risky','rmse_safe','rmse_risky','ratio']:check(f'{reg}/decile/{key}',dec[key],R['prospective_deciles'][key])
        check(f'{reg}/decile/p',dec['p'],R['prospective_deciles']['p_mannwhitney_one_sided'])
    cf=folder/'bound_check.csv';bound=None
    if cf.stat().st_size>10:
        b=pd.read_csv(cf);bound=dict(n=len(b),violations=int((b.error>b.bound*(1+1e-8)).sum()))
        check(f'{reg}/bound/n',len(b),R['bound_check']['n']);check(f'{reg}/bound/violations',bound['violations'],R['bound_check']['violations'])
                                                                               
                                                                              
    units=first.copy();units['run']=[s.split('|')[0] for s in units.index]
    cluster={}; cluster_p={}
    for m in ['kalman_var','hybrid','gnn','persistence']:
        if m not in units:continue
        dif=units[m]-units.st_sheaf
        runmeans=dif.groupby(units.run).mean().dropna()
        p=float(wilcoxon(runmeans,alternative='greater').pvalue)
        cluster_p[m]=p
        cluster[m]=dict(n_runs=len(runmeans),mean_difference=float(runmeans.mean()),ci=boot_mean(runmeans),p=p)
    for m,p in holm(cluster_p).items():cluster[m]['p_holm']=p
    results[reg]=dict(label=LABELS[reg],single_snapshot={m:dict(mean=float(singles[m].mean()),ci=boot_mean(singles[m].dropna()),n=int(singles[m].count())) for m in ['vector_sheaf','gaussian_cond','gnn']},mixed_snapshot={m:float(wide[m].mean()) for m in ['vector_sheaf','gaussian_cond','gnn']},channel={m:dict(mean=float(cl[m].mean()),ci=boot_mean(cl[m].dropna()),n=int(cl[m].count())) for m in ['vector_sheaf','gaussian_cond']},prolonged={m:dict(mean=float(first[m].mean()),ci=boot_mean(first[m].dropna()),n=int(first[m].count())) for m in ['st_sheaf','kalman_var','hybrid','gnn']},prolonged_long={m:float(entire[m].mean()) for m in ['st_sheaf','kalman_var']},n_runs=int(pro.run.nunique()),n_units=int(pro.unit.nunique()),horizon_counts=pro[pro.method=='st_sheaf'].groupby('unit').hours_elapsed.max().value_counts().sort_index().to_dict(),decile=dec,bound=bound,cluster_sensitivity=cluster)
    print(reg,'verified; prolonged units',len(first),'runs',len(cluster['kalman_var']) if False else results[reg]['n_runs'],flush=True)

adj=holm({reg:r['decile']['p'] for reg,r in results.items() if r['decile']})
for reg,p in adj.items():results[reg]['decile']['p_holm_across_networks']=p
hashes=[]
for R in FR['regions'].values():
    for name,expected in R['data_sha256'].items():
        if any(h['file']==name for h in hashes):continue
        file=SOURCE/name
        actual=hashlib.sha256(file.read_bytes()).hexdigest() if file.exists() else None
        hashes.append(dict(file=name,expected_prefix=expected,actual_sha256=actual,matched=bool(actual and actual.startswith(expected))))
record=dict(checks=checks,discrepancies=discrepancies,results=results,input_hashes=hashes,source_results_sha256=hashlib.sha256((SOURCE/'results/FINAL_RESULTS.json').read_bytes()).hexdigest())
(OUT/'audit').mkdir(exist_ok=True)
(OUT/'audit/numerical_audit.json').write_text(json.dumps(record,indent=2,default=lambda x:float(x)),encoding='utf8')
print('CHECKS',len(checks),'DISCREPANCIES',len(discrepancies),'HASHES',len(hashes),'MISSING',sum(h['actual_sha256'] is None for h in hashes))
print('single snapshot',sum(v.get('gaussian_cond',{}).get('rel_improvement',0)>0 for v in [r['P1_singles'] for r in FR['regions'].values()]),'significant',sum(r['P1_singles'].get('gaussian_cond',{}).get('significant_005_holm',False) for r in FR['regions'].values()),'median',np.median([r['P1_singles']['gaussian_cond']['rel_improvement'] for r in FR['regions'].values() if 'gaussian_cond' in r['P1_singles']]))
print('cluster significance',[(r['label'],r['cluster_sensitivity']['kalman_var']['p_holm']) for r in results.values()])
if discrepancies:print(json.dumps(discrepancies,indent=2))
