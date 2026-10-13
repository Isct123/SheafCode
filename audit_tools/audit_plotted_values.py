from pathlib import Path
import json
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1]
ROOT=BASE/'Final_Manuscript';TABLES=BASE/'evidence/sheafcode_iris_2026_final_result_tables/sheafcode_iris_2026_final/results'
SOURCE=BASE/'evidence/sheafcode_iris_2026_final/sheafcode_iris_2026_final'
FR=json.loads((SOURCE/'results/FINAL_RESULTS.json').read_text());checks=[]
def ck(label,a,b,tol=1e-8):
    ok=bool(np.isclose(a,b,atol=tol,rtol=tol,equal_nan=True));checks.append(dict(check=label,passed=ok))
    assert ok,(label,a,b)
for reg,R in FR['regions'].items():
    s=pd.read_csv(TABLES/f'v3_{reg}'/'test_results.csv');numeric=['rmse_z']+[f'rmse_{p}' for p in R['pollutants']]
    pl=s.groupby(['method','pattern']).agg({**{c:'mean' for c in numeric},'size':'first','c2_star':'first'}).reset_index();pl=pl[np.isfinite(pl.rmse_z)]
    for row in R['snapshot_by_size']:
        g=pl[(pl.method==row['method'])&(pl['size']==row['size'])]
        for col in numeric:ck(f'{reg}/snapshot_size_{row["size"]}/{row["method"]}/{col}',g[col].mean(),row[col])
    cv=pl[pl.method=='vector_sheaf']
    rho=cv[['c2_star','rmse_z']].corr(method='spearman').iloc[0,1]
    if 'c2_star' in R['P4_certificates']:
        ck(f'{reg}/certificate_spearman',rho,R['P4_certificates']['c2_star']['spearman'])
    p=pd.read_csv(TABLES/f'v3_{reg}'/'prolonged_results.csv');bins=pd.cut(p.hours_elapsed,[0,3,6,12,24,48,72],labels=['1-3','4-6','7-12','13-24','25-48','49-72'])
    g=p.groupby(['method',bins],observed=False).sq_err.mean().pow(.5)
    for method,curve in R['prolonged_by_horizon'].items():
        for horizon,val in curve.items():ck(f'{reg}/pooled_horizon/{method}/{horizon}',g.loc[(method,horizon)],val,tol=5.1e-5)
for name in ['snapshot','channel','prolonged']:
    d=pd.read_csv(ROOT/f'data/{name}.csv')
    for _,r in d.iterrows():
        ck(f'{name}/{r.network}/{r.method}/CI_order_low',min(r['mean'],r.ci_low),r.ci_low)
        ck(f'{name}/{r.network}/{r.method}/CI_order_high',max(r['mean'],r.ci_high),r.ci_high)
for reg in ['delhi','kolkata','india']:
    raw=pd.read_csv(TABLES/f'v3_{reg}'/'thinning.csv')
    for row in FR['studies'][reg]['structure']['B_thinning']:
        g=raw[np.isclose(raw.frac_edges,row['frac_edges'])]
        ck(f'{reg}/thinning/{row["frac_edges"]}',g.frac_identifiable_all.mean(),row['frac_identifiable_all'],tol=5.1e-5)
    faults=pd.read_csv(TABLES/f'v3_{reg}'/'fault_results.csv')
    curve=faults[faults.detector=='syndrome'].groupby('severity').top1.mean()
    for sev,row in FR['studies'][reg]['faults']['severity_curve_syndrome'].items():
        ck(f'{reg}/fault_top1/{sev}',curve.loc[float(sev)],row['top1'],tol=5.1e-5)
graph=pd.read_csv(TABLES/'v3_india/graph_study.csv').set_index('variant')
for row in FR['studies']['india']['structure']['A_graph_study']['table']:
    for metric in ['test_rmse','val_rmse','components','frac_identifiable']:
        ck(f'national_graph/{row["variant"]}/{metric}',graph.loc[row['variant'],metric],row[metric],tol=5.1e-5)
record=dict(checks=checks,passed=len(checks),failed=0)
(ROOT/'audit/plotted_values_audit.json').write_text(json.dumps(record,indent=2))
print('Supplementary raw-unit, size, correlation, horizon, and interval checks:',len(checks),'passed')
