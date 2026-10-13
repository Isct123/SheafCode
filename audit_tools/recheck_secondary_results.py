from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[1]
SOURCE=BASE/'evidence/sheafcode_iris_2026_final/sheafcode_iris_2026_final'
TABLES=BASE/'evidence/sheafcode_iris_2026_final_result_tables/sheafcode_iris_2026_final/results'
FR=json.loads((SOURCE/'results/FINAL_RESULTS.json').read_text())
LABELS=dict(zip(FR['regions'],['Region A','Region B','Region C','Region D','Region E','Region F','National','Raw 1','Raw 2']))
checks=[];fixed=[];coverage=[]
def ck(label,a,b,tol=1e-8):
    passed=bool(np.isclose(a,b,atol=tol,rtol=tol,equal_nan=True))
    checks.append({'check':label,'passed':passed});assert passed,(label,a,b)
for reg,R in FR['regions'].items():
    folder=TABLES/f'v3_{reg}'
    pro=pd.read_csv(folder/'prolonged_results.csv');pro['unit']=pro.run+'|'+pro.station
    recorded=R['temporal_ablation'];methods=list(recorded['rmse'])
    wide=pro[pro.method.isin(methods)].groupby(['unit','method']).sq_err.mean().pow(.5).unstack('method').dropna()
    ck(f'{reg}/temporal_ablation/common_units',len(wide),recorded['n_units'])
    for method,val in recorded['rmse'].items():ck(f'{reg}/temporal_ablation/{method}',wide[method].mean(),val,5.1e-5)
    snap=pd.read_csv(folder/'test_results.csv')
    pl=snap[snap.method=='vector_sheaf'].groupby('pattern').agg(rmse=('rmse_z','mean'),size=('size','first'),c2=('c2_star','first')).dropna()
    bysize=FR['studies'].get(reg,{}).get('extras',{}).get('c2star_spearman_by_size',{})
    for size,entry in bysize.items():
        group=pl[pl['size']==int(size)]
        rho=group[['c2','rmse']].corr(method='spearman').iloc[0,1]
        ck(f'{reg}/risk_fixed_size_{size}/n',len(group),entry['n'])
        ck(f'{reg}/risk_fixed_size_{size}/rho',rho,entry['spearman'])
        fixed.append({'network':LABELS[reg],'size':int(size),'n':len(group),'spearman':rho})
    interval=FR['studies'].get(reg,{}).get('intervals',{})
    for level in [50,90]:
        key=f'level_{level}'
        if key in interval:
            for metric in ['test_coverage_cells','test_coverage_day_means']:
                value=interval[key][metric]
                coverage.append({'network':LABELS[reg],'nominal_percent':level,'metric':metric,'observed_percent':100*value,'within_five_points':abs(100*value-level)<=5})
record={'passed':len(checks),'failed':0,'checks':checks,'fixed_size_correlations':fixed,'reported_interval_coverage':coverage,'intervals_independently_replayed':False}
(BASE/'Final_Manuscript/audit/report_secondary_recheck.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
pd.DataFrame(fixed).to_csv(BASE/'Final_Manuscript/data/risk_fixed_size_correlations.csv',index=False)
print(f'{len(checks)} additional temporal-ablation and fixed-size correlation comparisons passed.')
print('Single-station correlations:',[(r['network'],r['n'],round(r['spearman'],3)) for r in fixed if r['size']==1])
print('Interval entries outside five percentage points:',[(r['network'],r['nominal_percent'],r['metric'],round(r['observed_percent'],1)) for r in coverage if not r['within_five_points']])
