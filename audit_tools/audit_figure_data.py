from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]/'Final_Manuscript'
AUD=json.loads((ROOT/'audit/numerical_audit.json').read_text())['results']
BY_LABEL={v['label']:v for v in AUD.values()}
FR=json.loads((ROOT.parent/'evidence/sheafcode_iris_2026_final/sheafcode_iris_2026_final/results/FINAL_RESULTS.json').read_text())['regions']
LABEL_REG={v['label']:k for k,v in AUD.items()}
checks=[]
def ck(label,actual,expected):
    passed=bool(np.isclose(actual,expected,atol=1e-10,rtol=1e-10,equal_nan=True))
    checks.append({'check':label,'passed':passed})
    assert passed,(label,actual,expected)
methods={'Spatial sheaf':'vector_sheaf','Gaussian conditional':'gaussian_cond','Masked GCN':'gnn','Space-time sheaf':'st_sheaf','Kalman/VAR smoother':'kalman_var','Persistence + spatial':'hybrid'}
for name,key in [('snapshot','single_snapshot'),('channel','channel'),('prolonged','prolonged')]:
    for _,row in pd.read_csv(ROOT/f'data/{name}.csv').iterrows():
        expected=BY_LABEL[row.network][key][methods[row.method]]
        for column,value in [('mean',expected['mean']),('ci_low',expected['ci'][0]),('ci_high',expected['ci'][1]),('n',expected['n'])]:
            ck(f'{name}/{row.network}/{row.method}/{column}',row[column],value)
for _,row in pd.read_csv(ROOT/'data/certificate_deciles.csv').iterrows():
    expected=BY_LABEL[row.network]['decile'];suffix='safe' if row.group=='safe' else 'risky'
    for column,value in [('mean',expected[f'rmse_{suffix}']),('ci_low',expected[f'{suffix}_ci'][0]),('ci_high',expected[f'{suffix}_ci'][1]),('n',expected[f'n_{suffix}'])]:
        ck(f'certificates/{row.network}/{row.group}/{column}',row[column],value)
for _,row in pd.read_csv(ROOT/'data/figure_annotations.csv').iterrows():
    a=BY_LABEL[row.network];r=FR[LABEL_REG[row.network]]
    if row['plot']=='certificates':
        d=a['decile']
        for c,e in [('ratio',d['ratio']),('n',d['n_safe']),('n_high',d['n_risky']),('p_holm',d['p_holm_across_networks'])]:ck(f'annotation/{row.network}/certificate/{c}',row[c],e)
    else:
        plot=row['plot'];key={'snapshot':'single_snapshot','channel':'channel','prolonged':'prolonged'}[plot]
        ref='st_sheaf' if plot=='prolonged' else 'vector_sheaf';comp='kalman_var' if plot=='prolonged' else 'gaussian_cond'
        d=a[key];gain=100*(1-d[ref]['mean']/d[comp]['mean'])
        ck(f'annotation/{row.network}/{plot}/gain',row.gain_percent,gain)
        ck(f'annotation/{row.network}/{plot}/n',row['n'],d[ref]['n'])
        family={'snapshot':'P1_singles','channel':'P2','prolonged':'P3_1_24h'}[plot]
        p=a['cluster_sensitivity'][comp]['p_holm'] if plot=='prolonged' else r[family].get(comp,{}).get('p_holm',float('nan'))
        ck(f'annotation/{row.network}/{plot}/p_holm',row.p_holm,p)
        if plot=='prolonged':ck(f'annotation/{row.network}/{plot}/runs',row.runs,a['n_runs'])
record={'passed':len(checks),'failed':0,'checks':checks}
(ROOT/'audit/figure_data_crosscheck.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(f'{len(checks)} figure-data and annotation cross-checks passed.')
