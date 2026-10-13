from pathlib import Path
from fractions import Fraction
from collections import Counter
import json,re,gzip,numpy as np,pandas as pd,hashlib

out=Path(__file__).resolve().parent;root=out.parents[1];checks=[]
tex=(out/'SheafCode_IRIS_Natural_Writing_Technically_Corrected.tex').read_text(encoding='utf-8')
def ck(label,ok,detail=None):
    checks.append(dict(check=label,passed=bool(ok),detail=detail))
def js(p):return json.loads((root/p).read_text(encoding='utf-8'))
f=pd.read_csv(root/'Research_Release/decision_repair_results/RADIUS_CONFIGURATIONS.csv')
section=tex.split(r'\label{tab:repair-summary}')[1].split(r'\subsection{Archive transfer')[0]
plots=re.findall(r'\\addplot coordinates \{(.*?)\};',section,re.S)
ck('radius_plot/two_panels',len(plots)==2)
for budget,plot in enumerate(plots):
    shown=[tuple(map(float,p)) for p in re.findall(r'\(([^()]+)\)',plot) for p in [p.split(',')]]
    rows=f[(f.query_faults==budget)&(f.status=='certified_inside_radius')]
    actual=list(zip(rows.baseline_radius,rows.same_cost_radius))
    ck('radius_plot/all_points/'+str(budget),Counter(tuple(round(v,7) for v in p) for p in shown)==Counter(tuple(round(v,7) for v in p) for p in actual),len(shown))
selection=js('Research_Release/weakness_resolution_results/selection_guard/SUMMARY.json')
ck('selection/9_external_wins',sum(r['summary']['gain_percent']>0 for r in selection if r['cohort']!='india_long_test')==9)
legacy=(root/'Final_Manuscript/sections/09_results.tex').read_text(encoding='utf-8')
ck('legacy/retrospective_inference',all(s in legacy for s in ['median gain \\textbf{34.0\\%}','5/9 supplementary run-based tests']))
ck('legacy/snapshot_inference',all(s in legacy for s in ['gains in 7/8','median gain \\textbf{3.1\\%}','2/8 comparisons']))
pro=pd.read_csv(root/'Final_Manuscript/data/prolonged.csv').pivot(index='network',columns='method',values='mean')
gain=100*(1-pro['Space-time sheaf']/pro['Kalman/VAR smoother'])
ck('legacy/retrospective_median',round(gain.median(),1)==34.0, gain.median())
later=js('Research_Release/validation_results/external_summary.json')['comparison']['full_covariance_var']
ck('legacy/later_period',later['cases']==74 and later['days']==19 and round(later['rmse_improvement_percent'],1)==9.5)
replay=js('Research_Release/online_validation_results/replay_summary.json')['comparison']
ck('legacy/Delhi',replay['cases']==399 and replay['days']==58 and round(replay['gain_percent'],1)==23.2)
beijing=js('Research_Release/online_validation_results/beijing_corrected_summary.json')
ck('legacy/Beijing_cases',beijing['cases']==502)
weather=js('Research_Release/results/weather/weather_summary.json')
ck('legacy/weather_control',weather['weather_reconstruction']['rmse_relative_gain']<0)
records=json.loads(gzip.decompress((root/'Research_Release/math_integration_results/india_long_test_replay.json.gz').read_bytes()))
traces=[r['search']['upper']['history'] for r in records]
def number(v):return float(Fraction(v)) if v is not None else -np.inf
eligible=[t for t in traces if t and number(t[-1].get('lower'))!=-np.inf]
                                                                                              
if eligible:
    trace=max(eligible,key=lambda t:number(t[0]['upper'])-number(t[-1]['lower']))
    ck('search_trace/rounded_values',round(number(trace[0]['upper']),2)==3.58 and round(number(trace[6]['upper']),2)==2.85 and round(number(trace[7]['lower']),2)==2.85, {'root':number(trace[0]['upper']),'node7':number(trace[6]['upper']),'node8':number(trace[7]['lower'])})
metadata=js('SUBMISSION_WORKSPACE/Technical_Audit_20261013/BIBLIOGRAPHY_METADATA.json')
ck('references/Crossref_metadata',sum(r['status']=='retrieved' for r in metadata)==13)
result=dict(status='passed' if all(c['passed'] for c in checks) else 'failed',checks=checks)
(out/'ADDITIONAL_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
if result['status']!='passed':raise SystemExit(1)
