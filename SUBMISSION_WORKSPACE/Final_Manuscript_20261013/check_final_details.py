from pathlib import Path
from collections import Counter
from fractions import Fraction
import json,re,hashlib,numpy as np,pandas as pd

out=Path(__file__).resolve().parent;root=out.parents[1]
tex=(out/'SheafCode_IRIS_Final_Manuscript.tex').read_text(encoding='utf-8')
original=(out/'Original_Selected_Manuscript.tex').read_text(encoding='utf-8-sig')
audited=(root/'SUBMISSION_WORKSPACE/Technical_Audit_20261013/SheafCode_IRIS_Natural_Writing_Technically_Corrected.tex').read_text(encoding='utf-8')
checks=[]
def ck(name,ok,detail=None):checks.append(dict(check=name,passed=bool(ok),detail=detail))
def extract(s,kind):return re.findall(r'\\begin\{'+kind+r'\}(.*?)\\end\{'+kind+r'\}',s,re.S)
for kind,n in [('proof',7),('proposition',6),('theorem',1),('equation',2)]:
    now=extract(tex,kind)
    ck(kind+'/unchanged_from_selected_source',len(now)==n and now==extract(original,kind))
    ck(kind+'/identical_to_audited_mathematics',now==extract(audited,kind))
changes=json.loads((out/'FINAL_CHANGES.json').read_text(encoding='utf-8'))
rebuilt=original
for c in changes:rebuilt=rebuilt.replace(c['before'],c['after'])
ck('only_documented_changes',rebuilt==tex)
expected_abstract=extract(original,'abstract')[0].replace('Predictions use only information available at the time.','Predictions use no observations from later hours.')
ck('abstract_only_causality_scope_corrected',extract(tex,'abstract')==[expected_abstract])
ck('abstract_length',len(extract(tex,'abstract')[0].split())==234)
ck('weighted_bound_example',Fraction('1.49')*5+Fraction('.02')*2+Fraction('.51')*3==Fraction('9.02'))
ck('joint_segment_example',Fraction(2)/(2*Fraction('.75'))==Fraction(4,3))
ck('joint_separate_support_example',Fraction(2)/Fraction('1.5')>1 and Fraction(1,2)*2/Fraction('1.5')<1)
ck('stability_margins',[4/2-1,10/2-1]==[1,4])
ck('stability_boundary',max([0,4])<=5<max([4,6]) and abs(4-5)<=1 and abs(6-5)<=1)

                                                                          
pattern=r'\\fill\[[^\]]+\]\s*\(([-\d.]+),([-\d.]+)\)\s*rectangle[^;]+;\s*\\node[^;]+?\{(n/a|(?:\$-\$)?[\d.]+)\};'
cells=re.findall(pattern,tex)
ck('heatmap/cell_count',len(cells)==100,len(cells))
kinds=['same','opposite','random','unequal','drift','stuck','raw_gain','one_channel']
fault={};stress={}
for country,folder in [('India','delhi'),('US','epa_la')]:
    base=root/'Research_Release/recent_data_results'/folder
    f=pd.read_csv(base/'available_fault_summary.csv')
    fault[country]={(r.kind,int(r.true_count)):100*r.exact_support for r in f.itertuples()}
    h=pd.read_csv(base/'history_horizon_context_cases.csv')
    stress[country]={}
    for key,group in h.groupby(['history','horizon','context_loss']):
        p=group.pivot(index=['day','station'],columns='method',values='mse')[['sheaf_causal','var_causal']].dropna()
        stress[country][key]=100*(1-np.sqrt(p.sheaf_causal.sum()/p.var_causal.sum()))
for x,y,shown in cells:
    x=float(x);y=float(y)
    if y>=0:
        country='India' if x<5 else 'US';offset=0 if country=='India' else 7
        count=round((x-offset)/1.05);kind=kinds[round((3.5-y)/.5)]
        expected='n/a' if count==0 else f'{fault[country][kind,count]:.1f}'
        ck(f'heatmap/fault/{country}/{kind}/{count}',shown==expected,{'shown':shown,'expected':expected})
    else:
        country='India' if y>-5 else 'US';history=0 if y in [-3.5,-6.25] else 24
        context_group=round(x/4.4);horizon=[24,72,168][round((x-4.4*context_group)/1.05)]
        loss=[0.,.2,.5][context_group]
        value=stress[country][history,horizon,loss];expected=f'{value:.1f}'
        ck(f'heatmap/reconstruction/{country}/{history}/{horizon}/{loss}',shown.replace('$-$','-')==expected,{'shown':shown,'expected':expected})

radius=pd.read_csv(root/'Research_Release/decision_repair_results/RADIUS_CONFIGURATIONS.csv')
plots=re.findall(r'\\addplot coordinates \{(.*?)\};',tex,re.S)
ck('radius/two_panels',len(plots)==2)
for f,plot in enumerate(plots):
    shown=[tuple(map(float,p.split(','))) for p in re.findall(r'\(([^()]+)\)',plot)]
    actual=radius[(radius.query_faults==f)&(radius.status=='certified_inside_radius')]
    ck('radius/every_point/'+str(f),Counter(tuple(round(v,7) for v in p) for p in shown)==Counter(tuple(round(v,7) for v in p) for p in zip(actual.baseline_radius,actual.same_cost_radius)),len(shown))

alerts=pd.read_csv(root/'Research_Release/recent_data_results/delhi/hourly_alert_predictions.csv')
for method in ['sheaf_causal','var_causal']:
    f=alerts[alerts.method==method]
    ck('alerts/true_class/'+method,f.positive.eq(f.truth>60).all())
    ck('alerts/equality_count/'+method,int(f.truth.eq(60).sum())==8)
    ck('alerts/negative_count/'+method,int(f.truth.le(60).sum())==4012)
    summary=pd.read_csv(root/'Research_Release/recent_data_results/delhi/alert_summary.csv').set_index('method').loc[method]
    ck('alerts/false_alarm_reaggregation/'+method,np.isclose(f.loc[f.truth<=60,'alarm'].mean(),summary.false_alarm_rate))
    ck('alerts/sensitivity_reaggregation/'+method,np.isclose(f.loc[f.truth>60,'alarm'].mean(),summary.sensitivity))
joint=pd.read_csv(root/'Research_Release/joint_certification_results/CASES.csv')
ck('joint/260_feasible_12_inconsistent',int(joint.status.eq('verified_bounds').sum())==260)
ck('joint/117_abstentions',int(joint.decision.eq('abstain').sum())==117)
ck('selected_contents_preserved','17.24\\%' not in tex and '1.66\\%' not in tex and r'\subsection{Implementation verification}' not in tex)
ck('restored_heatmap_present',r'\label{fig:fault-outage}' in tex and r'\label{app:sensitivity}' in tex)
ck('threshold_equality_description','4,012 hours at or below the threshold' in tex)
result=dict(status='passed' if all(c['passed'] for c in checks) else 'failed',check_count=len(checks),checks=checks,source_sha256=hashlib.sha256((out/'SheafCode_IRIS_Final_Manuscript.tex').read_bytes()).hexdigest())
(out/'FINAL_DETAIL_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=result['status'],checks=len(checks),failures=[c for c in checks if not c['passed']]),indent=2))
if result['status']!='passed':raise SystemExit(1)
