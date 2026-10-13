                                                                           
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'Research_Release/expanded_validation_results'
OUT=ROOT/'Research_Release/weakness_resolution_results/threshold_grid'
sys.path.insert(0,str(HERE))
from uncertainty_decisions import roc,common
import pandas as pd,numpy as np

def main():
    OUT.mkdir(exist_ok=True);rows=[];checks=[]
    original=json.loads((OUT.parent/'uncertainty_decisions/ALERT_SUMMARY.json').read_text())
    for folder in sorted(p.parent for p in SOURCE.glob('*/hourly_alert_predictions.csv')):
        f=pd.read_csv(folder/'hourly_alert_predictions.csv')
        for threshold in [15.,25.,35.,60.,90.]:
            for method,g in f.groupby('method'):
                g=g.copy();g['positive']=g.truth>threshold;result=roc(g)
                row={'cohort':folder.name,'threshold':threshold,'method':method,'hours':len(g),
                     'positive_hours':int(g.positive.sum()),'negative_hours':int((~g.positive).sum()),
                     'auc':np.nan,'sensitivity_fpr05':np.nan,'average_precision':np.nan}
                if result is not None:
                    summary,_=result;row.update({'auc':summary['auc'],'sensitivity_fpr05':summary['sensitivity_under_fpr_cap']['0.05'],'average_precision':summary['average_precision']})
                    positive=g.positive.to_numpy();p=int(positive.sum());n=len(g)-p
                    rankauc=(g.prediction.rank(method='average')[positive].sum()-p*(p+1)/2)/(p*n)
                    if not np.isclose(rankauc,row['auc'],rtol=0,atol=1e-12):raise AssertionError('Threshold-grid AUC replay failed')
                    checks.append(folder.name+'/'+str(threshold)+'/'+method+'/rank_auc')
                if threshold==60:
                    old=next(r for r in original if r['cohort']==folder.name and r['method']==method)
                    if old['positive_hours']!=row['positive_hours'] or old['negative_hours']!=row['negative_hours']:raise AssertionError('Original threshold counts changed')
                    if 'auc' in old and not np.isclose(old['auc'],row['auc']):raise AssertionError('Original threshold ROC changed')
                    checks.append(folder.name+'/'+method+'/original_threshold')
                rows.append(row)
        print('Fixed threshold sensitivity',folder.name,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'SUMMARY.csv',index=False)
    (OUT/'AUDIT.json').write_text(json.dumps({'status':'passed','check_count':len(checks),'checks':checks,
          'source_sha256':common.sha(__file__),'protocol_sha256':common.sha(HERE/'THRESHOLD_GRID_PROTOCOL.md'),
          'input_sha256':{str(p.relative_to(ROOT)):common.sha(p) for p in SOURCE.glob('*/hourly_alert_predictions.csv')},
          'scope':'Fixed exploratory concentration grid; descriptive ROC only; no regulatory averaging or health claim'},indent=2))

if __name__=='__main__':main()
