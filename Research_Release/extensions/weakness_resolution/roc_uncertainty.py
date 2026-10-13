from pathlib import Path
import sys,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'Research_Release/expanded_validation_results'
OUT=ROOT/'Research_Release/weakness_resolution_results/uncertainty_decisions'
sys.path.insert(0,str(ROOT/'Research_Release/extensions/recent_data'))
import common

def metrics(score,label):
    npos=int(label.sum());nneg=len(label)-npos
    if npos==0 or nneg==0:return None
    order=np.argsort(-score,kind='stable');s=score[order];y=label[order]
    ends=np.r_[np.flatnonzero(np.diff(s)!=0),len(s)-1]
    tp=np.r_[0,np.cumsum(y)[ends]];fp=np.r_[0,np.cumsum(~y)[ends]]
    tpr=tp/npos;fpr=fp/nneg
    return np.array([np.trapezoid(tpr,fpr),np.max(tpr[fpr<=.05])])

def main():
                                                                           
    y=np.array([False,True])
    if not np.array_equal(metrics(np.array([0.,1.]),y),[1.,1.]):raise AssertionError('Perfect ROC')
    if not np.array_equal(metrics(np.array([1.,0.]),y),[0.,0.]):raise AssertionError('Inverted ROC')
    if not np.array_equal(metrics(np.array([1.,1.]),y),[.5,0.]):raise AssertionError('Tie ROC')
    rows=[]
    for folder in sorted(p.parent for p in SOURCE.glob('*/hourly_alert_predictions.csv')):
        f=pd.read_csv(folder/'hourly_alert_predictions.csv',dtype={'station':str})
        p=f.pivot(index=['timestamp','station','positive'],columns='method',values='prediction')
        p=p[['sheaf_causal','var_causal']].dropna()
        labels=p.index.get_level_values('positive').to_numpy(dtype=bool)
        arrays=p.to_numpy();point_a=metrics(arrays[:,0],labels);point_b=metrics(arrays[:,1],labels)
        for days in [7,14]:
            dates=pd.to_datetime(p.index.get_level_values('timestamp'))
            bins=(dates-dates.min()).days//days
            groups=[np.flatnonzero(bins==b) for b in np.unique(bins)]
            row={'cohort':folder.name,'bin_days':days,'calendar_bins':len(groups),'cases':len(p),
                 'positive_hours':int(labels.sum()),'negative_hours':int((~labels).sum()),
                 'auc_difference':None,'fpr05_sensitivity_difference':None,'auc_ci95':None,
                 'fpr05_sensitivity_ci95':None,'valid_draws':0,'planned_draws':500}
            if point_a is not None:
                diff=point_a-point_b;row['auc_difference']=float(diff[0]);row['fpr05_sensitivity_difference']=float(diff[1])
                if len(groups)>=5:
                    rng=np.random.default_rng(5371);boot=[]
                    for chosen in rng.integers(0,len(groups),(500,len(groups))):
                        ix=np.concatenate([groups[g] for g in chosen])
                        a=metrics(arrays[ix,0],labels[ix]);b=metrics(arrays[ix,1],labels[ix])
                        if a is not None:boot.append(a-b)
                    row['valid_draws']=len(boot)
                    if len(boot)==500:
                        limits=np.quantile(np.asarray(boot),[.025,.975],axis=0)
                        row['auc_ci95']=limits[:,0].tolist();row['fpr05_sensitivity_ci95']=limits[:,1].tolist()
            row['scope']='Paired exploratory calendar-bin bootstrap; ROC thresholds use test labels descriptively, not for deployment.'
            rows.append(row)
        print('Matched ROC uncertainty',folder.name,flush=True)
    (OUT/'PAIRED_ROC_UNCERTAINTY.json').write_text(json.dumps(common.clean(rows),indent=2))

if __name__=='__main__':main()
