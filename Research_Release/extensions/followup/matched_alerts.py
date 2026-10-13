                                                                            

                                                                              
   
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]

def sensitivity_at_fpr(truth,score,rate):
    truth=np.asarray(truth,bool);score=np.asarray(score,float)
    if not truth.any() or truth.all():return np.nan
    thresholds=np.r_[np.inf,np.unique(score)[::-1],-np.inf]
    pos=np.sort(score[truth]);neg=np.sort(score[~truth])
    tp=(len(pos)-np.searchsorted(pos,thresholds,side='left'))/len(pos)
    fp=(len(neg)-np.searchsorted(neg,thresholds,side='left'))/len(neg)
                                                                             
    unique=np.unique(fp);best=np.array([tp[fp==x].max() for x in unique])
    return float(np.interp(rate,unique,best))

def main():
    source=ROOT/'followup_results/validation_fixed_alerts.csv';f=pd.read_csv(source);out=ROOT/'constrained_results';out.mkdir(exist_ok=True);rows=[];rng=np.random.default_rng(901)
    for (region,missing),g in f.groupby(['region','extra_cell_missing_rate']):
        p=g.pivot(index=['timestamp','station','truth_positive'],columns='method',values='prediction').dropna().reset_index()
        p['day']=pd.to_datetime(p.timestamp).dt.floor('D');days=list(p.groupby('day').indices.values())
        for rate in [.01,.02,.05]:
            truth=p.truth_positive.to_numpy();a=sensitivity_at_fpr(truth,p.sheaf,rate);b=sensitivity_at_fpr(truth,p.city_mean,rate);draws=[]
            for _ in range(500):
                chosen=rng.integers(0,len(days),len(days));ix=np.concatenate([days[i] for i in chosen]);sample=p.iloc[ix]
                delta=sensitivity_at_fpr(sample.truth_positive,sample.sheaf,rate)-sensitivity_at_fpr(sample.truth_positive,sample.city_mean,rate)
                if np.isfinite(delta):draws.append(delta)
            ci=np.quantile(draws,[.025,.975]) if draws else [np.nan,np.nan]
            rows.append({'region':region,'missing_rate':missing,'target_fpr':rate,'cases':len(p),'days':len(days),'positive_cases':int(truth.sum()),
                'sheaf_sensitivity':a,'city_mean_sensitivity':b,'difference':a-b,'day_bootstrap_lower':float(ci[0]),'day_bootstrap_upper':float(ci[1]),'valid_bootstrap_draws':len(draws)})
    pd.DataFrame(rows).to_csv(out/'matched_false_alarm_comparisons.csv',index=False)
    (out/'matched_alert_scope.json').write_text(json.dumps({'scope':'Empirical matched-FPR ROC evaluation; interpolated rates describe randomized cutoffs, not a deployment policy',
        'limitations':['Archive previously examined','Day-cluster bootstrap is exploratory and not multiplicity adjusted','Rare positives yield wide uncertainty','Hourly threshold illustrative, not official AQI/GRAP','No emissions or health benefit established']},indent=2))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
