                                                                          
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'Research_Release/expanded_validation_results'
OUT=ROOT/'Research_Release/weakness_resolution_results'
sys.path.insert(0,str(ROOT/'Research_Release/extensions/recent_data'))
import common
import pandas as pd,numpy as np

def difference_ci(frame,days):
    if frame.empty:return None
    t=pd.to_datetime(frame.timestamp);bins=(t-t.min()).dt.days//days
    sums=frame.groupby(bins).agg(cells=('covered','size'),risk=('covered','sum'),control=('control_covered','sum')).to_numpy()
    if len(sums)<5:return None
    rng=np.random.default_rng(9127)
    draws=sums[rng.integers(0,len(sums),(1000,len(sums)))].sum(axis=1)
    return np.quantile((draws[:,1]-draws[:,2])/draws[:,0],[.025,.975]).tolist()

def main():
    dest=OUT/'width_control';dest.mkdir(exist_ok=True);summary=[];excluded=[]
    for folder in sorted(p.parent for p in SOURCE.glob('*/interval_calibration_predictions.csv')):
        name=folder.name;d=np.load(folder/'dataset.npz',allow_pickle=False)
        thresholds=dict(zip(d['stations'].astype(str),np.nanquantile(d['X'][d['split']==0,:,0],.75,axis=0)))
        cal=pd.read_csv(folder/'interval_calibration_predictions.csv',dtype={'station':str})
        cal['risk']=np.maximum(cal.prediction,cal.city_prediction)>cal.station.map(thresholds)
        cal['error']=abs(cal.truth-cal.prediction)
        f=pd.read_csv(OUT/'uncertainty_decisions'/name/'interval_cases.csv',dtype={'station':str},float_precision='round_trip')
        f=f[f.method.eq('risk_station')].copy();records=[]
        for (loss,station),test in f.groupby(['context_loss','station']):
            c=cal[cal.loss.eq(loss)&cal.station.eq(station)]
            q=[common.quantile(c.loc[c.risk.eq(risk),'error']) for risk in [False,True]]
            fixed=common.quantile(c.error)
            if not np.isfinite(q+[fixed]).all():
                excluded.append({'cohort':name,'loss':loss,'station':station,'test_cells':len(test),'reason':'Calibration cannot give finite quantiles in both risk groups'})
                continue
            radius=float(np.mean(np.where(c.risk,q[1],q[0])))
            test['risk_radius']=test.radius;test['control_radius']=radius
            test['control_covered']=abs(test.truth-test.prediction)<=radius
            test['control_width']=2*radius
            test['station_fixed_radius']=fixed
            test['control_decided']=(test.prediction-radius>60)|(test.prediction+radius<=60)
            records.append(test)
        frame=pd.concat(records,ignore_index=True) if records else pd.DataFrame()
        frame.to_csv(dest/(name+'_cases.csv'),index=False)
        if len(frame):
            for loss,g in frame.groupby('context_loss'):
                for group,h in [('all',g),('extreme',g[g.extreme])]:
                    summary.append({'cohort':name,'loss':loss,'group':group,'cells':len(h),'stations':h.station.nunique(),
                                    'risk_coverage':h.covered.mean(),'control_coverage':h.control_covered.mean(),
                                    'risk_width':h.width.mean(),'control_width':h.control_width.mean(),
                                    'risk_decided_fraction':h.decided.mean(),'control_decided_fraction':h.control_decided.mean(),
                                    'coverage_difference_ci7':difference_ci(h,7),'coverage_difference_ci14':difference_ci(h,14)})
        print('Calibration-width control',name,flush=True)
    pd.DataFrame(summary).to_csv(dest/'SUMMARY.csv',index=False)
    (dest/'EXCLUSIONS.json').write_text(json.dumps(excluded,indent=2))
    (dest/'status.json').write_text(json.dumps({'status':'completed','source_sha256':common.sha(__file__),
         'protocol_sha256':common.sha(HERE/'WIDTH_CONTROL_PROTOCOL.md'),'test_labels_choose_radius':False,
         'scope':'Exploratory calibration-mean-width control; paired calibration-supported cells; test widths can differ under shift'},indent=2))

if __name__=='__main__':main()
