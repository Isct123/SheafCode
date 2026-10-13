                                                                        
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'Research_Release/expanded_validation_results';OUT=ROOT/'Research_Release/weakness_resolution_results/uncertainty_decisions'
sys.path.insert(0,str(ROOT/'Research_Release/extensions/recent_data'))
import common
import numpy as np,pandas as pd

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def coverage_ci(frame,days=7):
    if frame.empty:return None
    t=pd.to_datetime(frame.timestamp)
    bins=(t-t.min()).dt.days//days
    group=frame.groupby(bins).covered.agg(['size','sum']).to_numpy()
    if len(group)<5:return None
    rng=np.random.default_rng(1379)
    selected=group[rng.integers(0,len(group),(2000,len(group)))].sum(axis=1)
    return np.quantile(selected[:,1]/selected[:,0],[.025,.975]).tolist()

def roc(frame):
                                                                   
    positive=int(frame.positive.sum());negative=len(frame)-positive
    if positive==0 or negative==0:return None
    g=frame.groupby('prediction',sort=True).positive.agg(['sum','size']).iloc[::-1]
    tp=np.r_[0,g['sum'].cumsum().to_numpy()];fp=np.r_[0,(g['size']-g['sum']).cumsum().to_numpy()]
    tpr=tp/positive;fpr=fp/negative
    auc=float(np.trapezoid(tpr,fpr))
    precision=np.divide(tp,tp+fp,out=np.ones(len(tp),float),where=tp+fp>0)
    ap=float(np.sum(np.diff(tpr)*precision[1:]))
    rows=pd.DataFrame({'fpr':fpr,'tpr':tpr,'precision':precision,
                       'threshold':np.r_[np.inf,g.index.to_numpy()]})
                                                                                 
    points={str(cap):float(np.max(tpr[fpr<=cap])) for cap in [.01,.05,.10,.20]}
                                                                                        
    xx=np.r_[fpr[fpr<.1],.1];yy=np.r_[tpr[fpr<.1],np.interp(.1,fpr,tpr)]
    partial=float(np.trapezoid(yy,xx)/.1)
    return {'positive_hours':positive,'negative_hours':negative,'auc':auc,'average_precision':ap,
            'normalized_partial_auc_0_10':partial,'sensitivity_under_fpr_cap':points},rows

def main():
    OUT.mkdir(parents=True,exist_ok=True);overview=[];alert_overview=[]
    for folder in sorted(p.parent for p in SOURCE.glob('*/uncertainty_cases.csv')):
        name=folder.name;dest=OUT/name;dest.mkdir(exist_ok=True)
        ds=np.load(folder/'dataset.npz',allow_pickle=False)
        stations=ds['stations'].astype(str)
        limits=dict(zip(stations,np.nanquantile(ds['X'][ds['split']==0,:,0],.75,axis=0)))
        scales=dict(zip(stations,ds['sd'][:,0]))
        cal=pd.read_csv(folder/'interval_calibration_predictions.csv',dtype={'station':str})
        test=pd.read_csv(folder/'uncertainty_cases.csv',dtype={'station':str})
        base=test[test.method.eq('station_fixed')].copy()
        for f in [cal,base]:
            f['high_risk']=np.maximum(f.prediction,f.city_prediction)>f.station.map(limits)
        cal['score']=abs(cal.truth-cal.prediction);cal['scaled_score']=cal.score/cal.station.map(scales)
        outputs=[]
        for loss in [0.,.2,.5]:
            cc=cal[cal.loss.eq(loss)];ff=base[base.context_loss.eq(loss)]
            for method in ['pooled_fixed','station_fixed','availability_scaled','risk_station','risk_pooled_scaled']:
                f=ff.copy();radii=[];counts=[];cache={}
                for station,risk,scale,gain in zip(f.station,f.high_risk,f.station.map(scales),f.pre_outcome_scale):
                    key=(station if method not in ['pooled_fixed','risk_pooled_scaled'] else '',risk if method in ['risk_station','risk_pooled_scaled'] else False)
                    if key not in cache:
                        if method=='pooled_fixed':c=cc;score=c.score
                        elif method=='risk_pooled_scaled':c=cc[cc.high_risk.eq(risk)];score=c.scaled_score
                        else:
                            c=cc[cc.station.eq(station)]
                            if method=='risk_station':c=c[c.high_risk.eq(risk)]
                            score=c.score/(c.pre_outcome_scale if method=='availability_scaled' else 1.)
                        cache[key]=(common.quantile(score),len(c))
                    q,count=cache[key]
                    mult=scale if method=='risk_pooled_scaled' else (gain if method=='availability_scaled' else 1.)
                    radii.append(q*mult);counts.append(count)
                f['method']=method;f['calibration_group_count']=counts;f['radius']=radii
                f['covered']=abs(f.truth-f.prediction)<=f.radius;f['width']=2*f.radius
                f['full_set']=~np.isfinite(f.radius);f['lower']=f.prediction-f.radius;f['upper']=f.prediction+f.radius
                f['decided']=(f.lower>60)|(f.upper<=60)
                f['correct_when_decided']=(f.lower>60)==(f.truth>60)
                outputs.append(f)
                                                                                 
                if method in ['pooled_fixed','station_fixed','availability_scaled']:
                    old=test[test.method.eq(method)&test.context_loss.eq(loss)]
                    a=f.merge(old,on=['timestamp','station','context_loss'],suffixes=('_new','_old'),validate='one_to_one')
                    if len(a)!=len(f) or not np.allclose(a.width_new,a.width_old,equal_nan=True):raise AssertionError('Baseline interval replay mismatch')
                for group,g in [('all',f),('extreme',f[f.extreme]),('predicted_high_risk',f[f.high_risk])]:
                    quarters=g.assign(quarter=pd.to_datetime(g.timestamp).dt.to_period('Q').astype(str)).groupby(['station','quarter']).covered.agg(['size','mean'])
                    finite=g.loc[~g.full_set,'width']
                    overview.append({'cohort':name,'context_loss':loss,'method':method,'group':group,'cells':len(g),
                                     'coverage':g.covered.mean() if len(g) else np.nan,'ci7':coverage_ci(g),
                                     'ci14':coverage_ci(g,14),'finite_mean_width':finite.mean(),
                                     'full_set_fraction':g.full_set.mean() if len(g) else np.nan,
                                     'decision_fraction':g.decided.mean() if len(g) else np.nan,
                                     'accuracy_when_decided':g.loc[g.decided,'correct_when_decided'].mean() if g.decided.any() else np.nan,
                                     'minimum_station_quarter_coverage':quarters['mean'].min() if len(quarters) else np.nan,
                                     'minimum_station_quarter_cells':int(quarters.loc[quarters['mean'].idxmin(),'size']) if len(quarters) else 0})
        pd.concat(outputs,ignore_index=True).to_csv(dest/'interval_cases.csv',index=False)
        alerts=pd.read_csv(folder/'hourly_alert_predictions.csv',dtype={'station':str})
        keys=None;truths=None
        for method,g in alerts.groupby('method'):
            k=set(map(tuple,g[['timestamp','station']].to_numpy()))
            if keys is None:keys=k;truths=g.set_index(['timestamp','station']).truth.sort_index()
            elif k!=keys or not np.array_equal(g.set_index(['timestamp','station']).truth.sort_index(),truths):raise AssertionError('Alert methods have different truth sets')
            result=roc(g)
            if result is None:
                alert_overview.append({'cohort':name,'method':method,'status':'undefined: no positive or no negative observations',
                                       'positive_hours':int(g.positive.sum()),'negative_hours':int((~g.positive).sum())})
            else:
                summary,curve=result;curve.to_csv(dest/(method+'_roc.csv'),index=False)
                alert_overview.append({'cohort':name,'method':method,'status':'evaluated',**summary})
        identity={'source_sha256':sha(Path(__file__)),'protocol_sha256':sha(HERE/'PROTOCOL.md'),
                  'input_sha256':{f:sha(folder/f) for f in ['dataset.npz','interval_calibration_predictions.csv','uncertainty_cases.csv','hourly_alert_predictions.csv']}}
        (dest/'status.json').write_text(json.dumps({'status':'completed','identity':identity,'risk_thresholds':limits,
            'test_label_feedback':False,'matched_baselines_replayed':True,
            'scope':'Risk groups are prediction-time covariates. Coverage remains empirical under temporal shift; ROC test thresholds are descriptive, not deployable.'},indent=2))
        print('Risk intervals and matched ROC',name,flush=True)
    frame=pd.DataFrame(overview);frame.to_csv(OUT/'INTERVAL_SUMMARY.csv',index=False)
    (OUT/'ALERT_SUMMARY.json').write_text(json.dumps(common.clean(alert_overview),indent=2))
    (OUT/'status.json').write_text(json.dumps({'status':'completed','cohorts':len(set(frame.cohort)),'scope':'Exploratory post-primary analyses; all results retained'}))

if __name__=='__main__':main()
