                                                                               
import sys,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
from evaluate import predict,score
from scipy.stats import spearmanr
def main():
    for region in regions():
        ds=load(region);m=models(region);Z=ds['Z'];n,p=Z.shape[1:];val=[]
        for day,ix in subset_days(blocks(ds,1),12):
            for v,station in enumerate(ds['stations']):
                good=np.isfinite(Z[ix[24:],v])
                if good.sum()<12*p:continue
                pred=predict(Z[ix],v,24,m,ds,False);val.append((good,Z[ix[24:],v],np.array([pred[k] for k in ['sheaf_causal','var_causal','seasonal']])))
        grid=[]
        for a in range(11):
            for b in range(11-a):
                w=np.array([a,b,10-a-b])/10;loss=np.mean([np.mean((np.einsum('a,aij->ij',w,pr)-truth)[good]**2) for good,truth,pr in val]);grid.append({'sheaf_weight':w[0],'var_weight':w[1],'seasonal_weight':w[2],'validation_mse':loss})
        best=min(grid,key=lambda x:x['validation_mse']);w=np.array([best[k] for k in ['sheaf_weight','var_weight','seasonal_weight']]);frame_save(region+'/hybrid_validation_grid.csv',pd.DataFrame(grid));save(region+'/hybrid_choice.json',best);rows=[]
        for day,ix in blocks(ds,2):
            for v,station in enumerate(ds['stations']):
                truth=Z[ix[24:],v];good=np.isfinite(truth)
                if good.sum()<12*p:continue
                pred=predict(Z[ix],v,24,m,ds,False);pr=np.array([pred[k] for k in ['sheaf_causal','var_causal','seasonal']]);hybrid=np.einsum('a,aij->ij',w,pr);rows.append({'day':day,'station':str(station),'method':'hybrid',**score(hybrid,truth,ds['sd'][v])})
        f=pd.DataFrame(rows);frame_save(region+'/hybrid_cases.csv',f);ref=pd.read_csv(OUT/region/'outage_cases.csv',dtype={'station':str});save(region+'/hybrid_summary.json',{method:paired(pd.concat([f,ref]),'hybrid',method) for method in ['sheaf_causal','var_causal','seasonal']})
                                                                                   
        cal=pd.read_csv(OUT/region/'interval_calibration_predictions.csv',dtype={'station':str});test=pd.read_csv(OUT/region/'uncertainty_cases.csv',dtype={'station':str});out=[]
        for loss in [0.,.2,.5]:
            cc=cal[cal.loss==loss];cc=cc.assign(error=abs(cc.truth-cc.prediction));ff=test[(test.context_loss==loss)&(test.method=='station_fixed')]
            for station,g in ff.groupby('station'):
                c=cc[cc.station==station];daily=c.groupby('day').error.max();radius=quantile(daily);week=((pd.to_datetime(c.day)-pd.to_datetime(c.day).min()).dt.days//7);weekly=c.groupby(week).error.max();weeklyradius=quantile(weekly)
                for method,r in [('day_max',radius),('week_max',weeklyradius)]:
                    gg=g.copy();gg['method']=method;gg['lower']=gg.prediction-r;gg['upper']=gg.prediction+r;gg['covered']=(gg.truth>=gg.lower)&(gg.truth<=gg.upper);gg['width']=2*r;gg['full_set']=not np.isfinite(r);gg['decided']=(gg.lower>60)|(gg.upper<=60);out.append(gg)
        f=pd.concat(out);frame_save(region+'/block_interval_cases.csv',f);frame_save(region+'/block_interval_summary.csv',f.groupby(['context_loss','method']).agg(cases=('covered','size'),coverage=('covered','mean'),width=('width','mean'),full_set_fraction=('full_set','mean'),decided_fraction=('decided','mean')).reset_index())
                                                                            
        rows=[]
        for station,g in ref[ref.method=='spatial_sheaf'].groupby('station'):
            v=int(np.flatnonzero(ds['stations']==station)[0]);A=m['sheaf'].D[:,v*p:(v+1)*p];sing=np.linalg.svd(A,compute_uv=False);rows.append({'station':station,'minimum_singular_value':sing[-1],'inverse_margin':1/sing[-1],'test_mean_mse':g.mse.mean(),'station_days':len(g)})
        frame_save(region+'/risk_diagnostic.csv',pd.DataFrame(rows));risk=spearmanr([r['inverse_margin'] for r in rows],[r['test_mean_mse'] for r in rows]);save(region+'/additional_status.json',{'status':'completed','hybrid_rows':len(f),'risk_station_count':len(rows),'risk_spearman':risk.statistic,'risk_pvalue':risk.pvalue,'scope':'Additional analyses specified after primary/interval outcomes; exploratory. Full-day/week coverage needs exchangeable calibration blocks and compatible truth availability; weekly calibration may yield infinite sets.'})
        print(region,'hybrid, block intervals and risk diagnostic completed',flush=True)
if __name__=='__main__':main()
