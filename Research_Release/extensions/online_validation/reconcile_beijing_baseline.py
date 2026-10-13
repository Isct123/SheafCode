                                                                   

                                                                      
                                                                          
                                                                      
   
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
from beijing import load_archive
from reference import QuadraticDecoder,FullVAR,graph_correlation,complete_training

def main():
    original=OUT/'beijing_cases.csv';frame=pd.read_csv(original)
    X,times,split,stations,excluded,mu,sd=load_archive(OUT/'beijing_original.zip');Z=(X-mu)/sd
    edges=graph_correlation(Z[split==0]);model=QuadraticDecoder(Z[split==0],edges)
    choice=json.loads((OUT/'beijing_choices.json').read_text())
    var=FullVAR(complete_training(Z[split==0]),len(stations),4,choice['ridge'],choice['noise'],choice['shrink'])
    np.savez_compressed(OUT/'beijing_model.npz',delta=model.D,P=model.P,mean=mu,std=sd,
        edges=np.array(edges),stations=np.array(stations),lambda_=np.array(model.lam),
        temporal_1=np.array(model.temporal[1]),temporal_24=np.array(model.temporal[24]),
        var_A=var.A,var_Q=var.Q,var_R=np.array(var.R))
    for row in frame.index[frame.method=='city_mean']:
        day=pd.Timestamp(frame.loc[row,'day']);v=stations.index(frame.loc[row,'station']);ix=np.flatnonzero(times.floor('D')==day)
        other=np.delete(X[ix],v,axis=1);counts=np.isfinite(other).sum(1)
        raw=np.divide(np.nansum(other,axis=1),counts,out=np.broadcast_to(mu[v],(24,4)).copy(),where=counts>0)
        estimate=(raw-mu[v])/sd[v];truth=Z[ix,v];good=np.isfinite(truth)
        frame.loc[row,'mse']=np.mean((estimate-truth)[good]**2)
        qgood=good[:,0];frame.loc[row,'pm25_raw_mse']=np.mean((raw[:,0]-X[ix,v,0])[qgood]**2) if qgood.any() else np.nan
    frame.to_csv(OUT/'beijing_corrected_cases.csv',index=False)
    summary=json.loads((OUT/'beijing_summary.json').read_text())
    summary['comparisons']={name:paired(frame,'temporal_sheaf',name) for name in ['full_covariance_var','seasonal','city_mean']}
    summary['pm25_raw_comparisons']={name:paired(frame.assign(mse=frame.pm25_raw_mse),'temporal_sheaf',name) for name in ['full_covariance_var','seasonal','city_mean']}
    summary['baseline_reconciliation']={'original_cases_sha256':sha(original),'corrected_cases_sha256':sha(OUT/'beijing_corrected_cases.csv'),
        'source_sha256':sha(__file__),'change':'Raw concentration city mean, standardized by the target training scale; original comparator had averaged other-station standardized coordinates. Only city-mean case rows change; all fitted models and primary VAR/seasonal comparisons preserved. No test-based method selection.'}
    save('beijing_corrected_summary.json',summary)
    print(summary['comparisons'],flush=True)

if __name__=='__main__':main()
