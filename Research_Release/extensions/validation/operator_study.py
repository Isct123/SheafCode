import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
from operator_uncertainty import inflate
from refined_bounds import envelope
def main():
    rng=np.random.default_rng(419);rows=[]
    for region in REGIONS:
        ds,m,folder=cohort(region);D=np.load(folder/'model.npz')['delta']
        for case in range(12):
            v=case%ds.n;A=D[:,v*ds.p:(v+1)*ds.p];B=[D[:,j*ds.p:(j+1)*ds.p] for j in range(ds.n) if j!=v];h=np.eye(ds.p)[0];z=rng.uniform(-1,1,ds.p);e=rng.uniform(-1,1,ds.p)
            for radius in [0.,.01,.05,.1]:
                RA=radius*abs(A);RB=[radius*abs(x) for x in B];trueA=A+rng.uniform(-1,1,A.shape)*RA;trueB=B[0]+rng.uniform(-1,1,B[0].shape)*RB[0];b=-trueA@z-trueB@e+rng.uniform(-.03,.03,len(D));eps=inflate(.03,[-3]*ds.p,[3]*ds.p,RA,0.,RB,1,1.)
                nominal=envelope(A,b,h,B,1,.03,1.,[-3]*ds.p,[3]*ds.p);robust=envelope(A,b,h,B,1,eps,1.,[-3]*ds.p,[3]*ds.p)
                contained=robust['lower']-1e-6<=h@z<=robust['upper']+1e-6
                if not contained:raise AssertionError('Uncertain-map envelope excluded a feasible quantity')
                rows.append({'region':region,'case':case,'relative_entry_radius':radius,'robust_width':robust['upper']-robust['lower'],'robust_covered':contained,'nominal_status':nominal['status'],'nominal_covered':nominal['status']!='inconsistent' and nominal['lower']-1e-6<=h@z<=nominal['upper']+1e-6,'scope':'Synthetic bounded perturbations of actual learned maps'})
        print(region,'operator uncertainty checks complete',flush=True)
    f=pd.DataFrame(rows);f.to_csv(OUT/'operator_uncertainty_cases.csv',index=False);f.groupby(['region','relative_entry_radius']).agg(cases=('robust_covered','size'),robust_coverage=('robust_covered','mean'),nominal_coverage=('nominal_covered','mean'),median_width=('robust_width','median')).reset_index().to_csv(OUT/'operator_uncertainty_summary.csv',index=False)
    save('operator_scope.json',{'source_sha256':digest(__file__),'solver_sha256':digest(HERE/'operator_uncertainty.py'),'scope':'Synthetic scenario radii, conditional mathematical containment, not estimates of real physical map error or field coverage'})
if __name__=='__main__':main()
