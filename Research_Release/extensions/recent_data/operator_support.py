import sys,itertools,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
sys.path.insert(0,str(ROOT/'extensions/validation'));from operator_uncertainty import inflate
sys.path.insert(0,str(ROOT/'extensions/followup'));from constrained import constrained_envelope
from exact_subspaces import exact_lattice
def main():
    for region in regions()+json.loads((OUT/'cohorts.json').read_text()).get('secondary',[]):
        ds=load(region);m=models(region);D=m['sheaf'].D;n,p=ds['Z'].shape[1:];rng=np.random.default_rng(1809);rows=[];lattices=[]
        for v in range(n):
            U=np.arange(v*p,(v+1)*p);A=D[:,U];B=[D[:,j*p:(j+1)*p] for j in range(n) if j!=v];start=time.perf_counter();states=exact_lattice(B,len(D),min(2,len(B)));count=sum(__import__('math').comb(len(B),k) for k in range(min(2,len(B))+1));lattices.append({'station':str(ds['stations'][v]),'support_count':count,'distinct_exact_decimal_spans':len(states),'seconds':time.perf_counter()-start,'scope':'Exact decimal-rational nominal raw nuisance spans; unrestricted coefficients only, not bounded fault images'})
        for repeat in range(12):
            v=repeat%n;U=np.arange(v*p,(v+1)*p);A=D[:,U];B=[D[:,j*p:(j+1)*p] for j in range(n) if j!=v];z=rng.uniform(-1,1,p);e=rng.uniform(-1,1,p);h=np.eye(p)[0]
            for radius in [0.,.01,.05,.1]:
                RA=radius*abs(A);RB=[radius*abs(x) for x in B];trueA=A+rng.uniform(-1,1,A.shape)*RA;trueB=B[repeat%len(B)]+rng.uniform(-1,1,B[0].shape)*RB[repeat%len(B)];b=-trueA@z-trueB@e+rng.uniform(-.03,.03,len(D));eps=inflate(.03,[-3]*p,[3]*p,RA,0.,RB,1,1.);nominal=constrained_envelope(A,b,h,B,1,.03,1.,[-3]*p,[3]*p);robust=constrained_envelope(A,b,h,B,1,eps,1.,[-3]*p,[3]*p);covered=robust['lower']-1e-6<=z[0]<=robust['upper']+1e-6
                if not covered:raise AssertionError('Map uncertainty inflation failed conditional containment')
                rows.append({'repeat':repeat,'radius':radius,'truth':z[0],'robust_lower':robust['lower'],'robust_upper':robust['upper'],'robust_covered':covered,'nominal_status':nominal['status'],'nominal_covered':nominal['status']=='bounded' and nominal['lower']-1e-6<=z[0]<=nominal['upper']+1e-6,'scope':'Synthetic declared perturbations, not measured map error'})
        frame_save(region+'/operator_uncertainty_cases.csv',pd.DataFrame(rows));frame_save(region+'/exact_nuisance_span_counts.csv',pd.DataFrame(lattices));save(region+'/operator_support_status.json',{'status':'completed','map_uncertainty_cases':len(rows),'exact_span_stations':len(lattices),'source_sha256':sha(__file__),'physical_uncertainty_estimated':False});print(region,'operator uncertainty and exact span checks complete',flush=True)
if __name__=='__main__':main()
