import sys,itertools,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
sys.path.insert(0,str(ROOT/'extensions/structured'));from holonomy import structured_components,outage_rank
sys.path.insert(0,str(ROOT/'extensions/functional'));from certificate import functional_interval
sys.path.insert(0,str(ROOT/'extensions/followup'));from constrained import constrained_envelope
from budget_dual import upper_certificate
sys.path.insert(0,str(ROOT/'extensions/joint_decisions'));from joint import joint_envelope
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
def main():
    for region in regions():
        ds=load(region);m=models(region);Z=ds['Z'];n,p=Z.shape[1:];D=m['sheaf'].D;result=structured_components(D,n,p,m['edges']);ranks=[];margin=[];thinning=[]
        for k in range(1,n+1):
            for F in itertools.combinations(range(n),k):
                cols=np.concatenate([np.arange(v*p,(v+1)*p) for v in F]);A=D[:,cols];rank=np.linalg.matrix_rank(A,tol=result['rank_absolute_tolerance']);pred=outage_rank(result,F);ranks.append({'support':','.join(map(str,F)),'size':k,'rank':int(rank),'predicted_rank':pred,'agree':rank==pred})
                if k<=4:margin.append({'support':','.join(map(str,F)),'size':k,'minimum_singular_value':float(np.linalg.svd(A,compute_uv=False)[-1]) if A.shape[0]>=len(cols) else 0.})
        for r in sorted(set([1,max(1,p//2),p])):
            DD=np.vstack([D[j*p:j*p+r] for j in range(len(m['edges']))])
            for k in range(1,n+1):
                for F in itertools.combinations(range(n),k):
                    cols=np.concatenate([np.arange(v*p,(v+1)*p) for v in F]);A=DD[:,cols];pattern=csr_matrix(np.abs(A)>1e-14);structural=int(np.sum(maximum_bipartite_matching(pattern,perm_type='column')>=0));actual=int(np.linalg.matrix_rank(A));thinning.append({'edge_rows':r,'support':','.join(map(str,F)),'size':k,'generic_pattern_rank':structural,'learned_rank':actual,'agree':structural==actual})
        frame_save(region+'/structured_rank_cases.csv',pd.DataFrame(ranks));frame_save(region+'/exhaustive_block_margins.csv',pd.DataFrame(margin));frame_save(region+'/thinning_cases.csv',pd.DataFrame(thinning));save(region+'/structured_math.json',result)
        full=np.isfinite(Z).all((1,2));cal=Z[(ds['split']==3)&full];eps2=quantile(np.linalg.norm(cal.reshape(len(cal),-1)@D.T,axis=1));epsinf=quantile(np.max(abs(cal.reshape(len(cal),-1)@D.T),axis=1));test=np.flatnonzero((ds['split']==2)&full);indices=test[np.unique(np.linspace(0,len(test)-1,min(40,len(test))).astype(int))];bounds=[];rng=np.random.default_rng(204)
        for repeat,t in enumerate(indices):
            v=repeat%n;U=np.arange(v*p,(v+1)*p);O=np.setdiff1d(np.arange(n*p),U);state=Z[t].ravel();A=D[:,U];b=D[:,O]@state[O];h=np.eye(p)[0];r=functional_interval(A,b,h,eps2);truth=state[U][0]
            blocks=[D[:,j*p:(j+1)*p] for j in range(n) if j!=v];exact=constrained_envelope(A,b,h,blocks,1,epsinf,1.,[-6]*p,[6]*p);upper=upper_certificate(A,b,h,blocks,1,epsinf,1.,[-6]*p,[6]*p);lower=upper_certificate(A,b,-h,blocks,1,epsinf,1.,[-6]*p,[6]*p)
            references={'PM2.5':60.,'PM10':100.,'O3':100.,'O3_ppm':.05,'NO2_ppb':40.};denom=np.array([references[c] for c in ds['channels']]);H=np.diag(ds['sd'][v]/denom);offset=ds['mu'][v]/denom;joint=joint_envelope(A,b,H,offset,blocks,1,epsinf,1.,[-6]*p,[6]*p)
            assumptions=bool(np.max(abs(state[U]))<=6 and np.max(abs(D@state))<=epsinf)
            bounds.append({'timestamp':str(ds['times'][t]),'station':str(ds['stations'][v]),'functional_status':r['status'],'functional_lower':r.get('lower'),'functional_upper':r.get('upper'),'functional_covered':r['status']=='bounded' and r['lower']<=truth<=r['upper'],'functional_assumptions_met':np.linalg.norm(D@state)<=eps2,'bounded_status':exact['status'],'bounded_lower':exact.get('lower'),'bounded_upper':exact.get('upper'),'bounded_covered':exact['status']=='bounded' and exact['lower']-1e-7<=truth<=exact['upper']+1e-7,'bounded_assumptions_met':assumptions,'dual_upper':upper.get('upper'),'dual_lower':-lower['upper'] if lower.get('upper') is not None else None,'joint_status':joint['status'],'joint_lower':joint.get('lower'),'joint_upper':joint.get('upper'),'joint_extra_lower':joint.get('joint_lower_gain'),'joint_decision':joint.get('decision'),'physical_scope':'Illustrative native-unit references, not AQI/GRAP'})
        frame_save(region+'/functional_joint_bounded_cases.csv',pd.DataFrame(bounds))
                                                                                       
        synthetic=[]
        for repeat in range(12):
            v=repeat%n;U=np.arange(v*p,(v+1)*p);A=D[:,U];blocks=[D[:,j*p:(j+1)*p] for j in range(n) if j!=v];truth=rng.uniform(-1,1,p);fault=rng.uniform(-1,1,p);b=-A@truth-blocks[repeat%len(blocks)]@fault+rng.uniform(-.03,.03,len(D));h=np.eye(p)[0];exact=constrained_envelope(A,b,h,blocks,1,.03,1.,[-3]*p,[3]*p);upper=upper_certificate(A,b,h,blocks,1,.03,1.,[-3]*p,[3]*p);lower=upper_certificate(A,b,-h,blocks,1,.03,1.,[-3]*p,[3]*p)
            if not exact['lower']-1e-6<=truth[0]<=exact['upper']+1e-6:raise AssertionError('Feasible planted truth excluded')
            if not -lower['upper']-1e-6<=exact['lower']<=exact['upper']<=upper['upper']+1e-6:raise AssertionError('Outer bound excludes sharp union')
            synthetic.append({'repeat':repeat,'truth':truth[0],'exact_lower':exact['lower'],'exact_upper':exact['upper'],'dual_lower':-lower['upper'],'dual_upper':upper['upper'],'scope':'Synthetic feasible state on recent fitted operator'})
        frame_save(region+'/synthetic_certificate_cases.csv',pd.DataFrame(synthetic));save(region+'/mathematics_status.json',{'status':'completed','all_structured_rank_agree':all(r['agree'] for r in ranks),'rank_subsets':len(ranks),'exhaustive_margin_subsets':len(margin),'thinning_subsets':len(thinning),'thinning_disagreements':sum(not r['agree'] for r in thinning),'actual_functional_cases':len(bounds),'synthetic_certificate_cases':len(synthetic),'calibration_complete_hours':len(cal),'residual_l2_budget':eps2,'residual_infinity_budget':epsinf,'priority_scope':'Previously stated derivations tested on recent maps; no newly invented general theorem claimed'})
        print(region,'mathematics',len(ranks),'subsets',len(bounds),'actual states',flush=True)
if __name__=='__main__':main()
