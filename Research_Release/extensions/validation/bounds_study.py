import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
from refined_bounds import envelope
from constrained import constrained_envelope
def main():
    rows=[];rng=np.random.default_rng(732)
                                                                           
                                                                               
    for region in REGIONS:
        ds,m,folder=cohort(region);ds.set_transform(m['choices']['sheaf']['transform']);D=np.load(folder/'model.npz')['delta'];q=ds.pollutants.index('PM2.5')
        for repeat in range(16):
            v=repeat%ds.n;U=np.arange(v*ds.p,(v+1)*ds.p);A=D[:,U];blocks=[D[:,j*ds.p:(j+1)*ds.p] for j in range(ds.n) if j!=v]
            z=rng.uniform(-1,1,ds.p);fault=rng.uniform(-1,1,ds.p);b=-A@z-blocks[repeat%len(blocks)]@fault+rng.uniform(-.03,.03,len(D));h=np.eye(ds.p)[q]
            start=time.perf_counter();exact=constrained_envelope(A,b,h,blocks,1,.03,1.,[-3]*ds.p,[3]*ds.p);exacttime=time.perf_counter()-start
            for budget in [1,7,31,127]:
                start=time.perf_counter();r=envelope(A,b,h,blocks,1,.03,1.,[-3]*ds.p,[3]*ds.p,budget);elapsed=time.perf_counter()-start
                if not r['lower']-1e-6<=exact['lower']<=exact['upper']<=r['upper']+1e-6:raise AssertionError('Refinement excluded exact envelope')
                if not r['lower']-1e-6<=h@z<=r['upper']+1e-6:raise AssertionError('Planted feasible state excluded')
                rows.append({'region':region,'repeat':repeat,'node_budget':budget,'nodes':r['lower_search']['nodes']+r['upper_search']['nodes'],'width':r['upper']-r['lower'],'exact_width':exact['upper']-exact['lower'],'seconds':elapsed,'exact_seconds':exacttime,'complete':r['lower_search']['complete'] and r['upper_search']['complete'],'quantity_truth':float(h@z),'scope':'Actual network maps, synthetic feasible states and faults'})
        print(region,'refinement optimization complete',flush=True)
    f=pd.DataFrame(rows);f.to_csv(OUT/'refinement_cases.csv',index=False);f.groupby(['region','node_budget']).agg(cases=('width','size'),median_width=('width','median'),median_exact_width=('exact_width','median'),median_seconds=('seconds','median'),completed_fraction=('complete','mean')).reset_index().to_csv(OUT/'refinement_summary.csv',index=False)
                                                                             
    prior=pd.read_csv(ROOT/'constrained_results/cases.csv')
    prior.groupby(['region','method','horizon']).agg(cases=('covered','size'),assumption_fraction=('assumptions_met','mean'),coverage=('covered','mean'),decision_fraction=('decision',lambda s:float((s!='abstain').mean()))).reset_index().to_csv(OUT/'certificate_assumption_diagnosis.csv',index=False)
    save('refinement_scope.json',{'source_sha256':digest(__file__),'solver_sha256':digest(HERE/'refined_bounds.py'),'theorem_sha256':digest(HERE/'REFINEMENT_THEOREM.md'),'scope':'Numerical anytime tightening benchmark on actual frozen operators with synthetic feasible states. Historical environmental width/assumption diagnosis is separate; no field benefit or first-ever claim.'})
if __name__=='__main__':main()
