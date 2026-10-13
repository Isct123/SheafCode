import sys,copy,itertools,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
from evaluate import score
sys.path.insert(0,str(ROOT/'extensions/followup'));from exact_certificate import verified_upper
from constrained import constrained_envelope
from budget_dual import upper_certificate
sys.path.insert(0,str(ROOT/'extensions/validation'));from refined_bounds import envelope
def main():
    for region in regions():
        ds=load(region);m=models(region);Z=ds['Z'];n,p=Z.shape[1:];tr=Z[ds['split']==0];original=m['sheaf'];variants={}
        for name,remove in [('no_lag24',24),('no_lag1',1)]:
            model=copy.deepcopy(original);model.temporal.pop(remove);model.precisions={};variants[name]=model
        identity=copy.deepcopy(original);rows=[]
        for u,v in m['edges']:
            block=np.zeros((p,n*p));block[:,u*p:(u+1)*p]=np.eye(p);block[:,v*p:(v+1)*p]=-np.eye(p);rows.append(block)
        identity.D=np.vstack(rows);identity.S=identity.D.T@identity.D+identity.lam*identity.P;identity.precisions={};variants['identity_spatial_maps']=identity
        scalar=copy.deepcopy(original);rows=[]
        for u,v in m['edges']:
            block=np.zeros((p,n*p))
            for q in range(p):
                good=np.isfinite(tr[:,u,q])&np.isfinite(tr[:,v,q]);a=tr[good,u,q];b=tr[good,v,q];rho=float(np.corrcoef(a,b)[0,1]);w=np.sqrt(np.clip(abs(rho),.001,.999)/(1-np.clip(abs(rho),.001,.999)));block[q,u*p+q]=w/max(np.std(a),.001);block[q,v*p+q]=-w*np.sign(rho)/max(np.std(b),.001)
            rows.append(block)
        scalar.D=np.vstack(rows);scalar.S=scalar.D.T@scalar.D+scalar.lam*scalar.P;scalar.precisions={};variants['scalar_spatial_maps']=scalar
        rows=[]
        for day,ix in blocks(ds,2,stride=3):
            for v,station in enumerate(ds['stations']):
                truth=Z[ix[24:],v];good=np.isfinite(truth)
                if good.sum()<12*p:continue
                x=Z[ix].copy();x[24:,v]=np.nan
                for name,model in variants.items():rows.append({'day':day,'station':str(station),'method':name,**score(model.causal(x,v),truth,ds['sd'][v])})
        frame_save(region+'/ablation_cases.csv',pd.DataFrame(rows));ref=pd.read_csv(OUT/region/'outage_cases.csv',dtype={'station':str});save(region+'/ablation_summary.json',{name:paired(pd.concat([pd.DataFrame(rows),ref[ref.primary]]),'sheaf_causal',name) for name in variants})
                                                                                               
        repairs=[];Q=original.S;Sigma=np.linalg.inv(Q);rng=np.random.default_rng(981)
        for day,ix in subset_days(blocks(ds,2),12):
            t=ix[24];x=Z[t:t+1];truth=Z[t]
            for pair in itertools.combinations(range(n),2):
                if not np.isfinite(truth[list(pair)]).all():continue
                U=np.concatenate([np.arange(v*p,(v+1)*p) for v in pair]);cov=np.linalg.inv(Q[np.ix_(U,U)]);tracegains=[];variances=[]
                for j,v in enumerate(pair):
                    R=np.arange(j*p,(j+1)*p);tracegains.append(float(np.trace(cov[:,R]@np.linalg.solve(cov[np.ix_(R,R)],cov[R]))));variances.append(float(np.trace(Sigma[v*p:(v+1)*p,v*p:(v+1)*p])))
                outcomes=[]
                for repair in pair:
                    target=pair[1] if repair==pair[0] else pair[0];hidden=x.copy();hidden[:,target]=np.nan;pred=static_solve(hidden,Q)[0,target];outcomes.append(float(np.mean((pred-truth[target])**2)))
                for method,k in [('trace_policy',int(np.argmax(tracegains))),('variance_policy',int(np.argmax(variances)))]:repairs.append({'day':day,'station':','.join(map(str,pair)),'method':method,'mse':outcomes[k],'repaired_index':pair[k]})
                repairs.append({'day':day,'station':','.join(map(str,pair)),'method':'expected_random','mse':np.mean(outcomes),'repaired_index':-1})
        frame_save(region+'/simulated_repair_cases.csv',pd.DataFrame(repairs));save(region+'/repair_summary.json',{name:paired(pd.DataFrame(repairs),name,'expected_random') for name in ['trace_policy','variance_policy']})
                                                                                             
        records=[];D=original.D
        for j in range(12):
            v=j%n;U=np.arange(v*p,(v+1)*p);A=D[:,U];B=[D[:,q*p:(q+1)*p] for q in range(n) if q!=v];z=rng.uniform(-1,1,p);e=rng.uniform(-1,1,p);b=-A@z-B[j%len(B)]@e+rng.uniform(-.03,.03,len(D));h=np.eye(p)[0];sharp=constrained_envelope(A,b,h,B,1,.03,1,[-3]*p,[3]*p);cert=upper_certificate(A,b,h,B,1,.03,1,[-3]*p,[3]*p);verified=verified_upper(A,b,h,B,1,.03,1,cert['w'],[-3]*p,[3]*p)
            if verified['upper_outward_float']<sharp['upper']-1e-6:raise AssertionError('Verified bound excludes nominal feasible maximum')
            for budget in [1,7,31]:
                start=time.perf_counter();r=envelope(A,b,h,B,1,.03,1,[-3]*p,[3]*p,budget)
                if not r['lower']-1e-6<=sharp['lower']<=sharp['upper']<=r['upper']+1e-6:raise AssertionError('Refinement invalid')
                records.append({'repeat':j,'node_budget':budget,'lower':r['lower'],'upper':r['upper'],'exact_lower':sharp['lower'],'exact_upper':sharp['upper'],'verified_upper':verified['upper_outward_float'],'verified_rational':verified['upper_rational'],'seconds':time.perf_counter()-start,'complete':r['lower_search']['complete'] and r['upper_search']['complete'],'scope':'Synthetic feasible case on recent learned operator'})
        frame_save(region+'/verified_refinement_cases.csv',pd.DataFrame(records));save(region+'/extensions_status.json',{'status':'completed','ablation_rows':len(rows),'simulated_repair_rows':len(repairs),'verified_refinement_rows':len(records),'physical_repairs_measured':False,'priority':'Previously derived mathematics and implementations tested on recent maps'})
        print(region,'ablations, repair and exact nominal refinement completed',flush=True)
if __name__=='__main__':main()
