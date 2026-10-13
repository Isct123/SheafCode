                                                                            
from pathlib import Path
import os,sys,csv,json,pickle,itertools,time,hashlib,gzip
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from scipy.optimize import linprog
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'extensions/online_validation'))
sys.path.insert(0,str(ROOT/'extensions/followup'))
sys.path.insert(0,str(ROOT/'extensions/joint_decisions'))
from constrained import constrained_envelope
from joint import joint_envelope
sys.path.insert(0,str(HERE))
from search import search,rat,feasible
from fractions import Fraction as Q
OUT=ROOT/'math_integration_results'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,Q):return str(x)
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x

def anchors(D,n,p,edges,tol):
                                                                                    
    if D.shape!=(len(edges)*p,n*p):return {'applicable':False,'reason':'row layout does not meet declared p-row edge assumption'},[]
    adj=[[] for _ in range(n)];worst=1.
    for j,(u,v) in enumerate(edges):
        row=D[j*p:(j+1)*p];M=row[:,u*p:(u+1)*p];N=-row[:,v*p:(v+1)*p]
        outside=[k for k in range(n*p) if k//p not in [u,v]]
        if np.any(row[:,outside]!=0):return {'applicable':False,'reason':'edge contains extra endpoint coordinates'},[]
        for block in [M,N]:
            s=np.linalg.svd(block,compute_uv=False);ratio=s[-1]/max(s[0],1e-300);worst=min(worst,ratio)
            if ratio<=tol:return {'applicable':False,'reason':'numerically singular selected endpoint','minimum_relative_singular_value':worst},[]
        adj[u].append((v,M,N));adj[v].append((u,N,M))
    done=set();comps=[];threshold=tol*max(float(np.linalg.norm(D,2)) if D.size else 0.,1.)
    for root in range(n):
        if root in done:continue
        transports={root:np.eye(p)};todo=[root];done.add(root)
        for u in todo:
            for v,M,N in adj[u]:
                if v in done:continue
                transports[v]=np.linalg.solve(N,M@transports[u]);done.add(v);todo.append(v)
        T=np.zeros((n*p,p))
        for v,value in transports.items():T[v*p:(v+1)*p]=value
        basis=np.linalg.qr(T)[0];sv=np.linalg.svd(D@basis,compute_uv=False)
        dim=p-int(np.sum(sv>threshold));comps.append((set(todo),dim))
    rows=[]
    for bits in range(1<<n):
        S=[v for v in range(n) if bits>>v&1];pred=sum(dim for C,dim in comps if C.issubset(S))
        cols=[j for v in S for j in range(v*p,(v+1)*p)]
        observed=p*len(S)-int(np.sum(np.linalg.svd(D[:,cols],compute_uv=False)>threshold)) if cols else 0
        rows.append({'mask':bits,'predicted_nullity':pred,'direct_nullity':observed,'agree':pred==observed})
    return {'applicable':True,'minimum_relative_singular_value':worst,'absolute_rank_tolerance':threshold,'components':[{'stations':sorted(C),'root_dimension':q} for C,q in comps],'checks':len(rows),'disagreements':sum(not r['agree'] for r in rows)},rows

def joint_bisect(A,b,H,offsets,blocks,eps,lo,hi):
    lower=float(max(np.sum(np.minimum(H*lo,H*hi),axis=1)+offsets));upper=float(max(np.sum(np.maximum(H*lo,H*hi),axis=1)+offsets))
    supports=[S for k in [0,1] for S in itertools.combinations(range(len(blocks)),k)]
    def exists(value):
        for S in supports:
            B=np.c_[*[blocks[j] for j in S]] if S else np.empty((len(A),0));M=np.c_[A,B]
            G=np.r_[M,-M,np.c_[H,np.zeros((len(H),B.shape[1]))]];rhs=np.r_[eps-b,eps+b,value-offsets]
            fit=linprog(np.zeros(M.shape[1]),A_ub=G,b_ub=rhs,bounds=list(zip(lo,hi))+[(-1,1)]*B.shape[1],method='highs')
            if fit.success:
                if np.max(G@fit.x-rhs)>1e-7:raise RuntimeError('Binary feasibility primal failure')
                return True
            if fit.status!=2:raise RuntimeError('Binary feasibility uncertain solver status')
        return False
    if not exists(upper):return None
    for _ in range(32):
        middle=(lower+upper)/2
        if exists(middle):upper=middle
        else:lower=middle
    return lower,upper

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'SUMMARY.json').exists():raise RuntimeError('Completed evidence exists; preserve it instead of overwriting')
    rows=list(csv.DictReader((ROOT/'expanded_validation_results/COHORT_RESULTS.csv').open()))
    cohorts=[(r['cohort'],ROOT/'expanded_validation_results'/r['cohort']) for r in rows]
    cohorts += [('original_india',ROOT/'recent_data_results/delhi'),('original_usa',ROOT/'recent_data_results/epa_la')]
    inputs={str(p.relative_to(ROOT)):sha(p) for _,folder in cohorts for p in [folder/'dataset.npz',folder/'models.pkl']}
    lock={'date':'2026-10-08','scope':'exploratory follow-up after prior archive access','inputs':inputs,'protocol_sha256':sha(HERE/'PROTOCOL.md'),'code_sha256':{p.name:sha(p) for p in [Path(__file__),HERE/'search.py']},'hours_per_role':8,'absolute_gap':1e-7,'nodes_limit':255}
    (OUT/'PROTOCOL_LOCK.json').write_text(json.dumps(lock,indent=2))
    all_anchor=[];cases=[];binary=[];statuses=[];start=time.perf_counter()
                                                                                    
    controls=[]
    for name,n,p,edges,singular in [('tree',4,2,[(0,1),(1,2),(2,3)],False),('disconnected',4,2,[(0,1),(2,3)],False),('isolated',4,2,[(0,1)],False),('singular',4,2,[(0,1),(1,2)],True)]:
        D=np.zeros((len(edges)*p,n*p))
        for j,(u,v) in enumerate(edges):D[j*p:(j+1)*p,u*p:(u+1)*p]=np.eye(p);D[j*p:(j+1)*p,v*p:(v+1)*p]=-np.eye(p)
        if singular:D[0]=0
        result,checks=anchors(D,n,p,edges,1e-8)
        assert result['applicable']==(not singular)
        if not singular:assert result['disagreements']==0
        controls.append({'name':name,'result':result,'cases':checks})
    (OUT/'ANCHOR_CONTROLS.json').write_text(json.dumps(plain(controls),indent=2))
    refs={'PM2.5':60.,'PM10':100.,'O3':100.,'O3_ppm':.05,'NO2_ppb':40.}
    for name,folder in cohorts:
        ds={k:v for k,v in np.load(folder/'dataset.npz',allow_pickle=False).items()}
        with (folder/'models.pkl').open('rb') as f:model=pickle.load(f)
        D=model['sheaf'].D;Z=ds['Z'];n,p=Z.shape[1:];edges=model['edges'];audit={}
        for tol in [1e-6,1e-8,1e-10]:
            result,checks=anchors(D,n,p,edges,tol);audit[str(tol)]=result
            all_anchor += [{'cohort':name,'tolerance':tol,**r} for r in checks]
        full=np.isfinite(Z).all((1,2));cal=Z[(ds['split']==3)&full];test=np.flatnonzero((ds['split']==2)&full)
        k=int(np.ceil(.9*(len(cal)+1)))
        if k>len(cal) or not len(test):
            statuses.append({'cohort':name,'status':'unavailable_complete_calibration_or_test','calibration_hours':len(cal),'test_hours':len(test),'anchor':audit});continue
        eps=float(np.sort(np.max(abs(cal.reshape(len(cal),-1)@D.T),axis=1))[k-1]);sample=test[np.unique(np.linspace(0,len(test)-1,min(8,len(test))).astype(int))]
        payload=[];count0=len(cases)
        for repeat,index in enumerate(sample):
            v=repeat%n;U=np.arange(v*p,(v+1)*p);O=np.setdiff1d(np.arange(n*p),U);survivors=[j for j in range(n) if j!=v]
            A=D[:,U];blocks=[D[:,j*p:(j+1)*p] for j in survivors];lo=np.full(p,-6.);hi=np.full(p,6.)
            channels=list(map(str,ds['channels']));channel=channels.index('PM2.5') if 'PM2.5' in channels else 0;h=np.eye(p)[channel]*ds['sd'][v,channel]/refs[channels[channel]];offset=ds['mu'][v,channel]/refs[channels[channel]]
            for scenario in ['clean','unit_shift']:
                state=Z[index].ravel().copy();fault=survivors[repeat%len(survivors)];shift=0. if scenario=='clean' else (1. if repeat%2==0 else -1.)
                state[fault*p:(fault+1)*p]+=shift;b=D[:,O]@state[O];bounds=constrained_envelope(A,b,h,blocks,1,eps,1.,lo,hi)
                scores={};search_payload={};tic=time.perf_counter()
                for direction,c in [('upper',h),('negative_lower',-h)]:
                    result=search(A,b,c,blocks,1,eps,1.,lo,hi);scores[direction]={k:v for k,v in result.items() if k!='records'};search_payload[direction]=result
                    if bounds['status']=='bounded':
                        optimum=bounds['upper'] if direction=='upper' else -bounds['lower']
                        assert result['upper'] is not None,'Exact infeasible contradicts numerical support LP'
                        assert float(Q(result['upper']))>=optimum-1e-6
                        if result['lower'] is not None:assert float(Q(result['lower']))<=optimum+1e-6
                    elif result['status']=='verified_gap':raise AssertionError('Exact feasible witness contradicts numerical infeasible support union')
                valid=bool(np.max(abs(Z[index,v]))<=6 and np.max(abs(D@Z[index].ravel()))<=eps)
                                                                                                   
                fixed={j:int(survivors[j]==fault and shift!=0) for j in range(len(survivors))}
                from search import node_model
                G,rhs,ll,hh=node_model(A,b,blocks,eps,1.,lo,hi,1,fixed)
                point=np.r_[Z[index,v],*[np.full(p,shift if j==fault else 0.) for j in survivors],list(fixed.values())]
                                                                                  
                point[p:p+p*len(survivors)]*=-1
                valid_exact=feasible(G,rhs,ll,hh,[rat(x) for x in point])
                lower=None if scores['negative_lower']['upper'] is None else -float(Q(scores['negative_lower']['upper']))+offset
                upper=None if scores['upper']['upper'] is None else float(Q(scores['upper']['upper']))+offset
                if valid_exact:
                    truth=float(h@Z[index,v]+offset);assert lower-1e-10<=truth<=upper+1e-10
                numerical_decision='inconsistent' if bounds['status']!='bounded' else ('above' if bounds['lower']+offset>1+1e-6 else ('below' if bounds['upper']+offset<=1-1e-6 else 'abstain'))
                certified_decision='inconsistent' if lower is None or upper is None else ('above' if lower>1 else ('below' if upper<=1 else 'abstain'))
                row={'cohort':name,'timestamp':str(ds['times'][index]),'target':str(ds['stations'][v]),'scenario':scenario,'channel':channels[channel],'calibration_hours':len(cal),'epsilon':eps,'exhaustive_status':bounds['status'],'numerical_lower':bounds.get('lower',0)+offset if bounds['status']=='bounded' else None,'numerical_upper':bounds.get('upper',0)+offset if bounds['status']=='bounded' else None,'verified_lower':lower,'verified_upper':upper,'numerical_decision':numerical_decision,'verified_decision':certified_decision,'exact_truth_assumptions':valid_exact,'upper_status':scores['upper']['status'],'lower_status':scores['negative_lower']['status'],'nodes':sum(s['nodes'] for s in scores.values()),'lp_solves':sum(s['lp_solves'] for s in scores.values()),'seconds':time.perf_counter()-tic}
                cases.append(row);payload.append({'case':row,'A':A,'b':b,'h':h,'blocks':blocks,'eps':eps,'lower':lo,'upper':hi,'search':search_payload})
            if repeat<4:
                b=D[:,O]@Z[index].ravel()[O];denom=np.array([refs[c] for c in channels]);H=np.diag(ds['sd'][v]/denom);offsets=ds['mu'][v]/denom
                epigraph=joint_envelope(A,b,H,offsets,blocks,1,eps,1.,lo,hi);bracket=joint_bisect(A,b,H,offsets,blocks,eps,lo,hi)
                assert (bracket is None)==(epigraph['status']=='inconsistent')
                if bracket is not None:assert bracket[0]-2e-6<=epigraph['lower']<=bracket[1]+2e-6
                binary.append({'cohort':name,'timestamp':str(ds['times'][index]),'epigraph_lower':epigraph.get('lower'),'binary_lower':None if bracket is None else bracket[0],'binary_upper':None if bracket is None else bracket[1],'scope':'Numerical cross-check, not rigorous infeasibility certification'})
        with gzip.open(OUT/(name+'_replay.json.gz'),'wt',encoding='utf-8') as f:json.dump(plain(payload),f)
        statuses.append({'cohort':name,'status':'completed','calibration_hours':len(cal),'complete_test_hours':len(test),'cases':len(cases)-count0,'anchor':audit})
        print(name,len(cases)-count0,'cases',round(time.perf_counter()-start,1),'seconds',flush=True)
        (OUT/'PROGRESS.json').write_text(json.dumps({'completed_roles':len(statuses),'cases':len(cases),'seconds':time.perf_counter()-start}))
    import pandas as pd
    pd.DataFrame(cases).to_csv(OUT/'CASES.csv',index=False);pd.DataFrame(all_anchor).to_csv(OUT/'ANCHOR_CASES.csv',index=False);pd.DataFrame(binary).to_csv(OUT/'BINARY_CROSSCHECK.csv',index=False)
    (OUT/'ROLE_STATUS.json').write_text(json.dumps(plain(statuses),indent=2))
    summary={'status':'completed','attempted_roles':len(cohorts),'completed_roles':sum(r['status']=='completed' for r in statuses),'real_archive_cases':len(cases),'directional_searches':2*len(cases),'anchor_subset_checks':len(all_anchor),'anchor_rank_disagreements':sum(not r['agree'] for r in all_anchor),'binary_crosschecks':len(binary),'seconds':time.perf_counter()-start,'scope':'Exploratory archive experiments, artificial hidden targets/injected survivor faults, exact nominal bounds; no physical model guarantee.'}
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
