                                                                            
from pathlib import Path
from fractions import Fraction as Q
import json,gzip,hashlib,itertools,csv,pickle,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'math_integration_results'
def r(x):return Q.from_float(float(x))
def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def build(case,fixed):
    A=np.array(case['A']);b=np.array(case['b']);blocks=[np.array(B) for B in case['blocks']];m,p=A.shape;n=len(blocks);q=sum(B.shape[1] for B in blocks)
    width=p+q+n;matrix=np.concatenate([A]+blocks,axis=1);G=np.zeros((2*m+2*q+1,width));G[:m,:p+q]=matrix;G[m:2*m,:p+q]=-matrix
    rhs=np.r_[case['eps']-b,case['eps']+b,np.zeros(2*q),1.];cursor=p
    row=2*m
    for v,B in enumerate(blocks):
        for j in range(B.shape[1]):
            G[row,cursor]=-1;G[row,p+q+v]=-1;row+=1
            G[row,cursor]=1;G[row,p+q+v]=-1;row+=1;cursor+=1
    G[-1,p+q:]=1;lo=np.r_[case['lower'],np.full(q,-1.),np.zeros(n)];hi=np.r_[case['upper'],np.ones(q),np.ones(n)]
    for v,a in fixed.items():
        lo[p+q+v]=hi[p+q+v]=a
        if a==0:
            j=p+sum(B.shape[1] for B in blocks[:v]);lo[j:j+blocks[v].shape[1]]=0.;hi[j:j+blocks[v].shape[1]]=0.
    return G,rhs,lo,hi

def main():
    lock=json.loads((OUT/'PROTOCOL_LOCK.json').read_text());checks=0
    for name,h in lock['inputs'].items():require(sha(ROOT/name)==h,'Input changed: '+name);checks+=1
    for name,h in lock['code_sha256'].items():require(sha(Path(__file__).with_name(name))==h,'Executed code changed: '+name);checks+=1
    require(sha(Path(__file__).with_name('PROTOCOL.md'))==lock['protocol_sha256'],'Protocol changed');checks+=1
    sys.path.insert(0,str(ROOT/'extensions/online_validation'))
    graph_checks=0
    for role in json.loads((OUT/'ROLE_STATUS.json').read_text()):
        name=role['cohort'];folder=ROOT/'recent_data_results'/('delhi' if name=='original_india' else 'epa_la') if name in ['original_india','original_usa'] else ROOT/'expanded_validation_results'/name
        with (folder/'models.pkl').open('rb') as f:model=pickle.load(f)
        ds=np.load(folder/'dataset.npz');n,p=ds['Z'].shape[1:];D=model['sheaf'].D
        for key,info in role['anchor'].items():
            require(info['applicable'],'Expected all fitted operators applicable');checks+=1
            tol=float(key);threshold=tol*max(np.linalg.norm(D,2),1.)
            for j,(u,v) in enumerate(model['edges']):
                for endpoint in [u,v]:
                    s=np.linalg.svd(D[j*p:(j+1)*p,endpoint*p:(endpoint+1)*p],compute_uv=False)
                    require(s[-1]>tol*s[0],'Endpoint invertibility assumption failed');checks+=1
                                                                                    
            for comp in info['components']:
                cols=[k for v in comp['stations'] for k in range(v*p,(v+1)*p)]
                dim=len(cols)-sum(np.linalg.svd(D[:,cols],compute_uv=False)>threshold)
                require(dim==comp['root_dimension'],'Root transport dimension mismatch');checks+=1
            for mask in range(1<<n):
                S=[v for v in range(n) if mask>>v&1];cols=[k for v in S for k in range(v*p,(v+1)*p)]
                actual=len(cols)-sum(np.linalg.svd(D[:,cols],compute_uv=False)>threshold) if cols else 0
                predicted=sum(c['root_dimension'] for c in info['components'] if set(c['stations']).issubset(S))
                require(actual==predicted,'Anchor prediction differs from direct subset rank');checks+=1;graph_checks+=1
    points=nodes=searches=phase_proofs=mixed=0;decision_checks=[];corrections=[];root_gaps=[]
    refs={'PM2.5':60.,'PM10':100.,'O3':100.,'O3_ppm':.05,'NO2_ppb':40.}
    for path in sorted(OUT.glob('*_replay.json.gz')):
        with gzip.open(path,'rt',encoding='utf-8') as f:cases=json.load(f)
        cohort=cases[0]['case']['cohort']
        folder=ROOT/'recent_data_results'/('delhi' if cohort=='original_india' else 'epa_la') if cohort in ['original_india','original_usa'] else ROOT/'expanded_validation_results'/cohort
        ds=np.load(folder/'dataset.npz');stations=list(map(str,ds['stations']));channels=list(map(str,ds['channels']))
        for case in cases:
            endpoints={}
            for direction,result in case['search'].items():
                searches+=1;c0=np.array(case['h'])*(1 if direction=='upper' else -1)
                initial=sum((max(r(h)*r(a),r(h)*r(b)) for h,a,b in zip(c0,case['lower'],case['upper'])),Q(0))
                active=[({},initial)];settled=[];inc=None;previous_upper=initial;previous_lower=None
                for record,history in zip(result['records'],result['history']):
                    nodes+=1;active.sort(key=lambda x:x[1]);fixed,parent=active.pop();saved_fixed={int(k):v for k,v in record['fixed'].items()}
                    require(fixed==saved_fixed and parent==Q(record['parent_bound']),'Frontier popped incorrect node');checks+=1
                    G,rhs,lo,hi=build(case,fixed);c=np.r_[c0,np.zeros(G.shape[1]-len(c0))]
                    if record.get('phase'):
                        G=np.c_[G,-np.ones(len(G))];lo=np.r_[lo,0.];hi=np.r_[hi,record['hi'][-1]];c=np.r_[np.zeros(len(c)),-1.]
                    require(np.array_equal(G,record['G']) and np.array_equal(rhs,record['rhs']) and np.array_equal(lo,record['lo']) and np.array_equal(hi,record['hi']) and np.array_equal(c,record['c']),'Saved node model mismatch');checks+=5
                    w=list(map(Q,record['weights']));require(all(a>=0 for a in w),'Negative inequality weight');checks+=1
                    coefficients=[r(a) for a in c]
                    for weight,row in zip(w,G):
                        for j,value in enumerate(row):coefficients[j]-=weight*r(value)
                    base=sum((a*r(b) for a,b in zip(w,rhs)),Q(0));correction=sum((max(a*r(l),a*r(u)) for a,l,u in zip(coefficients,lo,hi)),Q(0))
                    require(base==Q(record['base']) and correction==Q(record['correction']) and base+correction==Q(record['upper']),'Bound correction mismatch');checks+=3
                    if record['infeasible']:
                        require(record.get('phase') and base+correction<0,'Uncertified infeasibility prune');checks+=1;phase_proofs+=1
                    else:
                        bound=min(parent,base+correction);corrections.append(float(correction))
                        point=record.get('point')
                        if point is not None:
                            require(len(fixed)==len(case['blocks']),'Fractional relaxation used as genuine incumbent');checks+=1
                            x=list(map(Q,point));require(all(r(a)<=z<=r(b) for a,b,z in zip(lo,hi,x)),'Point violates bounds');checks+=len(x)
                            for row,b in zip(G,rhs):require(sum((r(a)*z for a,z in zip(row,x)),Q(0))<=r(b),'Point violates inequality');checks+=len(G)
                            value=sum((r(a)*z for a,z in zip(c,x)),Q(0));require(value<=bound,'Incumbent exceeds bound');checks+=1
                            inc=value if inc is None else max(inc,value);points+=1;mixed+=record.get('point_kind')=='interior_mix'
                        if inc is None or bound>inc:
                            if len(fixed)==len(case['blocks']):settled.append(bound)
                            else:
                                v=next(i for i in range(len(case['blocks'])) if i not in fixed)
                                for a in [0,1]:
                                    if a+sum(fixed.values())<=1:active.append(({**fixed,v:a},bound))
                    if inc is not None:active=[x for x in active if x[1]>inc];settled=[u for u in settled if u>inc]
                    remaining=[x[1] for x in active]+settled+([] if inc is None else [inc]);upper=max(remaining) if remaining else None
                    require((None if inc is None else str(inc))==history['lower'] and (None if upper is None else str(upper))==history['upper'],'Interruption bounds mismatch');checks+=2
                    require(history['active']==len(active) and history['unclosed_leaves']==len(settled),'Frontier count mismatch');checks+=2
                    require(upper is None or upper<=previous_upper,'Upper bound increased');checks+=1
                    require(previous_lower is None or inc>=previous_lower,'Lower bound decreased');checks+=1
                    previous_upper=upper;previous_lower=inc
                require(result['lower']==(None if inc is None else str(inc)) and result['upper']==(None if upper is None else str(upper)),'Final bounds mismatch');checks+=2
                require(len(result['records'])==len(result['history'])==result['nodes'],'Trace length mismatch');checks+=1
                if result['status']=='verified_gap':require(inc is not None and upper-inc<=r(1e-7),'False gap completion');checks+=1
                if result['status']=='certified_infeasible':require(upper is None,'False infeasible completion');checks+=1
                endpoints[direction]=upper
                if inc is not None:root_gaps.append({'initial_box_width':float(initial-inc),'final_gap':float(upper-inc)})
            row=case['case'];v=stations.index(row['target']);j=channels.index(row['channel']);offset=r(ds['mu'][v,j]/refs[channels[j]])
            L=endpoints['negative_lower'];U=endpoints['upper']
            label='inconsistent' if L is None or U is None else ('above' if -L+offset>1 else ('below' if U+offset<=1 else 'abstain'))
            require(label==row['verified_decision'],'Rounded decision disagrees with exact threshold check');checks+=1
            decision_checks.append({'cohort':cohort,'timestamp':row['timestamp'],'target':row['target'],'scenario':row['scenario'],'exact_decision':label,'exact_lower':None if L is None else str(-L+offset),'exact_upper':None if U is None else str(U+offset)})
    with (OUT/'EXACT_DECISION_REPLAY.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(decision_checks[0]));writer.writeheader();writer.writerows(decision_checks)
    report={'status':'passed','checks':checks,'independent_numerical_graph_checks':graph_checks,'directional_searches':searches,'node_certificates':nodes,'exact_infeasibility_certificates':phase_proofs,'verified_primal_witnesses':points,'interior_repaired_witnesses':mixed,'exact_threshold_rechecks':len(decision_checks),'max_positive_residual_correction':max(corrections,default=0.),'max_final_gap':max((r['final_gap'] for r in root_gaps),default=None),'scope':'Independent numerical graph-rank and exact-rational certificate, primal and full-frontier replay. Numerical LP reference optimum is not treated as an exact certificate.','verifier_sha256':sha(Path(__file__))}
    (OUT/'INDEPENDENT_REPLAY.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
