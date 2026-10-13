                                                                                
from pathlib import Path
from fractions import Fraction as Q
import json,gzip,hashlib,itertools,pickle,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'joint_certification_results'
def r(x):return Q.from_float(float(x))
def demand(ok,message):
    if not ok:raise RuntimeError(message)

def audit(case):
    checks=0;A=np.array(case['A']);b=np.array(case['b']);H=np.array(case['H']);offsets=np.array(case['offsets']);blocks=[np.array(B) for B in case['blocks']];p=A.shape[1]
    lo=np.array(case['lower']);hi=np.array(case['upper']);eps=case['eps'];cert=case['certification'];proofs=cert['proofs']
    supports=[S for k in range(min(1,len(blocks))+1) for S in itertools.combinations(range(len(blocks)),k)]
    groups={S:[] for S in supports}
    for proof in proofs:groups[tuple(proof['support'])].append(proof)
    lowers=[];upper_on_min=[];uppers=[];lower_on_max=[]
    for S,items in groups.items():
        demand(len([q for q in items if q['kind']=='joint_minimum'])==1,'Missing/duplicate support minimum');checks+=1
        B=np.concatenate([blocks[v] for v in S],axis=1) if S else np.empty((len(A),0));M=np.concatenate([A,B],axis=1);q=B.shape[1]
        G=np.r_[M,-M];rhs=np.r_[eps-b,eps+b];ll=np.r_[lo,np.full(q,-1.)];hh=np.r_[hi,np.ones(q)]
        minimum=next(q for q in items if q['kind']=='joint_minimum')
        if minimum['infeasible']:demand(len(items)==1,'Infeasible support still has maxima');checks+=1
        else:demand(sorted(q['channel'] for q in items if q['kind']=='pollutant_maximum')==list(range(len(H))),'Missing/duplicate pollutant maxima');checks+=1
        for proof in items:
            if proof['kind']=='joint_minimum':
                baseG=np.r_[np.c_[G,np.zeros(len(G))],np.c_[H,np.zeros((len(H),q)),-np.ones(len(H))]];base_rhs=np.r_[rhs,-offsets]
                actual_lo=max(sum((min(r(h)*r(l),r(h)*r(u)) for h,l,u in zip(row,lo,hi)),r(c)) for row,c in zip(H,offsets))
                actual_hi=max(sum((max(r(h)*r(l),r(h)*r(u)) for h,l,u in zip(row,lo,hi)),r(c)) for row,c in zip(H,offsets))
                ylow=proof['lo'][-2] if proof.get('phase') else proof['lo'][-1];yhigh=proof['hi'][-2] if proof.get('phase') else proof['hi'][-1]
                demand(r(ylow)<=actual_lo and r(yhigh)>=actual_hi,'Epigraph box excludes possible scores');checks+=1
                baselo=np.r_[ll,ylow];basehi=np.r_[hh,yhigh];objective=np.r_[np.zeros(len(ll)),-1.]
            else:
                baseG=G;base_rhs=rhs;baselo=ll;basehi=hh;objective=np.r_[H[proof['channel']],np.zeros(q)]
            if proof.get('phase'):
                baseG=np.c_[baseG,-np.ones(len(baseG))];baselo=np.r_[baselo,0.];basehi=np.r_[basehi,proof['hi'][-1]];objective=np.r_[np.zeros(len(objective)),-1.]
                demand(basehi[-1]>=0,'Invalid infeasibility slack box');checks+=1
            demand(np.array_equal(baseG,proof['G']) and np.array_equal(base_rhs,proof['rhs']) and np.array_equal(baselo,proof['lo']) and np.array_equal(basehi,proof['hi']) and np.array_equal(objective,proof['c']),'Joint node model mismatch '+str((S,proof['kind'],[np.array_equal(a,proof[k]) for a,k in [(baseG,'G'),(base_rhs,'rhs'),(baselo,'lo'),(basehi,'hi'),(objective,'c')]])));checks+=5
            weights=list(map(Q,proof['weights']));demand(all(w>=0 for w in weights),'Negative certificate weight');checks+=1
            residual=[r(c) for c in objective]
            for w,row in zip(weights,baseG):
                for j,value in enumerate(row):residual[j]-=w*r(value)
            base=sum((w*r(b) for w,b in zip(weights,base_rhs)),Q(0));corr=sum((max(d*r(l),d*r(u)) for d,l,u in zip(residual,baselo,basehi)),Q(0));upper=base+corr
            demand(upper==Q(proof['upper']) and base==Q(proof['base']) and corr==Q(proof['correction']),'Joint rational bound mismatch');checks+=3
            if proof['infeasible']:
                demand(proof.get('phase') and upper<0,'Unverified joint infeasibility');checks+=1;continue
            point=proof.get('point')
            if point is not None:
                x=list(map(Q,point))
                demand(all(r(l)<=z<=r(u) for l,u,z in zip(baselo,basehi,x)),'Joint incumbent violates bounds');checks+=len(x)
                for row,limit in zip(baseG,base_rhs):demand(sum((r(a)*z for a,z in zip(row,x)),Q(0))<=r(limit),'Joint incumbent violates constraints');checks+=len(baseG)
            if proof['kind']=='joint_minimum':
                lowers.append(-upper)
                if point is not None:upper_on_min.append(max(sum((r(a)*z for a,z in zip(row,x[:p])),r(c)) for row,c in zip(H,offsets)))
            else:
                channel=proof['channel'];uppers.append(upper+r(offsets[channel]))
                if point is not None:lower_on_max.append(sum((r(a)*z for a,z in zip(H[channel],x[:p])),r(offsets[channel])))
    if not lowers:demand(cert['status']=='certified_infeasible' and cert['decision']=='abstain','Incorrect joint infeasibility classification');checks+=1
    else:
        L=min(lowers);U=max(uppers);label='exceedance' if L>1 else ('below_or_equal' if U<=1 else 'abstain')
        demand(L==Q(cert['lower']) and U==Q(cert['upper']) and label==cert['decision'],'Joint endpoint/decision mismatch');checks+=3
        demand((None if not upper_on_min else min(upper_on_min))==(None if cert['upper_on_minimum'] is None else Q(cert['upper_on_minimum'])),'Minimum incumbent mismatch');checks+=1
        demand((None if not lower_on_max else max(lower_on_max))==(None if cert['lower_on_maximum'] is None else Q(cert['lower_on_maximum'])),'Maximum incumbent mismatch');checks+=1
        if case.get('metadata',{}).get('exact_truth_assumptions'):
            truth=max(sum((r(a)*r(z) for a,z in zip(row,case['truth_state'])),r(c)) for row,c in zip(H,offsets))
            demand(L<=truth<=U,'Joint truth containment failed');checks+=1
    return checks,len(proofs)

def main():
    lock=json.loads((OUT/'PROTOCOL_LOCK.json').read_text());checks=cases=proofs=0
    for name,h in lock['inputs'].items():demand(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,'Input hash mismatch: '+name);checks+=1
    for name,h in lock['code_sha256'].items():demand(hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h,'Executed source changed: '+name);checks+=1
    sys.path.insert(0,str(ROOT/'extensions/online_validation'))
    prior=json.loads((ROOT/'math_integration_results/PROTOCOL_LOCK.json').read_text())
    references={'PM2.5':60.,'PM10':100.,'O3':100.,'O3_ppm':.05,'NO2_ppb':40.}
    for path in sorted(OUT.glob('*_proofs.json.gz')):
        with gzip.open(path,'rt') as f:payload=json.load(f)
        name=payload[0]['metadata']['cohort'];folder=ROOT/'recent_data_results'/('delhi' if name=='original_india' else 'epa_la') if name in ['original_india','original_usa'] else ROOT/'expanded_validation_results'/name
        model_path=folder/'models.pkl';demand(hashlib.sha256(model_path.read_bytes()).hexdigest()==prior['inputs'][str(model_path.relative_to(ROOT))],'Original fitted model hash changed');checks+=1
        with model_path.open('rb') as f:model=pickle.load(f)
        ds=np.load(folder/'dataset.npz');D=model['sheaf'].D;n,p=ds['Z'].shape[1:];stations=list(map(str,ds['stations']));channels=list(map(str,ds['channels']));times=list(map(str,ds['times']));den=np.array([references[c] for c in channels])
        for position,case in enumerate(payload):
            v=stations.index(case['metadata']['target']);index=times.index(case['metadata']['timestamp']);U=np.arange(v*p,(v+1)*p);O=np.setdiff1d(np.arange(n*p),U);survivors=[j for j in range(n) if j!=v];repeat=position//2
            expected_H=np.diag(ds['sd'][v]/den);expected_offsets=ds['mu'][v]/den
            demand(np.array_equal(case['H'],expected_H) and np.array_equal(case['offsets'],expected_offsets),'Scores do not match archive units');checks+=2
            demand(np.array_equal(case['A'],D[:,U]) and all(np.array_equal(B,D[:,j*p:(j+1)*p]) for B,j in zip(case['blocks'],survivors)),'Stored inequalities do not match fitted maps');checks+=1
            observed=ds['Z'][index].ravel().copy();shift=0. if case['metadata']['scenario']=='clean' else (1. if repeat%2==0 else -1.);fault=survivors[repeat%len(survivors)];observed[fault*p:(fault+1)*p]+=shift
            demand(np.allclose(case['b'],D[:,O]@observed[O],rtol=0,atol=1e-12),'Observed/fault reconstruction mismatch');checks+=1
            demand(np.array_equal(case['truth_state'],ds['Z'][index,v]),'Truth changed from archive');checks+=1
            extra,k=audit(case);checks+=extra;proofs+=k;cases+=1
    for case in json.loads((OUT/'CONTROLS.json').read_text()):
        extra,k=audit(case);checks+=extra;proofs+=k;demand(case['certification']['decision']=='exceedance','Planted joint-only certificate failed');checks+=1
    report={'status':'passed','archive_cases':cases,'controls':2,'proof_records':proofs,'checks':checks,'scope':'Exact stored-coefficient joint inequalities, feasible witnesses, every fault support and decision classification; no solver imported.','verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'INDEPENDENT_REPLAY.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
