                                                                         
import random,sys,json,hashlib
from fractions import Fraction as Q
from itertools import combinations
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from radius_repair import radius,frontier,ambiguity_witness

def reference_feasible(states,labels,eta,mask,rho,f):
    stations=[v for v in range(len(states[0])) if mask>>v&1]
    faultsets=[set(s) for size in range(min(f,len(stations))+1) for s in combinations(stations,size)]
    for i,j in combinations(range(len(states)),2):
        if labels[i]==labels[j]:continue
        for corrupt_a in faultsets:
            for corrupt_b in faultsets:
                                                                       
                                                                               
                common=True
                for v in stations:
                    if v in corrupt_a or v in corrupt_b:continue
                    la=Q(states[i][v])-rho-eta;ua=Q(states[i][v])+rho+eta
                    lb=Q(states[j][v])-rho-eta;ub=Q(states[j][v])+rho+eta
                    if max(la,lb)>min(ua,ub):common=False;break
                if common:return False
    return True

def main():
    rng=random.Random(81026);checks=0;witnesses=0
    for trial in range(100):
        states=[[rng.randrange(13) for _ in range(4)] for _ in range(4)];labels=[0,1,0,1]
        eta=Q(rng.randrange(3),2);f=trial%2
        radii={0:Q(0),1:Q(1,2),2:Q(1)}
        for mask in range(16):
            r=radius(states,labels,eta,mask,f)
            tested=set(radii.values())
            if r['radius']>=0 and r['radius']!=float('inf'):
                tested.add(r['radius'])
                if r['radius']>0:tested.add(r['radius']/2)
                ambiguity_witness(states,r['witness'],eta,mask,r['radius'],f);witnesses+=1
            for rho in tested:
                assert reference_feasible(states,labels,eta,mask,rho,f)==(rho<r['radius']);checks+=1
        fr=frontier(states,labels,eta,f,costs=[1,3,2,1])
        for entry in fr:
            ref=max(radius(states,labels,eta,mask,f)['radius'] for mask in range(16)
                    if sum(c for v,c in enumerate([1,3,2,1]) if mask>>v&1)<=entry['budget'])
            assert ref==entry['radius'];checks+=1
                                                                               
    states=[[0,0],[4,10]];labels=[0,1]
    fr=frontier(states,labels,1)
    assert radius(states,labels,1,1)['radius']==1;checks+=1
    assert fr[1]['mask']==2 and fr[1]['radius']==4;checks+=1
    assert not reference_feasible(states,labels,Q(1),2,Q(4),0);checks+=1
    report={'status':'passed','random_banks':100,'checks':checks,'boundary_witnesses':witnesses,
            'scope':'Independent enumeration of corrupt-station assignments and closed-interval intersections; exact arithmetic.',
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('radius_repair.py')]}}
    out=Path(__file__).resolve().parents[3]/'Research_Release/decision_repair_results/RADIUS_VERIFICATION.json'
    out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
