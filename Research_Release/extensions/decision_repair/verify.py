from pathlib import Path
import sys,itertools,json,random
sys.path.insert(0,str(Path(__file__).parent))
from decision_repair import conflict_edges,master,plan,infer

def main():
    checks=0; rng=random.Random(917)
                                                                                          
    assert conflict_edges([[0.],[2.]],['safe','exceed'],1.)[0][0]==0;checks+=1
    assert plan(1,conflict_edges([[0.],[2.]],['safe','exceed'],1.))['status']=='infeasible';checks+=1
    assert plan(1,conflict_edges([[0.],[2.01]],['safe','exceed'],1.))['cost']==1;checks+=1
                                                                                          
    for n in range(1,5):
        edges=conflict_edges([[0.]*n,[3.]*n],[0,1],1.)
        result=plan(n,edges,faults=1)
        assert (result['status']=='optimal')==(n>=3);checks+=1
    for trial in range(160):
        n=rng.randrange(1,8);m=rng.randrange(2,9)
        states=[[float(rng.randrange(7)) for _ in range(n)] for _ in range(m)]
        labels=[rng.randrange(2) for _ in range(m)];eta=rng.choice([0.,.5,1.])
        edges=conflict_edges(states,labels,eta);costs=[float(rng.randrange(1,5)) for _ in range(n)]
        for f in [0,1]:
            lazy=plan(n,edges,costs,f); exact=master(n,costs,[e for e,_ in edges],2*f+1)
            assert (exact is None)==(lazy['status']=='infeasible');checks+=1
            if exact is not None:
                assert lazy['cost']==exact[1];checks+=1
                lows=[r['lower'] for r in lazy['trace']];assert lows==sorted(lows);checks+=1
                interrupted=plan(n,edges,costs,f,max_refinements=0)
                assert interrupted['lower']<=exact[1]<=interrupted['upper'];checks+=1
                                                                                              
                for s,label in zip(states,labels):
                    for sign in [-1.,1.]:
                        y={v:s[v]+sign*eta for v in lazy['stations']}
                        assert infer(states,labels,eta,y,f)==label;checks+=1
                        if f:
                            for bad in lazy['stations']:
                                yy=y.copy();yy[bad]=1000.
                                assert infer(states,labels,eta,yy,f)==label;checks+=1
    report={'status':'passed','checks':checks,'random_banks':160,'scope':'Finite bank, noise boundaries, corruption union, optimality and interruption bounds; not mathematical priority or physical-model verification.'}
    out=Path(__file__).resolve().parents[2]/'decision_repair_results';out.mkdir(exist_ok=True)
    (out/'VERIFICATION.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
