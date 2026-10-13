                                                                               
from pathlib import Path
import sys,itertools
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'structured'))
import numpy as np
from holonomy import structured_components,outage_rank

def main():
    checks=0;instances=0
    for p in [1,2,3]:
        n=4;edges=[(0,1),(2,3)]
        for removed in range(p+1):
            rows=[];counts=[]
            for u,v in edges:
                edge=np.zeros((p+(removed if u==0 else 0),n*p));edge[:p,u*p:(u+1)*p]=np.eye(p);edge[:p,v*p:(v+1)*p]=-np.eye(p)
                if u==0:
                    for j in range(removed):edge[p+j,j]=1.
                rows.append(edge);counts.append(len(edge))
            D=np.vstack(rows);result=structured_components(D,n,p,edges,counts)
            assert result['applicable']
            assert [c['kernel_dimension'] for c in result['components']]==[p-removed,p];checks+=2
            for k in range(n+1):
                for S in itertools.combinations(range(n),k):
                    cols=[j for v in S for j in range(v*p,(v+1)*p)]
                    actual=int(np.linalg.matrix_rank(D[:,cols])) if cols else 0
                    assert outage_rank(result,S)==actual;checks+=1
            instances+=1
    print(instances,'extra-row graph controls;',checks,'checks passed')
    return {'instances':instances,'checks':checks,'status':'passed','scope':'Identity endpoint maps, extra independent root constraints, disconnected components and every outage subset.'}

if __name__=='__main__':main()
