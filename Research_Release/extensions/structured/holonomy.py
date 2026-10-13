                                                                   

                                                                            
                                                                          
   
from collections import deque
import numpy as np

def structured_components(delta,n,p,edges,row_counts=None,tol=1e-8):
    D=np.asarray(delta,float)
    if D.ndim!=2 or D.shape[1]!=n*p or not np.isfinite(D).all():raise ValueError('Invalid coboundary dimensions')
    counts=[p]*len(edges) if row_counts is None else list(row_counts)
    if len(counts)!=len(edges) or sum(counts)!=len(D):raise ValueError('Edge row counts do not match operator')
    adjacency=[[] for _ in range(n)];offset=0;maps=[]
    for e,((u,v),r) in enumerate(zip(edges,counts)):
        if not (0<=u<n and 0<=v<n) or u==v:raise ValueError('Invalid edge')
        if r<p:return {'applicable':False,'reason':'edge has fewer rows than station coordinates'}
        block=D[offset:offset+r];offset+=r
        forbidden=np.setdiff1d(np.arange(n*p),np.r_[np.arange(u*p,(u+1)*p),np.arange(v*p,(v+1)*p)])
        if np.any(abs(block[:,forbidden])>1e-12):raise ValueError('Rows touch coordinates beyond edge endpoints')
        Au=block[:p,u*p:(u+1)*p];Av=-block[:p,v*p:(v+1)*p]
        for A in [Au,Av]:
            s=np.linalg.svd(A,compute_uv=False)
            if s[-1]<=tol*max(s[0],1e-300):return {'applicable':False,'reason':'selected endpoint submap numerically singular'}
        maps.append((Au,Av));adjacency[u].append((v,e,True));adjacency[v].append((u,e,False))
    threshold=tol*max(np.linalg.norm(D,2) if len(D) else 0.,1.)
    visited=set();components=[]
    for root in range(n):
        if root in visited:continue
        transport={root:np.eye(p)};visited.add(root);queue=deque([root])
        while queue:
            u=queue.popleft()
            for v,e,forward in adjacency[u]:
                if v in visited:continue
                Au,Av=maps[e]
                transport[v]=np.linalg.solve(Av if forward else Au,(Au if forward else Av)@transport[u])
                visited.add(v);queue.append(v)
        vertices=sorted(transport);T=np.zeros((n*p,p))
        for v in vertices:T[v*p:(v+1)*p]=transport[v]
                                                                             
        basis=np.linalg.qr(T,mode='reduced')[0];H=D@basis
        s=np.linalg.svd(H,compute_uv=False);rank=int(np.sum(s>threshold));nullity=p-rank
        components.append({'vertices':vertices,'root':root,'root_constraint_rank':rank,'kernel_dimension':nullity,
            'root_constraint_singular_values':s.tolist(),'root_constraint_shape':list(H.shape)})
    kernel=sum(c['kernel_dimension'] for c in components)
    vulnerable=[len(c['vertices']) for c in components if c['kernel_dimension']>0]
    return {'applicable':True,'n':n,'p':p,'components':components,'predicted_rank':n*p-kernel,'kernel_dimension':kernel,
        'block_distance':min(vulnerable) if vulnerable else float('inf'),'rank_absolute_tolerance':threshold,
        'scope':'actual invertible endpoint submaps; unregularized identifiability, not real-data accuracy'}

def outage_rank(result,stations):
    if not result['applicable']:raise ValueError('Structured theorem not applicable')
    S=set(stations)
    if any(v<0 or v>=result['n'] for v in S):raise ValueError('Invalid station')
    loss=sum(c['kernel_dimension'] for c in result['components'] if set(c['vertices']).issubset(S))
    return result['p']*len(S)-loss
