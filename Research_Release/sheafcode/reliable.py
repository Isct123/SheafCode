                                                                   

                                                                       
                                                                       
                                                                       
   
from bisect import bisect_left, insort
from itertools import combinations
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


def structural_rank(matrix):
    graph = csr_matrix(np.asarray(matrix) != 0)
    return int(np.sum(maximum_bipartite_matching(graph, perm_type='column') >= 0))


def union_margin(delta, p, max_blocks=4):
                                                                        
    n = delta.shape[1] // p
    result = {}
    for k in range(1, min(max_blocks, n)+1):
        worst = np.inf; arg = None
        for S in combinations(range(n), k):
            cols = np.concatenate([np.arange(v*p,(v+1)*p) for v in S])
            sv = np.linalg.svd(delta[:, cols], compute_uv=False)
            val = float(sv[-1]) if len(sv) == len(cols) else 0.
            if val < worst:
                worst, arg = val, list(S)
        result[str(k)] = {'sigma_min':worst, 'support':arg,
                           'sets_checked':__import__('math').comb(n,k)}
    return result


class AdaptiveQuantile:
                                                                            

                                                                        
                                                                        
                                                                      
                                                 
       
    def __init__(self, scores, alpha=.1, gamma=.005):
        self.scores = sorted(float(x) for x in scores if np.isfinite(x))
        if not self.scores:
            raise ValueError('Empty calibration scores')
        self.alpha = float(alpha); self.level = float(alpha)
        self.gamma = float(gamma); self.n=0; self.errors=0

    def quantile(self, adaptive=True):
        a = self.level if adaptive else self.alpha
        if a <= 0:
            return np.inf
        if a >= 1:
            return -np.inf
        k = int(np.ceil((len(self.scores)+1)*(1-a)))
        return np.inf if k > len(self.scores) else self.scores[max(k-1,0)]

    def update(self, score, threshold):
                                                                           
        error = int(score > threshold)
        self.level += self.gamma*(self.alpha-error)
        insort(self.scores,float(score))
        self.n += 1; self.errors += error


class CachedSpatial:
                                                                     

                                                                       
                                                                     
                                                    
       
    def __init__(self, precision, n, p, mean=None, cache_limit=512):
        self.Q=np.asarray(precision); self.n=n; self.p=p
        self.mean=np.zeros(n*p) if mean is None else np.asarray(mean).reshape(-1)
        self.cache={}; self.limit=cache_limit

    def predict(self, Z, stations):
        X=np.asarray(Z).reshape(-1,self.n*self.p).copy()
        cols=np.concatenate([np.arange(v*self.p,(v+1)*self.p) for v in stations])
        X[:,cols]=np.nan
        X -= self.mean
        groups={}
        for t in range(len(X)):
            key=(~np.isfinite(X[t])).tobytes()
            groups.setdefault(key,[]).append(t)
        out=X.copy(); scale=np.ones((len(X),len(stations),self.p))
        for key,ts in groups.items():
            if key not in self.cache:
                U=np.flatnonzero(~np.isfinite(X[ts[0]])); O=np.flatnonzero(np.isfinite(X[ts[0]]))
                B=-np.linalg.solve(self.Q[np.ix_(U,U)],self.Q[np.ix_(U,O)])
                if len(self.cache)>=self.limit:
                    self.cache.clear()
                self.cache[key]=(U,O,B)
            U,O,B=self.cache[key]
            out[np.ix_(ts,U)]=np.nan_to_num(X[np.ix_(ts,O)])@B.T
                                                                         
            positions=np.searchsorted(U,cols)
            gain=np.sqrt(np.sum(B[positions]**2,axis=1)+.01)
            scale[ts]=gain.reshape(len(stations),self.p)
        out += self.mean
        return out[:,cols].reshape(-1,len(stations),self.p),scale


def paired_bootstrap(frame, method, reference, value='mse', unit='day', seed=29, repeats=2000):
                                                                            
    import pandas as pd
    keys=[c for c in ['day','station','mode','pattern'] if c in frame]
    table=frame.pivot_table(index=keys,columns='method',values=value,aggfunc='mean')
    if method not in table or reference not in table:
        return {'n':0}
    t=table[[method,reference]].dropna().reset_index()
    if len(t)==0:
        return {'n':0}
    clusters=t.groupby(unit)[[method,reference]].agg(['sum','count'])
    sums=clusters.xs('sum',axis=1,level=1).to_numpy()
    counts=clusters.xs('count',axis=1,level=1).to_numpy()
    means=sums.sum(0)/counts.sum(0)
    rng=np.random.default_rng(seed)
    ids=rng.integers(0,len(clusters),(repeats,len(clusters)))
    boot=sums[ids].sum(1)/counts[ids].sum(1)
    delta=boot[:,1]-boot[:,0]
                                                                        
    return {'n':len(t),'clusters':len(clusters),'method_mean':float(means[0]),
            'reference_mean':float(means[1]),'difference_reference_minus_method':float(means[1]-means[0]),
            'difference_ci95':np.quantile(delta,[.025,.975]).tolist(),
            'rmse_relative_gain':float(1-np.sqrt(means[0]/means[1])) if means[1]>0 else None,
            'unit':unit, 'bootstrap_repeats':repeats}
