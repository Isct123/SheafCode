                                                                            

                                                                           
                                                                              
                                                 
   
import numpy as np

def conditional_covariance(precision, unknown):
    Q=np.asarray(precision,dtype=float);U=np.asarray(unknown,dtype=int)
    if Q.ndim!=2 or Q.shape[0]!=Q.shape[1] or not np.allclose(Q,Q.T):
        raise ValueError('Precision must be square and symmetric')
    if len(np.unique(U))!=len(U) or np.any(U<0) or np.any(U>=len(Q)):
        raise ValueError('Unknown coordinates must be unique valid indices')
    if not len(U):return np.empty((0,0))
    A=Q[np.ix_(U,U)];np.linalg.cholesky(A)
    C=np.linalg.solve(A,np.eye(len(U)))
    return (C+C.T)/2

def reveal_block(covariance, positions):
                                                                           

                                                                           
                                                              
       
    C=np.asarray(covariance,dtype=float);R=np.asarray(positions,dtype=int)
    if C.ndim!=2 or C.shape[0]!=C.shape[1] or not np.allclose(C,C.T):
        raise ValueError('Covariance must be square and symmetric')
    if len(np.unique(R))!=len(R) or np.any(R<0) or np.any(R>=len(C)):
        raise ValueError('Revealed positions must be unique valid indices')
    if not len(R):return C.copy()
    CR=C[:,R];A=C[np.ix_(R,R)];np.linalg.cholesky(A)
    updated=C-CR@np.linalg.solve(A,CR.T)
    updated[R,:]=0;updated[:,R]=0
    return (updated+updated.T)/2

def trace_reduction(covariance, positions, weights=None):
                                                                       
    C=np.asarray(covariance,dtype=float)
    w=np.ones(len(C)) if weights is None else np.asarray(weights,dtype=float)
    if w.shape!=(len(C),) or not np.isfinite(w).all() or np.any(w<0):
        raise ValueError('Weights must be finite and nonnegative')
    after=reveal_block(C,positions)
    gain=float(w@(np.diag(C)-np.diag(after)))
    if gain < -1e-9:raise ArithmeticError('Observation increased model variance')
    return max(gain,0.)

def choose_repair(covariance, blocks, weights=None):
                                                                               
    if not blocks:raise ValueError('No repair candidates')
    gains={key:trace_reduction(covariance,value,weights) for key,value in blocks.items()}
    selected=max(gains,key=gains.get)
    return selected,gains
