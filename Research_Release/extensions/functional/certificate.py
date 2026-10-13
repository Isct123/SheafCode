                                                                             

                                                                   
                                                                      
   
import numpy as np

def functional_interval(A,b,h,epsilon,rcond=1e-12):
    A=np.asarray(A,dtype=float);b=np.asarray(b,dtype=float);h=np.asarray(h,dtype=float)
    if A.ndim!=2 or b.shape!=(A.shape[0],) or h.shape!=(A.shape[1],):raise ValueError('Incompatible dimensions')
    if not all(np.isfinite(x).all() for x in [A,b,h]) or not np.isfinite(epsilon) or epsilon<0:raise ValueError('Finite inputs and nonnegative epsilon required')
    u,s,vt=np.linalg.svd(A,full_matrices=False);rank=int(np.sum(s>rcond*(s[0] if len(s) else 1)))
    ur=u[:,:rank];vr=vt[:rank].T;sr=s[:rank]
    center=-(vr@((ur.T@b)/sr)) if rank else np.zeros(A.shape[1])
    residual=float(np.linalg.norm(A@center+b));null_h=h-vr@(vr.T@h)
    tolerance=1e-10*max(1.,np.linalg.norm(b),float(epsilon))
    info={'rank':rank,'unknown_coordinates':A.shape[1],'minimum_residual':residual,'null_functional_norm':float(np.linalg.norm(null_h)),
        'rank_relative_tolerance':rcond,'guarantee':'conditional on fixed observed inputs and true residual norm <= epsilon'}
    if residual>epsilon+tolerance:return {**info,'status':'inconsistent','lower':None,'upper':None}
    if np.linalg.norm(null_h)>1e-10*max(1.,np.linalg.norm(h)):
        return {**info,'status':'unbounded','lower':-np.inf,'upper':np.inf}
    dual=(vr.T@h)/sr if rank else np.empty(0)
    radius=float(np.sqrt(max(epsilon**2-residual**2,0.))*np.linalg.norm(dual))
    mid=float(h@center)
    return {**info,'status':'bounded','lower':mid-radius,'upper':mid+radius,'center':mid,'halfwidth':radius}
