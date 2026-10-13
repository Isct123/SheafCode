                                                                              

                                                                       
                                                                          
   
import numpy as np
from certificate import functional_interval

def worst_case_repair_halfwidth(A,b,h,epsilon,repaired):
    A=np.asarray(A,float);b=np.asarray(b,float);h=np.asarray(h,float)
    original=functional_interval(A,b,h,epsilon)
    if original['status']=='inconsistent':return {'status':'inconsistent','halfwidth':None}
    R=np.asarray(repaired,int)
    if len(np.unique(R))!=len(R) or np.any(R<0) or np.any(R>=A.shape[1]):raise ValueError('Invalid repair coordinates')
    K=np.setdiff1d(np.arange(A.shape[1]),R)
                                                                           
                                                                       
    center=-np.linalg.pinv(A,rcond=1e-12)@b
    shifted=b+A[:,R]@center[R]
    after=functional_interval(A[:,K],shifted,h[K],epsilon)
    if after['status']=='inconsistent':raise ArithmeticError('Center repair unexpectedly infeasible')
    return {'status':after['status'],'halfwidth':after.get('halfwidth',np.inf),
            'attaining_repair_reading':center[R].tolist(),'remaining_coordinates':K.tolist(),
            'scope':'supremum over feasible exact repaired readings; fixed A,b,h and epsilon'}
