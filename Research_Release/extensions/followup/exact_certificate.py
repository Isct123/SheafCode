                                                                           

                                                                            
                                                                         
   
from fractions import Fraction
import math
import sys
import numpy as np

def rational(x):
    return Fraction.from_float(float(x))

def verified_upper(A,b,h,blocks,t,epsilon,beta,w,lower,upper):
    A=np.asarray(A,float);b=np.asarray(b,float);h=np.asarray(h,float);w=np.asarray(w,float)
    if A.ndim!=2 or b.shape!=(len(A),) or w.shape!=b.shape or h.shape!=(A.shape[1],):raise ValueError('Invalid dimensions')
    lo=np.asarray(lower,float);hi=np.asarray(upper,float);eps=np.broadcast_to(np.asarray(epsilon,float),b.shape)
    arrays=[A,b,h,w,lo,hi,eps]+[np.asarray(B,float) for B in blocks]
    if lo.shape!=h.shape or hi.shape!=h.shape or not all(np.isfinite(x).all() for x in arrays):raise ValueError('Finite trusted box and inputs required')
    if (lo>hi).any() or (eps<0).any() or not np.isfinite(beta) or beta<0 or not isinstance(t,int) or not 0<=t<=len(blocks):raise ValueError('Invalid bounds')
    wr=[rational(x) for x in w];delta=[]
    for j in range(A.shape[1]):delta.append(rational(h[j])-sum((rational(A[i,j])*wr[i] for i in range(len(A))),Fraction(0)))
    value=sum((-rational(b[i])*wr[i]+rational(eps[i])*abs(wr[i]) for i in range(len(A))),Fraction(0))
    leverage=[]
    for B in blocks:
        B=np.asarray(B,float)
        if B.ndim!=2 or B.shape[0]!=len(A):raise ValueError('Invalid fault block')
        leverage.append(sum((abs(sum((rational(B[i,j])*wr[i] for i in range(len(A))),Fraction(0))) for j in range(B.shape[1])),Fraction(0)))
    value+=rational(beta)*sum(sorted(leverage,reverse=True)[:t],Fraction(0))
    correction=sum((max(delta[j]*rational(lo[j]),delta[j]*rational(hi[j])) for j in range(len(h))),Fraction(0))
    value+=correction
    try:rounded=float(value)
    except OverflowError:rounded=math.inf if value>0 else -sys.float_info.max
    if rounded==-math.inf:rounded=-sys.float_info.max
    if math.isfinite(rounded) and rational(rounded)<value:rounded=math.nextafter(rounded,math.inf)
    return {'status':'exact_nominal_upper','upper_outward_float':rounded,'upper_rational':str(value),'residual_correction_rational':str(correction),
        'dual_equation_residual_rational':[str(x) for x in delta],
        'scope':'Exact rational bound for stored nominal coefficients and finite trusted state box; not a certificate of physical model validity or LP optimality'}
