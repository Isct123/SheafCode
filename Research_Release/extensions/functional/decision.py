                                                                                
import numpy as np
from certificate import functional_interval

def certified_decision(A,b,h,epsilon,prediction=None):
    result=functional_interval(A,b,h,epsilon)
    if prediction is not None and not np.isfinite(prediction):raise ValueError('Prediction must be finite')
    if result['status']!='bounded':
        return {**result,'minimax_estimate':None,'minimax_absolute_error':None,
            'projected_prediction':None,'decision':'abstain'}
    center=.5*(result['lower']+result['upper'])
    projected=None if prediction is None else float(np.clip(prediction,result['lower'],result['upper']))
    return {**result,'minimax_estimate':center,'minimax_absolute_error':.5*(result['upper']-result['lower']),
        'projected_prediction':projected,'decision':'conditionally_bounded',
        'scope':'minimax over the residual-feasible set; not population-average or shift-valid risk'}
