                                                                                       
from fractions import Fraction
import math

def exact(value):
    if isinstance(value,Fraction):return value
    if not math.isfinite(float(value)):raise ValueError('Finite measurements and bounds required')
    return Fraction.from_float(float(value))

def assess(stations,readings,threshold=35.,noise=1.,faults=0):
                                                                                      

                                                                             
                                                                             
                                                                              
       
    stations=tuple(stations)
    if not stations or len(set(stations))!=len(stations):raise ValueError('Nonempty unique station universe required')
    if not isinstance(faults,int) or isinstance(faults,bool) or faults<0:raise ValueError('Nonnegative integer fault budget required')
    if set(readings)-set(stations):raise ValueError('Reading outside declared station universe')
    tau=exact(threshold)
    bounds={v:exact(noise[v] if isinstance(noise,dict) else noise) for v in stations}
    if any(x<0 for x in bounds.values()):raise ValueError('Negative clean-error bound')
    measured={v:exact(y) for v,y in readings.items()}
    high=[v for v in stations if v in measured and measured[v]-bounds[v]>tau]
    missing=[v for v in stations if v not in measured]
    label='abstain';witness=[];reason='insufficient_separating_measurements'
    if len(high)>faults:
        label='exceedance';witness=high[:faults+1];reason='at_least_one_high_reading_is_clean'
    elif faults==0 and not missing and all(measured[v]+bounds[v]<=tau for v in stations):
        label='safe';witness=list(stations);reason='all_upper_bounds_are_safe'
    elif faults>0 and not missing:reason='arbitrary_corruption_prevents_uniform_safe_certificate'
    return {'label':label,'witness_stations':witness,'queried':len(measured),'unqueried':missing,'reason':reason,
            'scope':'Conditional on real queried readings, the declared station universe and clean-error/fault bounds; not AQI or an emissions intervention.'}
