                                                                 
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from fractions import Fraction as Q
from search import search

def test_known_optimum_and_zero_fault_budget():
                                                                     
    for t,maximum in [(0,.25),(1,1.25)]:
        result=search(np.array([[1.]]),np.array([0.]),np.array([1.]),[np.array([[1.]])],t,.25,1.,[-2.],[2.])
        assert result['status']=='verified_gap'
        assert Q(result['lower'])<=Q.from_float(maximum)<=Q(result['upper'])
        assert float(Q(result['upper'])-Q(result['lower']))<=1e-7

def test_exact_infeasibility():
                                                              
    result=search(np.array([[1.]]),np.array([4.]),np.array([1.]),[np.array([[1.]])],1,.25,1.,[-1.],[1.])
    assert result['status']=='certified_infeasible'
    assert all(r['upper']<0 for r in result['records'] if r['infeasible'])

def test_interruption_retains_a_valid_bound():
    result=search(np.array([[1.]]),np.array([0.]),np.array([1.]),[np.array([[1.]])]*3,2,.25,1.,[-4.],[4.],max_nodes=1)
    assert result['status']=='unresolved' and result['active_nodes']==2
    assert Q(result['upper'])>=Q(9,4)

def test_two_faults_and_monotone_trace():
    result=search(np.array([[1.]]),np.array([0.]),np.array([1.]),[np.array([[1.]])]*3,2,.25,1.,[-4.],[4.])
    assert result['status']=='verified_gap'
    assert Q(result['lower'])<=Q(9,4)<=Q(result['upper'])
    uppers=[Q(h['upper']) for h in result['history']]
    lowers=[Q(h['lower']) for h in result['history'] if h['lower'] is not None]
    assert all(a>=b for a,b in zip(uppers,uppers[1:]))
    assert all(a<=b for a,b in zip(lowers,lowers[1:]))

if __name__=='__main__':
    tests=[v for k,v in list(globals().items()) if k.startswith('test_')]
    for test in tests:test()
    print(len(tests),'known-optimum/interruption controls passed')
