from fractions import Fraction
import numpy as np
import pytest
from exact_certificate import verified_upper,rational

def test_small_dual_residual_can_matter_for_large_state():
    result=verified_upper(np.ones((1,1)),np.array([-1e9]),np.ones(1),[],0,0.,0.,np.array([.999999999]),[1e9],[1e9])
    assert Fraction(result['upper_rational'])==Fraction(1000000000)
    assert float(Fraction(result['residual_correction_rational']))>.9

def test_exact_arithmetic_bound_contains_feasible_random_examples():
    rng=np.random.default_rng(137)
    for _ in range(100):
        A=rng.normal(size=(6,3));z=rng.uniform(-1,1,3);h=rng.normal(size=3);B=rng.normal(size=(6,2));e=rng.uniform(-.5,.5,2)
        b=-A@z-B@e;w=rng.normal(size=6)
        result=verified_upper(A,b,h,[B],1,.001,.5,w,[-1,-1,-1],[1,1,1])
        actual=sum((rational(a)*rational(c) for a,c in zip(h,z)),Fraction(0))
        assert actual<=Fraction(result['upper_rational'])
        assert rational(result['upper_outward_float'])>=Fraction(result['upper_rational'])

def test_reject_unbounded_box():
    with pytest.raises(ValueError):verified_upper(np.ones((1,1)),np.zeros(1),np.ones(1),[],0,0.,0.,np.ones(1),[0],[np.inf])
