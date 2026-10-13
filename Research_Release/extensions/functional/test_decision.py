import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from decision import certified_decision

def test_sharp_minimax_center_and_projection_dominate_only_inside_interval():
    r=certified_decision(np.array([[1.,0.]]),np.array([-2.]),np.array([1.,0.]),.5,prediction=5.)
    assert r['minimax_estimate']==2. and r['minimax_absolute_error']==.5
    assert r['projected_prediction']==2.5
    for estimate in np.linspace(-5,5,100):
        assert max(abs(estimate-r['lower']),abs(estimate-r['upper']))>=r['minimax_absolute_error']
    for truth in np.linspace(r['lower'],r['upper'],100):
        assert abs(r['projected_prediction']-truth)<=abs(5.-truth)
                                                                               
    assert abs(r['projected_prediction']-5.)>abs(5.-5.)

def test_inconsistent_or_unbounded_functionals_abstain():
    assert certified_decision(np.array([[1.,0.]]),np.zeros(1),np.array([0.,1.]),1.)['decision']=='abstain'
    assert certified_decision(np.zeros((1,2)),np.ones(1),np.ones(2),0.)['decision']=='abstain'
