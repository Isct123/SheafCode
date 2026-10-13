import numpy as np
from matched_alerts import sensitivity_at_fpr

def test_perfect_ranking_and_ties():
    assert sensitivity_at_fpr([0,0,1,1],[0,1,2,3],0)==1.
    assert abs(sensitivity_at_fpr([0,0,1,1],[1,1,1,1],.05)-.05)<1e-12
    assert np.isnan(sensitivity_at_fpr([0,0],[1,2],.05))
