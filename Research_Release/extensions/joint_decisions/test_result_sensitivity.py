import numpy as np
import pandas as pd
import pytest
from result_sensitivity import pair_cases, cluster_sensitivity, leave_cluster_out


def test_constant_ratio_is_invariant_to_cluster_resampling():
    pair=pd.DataFrame({'day':pd.date_range('2026-01-01',periods=30),
                       'station':['s']*30,'method_mse':np.arange(1,31)*.25,
                       'reference_mse':np.arange(1,31)})
    for days in [1,7,14]:
        r=cluster_sensitivity(pair,days,draws=200)
        assert r['gain_percent']==pytest.approx(50)
        assert r['nominal_95_low']==pytest.approx(50)
        assert r['family_sensitivity_high']==pytest.approx(50)
    assert all(r['gain_percent']==pytest.approx(50) for r in leave_cluster_out(pair))


def test_common_cases_only_and_duplicate_seed_rows_rejected():
    frame=pd.DataFrame([['2026-01-01','a','m',1],['2026-01-01','a','r',4],
                        ['2026-01-02','a','m',5]],columns=['day','station','method','mse'])
    assert len(pair_cases(frame,'m','r'))==1
    with pytest.raises(ValueError): pair_cases(pd.concat([frame,frame]),'m','r')


def test_calendar_gaps_do_not_become_adjacent_week_clusters():
    pair=pd.DataFrame({'day':['2026-01-05','2026-02-02'], 'station':['s','s'],
                       'method_mse':[1,1], 'reference_mse':[2,3]})
    r=cluster_sensitivity(pair,7,draws=200)
    assert r['clusters']==2 and r['small_cluster_warning']
    assert r['family_sensitivity_low']<=r['nominal_95_low']
    assert r['family_sensitivity_high']>=r['nominal_95_high']
