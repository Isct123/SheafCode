import pandas as pd
from review_expanded import week_cluster

def test_calendar_clusters_independent_of_timestamp_resolution():
    dates=pd.Series(pd.to_datetime(['2026-01-01','2026-01-15','2026-02-01']))
    a=week_cluster(dates.astype('datetime64[us]'))
    b=week_cluster(dates.astype('datetime64[ns]'))
    assert a.equals(b)
    assert a.nunique()==3
