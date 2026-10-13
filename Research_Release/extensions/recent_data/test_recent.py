import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
def fixture():
    rng=np.random.default_rng(17);a=rng.normal(size=(260,3,2));a[:,1]=a[:,0]+rng.normal(scale=.3,size=(260,2));a[:,2]=.5*a[:,0]+rng.normal(scale=.3,size=(260,2));return a
def test_hidden_target_and_future_do_not_change_causal_prediction():
    z=fixture();m=QuadraticDecoder(z,[(0,1),(1,2),(0,2)]);x=z[:48].copy();a=m.causal(x,0);b=x.copy();b[24:,0]=10000;assert np.allclose(a,m.causal(b,0))
    changed=x.copy();changed[40:,1:]=-7000;assert np.allclose(a[:16],m.causal(changed,0)[:16])
def test_prior_cross_term_stationarity():
    z=fixture();m=QuadraticDecoder(z,[(0,1),(1,2)]);x=z[:1].copy();x[0,0,0]=np.nan;pred=static_solve(x,m.S).ravel();assert abs((m.S@pred)[0])<1e-9
def test_city_averages_raw_units_before_normalization():
    x=np.array([[[np.nan],[1.],[1.]]]);mu=np.array([[10.],[20.],[100.]]);sd=np.array([[2.],[2.],[10.]]);pred=raw_city(x,0,mu,sd);assert np.allclose(pred,(66.-10)/2)
def test_order_statistic_boundary_returns_infinite():
    assert np.isinf(quantile([1,2,3],.1));assert quantile(np.arange(100),.1)==90
def test_train_selected_dates_are_recent_and_disjoint():
    for region in regions()+['epa_la']:
        ds=load(region);times=ds['times'];s=ds['split'];assert times.min()>=pd.Timestamp('2026-01-01');assert times.max()<=pd.Timestamp('2026-10-05 23:00')
        assert times[s==0].max()<times[s==1].min()<times[s==1].max()<times[s==3].min()<times[s==3].max()<times[s==2].min()
        assert np.allclose(ds['mu'],np.nanmean(ds['X'][s==0],0));assert np.allclose(ds['sd'],np.nanstd(ds['X'][s==0],0,ddof=1))
def test_var_filter_preserves_observed_values_and_causality():
    z=fixture();v=FullVAR(complete_training(z),3,2);x=z[:48].copy();x[24:,0]=np.nan;a=v.filter(x);b=x.copy();b[40:,1:]=10000;assert np.allclose(a[:40],v.filter(b)[:40]);good=np.isfinite(x);assert np.array_equal(a[good],x[good])
