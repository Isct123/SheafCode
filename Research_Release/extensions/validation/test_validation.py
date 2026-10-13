import numpy as np
from refined_bounds import envelope
from common import ROOT
import sys
sys.path.insert(0,str(ROOT/'extensions/followup'))
from constrained import constrained_envelope
def test_refinement_matches_exact_and_contains_planted_truth():
    rng=np.random.default_rng(721)
    for _ in range(8):
        A=rng.normal(size=(6,2));B=[rng.normal(size=(6,1)) for j in range(3)];z=rng.uniform(-1,1,2);b=-A@z-B[0][:,0]*.4;h=rng.normal(size=2)
        exact=constrained_envelope(A,b,h,B,1,.05,1,[-2,-2],[2,2]);r=envelope(A,b,h,B,1,.05,1,[-2,-2],[2,2],128)
        assert r['lower']-1e-7<=h@z<=r['upper']+1e-7
        assert np.allclose([r['lower'],r['upper']],[exact['lower'],exact['upper']],atol=1e-6)
def test_small_node_budget_is_outer_bound():
    rng=np.random.default_rng(25);A=rng.normal(size=(7,2));B=[rng.normal(size=(7,1)) for j in range(4)];b=-A@np.ones(2);h=np.ones(2)
    exact=constrained_envelope(A,b,h,B,2,.1,1,[-3,-3],[3,3]);r=envelope(A,b,h,B,2,.1,1,[-3,-3],[3,3],1)
    assert r['lower']<=exact['lower']+1e-6 and r['upper']>=exact['upper']-1e-6

def test_alert_threshold_is_infinite_with_insufficient_negatives():
    from alerts import negative_cutoff
    assert np.isinf(negative_cutoff([1,2],[False,False]))

def test_episode_delay_and_missing_truth_boundary():
    from alerts import episodes
    out=episodes(np.array([1,1,1,1],bool),np.array([0,1,0,0],bool),np.array([1,1,0,1],bool))
    assert len(out)==2 and out[0]['delay_hours']==1 and not out[1]['detected']

def test_unknown_count_classifier_probabilities_sum_to_one():
    from faults import fit_classifier,probabilities
    rng=np.random.default_rng(1);X=rng.normal(size=(100,3));y=(np.arange(100)%3)
    P=probabilities(X,fit_classifier(X,y,1.))
    assert np.allclose(P.sum(1),1) and (P>=0).all()

def test_uncertain_maps_inflation_prevents_nominal_exclusion():
    from operator_uncertainty import inflate
    eps=inflate(0.,[0.],[2.],np.ones((1,1)),np.zeros(1),[],0,0.)
    r=envelope(np.ones((1,1)),np.array([-2.]),np.ones(1),[],0,eps,0.,[0.],[2.])
    assert r['lower']<=1<=r['upper']
                                                                               
    naive=envelope(np.ones((1,1)),np.array([-2.]),np.ones(1),[],0,0.,0.,[0.],[2.])
    assert naive['lower']>1

def test_operator_inflation_top_budget_contains_random_perturbations():
    from operator_uncertainty import inflate
    rng=np.random.default_rng(105)
    for _ in range(30):
        RA=rng.uniform(0,.1,(5,2));RB=[rng.uniform(0,.1,(5,2)) for j in range(3)];z=rng.uniform(-2,2,2);e=rng.uniform(-1,1,2)
        dA=rng.uniform(-1,1,RA.shape)*RA;dB=rng.uniform(-1,1,RB[1].shape)*RB[1]
        extra=dA@z+dB@e;inflated=inflate(0.,[-2,-2],[2,2],RA,0.,RB,1,1.)
        assert (abs(extra)<=inflated+1e-12).all()
