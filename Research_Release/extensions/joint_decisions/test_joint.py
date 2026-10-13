import numpy as np
import pytest
from joint import joint_envelope


def test_joint_exceedance_can_be_certified_without_identifying_pollutant():
    r = joint_envelope([[1,1]], [-2], np.eye(2)/.75, [0,0], [], 0,
                       0, 0, [0,0], [2,2])
    assert r['lower'] == pytest.approx(4/3)
    assert r['upper'] == pytest.approx(8/3)
    assert r['independent_lower'] == pytest.approx(0)
    assert r['decision'] == 'exceedance'


def test_inconsistency_is_abstention_and_boundary_is_guarded():
    r = joint_envelope([[1]], [-3], [[1]], [0], [], 0, 0, 0, [0], [2])
    assert r['status'] == 'inconsistent' and r['decision'] == 'abstain'
    r = joint_envelope([[1]], [-1], [[1]], [0], [], 0, 0, 0, [0], [2])
    assert r['lower'] == r['upper'] == 1 and r['decision'] == 'abstain'


def test_support_union_extremes_and_affine_offset():
    r = joint_envelope([[1],[1]], [-1,-1], [[2]], [-1],
                       [np.array([[1.],[0.]])], 1, .1, 1, [0], [2])
    assert r['lower'] == pytest.approx(.8)
    assert r['upper'] == pytest.approx(1.2)
    assert r['decision'] == 'abstain'


def test_random_planted_truth_and_attaining_fault_witnesses():
    rng = np.random.default_rng(719)
    for _ in range(20):
        A = rng.normal(size=(5,2)); blocks = [rng.normal(size=(5,1)) for j in range(2)]
        z = rng.uniform(-1,1,2); e = .5; b = -A@z-blocks[0][:,0]*e
        H = rng.normal(size=(3,2)); c = rng.normal(size=3)
        r = joint_envelope(A,b,H,c,blocks,1,.1,1,[-2,-2],[2,2])
        assert r['lower']-1e-6 <= np.max(H@z+c) <= r['upper']+1e-6
        assert r['lower'] >= r['independent_lower']-1e-6
        for end in ['lower','upper']:
            w = r[end+'_witness']; x = np.array(w['state']); fault = np.array(w['fault'])
            B = np.concatenate([blocks[j] for j in w['support']],axis=1) if w['support'] else np.empty((5,0))
            assert np.max(abs(A@x+b+B@fault)) <= .1+1e-6
            assert np.max(H@x+c) == pytest.approx(r[end], abs=1e-6)


def test_input_validation_and_enumeration_limit():
    with pytest.raises(ValueError):
        joint_envelope([[1]], [0], [[1]], [0], [], 0, 0, 0, [-np.inf], [1])
    with pytest.raises(ValueError):
        joint_envelope([[1]], [0], [[1]], [0], [np.ones((1,1))]*5, 2,
                       0, 1, [0], [1], max_supports=2)
