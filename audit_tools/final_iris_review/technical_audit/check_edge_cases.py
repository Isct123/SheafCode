                                                                                    
from pathlib import Path
from fractions import Fraction as Q
from itertools import combinations
import json, random, sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Research_Release/extensions/decision_repair'))
sys.path.insert(0, str(ROOT / 'Research_Release/extensions/math_integration'))
from decision_repair import conflict_edges, plan
from radius_repair import radius, ambiguity_witness
from adaptive_repair import solve
from verify_adaptive import brute
from joint_certified import certify_joint


def common_response(states, labels, noise, selected, faults, rho=Q(0)):
                                                                                
    supports = [set(s) for k in range(min(faults, len(selected)) + 1)
                for s in combinations(selected, k)]
    for i, j in combinations(range(len(states)), 2):
        if labels[i] == labels[j]:
            continue
        for first in supports:
            for second in supports:
                if all(all(abs(a - b) <= 2 * (noise[v] + rho)
                           for a, b in zip(states[i][v], states[j][v]))
                       for v in selected if v not in first | second):
                    return True
    return False


def main():
    rng = random.Random(10102026)
    counts = {'decoder': 0, 'vector_repair': 0, 'radius': 0,
              'radius_witness': 0, 'adaptive': 0, 'joint_controls': 0}
                                                                                    
    for _ in range(50):
        D = np.array([[rng.uniform(-2, 2) for _ in range(5)] for _ in range(7)])
        C = np.array([[rng.uniform(-1, 1) for _ in range(5)] for _ in range(5)])
        P = C.T @ C + np.eye(5)
        U, O = [0, 2, 4], [1, 3]
        obs = np.array([rng.uniform(-2, 2) for _ in O]); weight = .3
        A = D[:, U].T @ D[:, U] + weight * P[np.ix_(U, U)]
        H = D[:, U].T @ D[:, O] + weight * P[np.ix_(U, O)]
        optimum = np.linalg.solve(A, -H @ obs)
        x = np.zeros(5); x[U] = optimum; x[O] = obs
        gradient = (D.T @ D @ x + weight * P @ x)[U]
        assert np.max(abs(gradient)) < 1e-10
        delta = np.array([rng.uniform(-1, 1) for _ in U])
        y = x.copy(); y[U] += delta
        objective = lambda w: np.linalg.norm(D @ w)**2 + weight * w @ P @ w
        assert abs(objective(y) - objective(x) - delta @ A @ delta) < 1e-9
        counts['decoder'] += 1
                                                                                     
    for trial in range(80):
        states = [[[Q(rng.randrange(9)) for _ in range(2)] for _ in range(4)]
                  for _ in range(4)]
        labels = [0, 1, 0, 1]
        noise = [Q(rng.randrange(4), 2) for _ in range(4)]
        edges = conflict_edges(states, labels, noise)
        for faults in [0, 1]:
            possible = []
            for mask in range(16):
                selected = [v for v in range(4) if mask >> v & 1]
                feasible = not common_response(states, labels, noise, selected, faults)
                if feasible:
                    possible.append(len(selected))
                assert feasible == all((mask & e).bit_count() >= 2*faults+1 for e, _ in edges)
                counts['vector_repair'] += 1
            answer = plan(4, edges, faults=faults)
            assert (answer['status'] == 'infeasible') == (not possible)
            if possible:
                assert answer['cost'] == min(possible)
        scalar = [[station[0] for station in s] for s in states]
        vectors = [[[v] for v in s] for s in scalar]
        faults = trial % 2
        for mask in range(16):
            selected = [v for v in range(4) if mask >> v & 1]
            result = radius(scalar, labels, noise, mask, faults)
            critical = result['radius']
            values = {Q(0), Q(1, 2), Q(2)}
            if critical >= 0 and critical != float('inf'):
                values.add(critical)
                if critical > 0:
                    values.add(critical / 2)
                witness = ambiguity_witness(scalar, result['witness'], noise, mask, critical, faults)
                assert all(abs(a-b) <= critical for a, b in zip(witness['a'], scalar[result['witness'][0]]))
                assert all(abs(a-b) <= critical for a, b in zip(witness['b'], scalar[result['witness'][1]]))
                counts['radius_witness'] += 1
            for rho in values:
                assert (rho < critical) == (not common_response(vectors, labels, noise, selected, faults, rho))
                counts['radius'] += 1
    for trial in range(60):
        states = [[rng.randrange(6) for _ in range(3)] for _ in range(4)]
        labels = [0, 1, 0, 1]; noise = Q(trial % 3, 2); faults = trial % 2
        reference = brute(states, labels, noise, faults)
        answer = solve(states, labels, noise, faults)
        assert (answer['status'] == 'infeasible') == (reference == float('inf'))
        if reference != float('inf'):
            assert answer['cost'] == reference
        counts['adaptive'] += 1
                                                                               
    for epsilon in [0., .01]:
        result = certify_joint(np.array([[1., 1.]]), np.array([-2.]),
                               np.eye(2) / .75, [0., 0.], [], 0, epsilon, 1., [0., 0.], [2., 2.])
        assert result['decision'] == 'exceedance'
        assert result['lower'] <= Q.from_float((2-epsilon)/1.5) + Q(1, 10**12)
        counts['joint_controls'] += 1
                                                                             
    assert radius([[0], [4]], [0, 1], 0, 1)['radius'] == 2
                                                                           
    assert max(Q(-1), Q(1)) <= 1 and Q(3) > 1
                                                                                       
    singular_before = .1; singular_after = np.sqrt(.02); lam = 1.
    amp_before = singular_before / (singular_before**2 + lam)
    amp_after = singular_after / (singular_after**2 + lam)
    assert amp_after > amp_before
    report = {'status': 'passed', 'checks': counts,
              'G_boundary_counterexample': {'centres': [[0], [4]], 'threshold': 1,
                                           'G': 1, 'rho': 1, 'separation_radius': 2,
                                           'certification_still_possible': True},
              'archive_docstring_counterexample': {'lambda': lam,
                                                   'amplification_before': amp_before,
                                                   'amplification_after': amp_after},
              'scope': 'Small-instance formula and adversarial-response checks; not a replacement for general proofs or field validation.'}
    joint = pd.read_csv(ROOT / 'Research_Release/joint_certification_results/CASES.csv')
    covered = joint.exact_truth_assumptions
    report['joint_truth_membership'] = {
        'total': len(joint), 'covered': int(covered.sum()), 'outside': int((~covered).sum()),
        'outside_with_feasible_model': int(((~covered) & (joint.status == 'verified_bounds')).sum()),
        'outside_issued_below': int(((~covered) & (joint.decision == 'below_or_equal')).sum()),
        'covered_issued_below': int((covered & (joint.decision == 'below_or_equal')).sum()),
        'outside_wrong': int(joint.loc[~covered, 'wrong'].sum())}
    cases = pd.read_csv(ROOT / 'Research_Release/recent_data_results/delhi/outage_cases.csv')
    primary = cases[cases.primary]
    headlines = {}
    for column in ['mse', 'pm25_raw_mse']:
        matched = primary.pivot(index=['day', 'station'], columns='method', values=column)
        matched = matched[['sheaf_causal', 'var_causal']].dropna()
        gain = 100 * (1 - np.sqrt(matched.sheaf_causal.sum() / matched.var_causal.sum()))
        headlines[column] = {'cases': len(matched), 'gain_percent': float(gain)}
    assert round(headlines['mse']['gain_percent'], 2) == 16.48
    assert round(headlines['pm25_raw_mse']['gain_percent'], 2) == 10.45
    cohorts = pd.read_csv(ROOT / 'Research_Release/expanded_validation_results/COHORT_RESULTS.csv')
    external = cohorts[cohorts.provider.isin(['EPA', 'AURN', 'NSW'])]
    forest = pd.read_csv(ROOT / 'Research_Release/weakness_resolution_results/NONLINEAR_SUMMARY.csv')
    forest_external = forest[forest.external]
    assert len(external) == 20 and int((external.gain > 0).sum()) == 13
    assert len(forest_external) == 20 and int((forest_external.gain > 0).sum()) == 18
    report['reconstruction_recalculation'] = headlines
    report['external_counts'] = {'cohorts': 20, 'wins_vs_VAR': 13, 'wins_vs_RF': 18}
    target = Path(__file__).with_name('edge_case_results.json')
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
