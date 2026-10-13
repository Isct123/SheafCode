                                                                                    
from fractions import Fraction as Q
from itertools import combinations
from collections import deque
from pathlib import Path
import json

def main():
    checks = []
                                                                                  
    for a in (Q(-2), Q(0), Q(3, 2)):
        path = [a, a, a]
        assert all(path[i] == path[i+1] for i in range(2))
        checks.append('identity-path-root-' + str(a))
                                                             
    blocks = [(Q(1), Q(0)), (Q(2), Q(0)), (Q(0), Q(1))]
    def span(support):
        if not support:
            return 'zero'
        xs = [blocks[j] for j in support]
        if any(a[0]*b[1] != a[1]*b[0] for a,b in combinations(xs,2)):
            return 'plane'
        return 'horizontal' if xs[0][1] == 0 else 'vertical'
    for t in range(4):
        queue = deque([frozenset()]); seen = {'zero': frozenset()}
        while queue:
            support = queue.popleft()
            if len(support) >= t:
                continue
            for j in range(3):
                new = support | {j}; state = span(new)
                if state not in seen:
                    seen[state] = new; queue.append(new)
        exhaustive = {span(c) for k in range(t+1) for c in combinations(range(3),k)}
        assert set(seen) == exhaustive
        checks.append('quotient-bfs-budget-' + str(t))
    assert (Q(-1),Q(1)) != (Q(-2),Q(2))
    checks.append('bounded-effects-not-equivalent')
                                                                     
    threshold = Q(3,4)
    sharp_joint = Q(1) / threshold
    weighted_certificate = (Q(1,2)*Q(2)) / threshold
    assert sharp_joint == weighted_certificate == Q(4,3) > 1
    assert min(Q(0),Q(2)/threshold) == 0                                  
    checks.append('segment-joint-versus-individual-minima')
                                                                                     
    scores = [(Q(2),Q(0)), (Q(0),Q(2))]
    L = min(max(s) for s in scores)
    assert L == 2
                                                                       
    assert Q(2)*Q(1,2) == 1 < L
    assert sum([Q(1)*scores[0][0],Q(0)*scores[0][1]]) == L
    assert sum([Q(0)*scores[1][0],Q(1)*scores[1][1]]) == L
    checks.append('support-specific-weight-strict-gap')
    for lo, hi, label in [(Q(1),Q(1),'safe-boundary'), (Q(1,2),Q(2),'ambiguous'), (Q(4,3),Q(2),'exceedance')]:
        decision = 'exceedance' if lo > 1 else ('safe-boundary' if hi <= 1 else 'ambiguous')
        assert decision == label
        checks.append('strict-threshold-' + label)
    report = {'status':'passed','checks':len(checks),'examples':checks,
              'scope':'Exact rational illustrative checks; no field-data performance or mathematical priority claim.'}
    target = Path(__file__).resolve().parents[1] / 'IRIS_Submission_Recent_Data/Review_Notes/CONCISE_MATH_EXAMPLES.json'
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
