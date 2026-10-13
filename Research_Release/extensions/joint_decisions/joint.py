                                                                            

                                                                               
   
from itertools import combinations
from math import comb
import numpy as np
from scipy.optimize import linprog


def joint_envelope(A, b, H, offsets, blocks, t, eps, beta, lower, upper,
                   max_supports=10000, feasibility_tolerance=1e-7):
                                                                      

                                                                              
                                                                               
       
    A, b, H, offsets = map(lambda x: np.asarray(x, float), (A, b, H, offsets))
    if A.ndim != 2 or not A.shape[1] or b.shape != (A.shape[0],):
        raise ValueError('Invalid equation dimensions')
    m, p = A.shape
    if H.ndim != 2 or not H.shape[0] or H.shape[1] != p or offsets.shape != (len(H),):
        raise ValueError('Invalid score dimensions')
    if not isinstance(t, (int, np.integer)) or not 0 <= t <= len(blocks):
        raise ValueError('Invalid station fault budget')
    if sum(comb(len(blocks), k) for k in range(t + 1)) > max_supports:
        raise ValueError('Support enumeration exceeds declared budget')
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    eps = np.broadcast_to(np.asarray(eps, float), (m,))
    blocks = [np.asarray(B, float) for B in blocks]
    if any(B.ndim != 2 or B.shape[0] != m or not B.shape[1] for B in blocks):
        raise ValueError('Invalid fault block')
    if lo.shape != (p,) or hi.shape != (p,) or (lo > hi).any():
        raise ValueError('Invalid state box')
    values = np.concatenate([A.ravel(), b, H.ravel(), offsets, eps, lo, hi,
                             [beta, feasibility_tolerance],
                             *[B.ravel() for B in blocks]])
    if not np.isfinite(values).all() or (eps < 0).any() or beta < 0 or feasibility_tolerance <= 0:
        raise ValueError('Finite values and nonnegative bounds required')
    marginal_low = np.full(len(H), np.inf)
    score_low, score_high = np.inf, -np.inf
    low_witness = high_witness = None
    considered = feasible = solves = 0

    def solve(cost, G, rhs, bounds):
        nonlocal solves
        solves += 1
        fit = linprog(cost, A_ub=G, b_ub=rhs, bounds=bounds, method='highs')
        if fit.status == 2:
            return None
        if not fit.success:
            raise RuntimeError(fit.message)
        violation = np.max(G @ fit.x - rhs, initial=0.)
        for value, (a, c) in zip(fit.x, bounds):
            if a is not None: violation = max(violation, a-value)
            if c is not None: violation = max(violation, value-c)
        if violation > feasibility_tolerance:
            raise RuntimeError('LP primal feasibility exceeds tolerance')
        return fit

    def witness(fit, S, q):
        return {'state': fit.x[:p].tolist(), 'support': list(S),
                'fault': fit.x[p:p+q].tolist()}

    for k in range(t + 1):
        for S in combinations(range(len(blocks)), k):
            considered += 1
            B = np.concatenate([blocks[j] for j in S], axis=1) if S else np.empty((m, 0))
            q = B.shape[1]
            M = np.c_[A, B]
            G, rhs = np.r_[M, -M], np.r_[eps-b, eps+b]
            bounds = list(zip(lo, hi)) + [(-beta, beta)] * q
                                                                          
            J = np.c_[H, np.zeros((len(H), q)), -np.ones(len(H))]
            low = solve(np.r_[np.zeros(p+q), 1.],
                        np.r_[np.c_[G, np.zeros(2*m)], J],
                        np.r_[rhs, -offsets], bounds + [(None, None)])
            if low is None:
                continue
            feasible += 1
            val = float(np.max(H @ low.x[:p] + offsets))
            if abs(val-low.fun) > feasibility_tolerance * 10:
                raise RuntimeError('Epigraph endpoint mismatch')
            if val < score_low:
                score_low, low_witness = val, witness(low, S, q)
            for j, h in enumerate(H):
                c = np.r_[h, np.zeros(q)]
                small, large = solve(c, G, rhs, bounds), solve(-c, G, rhs, bounds)
                if small is None or large is None:
                    raise RuntimeError('Support feasibility inconsistent across objectives')
                marginal_low[j] = min(marginal_low[j], float(small.fun+offsets[j]))
                val = float(-large.fun+offsets[j])
                if val > score_high:
                    score_high, high_witness = val, witness(large, S, q)
    if not feasible:
        return {'status': 'inconsistent', 'decision': 'abstain',
                'supports': considered, 'feasible_supports': 0, 'lp_solves': solves}
    independent_lower = float(np.max(marginal_low))
    guard = 10 * feasibility_tolerance
    decision = ('exceedance' if score_low > 1+guard else
                'below_or_equal' if score_high < 1-guard else 'abstain')
    return {'status': 'bounded', 'lower': score_low, 'upper': score_high,
            'marginal_lower': marginal_low.tolist(),
            'independent_lower': independent_lower,
            'joint_lower_gain': score_low-independent_lower,
            'lower_witness': low_witness, 'upper_witness': high_witness,
            'decision': decision, 'supports': considered,
            'feasible_supports': feasible, 'lp_solves': solves,
            'numerical_guard': guard}
