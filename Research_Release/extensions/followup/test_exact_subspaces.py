from itertools import combinations
import numpy as np
from exact_subspaces import canonical_span,exact_lattice

def test_exact_lattice_equals_exhaustive_rational_spans():
    blocks=[np.array([[1],[0],[0]]),np.array([[2],[0],[0]]),np.array([[0],[1],[0]]),np.array([[1],[1],[0]])]
    brute=set()
    for k in range(3):
        for support in combinations(range(len(blocks)),k):
            rows=[row for j in support for row in blocks[j].T.tolist()]
            brute.add(canonical_span(rows,3))
    states=exact_lattice(blocks,3,2)
    assert set(states)==brute
    assert len(states)==5

def test_nearly_equal_spaces_are_not_merged():
    a=np.array([['1'],['0']],dtype=object)
    b=np.array([['1'],['0.00000000000001']],dtype=object)
    states=exact_lattice([a,b],2,1)
    assert len(states)==3

def test_representation_independence():
    assert canonical_span([[1,2,3],[0,1,1]],3)==canonical_span([[2,5,7],[0,-2,-2]],3)
