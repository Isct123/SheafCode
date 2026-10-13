                                                                      
                                                                       
                                                                    
from __future__ import annotations

import numpy as np

from .data import haversine_km


def knn_edges(D: np.ndarray, k: int) -> list[tuple[int, int]]:
    n = D.shape[0]
    edges = set()
    for i in range(n):
        order = np.argsort(D[i])
        for j in order[1:k + 1]:
            a, b = (i, int(j)) if i < j else (int(j), i)
            edges.add((a, b))
    return sorted(edges)


def geographic_graph(coords, k: int):
    D = haversine_km(coords)
    return knn_edges(D, k), D


def correlation_graph(Ztrain: np.ndarray, k: int):
                                                                          
                                    
    T, n, p = Ztrain.shape
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cs = [abs(np.corrcoef(Ztrain[:, i, q], Ztrain[:, j, q])[0, 1]) for q in range(p)]
            C[i, j] = np.nanmean(cs)
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    return knn_edges(D, k), D


def complete_graph(n: int):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def distance_weights(D: np.ndarray, edges, length_scale=None) -> np.ndarray:
                                                                      
                          
    d = np.array([D[u, v] for u, v in edges])
    if length_scale is None:
        length_scale = np.median(d) if len(d) else 1.0
    return np.exp(-(d / length_scale) ** 2)
