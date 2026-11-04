"""Rectangular minimum-cost assignment without adding SciPy to the runtime image."""
import numpy as np


def linear_sum_assignment(cost):
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or not np.isfinite(cost).all():
        raise ValueError('Assignment requires a finite matrix')
    transpose = cost.shape[0] > cost.shape[1]
    if transpose:
        cost = cost.T
    n, m = cost.shape
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p, way = np.zeros(m + 1, int), np.zeros(m + 1, int)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minimum, used = np.full(m + 1, np.inf), np.zeros(m + 1, bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            available = np.flatnonzero(~used[1:]) + 1
            reduced = cost[i0 - 1, available - 1] - u[i0] - v[available]
            improved = reduced < minimum[available]
            minimum[available[improved]] = reduced[improved]
            way[available[improved]] = j0
            j1 = available[np.argmin(minimum[available])]
            delta = minimum[j1]
            u[p[used]] += delta
            v[used] -= delta
            minimum[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    cols = np.flatnonzero(p[1:])
    rows = p[cols + 1] - 1
    if transpose:
        rows, cols = cols, rows
    order = np.argsort(rows)
    return rows[order], cols[order]
