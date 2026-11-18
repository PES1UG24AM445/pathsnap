import numpy as np
import shapely

def arc_lines(trip, idx=None):
    a = trip.arcs if idx is None else trip.arcs[idx]
    return shapely.linestrings(np.stack([trip.nodes_xy[a[:, 0]], trip.nodes_xy[a[:, 1]]], axis=1))

def reference_arcs(trip):
    route = trip.route
    d = shapely.distance(shapely.points(trip.xy)[:, None], arc_lines(trip, route)[None, :])
    n, r = d.shape
    cost = np.empty_like(d)
    back = np.zeros((n, r), dtype=int)
    cost[0] = d[0]
    idx = np.arange(r)
    for i in range(1, n):
        pm = np.minimum.accumulate(cost[i - 1])
        am = np.maximum.accumulate(np.where(cost[i - 1] == pm, idx, 0))
        cost[i] = d[i] + pm
        back[i] = am
    j = int(cost[-1].argmin())
    path = np.empty(n, dtype=int)
    for i in range(n - 1, -1, -1):
        path[i] = j
        j = back[i, j]
    return route[path]
