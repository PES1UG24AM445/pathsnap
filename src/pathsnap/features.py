import numpy as np
from .candidates import build_candidates
from .labels import reference_arcs

FEATURES = [
    'dist', 'dist_rel', 'rank', 'n_cand', 'along_frac', 'remaining', 'arc_len',
    'ang_diff', 'cos_diff', 'has_rev', 'out_deg', 'in_deg',
    'speed', 'step_prev', 'gap_prev', 'turn'
]

def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def build_table(trip, with_labels=True):
    c = build_candidates(trip)
    n = len(trip.xy)
    x, y, t = trip.xy[:, 0], trip.xy[:, 1], trip.t
    i = np.arange(n)
    a, b = np.maximum(i - 1, 0), np.minimum(i + 1, n - 1)
    hx, hy = x[b] - x[a], y[b] - y[a]
    speed = np.hypot(hx, hy) / np.maximum(t[b] - t[a], 1e-3)
    head = np.arctan2(hy, hx)
    turn = np.abs(_wrap(head - np.r_[head[0], head[:-1]]))
    step_prev = np.r_[0, np.hypot(np.diff(x), np.diff(y))]
    gap_prev = np.r_[0, np.diff(t)]

    A = trip.arcs
    N = len(trip.nodes_xy)
    s, e = trip.nodes_xy[A[:, 0]], trip.nodes_xy[A[:, 1]]
    dx, dy = e[:, 0] - s[:, 0], e[:, 1] - s[:, 1]
    arc_len, bearing = np.hypot(dx, dy), np.arctan2(dy, dx)

    key = A[:, 0].astype(np.int64) * N + A[:, 1]
    rkey = A[:, 1].astype(np.int64) * N + A[:, 0]
    has_rev = np.isin(key, rkey).astype(int)  # reverse-direction arc exists
    out_deg = np.bincount(A[:, 0], minlength=N)[A[:, 1]]  # arcs leaving this arc's end node
    in_deg = np.bincount(A[:, 1], minlength=N)[A[:, 0]]   # arcs entering this arc's start node

    o, ar = c['obs'].to_numpy(), c['arc'].to_numpy()
    c['dist_rel'] = c['dist'] - c.groupby('obs')['dist'].transform('min')
    c['n_cand'] = c.groupby('obs')['arc'].transform('size')
    c['arc_len'] = arc_len[ar]
    c['along_frac'] = (c['along'] / np.maximum(arc_len[ar], 1e-6)).clip(0, 1)
    c['remaining'] = arc_len[ar] - c['along']
    ang = np.abs(_wrap(head[o] - bearing[ar]))
    c['ang_diff'], c['cos_diff'] = ang, np.cos(ang)
    c['has_rev'], c['out_deg'], c['in_deg'] = has_rev[ar], out_deg[ar], in_deg[ar]
    c['speed'], c['step_prev'], c['gap_prev'], c['turn'] = speed[o], step_prev[o], gap_prev[o], turn[o]
    c.insert(0, 'tid', trip.tid)
    c['label'] = -1
    if with_labels and trip.route is not None:
        c['label'] = (ar == reference_arcs(trip)[o]).astype(int)
    return c
