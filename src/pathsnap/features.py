"""Feature definitions and table builder for PathSnap.

Owned by: Rishi (feat/rishi)
Shared Table Contract:
columns: tid, obs, arc, label, along, rank, ...FEATURES
label=-1 means unknown (e.g. unlabelled evaluation trips).
"""

FEATURES = [
    'dist',
    'dist_rel',
    'rank',
    'n_cand',
    'along_frac',
    'remaining',
    'arc_len',
    'ang_diff',
    'cos_diff',
    'has_rev',
    'out_deg',
    'in_deg',
    'speed',
    'step_prev',
    'gap_prev',
    'turn'
]
