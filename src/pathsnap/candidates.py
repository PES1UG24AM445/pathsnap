import numpy as np
import pandas as pd
import shapely
from .config import RADIUS_M, K_CAND
from .labels import arc_lines

def build_candidates(trip, radius=RADIUS_M, k=K_CAND):
    lines = arc_lines(trip)
    tree = shapely.STRtree(lines)
    pts = shapely.points(trip.xy)
    qi, ai = tree.query(pts, predicate='dwithin', distance=radius)
    miss = np.setdiff1d(np.arange(len(pts)), qi)
    if len(miss):  # no arc within radius: take nearest arc
        qi = np.r_[qi, miss]
        ai = np.r_[ai, tree.nearest(pts[miss])]
    d = shapely.distance(pts[qi], lines[ai])
    al = shapely.line_locate_point(lines[ai], pts[qi])
    df = pd.DataFrame(dict(obs=qi, arc=ai, dist=d, along=al)).sort_values(['obs', 'dist'])
    df = df.groupby('obs', sort=True).head(k).reset_index(drop=True)
    df['rank'] = df.groupby('obs').cumcount()
    return df
