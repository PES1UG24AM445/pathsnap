from dataclasses import dataclass
from pathlib import Path
import numpy as np
from pyproj import Transformer
from .config import RAW

@dataclass
class Trip:
    tid: str
    part: str
    xy: np.ndarray        # GPS, projected metres (n,2)
    t: np.ndarray         # seconds (n,)
    lonlat: np.ndarray    # GPS lon, lat (n,2)
    nodes_xy: np.ndarray  # projected metres (N,2)
    nodes_ll: np.ndarray  # lon, lat (N,2)
    arcs: np.ndarray      # (M,2) int from_node, to_node; row index = arc_id
    route: object         # reference arc_id sequence or None

def partition_of(rel):
    s = str(rel).lower()
    if 'eval' in s:
        return 'evaluator'
    if 'student' in s:
        return 'student'
    return 'train'

def find_trips(root=RAW):
    root = Path(root)
    out = []
    for t in sorted(root.rglob('*.track')):
        d, stem = t.parent, t.stem
        def sib(ext):
            p = d / (stem + ext)
            if p.exists():
                return p
            c = sorted(d.glob('*' + ext))
            return c[0] if c else None
        rel = t.relative_to(root)
        out.append(dict(
            tid='__'.join(rel.with_suffix('').parts),
            part=partition_of(rel),
            track=t,
            nodes=sib('.nodes'),
            arcs=sib('.arcs'),
            route=sib('.route')
        ))
    return out

def _tf(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return Transformer.from_crs(4326, (32600 if lat >= 0 else 32700) + zone, always_xy=True)

def _read(p, dtype=float):
    return np.loadtxt(p, dtype=dtype, ndmin=2)

def load_trip(info, part=None, with_route=True):
    g = _read(info['track'])  # lon, lat, t
    g = g[np.isfinite(g).all(axis=1)]
    g = g[(np.abs(g[:, 0]) <= 180) & (np.abs(g[:, 1]) <= 90)]
    nd = _read(info['nodes'])  # lon, lat
    ar = _read(info['arcs'], int)  # from, to
    tf = _tf(nd[:, 0].mean(), nd[:, 1].mean())
    gx, gy = tf.transform(g[:, 0], g[:, 1])
    nx, ny = tf.transform(nd[:, 0], nd[:, 1])
    route = None
    if with_route and info.get('route') is not None:
        route = _read(info['route'], int).ravel()
    return Trip(
        info['tid'],
        part or info['part'],
        np.c_[gx, gy],
        g[:, 2],
        g[:, :2],
        np.c_[nx, ny],
        nd[:, :2],
        ar[:, :2],
        route
    )
