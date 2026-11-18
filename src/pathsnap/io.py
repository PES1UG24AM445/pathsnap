"""Dataset I/O and projection utilities for PathSnap.

Owned by: Rishi (feat/rishi)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from pyproj import Transformer

from .config import RAW


@dataclass
class Trip:
    tid: str
    part: str
    xy: np.ndarray          # GPS, projected UTM metres (n, 2)
    t: np.ndarray           # timestamp seconds (n,)
    lonlat: np.ndarray      # GPS lon, lat (n, 2)
    nodes_xy: np.ndarray    # projected road nodes metres (N, 2)
    nodes_ll: np.ndarray    # road nodes lon, lat (N, 2)
    arcs: np.ndarray        # (M, 2) int from_node, to_node; row index = arc_id
    route: Optional[np.ndarray] = None  # reference arc_id sequence or None


def partition_of(rel: Path) -> str:
    """Determine partition ('train', 'student', or 'evaluator') from relative file path."""
    s = str(rel).lower()
    if 'eval' in s:
        return 'evaluator'
    if 'student' in s:
        return 'student'
    return 'train'


def find_trips(root: Path = RAW) -> List[Dict[str, Any]]:
    """Scan raw directory for trip trajectory and road network files."""
    root = Path(root)
    if not root.exists():
        return []

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


def _tf(lon: float, lat: float) -> Transformer:
    """Compute UTM zone dynamically and return Transformer from WGS84 (EPSG:4326) to UTM."""
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return Transformer.from_crs(4326, epsg, always_xy=True)


def _read(p: Path, dtype: type = float) -> np.ndarray:
    """Read whitespace-delimited numerical table into a 2D numpy array."""
    return np.loadtxt(p, dtype=dtype, ndmin=2)


def load_trip(info: Dict[str, Any], part: Optional[str] = None, with_route: bool = True) -> Trip:
    """Load, project, and clean raw GPS and road network files for a single trip."""
    g = _read(info['track'])  # lon, lat, t
    g = g[np.isfinite(g).all(axis=1)]
    g = g[(np.abs(g[:, 0]) <= 180) & (np.abs(g[:, 1]) <= 90)]

    nd = _read(info['nodes'])  # lon, lat
    ar = _read(info['arcs'], int)  # from, to

    # Automatic local UTM projection
    tf = _tf(float(nd[:, 0].mean()), float(nd[:, 1].mean()))
    gx, gy = tf.transform(g[:, 0], g[:, 1])
    nx, ny = tf.transform(nd[:, 0], nd[:, 1])

    route = None
    if with_route and info.get('route') is not None:
        route = _read(info['route'], int).ravel()

    return Trip(
        tid=info['tid'],
        part=part or info.get('part', 'train'),
        xy=np.c_[gx, gy],
        t=g[:, 2],
        lonlat=g[:, :2],
        nodes_xy=np.c_[nx, ny],
        nodes_ll=nd[:, :2],
        arcs=ar[:, :2],
        route=route
    )
