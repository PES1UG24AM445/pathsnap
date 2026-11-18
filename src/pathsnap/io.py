from dataclasses import dataclass
from pathlib import Path
import re
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

PARTITIONS = {
    "train": {"train", "training", "training data", "train data", "training trips"},
    "student": {"student", "student test", "student testing", "student testing data"},
    "evaluator": {"eval", "evaluation", "evaluator", "evaluator test", "evaluator testing",
                  "evaluator testing data"},
}


def partition_of(rel):
    parts = {re.sub(r"[^a-z0-9]+", " ", p.lower()).strip() for p in rel.parent.parts}
    matches = [name for name, aliases in PARTITIONS.items() if parts & aliases]
    if len(matches) != 1:
        raise ValueError(f"Cannot determine partition for {rel}. Use train, student, or evaluator folders.")
    return matches[0]


def find_trips(root=RAW, *, partition=None):
    root = Path(root)
    out = []
    for track in sorted(root.rglob("*.track")):
        rel = track.relative_to(root)
        part = partition if partition is not None else partition_of(rel)
        out.append({"tid": "__".join(rel.with_suffix("").parts), "part": part, "track": track})
        if part == "evaluator":
            continue
        for ext, key in ((".nodes", "nodes"), (".arcs", "arcs"), (".route", "route")):
            exact = track.with_suffix(ext)
            if exact.is_file():
                out[-1][key] = exact
                continue
            siblings = list(track.parent.glob("*" + ext))
            single_track = len(list(track.parent.glob("*.track"))) == 1
            if len(siblings) == 1 and single_track:
                out[-1][key] = siblings[0]
            elif siblings or key != "route":
                raise ValueError(f"{track}: missing or ambiguous {ext} file")
            else:
                out[-1][key] = None
    tids = [(info["part"], info["tid"]) for info in out]
    if len(set(tids)) != len(tids):
        raise ValueError(f"{root}: duplicate trip identifiers")
    return out


def _tf(lon, lat):
    zone = min(60, int((lon + 180) // 6) + 1)
    return Transformer.from_crs(4326, (32600 if lat >= 0 else 32700) + zone, always_xy=True)

def _read(p, dtype=float):
    return np.loadtxt(p, dtype=dtype, ndmin=2)

def load_trip(info, part=None, with_route=True):
    g = _read(info['track'])  # lon, lat, t
    g = g[np.isfinite(g).all(axis=1)]
    g = g[(np.abs(g[:, 0]) <= 180) & (np.abs(g[:, 1]) <= 90)]
    if not len(g):
        raise ValueError(f"{info['tid']}: no valid GPS observations")
    g = g[np.argsort(g[:, 2], kind="stable")]
    nd = _read(info['nodes'])  # lon, lat
    ar = _read(info['arcs'], int)  # from, to
    if (not len(nd) or not len(ar) or not np.isfinite(nd[:, :2]).all()
            or (np.abs(nd[:, 0]) > 180).any() or (np.abs(nd[:, 1]) > 90).any()
            or (ar[:, :2] < 0).any() or (ar[:, :2] >= len(nd)).any()):
        raise ValueError(f"{info['tid']}: invalid road network")
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
