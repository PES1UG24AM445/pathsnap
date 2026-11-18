"""Summarize training and student-test GPS data and networks."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap.config import RES
from pathsnap.io import _read, find_trips, load_trip

rows = []
for info in find_trips():
    if info["part"] not in ("train", "student"):
        continue
    gps = _read(info["track"])
    trip = load_trip(info)
    graph = csr_matrix((np.ones(len(trip.arcs)), trip.arcs.T),
                       shape=(len(trip.nodes_xy), len(trip.nodes_xy)))
    dt = np.diff(trip.t)
    rows.append({
        "tid": trip.tid, "partition": trip.part, "raw_observations": len(gps),
        "valid_observations": len(trip.xy), "removed_observations": len(gps) - len(trip.xy),
        "nonfinite_gps_rows": int((~np.isfinite(gps).all(axis=1)).sum()),
        "nonincreasing_raw_steps": int((np.diff(gps[:, 2]) <= 0).sum()),
        "max_gap_seconds": max(dt, default=0), "nodes": len(trip.nodes_xy),
        "arcs": len(trip.arcs), "weak_components": connected_components(graph, directed=True)[0],
        "reference_arcs": len(trip.route) if trip.route is not None else 0,
        "span_x_m": np.ptp(trip.xy[:, 0]), "span_y_m": np.ptp(trip.xy[:, 1]),
    })
if not rows:
    raise SystemExit("No training or student-test trips found.")
summary = pd.DataFrame(rows)
summary.to_csv(RES / "eda_inventory.csv", index=False)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
for ax, key, title in zip(axes, ("valid_observations", "max_gap_seconds", "arcs"),
                          ("Observations per trip", "Longest GPS gap (s)", "Road arcs per trip")):
    for part, trips in summary.groupby("partition"):
        ax.hist(trips[key], alpha=0.55, label=part)
    ax.set_title(title)
axes[0].set_ylabel("Trips")
axes[-1].legend()
fig.savefig(RES / "figs" / "eda_inventory.png", dpi=160)
