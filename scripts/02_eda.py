import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from pathsnap.io import find_trips, load_trip, _read
from pathsnap.config import RES

rows = []
for info in find_trips():
    if info['part'] == 'evaluator':
        continue
    raw = _read(info['track'])
    tr = load_trip(info)
    dt = np.diff(raw[:, 2])
    A, N = tr.arcs, len(tr.nodes_xy)
    g = csr_matrix((np.ones(len(A)), (A[:, 0], A[:, 1])), shape=(N, N))
    ncomp, _ = connected_components(g, directed=False)
    deg = np.bincount(A.ravel(), minlength=N)
    r = tr.route
    ok = r is not None and len(r) > 0 and r.min() >= 0 and r.max() < len(A)
    rows.append(dict(
        tid=tr.tid,
        part=tr.part,
        n_obs=len(raw),
        n_nan=int(np.isnan(raw).sum()),
        n_bad_range=int(((np.abs(raw[:, 0]) > 180) | (np.abs(raw[:, 1]) > 90)).sum()),
        duration_s=raw[:, 2].max() - raw[:, 2].min(),
        pct_dt_1s=float((dt == 1).mean()) if dt.size else np.nan,
        max_dt=float(dt.max()) if dt.size else 0.0,
        n_dt_nonpositive=int((dt <= 0).sum()),
        extent_x_km=np.ptp(tr.xy[:, 0]) / 1000,
        extent_y_km=np.ptp(tr.xy[:, 1]) / 1000,
        n_nodes=N,
        n_arcs=len(A),
        n_components=ncomp,
        n_isolated=int((deg == 0).sum()),
        route_len=-1 if r is None else len(r),
        route_ids_valid=bool(ok),
        route_connected_pct=float((A[r[:-1], 1] == A[r[1:], 0]).mean()) if ok and len(r) > 1 else np.nan
    ))

if not rows or not any(row["part"] == "train" for row in rows):
    raise SystemExit("Place the official training and student-test data in data/raw first.")
df = pd.DataFrame(rows)
df.to_csv(RES / 'eda_inventory.csv', index=False)
print(df.groupby('part').size())
print(df.describe().T[['count', 'mean', 'std', 'min', '50%', 'max']].to_string())

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
df.n_obs.hist(ax=ax[0])
ax[0].set_title('GPS points per trip')
df.max_dt.hist(ax=ax[1])
ax[1].set_title('Max time gap (s)')
df.n_arcs.hist(ax=ax[2])
ax[2].set_title('Road arcs per trip')
plt.tight_layout()
plt.savefig(RES / 'figs' / 'eda_hist.png', dpi=150)

info = [i for i in find_trips() if i['part'] == 'train'][0]
tr = load_trip(info)
plt.figure(figsize=(7, 7))
plt.scatter(*tr.nodes_xy.T, s=1, c='lightgrey', label='road nodes')
plt.plot(*tr.xy.T, 'b.', ms=2, label='GPS')
plt.legend()
plt.axis('equal')
plt.title(tr.tid + ' (train)')
plt.savefig(RES / 'figs' / 'eda_trip.png', dpi=150)
