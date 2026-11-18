"""Global configuration and paths for PathSnap."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data' / 'raw'
PROC = ROOT / 'data' / 'processed'
ART = ROOT / 'artifacts'
RES = ROOT / 'results'
SEED = 42
RADIUS_M = 150.0
K_CAND = 12

for _p in (
    PROC,
    ART / 'models',
    RES / 'figs',
    RES / 'maps',
    RES / 'models',
    RES / 'preds',
    RES / 'oof',
):
    _p.mkdir(parents=True, exist_ok=True)
