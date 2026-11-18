"""Unified Experiment and Cross-Validation Harness for PathSnap.

Owned by: Sanjit (feat/sanjit)
Responsibilities:
- 5-fold GroupKFold by trip (tid) on 80 training trips (prevents spatial/temporal leakage).
- Fit on full training data, evaluate on student-testing trips (10 trips).
- Evaluate candidate-level Macro F1, Observation-level Top-1 Accuracy, Route Continuity, Log Loss.
- Benchmark training time and inference latency (ms / 1k rows).
- Compute in-sample train score (first 20 trips) for overfitting/underfitting diagnostic (Report Section 3.2, Q.C9).
- Cache network arc graphs to avoid repetitive disk I/O.
- Persist model joblib, per-fold metrics, OOF predictions, and test predictions.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

from .config import ART, PROC, RES, SEED
from .metrics import continuity, macro_f1_score, pred_arcs, top1

# Gracefully import FEATURES; fallback to frozen standard list if features.py is pending merge
try:
    from .features import FEATURES
except ImportError:
    FEATURES = [
        'dist', 'dist_rel', 'rank', 'n_cand', 'along_frac', 'remaining', 'arc_len',
        'ang_diff', 'cos_diff', 'has_rev', 'out_deg', 'in_deg',
        'speed', 'step_prev', 'gap_prev', 'turn'
    ]

# Columns to persist in prediction parquet files for downstream HMM decoding
KEEP = ['tid', 'obs', 'arc', 'label', 'along', 'arc_len']

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger("pathsnap.experiment")

# In-memory cache for road network arcs to avoid redundant disk parsing
_ARCS_CACHE: Dict[str, Dict[str, np.ndarray]] = {}


def load(part: str) -> pd.DataFrame:
    """Load preprocessed candidate parquet dataset for a given partition ('train' or 'student')."""
    parquet_path = PROC / f"{part}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Candidate table {parquet_path} not found! "
            "Please run 'python scripts/03_build_features.py' first."
        )
    return pd.read_parquet(parquet_path).reset_index(drop=True)


def arcs_by_tid(part: str) -> Dict[str, np.ndarray]:
    """Retrieve road network arcs by trip ID with memory caching.
    
    Returns a dict mapping: trip_id -> arcs array (M, 2).
    """
    global _ARCS_CACHE
    if part in _ARCS_CACHE:
        return _ARCS_CACHE[part]

    try:
        from .io import find_trips, load_trip
        trips = {
            i['tid']: load_trip(i, with_route=False).arcs
            for i in find_trips()
            if i.get('part') == part
        }
        _ARCS_CACHE[part] = trips
        return trips
    except Exception as e:
        logger.warning("Could not load raw trips for arcs_by_tid('%s'): %s. Continuity may be unavailable.", part, e)
        return {}


def _fit(make: Callable[[], Any], df: pd.DataFrame, cap: Optional[int] = None) -> Any:
    """Instantiate and fit a model, with optional random subsampling for expensive algorithms (e.g., KNN)."""
    if cap and len(df) > cap:
        logger.info("Subsampling training rows from %d to %d for computational tractability.", len(df), cap)
        df_fit = df.sample(cap, random_state=SEED)
    else:
        df_fit = df

    model = make()
    model.fit(df_fit[FEATURES], df_fit['label'])
    return model


def cv_oof(make: Callable[[], Any], tr: pd.DataFrame, cap: Optional[int] = None, n_splits: int = 5) -> Tuple[np.ndarray, pd.DataFrame]:
    """Execute n-fold GroupKFold cross-validation grouped strictly by trip ID (tid).
    
    Guarantees that observations from the same trip never appear in both train and validation folds.
    """
    oof = np.zeros(len(tr), dtype=np.float64)
    folds = []
    gkf = GroupKFold(n_splits=n_splits)

    logger.info("Starting %d-fold GroupKFold CV across %d unique trips (%d total rows)...",
                n_splits, tr['tid'].nunique(), len(tr))

    for k, (train_idx, val_idx) in enumerate(gkf.split(tr, groups=tr['tid'])):
        t_fold_start = time.time()
        m = _fit(make, tr.iloc[train_idx], cap)
        
        # Predict positive class probabilities on held-out validation fold
        val_sub = tr.iloc[val_idx]
        p_val = m.predict_proba(val_sub[FEATURES])[:, 1]
        oof[val_idx] = p_val
        
        v = val_sub.assign(p=p_val)
        fold_top1 = top1(v)
        fold_f1 = macro_f1_score(v['label'].to_numpy(), p_val, threshold=0.5)
        
        fold_elapsed = time.time() - t_fold_start
        logger.info("Fold %d/%d completed in %.1fs | Top-1: %.4f | Macro F1: %.4f",
                    k + 1, n_splits, fold_elapsed, fold_top1, fold_f1)

        folds.append(dict(fold=k, macro_f1=fold_f1, top1=fold_top1))

    return oof, pd.DataFrame(folds)


def evaluate(name: str, family: str, config: Dict[str, Any], make: Callable[[], Any], cap: Optional[int] = None) -> Dict[str, Any]:
    """Run full evaluation suite for a candidate model:
    
    1. 5-fold GroupKFold on training data -> records CV Top-1, CV Macro F1.
    2. Fit on full training data -> benchmarks training execution time.
    3. Benchmark inference latency and student-test Top-1, Macro F1, Log Loss, and Route Continuity.
    4. Compute in-sample training performance on 20 trips for overfitting diagnosis.
    5. Persist model artifact (.joblib), metrics JSON, and OOF/test parquet predictions.
    """
    logger.info("=================================================================")
    logger.info("Running evaluation for model: %s (%s)", name, family)
    logger.info("=================================================================")

    tr = load('train')
    st = load('student')

    # Step 1: Group-aware Cross Validation
    oof, folds = cv_oof(make, tr, cap=cap, n_splits=5)

    # Step 2: Fit on full train data
    t0 = time.time()
    m = _fit(make, tr, cap=cap)
    train_seconds = round(time.time() - t0, 3)

    # Step 3: Student-test prediction & latency benchmarking
    t0 = time.time()
    p_st = m.predict_proba(st[FEATURES])[:, 1]
    infer_seconds = time.time() - t0
    infer_ms_per_1k = round((infer_seconds / max(len(st), 1)) * 1e6, 3)

    st_res = st.assign(p=p_st)

    # Step 4: In-sample performance on 20 training trips for overfitting evaluation (Q.C9)
    sample_tids = tr['tid'].unique()[:20]
    tr_sub = tr[tr['tid'].isin(sample_tids)].copy()
    tr_sub['p'] = m.predict_proba(tr_sub[FEATURES])[:, 1]
    train_top1_20trips = top1(tr_sub)

    # Step 5: Route Continuity and Loss computation
    student_arcs = arcs_by_tid('student')
    pred_st = pred_arcs(st_res)
    st_continuity = continuity(pred_st, student_arcs) if student_arcs else 0.0

    st_logloss = float(log_loss(st_res['label'], np.clip(p_st, 1e-6, 1.0 - 1e-6)))
    st_macro_f1 = macro_f1_score(st_res['label'].to_numpy(), p_st, threshold=0.5)
    st_top1_acc = top1(st_res)

    row = {
        'model': name,
        'family': family,
        'config': json.dumps(config),
        'cv_macro_f1': float(folds['macro_f1'].mean()),
        'cv_macro_f1_std': float(folds['macro_f1'].std()),
        'cv_top1': float(folds['top1'].mean()),
        'cv_top1_std': float(folds['top1'].std()),
        'train_top1_20trips': float(train_top1_20trips),
        'st_top1': float(st_top1_acc),
        'st_macro_f1': float(st_macro_f1),
        'st_log_loss': float(st_logloss),
        'st_route_continuity': float(st_continuity),
        'train_seconds': float(train_seconds),
        'infer_ms_per_1k_rows': float(infer_ms_per_1k)
    }

    # Ensure output directories exist
    (ART / 'models').mkdir(parents=True, exist_ok=True)
    (RES / 'models').mkdir(parents=True, exist_ok=True)
    (RES / 'oof').mkdir(parents=True, exist_ok=True)
    (RES / 'preds').mkdir(parents=True, exist_ok=True)

    # Step 6: Serialize model and experimental metrics
    joblib_path = ART / 'models' / f"{name}.joblib"
    joblib.dump(m, joblib_path, compress=3)

    json_path = RES / 'models' / f"{name}.json"
    json_path.write_text(json.dumps(row, indent=2))

    folds_path = RES / 'models' / f"{name}_folds.csv"
    folds.to_csv(folds_path, index=False)

    oof_path = RES / 'oof' / f"{name}.parquet"
    tr_cols = [c for c in KEEP if c in tr.columns] + ['p']
    tr.assign(p=oof)[tr_cols].to_parquet(oof_path, index=False)

    preds_path = RES / 'preds' / f"{name}_student.parquet"
    st_cols = [c for c in KEEP if c in st.columns] + ['p']
    st_res[st_cols].to_parquet(preds_path, index=False)

    logger.info("Evaluation complete for %s:", name)
    logger.info("  CV Top-1: %.4f (±%.4f) | Student Top-1: %.4f",
                row['cv_top1'], row['cv_top1_std'], row['st_top1'])
    logger.info("  CV Macro F1: %.4f | Student Macro F1: %.4f",
                row['cv_macro_f1'], row['st_macro_f1'])
    logger.info("  Student Route Continuity: %.4f | Log Loss: %.4f",
                row['st_route_continuity'], row['st_log_loss'])
    logger.info("  Train Time: %.2fs | Inference: %.2f ms / 1k rows",
                row['train_seconds'], row['infer_ms_per_1k_rows'])

    return row
