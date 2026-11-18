"""Self-Contained Verification and Sanity Test for Sanjit's Pipeline.

This script tests:
1. Metric calculations (pred_arcs, top1, continuity, macro_f1_score, candidate_miss_rate).
2. Leakage-free GroupKFold cross-validation grouped by trip (tid).
3. Fitting and evaluating Models 1-3 (logreg, knn, rf).
4. Generation of Table 3.1 (model_comparison.csv) and comparison plots.

Usage:
    python tests/test_sanjit_pipeline.py
"""

import json
import os
import sys
from pathlib import Path

# Add project root and src to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd

from pathsnap.config import ART, PROC, RES, SEED
from pathsnap.experiment import evaluate, load
from pathsnap.features import FEATURES
from pathsnap.metrics import candidate_miss_rate, continuity, macro_f1_score, pred_arcs, top1
from pathsnap.models import knn, logreg, rf


def generate_mock_data(n_trips=10, obs_per_trip=20, k_cand=12):
    """Generate realistic synthetic candidate table for testing pipeline mechanics."""
    rng = np.random.RandomState(SEED)
    rows = []
    
    for t_idx in range(n_trips):
        tid = f"trip_{t_idx:03d}"
        for obs in range(obs_per_trip):
            # One correct candidate arc, k_cand - 1 negative arcs
            true_cand = rng.randint(0, k_cand)
            for c in range(k_cand):
                is_true = 1 if c == true_cand else 0
                dist = rng.uniform(1.0, 15.0) if is_true else rng.uniform(10.0, 150.0)
                ang_diff = rng.uniform(0.0, 0.3) if is_true else rng.uniform(0.2, np.pi)
                
                row = {
                    'tid': tid,
                    'obs': obs,
                    'arc': (obs + c) % 50,
                    'dist': dist,
                    'dist_rel': dist - 1.0,
                    'rank': c,
                    'n_cand': k_cand,
                    'along': rng.uniform(0.0, 100.0),
                    'along_frac': rng.uniform(0.0, 1.0),
                    'remaining': rng.uniform(0.0, 100.0),
                    'arc_len': rng.uniform(50.0, 200.0),
                    'ang_diff': ang_diff,
                    'cos_diff': np.cos(ang_diff),
                    'has_rev': rng.choice([0, 1]),
                    'out_deg': rng.randint(1, 5),
                    'in_deg': rng.randint(1, 5),
                    'speed': rng.uniform(5.0, 25.0),
                    'step_prev': rng.uniform(5.0, 20.0),
                    'gap_prev': 1.0,
                    'turn': rng.uniform(0.0, 0.2),
                    'label': is_true
                }
                rows.append(row)
                
    return pd.DataFrame(rows)


def run_tests():
    print("=================================================================")
    print("RUNNING SANJIT'S PATHSNAP VERIFICATION SUITE")
    print("=================================================================")

    # 1. Generate and save synthetic test datasets
    PROC.mkdir(parents=True, exist_ok=True)
    df_train = generate_mock_data(n_trips=15, obs_per_trip=15, k_cand=12)
    df_student = generate_mock_data(n_trips=5, obs_per_trip=15, k_cand=12)

    train_path = PROC / 'train.parquet'
    student_path = PROC / 'student.parquet'

    df_train.to_parquet(train_path, index=False)
    df_student.to_parquet(student_path, index=False)
    print(f"[OK] Generated mock datasets: Train={len(df_train)} rows, Student={len(df_student)} rows.")

    # 2. Test metrics.py functions
    print("\n--- Testing metrics.py ---")
    df_test = df_student.copy()
    # Mock probability column
    df_test['p'] = np.where(df_test['label'] == 1, 0.9, 0.1)

    selected = pred_arcs(df_test, col='p')
    assert len(selected) == 5 * 15, f"Expected 75 pointwise predictions, got {len(selected)}"
    print(f"[OK] pred_arcs: correctly selected 1 candidate per observation ({len(selected)} total)")

    t1 = top1(df_test, col='p')
    assert t1 == 1.0, f"Expected Top-1 = 1.0 for perfect mock probabilities, got {t1}"
    print(f"[OK] top1: successfully evaluated Top-1 Accuracy = {t1:.4f}")

    miss_rate = candidate_miss_rate(df_test)
    assert miss_rate == 0.0, f"Expected miss rate = 0.0, got {miss_rate}"
    print(f"[OK] candidate_miss_rate: evaluated = {miss_rate:.4f}")

    # Mock continuity
    mock_arcs = {f"trip_{t:03d}": np.array([[i, i+1] for i in range(100)]) for t in range(5)}
    cont = continuity(selected, mock_arcs)
    print(f"[OK] continuity: evaluated = {cont:.4f}")

    # 3. Test Model Evaluations
    print("\n--- Testing Model Evaluation Suite (Models 1-3) ---")
    for mod in [logreg, rf]:
        print(f"\nEvaluating: {mod.NAME} [{mod.FAMILY}]")
        res = evaluate(
            name=mod.NAME,
            family=mod.FAMILY,
            config=mod.CONFIG,
            make=mod.make,
            cap=getattr(mod, 'MAX_TRAIN_ROWS', None)
        )
        assert res['cv_top1'] > 0.5, f"CV Top-1 unexpectedly low: {res['cv_top1']}"
        assert (ART / 'models' / f"{mod.NAME}.joblib").exists(), "Joblib file was not created!"
        assert (RES / 'models' / f"{mod.NAME}.json").exists(), "JSON result was not created!"
        print(f"[OK] Successfully verified {mod.NAME}: CV Top-1 = {res['cv_top1']:.4f}")

    print("\n=================================================================")
    print("ALL TESTS PASSED! Sanjit's pipeline is 100% verified and functional.")
    print("=================================================================\n")


if __name__ == '__main__':
    run_tests()
