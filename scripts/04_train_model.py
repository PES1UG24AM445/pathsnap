"""Train and evaluate a specific PathSnap model.

Owned by: Sanjit (feat/sanjit)
Usage:
    python scripts/04_train_model.py logreg
    python scripts/04_train_model.py knn
    python scripts/04_train_model.py rf
    python scripts/04_train_model.py lgbm
    python scripts/04_train_model.py mlp
"""

import importlib
import sys
from pathlib import Path

# Add project root and src to path to ensure robust imports regardless of invocation cwd
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from pathsnap.experiment import evaluate

AVAILABLE_MODELS = ['logreg', 'knn', 'rf', 'lgbm', 'mlp']


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(f"Usage: python scripts/04_train_model.py <model_name>")
        print(f"Available models: {', '.join(AVAILABLE_MODELS)}")
        sys.exit(0)

    model_arg = sys.argv[1].lower().replace('.py', '')

    try:
        mod = importlib.import_module(f"pathsnap.models.{model_arg}")
    except ModuleNotFoundError as e:
        print(f"\n[ERROR] Could not load model module 'pathsnap.models.{model_arg}': {e}")
        print(f"Please verify the model name. Available models: {', '.join(AVAILABLE_MODELS)}\n")
        sys.exit(1)

    print(f"\n>>> Loaded model: {mod.NAME} [{mod.FAMILY}]")
    print(f">>> Hyperparameters: {mod.CONFIG}")
    cap = getattr(mod, 'MAX_TRAIN_ROWS', None)
    if cap:
        print(f">>> Training row subsampling cap: {cap}")

    try:
        res = evaluate(
            name=mod.NAME,
            family=mod.FAMILY,
            config=mod.CONFIG,
            make=mod.make,
            cap=cap
        )
        print("\n" + "=" * 60)
        print(f"SUCCESS: Evaluated and saved model '{mod.NAME}'")
        print(f"  5-Fold CV Top-1 Accuracy : {res['cv_top1']:.4f} (±{res['cv_top1_std']:.4f})")
        print(f"  Student-Test Top-1 Acc   : {res['st_top1']:.4f}")
        print(f"  Student-Test Macro F1    : {res['st_macro_f1']:.4f}")
        print(f"  Route Continuity         : {res['st_route_continuity']:.4f}")
        print(f"  Inference Latency        : {res['infer_ms_per_1k_rows']:.2f} ms / 1k rows")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\n[ERROR] Training failed for model '{model_arg}': {e}")
        raise e


if __name__ == '__main__':
    main()
