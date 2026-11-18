"""Compare all trained PathSnap models, generate summary tables and visualization plots.

Owned by: Sanjit (feat/sanjit)
Generates:
- results/model_comparison.csv (Matches Deliverables.pdf Section 3.1, Table 3.1)
- results/figs/cmp_cv_macro_f1.png (5-fold Group CV Macro F1 with std error bars)
- results/figs/cmp_student_top1_f1.png (Student-test Top-1 Accuracy and Macro F1 comparison)
- results/figs/cmp_log_loss.png (Student-test binary log loss)
- results/figs/confusion_matrices.png (Side-by-side student-test confusion matrices)
"""

import glob
import json
import sys
from pathlib import Path

# Add project root and src to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from pathsnap.config import RES

# Ensure figures output directory exists
FIG_DIR = RES / 'figs'
FIG_DIR.mkdir(parents=True, exist_ok=True)


def finish_plot(title: str, fname: str, x: np.ndarray, labels: list):
    """Format, label, and export matplotlib figures cleanly."""
    plt.xticks(x, labels, rotation=25, ha='right', fontsize=10)
    plt.title(title, fontsize=12, fontweight='bold', pad=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=180)
    plt.close()
    print(f"Saved figure: {FIG_DIR / fname}")


def main():
    model_json_files = sorted(glob.glob(str(RES / 'models' / '*.json')))
    # Exclude non-model or tuning files if any
    model_json_files = [f for f in model_json_files if not f.endswith('best_params.json')]

    if not model_json_files:
        print(f"[WARNING] No model result files found in {RES / 'models'}. Please train models first using 04_train_model.py.")
        sys.exit(0)

    rows = []
    for f in model_json_files:
        try:
            with open(f, 'r') as fp:
                rows.append(json.load(fp))
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("[ERROR] DataFrame is empty after reading model JSON files.")
        sys.exit(1)

    # Rank by primary selection metric (cv_top1 descending)
    df = df.sort_values('cv_top1', ascending=False).reset_index(drop=True)

    # Export machine-readable internal comparison
    df.to_csv(RES / 'model_comparison_raw.csv', index=False)

    # Format Table 3.1 strictly according to Deliverables.pdf Section 3.1
    # Required columns: Model, Algorithm Family, Key Configuration, CV Macro F1-score, Top-1 Accuracy,
    # Route continuity, Student-Test Macro F1-score, Log Loss, Inference Time
    report_table = pd.DataFrame({
        'Model': df['model'],
        'Algorithm Family': df['family'],
        'Key Configuration': df['config'],
        'CV Macro F1-score': df.apply(lambda r: f"{r['cv_macro_f1']:.4f} ± {r['cv_macro_f1_std']:.4f}", axis=1),
        'Top-1 Accuracy': df['st_top1'].apply(lambda v: f"{v:.4f}"),
        'Route continuity': df['st_route_continuity'].apply(lambda v: f"{v:.4f}"),
        'Student-Test Macro F1-score': df['st_macro_f1'].apply(lambda v: f"{v:.4f}"),
        'Log Loss': df['st_log_loss'].apply(lambda v: f"{v:.4f}"),
        'Inference Time (ms/1k)': df['infer_ms_per_1k_rows'].apply(lambda v: f"{v:.2f}"),
        'Train Time (s)': df['train_seconds'].apply(lambda v: f"{v:.2f}")
    })

    report_table.to_csv(RES / 'model_comparison.csv', index=False)
    print("\n" + "=" * 90)
    print("CONSOLIDATED MODEL COMPARISON TABLE (Table 3.1)")
    print("=" * 90)
    print(report_table.to_string(index=False))
    print("=" * 90 + "\n")

    # Generate Required Comparison Figures
    n_models = len(df)
    x = np.arange(n_models)
    model_names = df['model'].tolist()

    # 1. 5-Fold Group CV Macro F1 Comparison (with error bars)
    plt.figure(figsize=(max(7, n_models * 1.5), 4.5))
    plt.bar(x, df['cv_macro_f1'], yerr=df['cv_macro_f1_std'], capsize=4, color='#2b5c8f', edgecolor='black', alpha=0.85)
    plt.ylabel('Macro F1-Score', fontsize=11)
    finish_plot('5-Fold Group CV Macro F1 (80 Training Trips)', 'cmp_cv_macro_f1.png', x, model_names)

    # 2. Student-Test: Top-1 vs Macro F1 Comparison
    plt.figure(figsize=(max(8, n_models * 1.6), 4.5))
    bar_width = 0.35
    plt.bar(x - bar_width / 2, df['st_top1'], width=bar_width, label='Top-1 Accuracy (Primary)', color='#2ca02c', edgecolor='black', alpha=0.85)
    plt.bar(x + bar_width / 2, df['st_macro_f1'], width=bar_width, label='Macro F1-Score (Threshold 0.5)', color='#ff7f0e', edgecolor='black', alpha=0.85)
    plt.ylabel('Score', fontsize=11)
    plt.legend(frameon=True)
    finish_plot('Student-Test Performance: Top-1 vs Macro F1', 'cmp_student_top1_f1.png', x, model_names)

    # 3. Student-Test Log Loss Comparison
    plt.figure(figsize=(max(7, n_models * 1.5), 4.5))
    plt.bar(x, df['st_log_loss'], color='#d62728', edgecolor='black', alpha=0.85)
    plt.ylabel('Binary Log Loss (lower is better)', fontsize=11)
    finish_plot('Student-Test Log Loss Across Models', 'cmp_log_loss.png', x, model_names)

    # 4. Side-by-Side Confusion Matrices
    available_preds = [name for name in df['model'] if (RES / 'preds' / f"{name}_student.parquet").exists()]
    if available_preds:
        fig, axes = plt.subplots(1, len(available_preds), figsize=(4.0 * len(available_preds), 3.8))
        if len(available_preds) == 1:
            axes = [axes]
        
        for ax, name in zip(axes, available_preds):
            pred_file = RES / 'preds' / f"{name}_student.parquet"
            p_df = pd.read_parquet(pred_file)
            cm = confusion_matrix(p_df['label'], p_df['p'] >= 0.5)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0 (Other)', '1 (Ref)'])
            disp.plot(ax=ax, colorbar=False, cmap='Blues')
            ax.set_title(name, fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'confusion_matrices.png', dpi=180)
        plt.close()
        print(f"Saved figure: {FIG_DIR / 'confusion_matrices.png'}")


if __name__ == '__main__':
    main()
