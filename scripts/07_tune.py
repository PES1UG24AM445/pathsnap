"""Run or resume a model-specific Bayesian search using grouped training CV."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import optuna
import pandas as pd

from pathsnap.config import PROC, RES, SEED
from pathsnap.experiment import cv_oof, evaluate, load
from pathsnap.features import FEATURES
from pathsnap import model_selection as selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20, help="Target completed trials")
    parser.add_argument("--timeout", type=float, help="Time budget in seconds for this invocation")
    args = parser.parse_args()
    if args.trials < 6 or (args.timeout is not None and args.timeout <= 0):
        parser.error("Use at least six trials and a positive timeout.")

    winner = selection.best_baseline()
    name = winner["model"]
    module = selection.baseline(name)
    cap = getattr(module, "MAX_TRAIN_ROWS", None)
    train, student = load("train"), load("student")
    selection.validate_table(train, FEATURES, "train")
    selection.validate_table(student, FEATURES, "student")
    if set(train["tid"]) & set(student["tid"]):
        raise ValueError("Training and student trip IDs overlap.")

    def objective(trial):
        overrides = selection.search_params(name, trial)
        _, folds = cv_oof(lambda: selection.configured_model(name, overrides), train, cap)
        trial.set_user_attr("cv_macro_f1", float(folds["macro_f1"].mean()))
        trial.set_user_attr("fold_top1", folds["top1"].tolist())
        trial.set_user_attr("fold_macro_f1", folds["macro_f1"].tolist())
        return float(folds["top1"].mean())

    fingerprint = hashlib.sha256()
    sources = [PROC / "train.parquet", Path(__file__), Path(selection.__file__),
               Path(module.__file__), Path(cv_oof.__code__.co_filename)]
    for source in sources:
        with source.open("rb") as stream:
            fingerprint.update(hashlib.file_digest(stream, "sha256").digest())
    study_name = f"{name}-{fingerprint.hexdigest()[:16]}"
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5),
        study_name=study_name,
        storage=f"sqlite:///{(RES / 'optuna.db').resolve().as_posix()}",
        load_if_exists=True,
    )
    complete = optuna.trial.TrialState.COMPLETE
    remaining = args.trials - len(study.get_trials(states=(complete,)))
    started = time.perf_counter()
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, timeout=args.timeout)
    study.trials_dataframe().to_csv(RES / "optuna_trials.csv", index=False)
    completed = study.get_trials(states=(complete,))
    if len(completed) <= 5:
        raise RuntimeError("Search record saved. Resume until at least one TPE-guided trial completes.")

    tuned_name = f"{name}_tuned"
    best_trial = max(
        completed,
        key=lambda trial: (trial.value, trial.user_attrs["cv_macro_f1"], -trial.number),
    )
    params = best_trial.params
    tuned = evaluate(
        tuned_name, f"{module.FAMILY} (Bayesian-tuned)", module.CONFIG | params,
        lambda: selection.configured_model(name, params), cap=cap,
    )
    pd.DataFrame([winner, tuned])[selection.COMPARISON_COLUMNS].to_csv(
        RES / "tuned_vs_untuned.csv", index=False
    )
    metadata = {
        "study": study_name, "baseline": name, "tuned": tuned_name,
        "objective": "mean five-fold trip-grouped training Top-1; ties use CV Macro F1",
        "completed_trials": len(completed), "best_trial": best_trial.number,
        "best_cv_top1": best_trial.value, "best_params": params,
        "best_cv_macro_f1": best_trial.user_attrs["cv_macro_f1"],
        "best_fold_top1": best_trial.user_attrs["fold_top1"],
        "best_fold_macro_f1": best_trial.user_attrs["fold_macro_f1"],
        "full_config": module.CONFIG | params,
        "completed_trial_seconds": sum(t.duration.total_seconds() for t in completed),
        "invocation_seconds_including_refit": time.perf_counter() - started,
        "search_space": {key: str(value) for key, value in best_trial.distributions.items()},
    }
    (RES / "best_params.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Compared {name} and {tuned_name} after {len(completed)} completed trials.")


if __name__ == "__main__":
    main()
