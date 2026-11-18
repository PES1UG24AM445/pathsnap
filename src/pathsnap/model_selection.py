"""Model selection and search spaces for the five Build Guide baselines."""

from importlib import import_module
import json
import math

from sklearn.pipeline import Pipeline

from .config import RES

MODULES = {
    "logreg": "logreg",
    "knn": "knn",
    "random_forest": "rf",
    "lightgbm": "lgbm",
    "mlp": "mlp",
}
COMPARISON_COLUMNS = [
    "model", "family", "config", "cv_macro_f1", "cv_top1", "st_top1",
    "st_macro_f1", "st_log_loss", "st_route_continuity", "train_seconds",
    "infer_ms_per_1k_rows",
]


def baseline(name):
    return import_module(f"pathsnap.models.{MODULES[name]}")


def best_baseline():
    rows = [read_result(name) for name in MODULES]
    return min(rows, key=lambda row: (-row["cv_top1"], -row["cv_macro_f1"], row["model"]))


def read_result(name):
    path = RES / "models" / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing experiment result: {path}. Run model training first.")
    row = json.loads(path.read_text(encoding="utf-8"))
    if row["model"] != name or not all(
        math.isfinite(row[key]) and 0 <= row[key] <= 1
        for key in ("cv_top1", "cv_macro_f1")
    ):
        raise ValueError(f"Invalid model identity or CV scores in {path}")
    return row


def search_params(name, trial):
    if name == "logreg":
        return {"C": trial.suggest_float("C", 0.01, 30, log=True)}
    if name == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 5, 45, step=4),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_categorical("p", [1, 2]),
        }
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 8, 24, step=4),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 40, log=True),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.75, 1.0]),
        }
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 550, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.025, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 95),
            "min_child_samples": trial.suggest_int("min_child_samples", 15, 150, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0001, 2, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10, log=True),
        }
    if name == "mlp":
        return {
            "alpha": trial.suggest_float("alpha", 0.00001, 0.03, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 0.0002, 0.004, log=True),
            "max_iter": trial.suggest_int("max_iter", 30, 70, step=10),
        }
    raise ValueError(f"Unknown model: {name}")


def configured_model(name, overrides):
    estimator = baseline(name).make()
    if isinstance(estimator, Pipeline):
        step_name = estimator.steps[-1][0]
        return estimator.set_params(**{f"{step_name}__{key}": value for key, value in overrides.items()})
    return estimator.set_params(**overrides)


def validate_table(df, features, partition):
    import numpy as np

    expected = {"train": 80, "student": 10}[partition]
    if df["tid"].isna().any() or df["tid"].nunique() != expected:
        raise ValueError(f"{partition}: expected exactly {expected} distinct named trips")
    if df[["obs", "arc"]].isna().any().any() or df.duplicated(["tid", "obs", "arc"]).any():
        raise ValueError(f"{partition}: missing or duplicate candidate keys")
    if not df["label"].isin([0, 1]).all() or df["label"].nunique() != 2:
        raise ValueError(f"{partition}: expected known binary candidate labels")
    if df.groupby(["tid", "obs"])["label"].sum().gt(1).any():
        raise ValueError(f"{partition}: multiple positive arcs for an observation")
    if not np.isfinite(df[features].to_numpy(dtype=float)).all():
        raise ValueError(f"{partition}: candidate features must be finite")
