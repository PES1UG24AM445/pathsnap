"""Boosted-tree candidate classifier."""

from lightgbm import LGBMClassifier

from ..config import SEED

NAME = "lightgbm"
FAMILY = "Gradient-boosted trees"
CONFIG = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 30,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 0.5,
}


def make(**overrides):
    return LGBMClassifier(
        **(CONFIG | overrides), random_state=SEED, n_jobs=4, verbosity=-1
    )
