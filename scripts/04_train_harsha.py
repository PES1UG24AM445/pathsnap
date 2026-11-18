"""Validate the shared features and train Harsha's two baseline models."""

import argparse

from pathsnap.experiment import evaluate, load
from pathsnap.features import FEATURES
from pathsnap.model_selection import baseline, validate_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["lightgbm", "mlp"], action="append")
    args = parser.parse_args()
    train, student = load("train"), load("student")
    validate_table(train, FEATURES, "train")
    validate_table(student, FEATURES, "student")
    if set(train["tid"]) & set(student["tid"]):
        raise ValueError("Training and student trip IDs overlap.")
    for name in args.model or ("lightgbm", "mlp"):
        module = baseline(name)
        evaluate(module.NAME, module.FAMILY, module.CONFIG, module.make,
                 cap=getattr(module, "MAX_TRAIN_ROWS", None))


if __name__ == "__main__":
    main()
