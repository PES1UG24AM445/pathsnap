"""Plot measured training and validation loss on one held-out training-trip split."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit

from pathsnap.config import RES, SEED
from pathsnap.experiment import load
from pathsnap.features import FEATURES
from pathsnap.model_selection import validate_table
from pathsnap.models import lgbm, mlp


def save(name, unit, train_loss, validation_loss):
    history = pd.DataFrame({"train": train_loss, "validation": validation_loss})
    history.index = np.arange(1, len(history) + 1)
    history.index.name = unit
    history.to_csv(RES / "figs" / f"curve_{name}.csv")
    ax = history.plot(title=f"{name} · held-out training trips")
    ax.set_ylabel("Binary log loss")
    ax.figure.tight_layout()
    ax.figure.savefig(RES / "figs" / f"curve_{name}.png", dpi=150)
    plt.close(ax.figure)


def main():
    table = load("train")
    validate_table(table, FEATURES, "train")
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train, valid = next(split.split(table, groups=table["tid"]))
    train_x, valid_x = table.iloc[train][FEATURES], table.iloc[valid][FEATURES]
    train_y, valid_y = table.iloc[train]["label"], table.iloc[valid]["label"]
    (RES / "figs").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "tid": sorted(table["tid"].unique()),
    }).assign(validation=lambda df: df["tid"].isin(table.iloc[valid]["tid"])).to_csv(
        RES / "figs" / "curve_trip_split.csv", index=False
    )

    booster = lgbm.make()
    booster.fit(
        train_x, train_y,
        eval_set=[(train_x, train_y), (valid_x, valid_y)],
        eval_names=["train", "validation"],
        eval_metric="binary_logloss",
    )
    history = booster.evals_result_
    save("lightgbm", "boosting round", history["train"]["binary_logloss"],
         history["validation"]["binary_logloss"])

    (_, scaler), (_, network) = mlp.make().steps
    network.set_params(random_state=np.random.RandomState(SEED))
    train_x = scaler.fit_transform(train_x)
    valid_x = scaler.transform(valid_x)
    train_loss, valid_loss = [], []
    for _ in range(mlp.CONFIG["max_iter"]):
        network.partial_fit(train_x, train_y, classes=[0, 1])
        train_loss.append(log_loss(train_y, network.predict_proba(train_x), labels=[0, 1]))
        valid_loss.append(log_loss(valid_y, network.predict_proba(valid_x), labels=[0, 1]))
    save("mlp", "epoch", train_loss, valid_loss)


if __name__ == "__main__":
    main()
