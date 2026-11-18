"""Measure candidate coverage, ranking errors, and paired gains by trip."""

import json

import numpy as np
import pandas as pd

from pathsnap.config import ART, RES, SEED
from pathsnap.experiment import load
from pathsnap.features import FEATURES
from pathsnap.model_selection import validate_table


KEYS = ["tid", "obs", "arc"]
OBS = KEYS[:2]


def predictions_for(name, part, reference):
    directory = RES / ("oof" if part == "train" else "preds")
    filename = f"{name}.parquet" if part == "train" else f"{name}_student.parquet"
    path = directory / filename
    scores = pd.read_parquet(path, columns=KEYS + ["label", "p"])
    aligned = scores.set_index(KEYS)
    if not aligned.index.is_unique or len(aligned) != len(reference):
        raise ValueError(f"{path}: candidate count or unique keys differ from processed data.")
    aligned = aligned.reindex(reference.index)
    if not aligned["label"].eq(reference["label"]).all():
        raise ValueError(f"{path}: candidate keys or labels differ. Regenerate predictions.")
    if not aligned["p"].between(0, 1).all():
        raise ValueError(f"{path}: expected a finite probability for every candidate.")
    # Preserve saved row order on ties, consistent with the harness's idxmax.
    return scores.sort_values("p", ascending=False, kind="stable")


def rank_observations(scores, gaps):
    groups = scores.groupby(OBS, sort=False)
    rank = groups.cumcount().add(1)
    top = scores.loc[rank.eq(1), OBS + ["p"]].set_index(OBS).rename(columns={"p": "top_score"})
    runner = scores.loc[rank.eq(2)].set_index(OBS)["p"]
    positive = scores.loc[scores["label"].eq(1), OBS].assign(rank=rank)
    positive_rank = positive.set_index(OBS)["rank"].reindex(top.index)
    top["covered"] = positive_rank.notna()
    top["top1"] = positive_rank.eq(1)
    top["top3"] = positive_rank.le(3)
    top["reciprocal_rank"] = (1 / positive_rank).fillna(0)
    top["margin"] = top["top_score"] - runner
    if gaps is not None:
        top["gap"] = gaps.reindex(top.index)
    return top


def summarize(observations):
    size = len(observations)
    covered = int(observations["covered"].sum())
    correct = int(observations["top1"].sum())
    return {
        "observations": size,
        "candidate_recall": covered / size,
        "candidate_misses": size - covered,
        "ranking_errors_with_reference_present": covered - correct,
        "top1": correct / size,
        "top1_when_reference_present": correct / covered if covered else None,
        "top3": float(observations["top3"].mean()),
        "mean_reciprocal_rank": float(observations["reciprocal_rank"].mean()),
    }


def paired_gains(untuned, tuned):
    delta = (tuned["top1"].astype(int) - untuned["top1"].astype(int)).groupby("tid")
    trip_gains = delta.mean().to_numpy()
    rng = np.random.default_rng(SEED)
    bootstrap = rng.choice(trip_gains, size=(10_000, len(trip_gains)), replace=True)
    bounds = np.quantile(bootstrap.mean(axis=1), [0.025, 0.975])
    return {
        "trips": len(trip_gains),
        "trips_improved": int((trip_gains > 0).sum()),
        "trips_regressed": int((trip_gains < 0).sum()),
        "mean_trip_top1_change_percentage_points": float(100 * trip_gains.mean()),
        "exploratory_95_percent_paired_trip_bootstrap_interval_pp": (100 * bounds).tolist(),
    }


def plot_student_summary(student, trips, output):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, 4), layout="constrained")
    FigureCanvasAgg(fig)
    errors, change = fig.subplots(1, 2)
    names = list(student)
    left = np.zeros(len(names))
    for label, color, share in (
        ("Correct", "#177E89", lambda m: m["top1"]),
        ("Ranking error", "#D88832", lambda m: m["candidate_recall"] - m["top1"]),
        ("Candidate missing", "#78818F", lambda m: 1 - m["candidate_recall"]),
    ):
        widths = 100 * np.array([share(student[name]) for name in names])
        errors.barh(names, widths, left=left, color=color, label=label)
        left += widths
    errors.set(xlabel="Student-test observations (%)", title="Top-1 error breakdown", xlim=(0, 100))
    errors.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False, ncol=2)

    trip_top1 = trips.loc[trips["partition"].eq("student_test")].pivot(
        index="tid", columns="model", values="top1"
    )
    gains = (100 * (trip_top1[names[1]] - trip_top1[names[0]])).sort_values()
    positions = np.arange(1, len(gains) + 1)
    change.axhline(0, color="#78818F", linewidth=1)
    change.bar(positions, gains, color=np.where(gains >= 0, "#177E89", "#D88832"))
    change.set(xlabel="Student-test trips, sorted by change", ylabel="Top-1 change (pp)",
               title="Tuned model vs. baseline", xticks=positions)
    fig.savefig(output / "student_error_breakdown.png", dpi=160)


def main():
    manifest = json.loads((ART / "final_model_manifest.json").read_text(encoding="utf-8"))
    baseline = manifest["baseline"]
    tuned = manifest["tuned_candidate"]
    output = RES / "diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    per_trip, slices = [], []
    report = {
        "baseline": baseline,
        "tuned_candidate": tuned,
        "exported": manifest["model"],
        "partitions": {},
        "uncertainty_note": (
            "The paired interval resamples whole student-test trips and weights each trip equally. "
            "With ten trips it is descriptive. The training OOF scores informed model selection."
        ),
    }

    for part in ("train", "student"):
        table = load(part)
        validate_table(table, FEATURES, part)
        reference = table.set_index(KEYS)[["label"]]
        gaps = table.groupby(OBS)["gap_prev"].first() if "gap_prev" in table else None
        partition = "train_oof" if part == "train" else "student_test"
        outcomes = {}
        report["partitions"][partition] = {}
        for name in (baseline, tuned):
            scores = predictions_for(name, part, reference)
            observations = rank_observations(scores, gaps)
            outcomes[name] = observations
            report["partitions"][partition][name] = summarize(observations)
            for tid, trip in observations.groupby("tid"):
                per_trip.append(dict(partition=partition, model=name, tid=tid, **summarize(trip)))

            margin = observations["margin"]
            masks = {
                "single_candidate": margin.isna(),
                "margin_at_most_0.10": margin.le(0.10),
                "margin_above_0.10": margin.gt(0.10),
            }
            if gaps is not None:
                masks["gps_gap_above_1.5s"] = observations["gap"].gt(1.5)
                masks["gps_gap_at_most_1.5s"] = observations["gap"].le(1.5)
            for category, mask in masks.items():
                if mask.any():
                    slices.append(dict(partition=partition, model=name, slice=category,
                                       **summarize(observations.loc[mask])))
        if part == "student":
            report["paired_student_top1"] = paired_gains(outcomes[baseline], outcomes[tuned])

    trips = pd.DataFrame(per_trip)
    trips.to_csv(output / "per_trip.csv", index=False)
    pd.DataFrame(slices).to_csv(output / "error_slices.csv", index=False)
    (output / "model_diagnostics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    plot_student_summary(report["partitions"]["student_test"], trips, output)
    print(f"Candidate coverage and paired trip diagnostics saved to {output}.")


if __name__ == "__main__":
    main()
