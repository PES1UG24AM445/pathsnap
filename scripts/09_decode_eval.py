"""Tune the decoder with train OOF scores and compare on student-test trips."""

import argparse
import itertools
import json

import numpy as np
import pandas as pd

from pathsnap.config import ART, RES, SEED
from pathsnap.evaluation import evaluate_trip, sequence_f1
from pathsnap.hmm import build_graph, decode, prepare_transition_costs
from pathsnap.io import find_trips, load_trip
from pathsnap.labels import reference_arcs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune-trips", type=int, default=8)
    args = parser.parse_args()
    if args.tune_trips < 1:
        parser.error("--tune-trips must be positive.")
    manifest = json.loads((ART / "final_model_manifest.json").read_text(encoding="utf-8"))
    model = manifest["model"]
    names = ("tid", "obs", "arc", "along", "p")
    oof = pd.read_parquet(RES / "oof" / f"{model}.parquet", columns=list(names))
    student = pd.read_parquet(RES / "preds" / f"{model}_student.parquet", columns=list(names))
    permitted = {item["tid"]: item for item in find_trips() if item["part"] in ("train", "student")}

    def load_examples(table, ids, partition):
        for tid, rows in table[table["tid"].isin(ids)].groupby("tid"):
            info = permitted[tid]
            if info["part"] != partition:
                raise ValueError(f"{tid}: predictions are from the wrong partition.")
            trip = load_trip(info)
            yield trip, rows, build_graph(trip), reference_arcs(trip)

    ids = np.array(sorted(oof["tid"].unique()))
    train_ids = np.random.default_rng(SEED).choice(ids, min(len(ids), args.tune_trips), replace=False)
    examples = [
        (trip, rows, graph, reference, prepare_transition_costs(trip, rows, graph))
        for trip, rows, graph, reference in load_examples(oof, train_ids, "train")
    ]
    results = []
    for beta, w, jump in itertools.product((15., 40.), (0.5, 1.0, 2.0), (-20., -50.)):
        config = {"beta": beta, "w": w, "jump": jump}
        correct = count = 0
        route = []
        for trip, rows, graph, reference, costs in examples:
            decoded = decode(trip, rows, g=graph, transition_costs=costs, **config)
            correct += int((decoded.arcs == reference[decoded.observations]).sum())
            count += len(decoded.arcs)
            route.append(sequence_f1(decoded.segments, trip.route))
        results.append({**config, "top1": correct / count, "route_f1": float(np.mean(route))})

    grid = pd.DataFrame(results).sort_values(["top1", "route_f1"], ascending=False, kind="stable")
    settings = {name: float(grid.iloc[0][name]) for name in ("beta", "w", "jump")}
    rows = []
    for trip, scores, graph, reference in load_examples(student, student["tid"].unique(), "student"):
        metrics = evaluate_trip(trip, scores, graph, reference, settings)
        rows.append({"tid": trip.tid, **metrics})
    comparison = pd.DataFrame(rows)
    if len(comparison) != 10:
        raise ValueError("Expected decoder metrics for exactly ten student trips.")
    summary = comparison.drop(columns=["tid"]).mean().to_frame("mean_per_trip").T
    grid.to_csv(RES / "hmm_grid.csv", index=False)
    comparison.to_csv(RES / "hmm_vs_classifier_per_trip.csv", index=False)
    summary.to_csv(RES / "hmm_vs_classifier.csv", index=False)
    (ART / "hmm_config.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    (ART / "hmm_model_sha256.txt").write_text(manifest["sha256"] + "\n", encoding="utf-8")
    (RES / "hmm_tuning_trips.json").write_text(json.dumps(train_ids.tolist()) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
