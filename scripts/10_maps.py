
"""Generate Folium comparison maps only for student-test trips."""

import argparse
import json
import re

import pandas as pd

from pathsnap.config import ART, RES
from pathsnap.hmm import build_graph, classifier_route, decode
from pathsnap.io import find_trips, load_trip
from pathsnap.viz import make_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trip_ids",
        nargs="*",
        help="Student trip IDs. Defaults to weakest and strongest HMM changes."
    )

    args = parser.parse_args()

    # Load final model and HMM configuration
    manifest = json.loads(
        (ART / "final_model_manifest.json").read_text(encoding="utf-8")
    )

    settings = json.loads(
        (ART / "hmm_config.json").read_text(encoding="utf-8")
    )

    # Only student-test trips are allowed
    infos = {
        item["tid"]: item
        for item in find_trips()
        if item["part"] == "student"
    }

    # Select trips
    if args.trip_ids:
        trip_ids = args.trip_ids
    else:
        metrics = pd.read_csv(RES / "hmm_vs_classifier_per_trip.csv")

        spread = metrics["hmm_top1"] - metrics["classifier_top1"]

        trip_ids = list(dict.fromkeys([
            metrics.loc[spread.idxmin(), "tid"],
            metrics.loc[spread.idxmax(), "tid"]
        ]))

    # Load student predictions
    scores = pd.read_parquet(
        RES / "preds" / f"{manifest['model']}_student.parquet"
    )

    for tid in trip_ids:

        if tid not in infos:
            raise ValueError(f"{tid} is not a valid student-test trip.")

        trip = load_trip(infos[tid])

        candidates = scores[scores["tid"] == tid]

        if candidates.empty:
            raise ValueError(f"No prediction candidates found for {tid}.")

        graph = build_graph(trip)

        # Classifier-only route
        raw = classifier_route(trip, candidates, graph)

        # HMM/Viterbi route
        matched = decode(
            trip,
            candidates,
            g=graph,
            **settings
        )

        # Generate map
        slug = re.sub(r"[^a-zA-Z0-9._-]", "_", str(tid))

        output_path = RES / "maps" / f"{slug}.html"

        make_map(
            trip,
            raw,
            matched,
            output_path
        )

        print(f"Map generated: {output_path}")


if __name__ == "__main__":
    main()
