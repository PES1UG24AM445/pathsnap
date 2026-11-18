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
    parser.add_argument("trip_ids", nargs="*", help="Defaults to weakest and strongest HMM changes")
    args = parser.parse_args()
    manifest = json.loads((ART / "final_model_manifest.json").read_text(encoding="utf-8"))
    settings = json.loads((ART / "hmm_config.json").read_text(encoding="utf-8"))
    infos = {i["tid"]: i for i in find_trips() if i["part"] == "student"}
    if args.trip_ids:
        trip_ids = args.trip_ids
    else:
        metrics = pd.read_csv(RES / "hmm_vs_classifier_per_trip.csv")
        spread = metrics["hmm_top1"] - metrics["classifier_top1"]
        trip_ids = list(dict.fromkeys(metrics.loc[[spread.idxmin(), spread.idxmax()], "tid"]))
    scores = pd.read_parquet(RES / "preds" / f"{manifest['model']}_student.parquet")

    for tid in trip_ids:
        trip = load_trip(infos[tid])
        candidates = scores[scores["tid"] == tid]
        graph = build_graph(trip)
        raw = classifier_route(trip, candidates, graph)
        matched = decode(trip, candidates, g=graph, **settings)
        reference_segments = [[]]
        for arc in trip.route:
            current = reference_segments[-1]
            if current and trip.arcs[current[-1], 1] != trip.arcs[arc, 0]:
                reference_segments.append([])
            reference_segments[-1].append(int(arc))
        map_ = make_map(trip, matched.segments, classifier=raw.segments,
                        reference=reference_segments)
        slug = re.sub(r"[^a-zA-Z0-9._-]", "_", tid)
        map_.save(RES / "maps" / f"{slug}.html")


if __name__ == "__main__":
    main()
