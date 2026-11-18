"""Generate Folium comparison maps only for student-test trips."""

import argparse
import json
import re

import folium
import pandas as pd

from pathsnap.config import ART, RES
from pathsnap.hmm import build_graph, classifier_route, decode, route_coords
from pathsnap.io import find_trips, load_trip


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
        gps = trip.lonlat[:, ::-1]
        map_ = folium.Map(tiles="OpenStreetMap")
        road = folium.FeatureGroup(name="Road network", show=False)
        for start, end in trip.arcs:
            folium.PolyLine(trip.nodes_ll[[start, end], ::-1], color="#8a929b", weight=1).add_to(road)
        road.add_to(map_)
        points = folium.FeatureGroup(name="Noisy GPS")
        for point in gps[::max(1, len(gps) // 1500)]:
            folium.CircleMarker(point, radius=2, color="#2459b5", fill=True).add_to(points)
        points.add_to(map_)
        reference_segments = [[]]
        for arc in trip.route:
            current = reference_segments[-1]
            if current and trip.arcs[current[-1], 1] != trip.arcs[arc, 0]:
                reference_segments.append([])
            reference_segments[-1].append(int(arc))
        for name, segments, color in (
            ("Reference", reference_segments, "#23814b"),
            ("Classifier", raw.segments, "#a86100"),
            ("HMM/Viterbi", matched.segments, "#ce3344"),
        ):
            layer = folium.FeatureGroup(name=name, show=name != "Classifier")
            for line in route_coords(trip, segments):
                folium.PolyLine(line, color=color, weight=3, tooltip=name).add_to(layer)
            layer.add_to(map_)
        map_.fit_bounds([gps.min(axis=0).tolist(), gps.max(axis=0).tolist()])
        folium.LayerControl().add_to(map_)
        slug = re.sub(r"[^a-zA-Z0-9._-]", "_", tid)
        map_.save(RES / "maps" / f"{slug}.html")


if __name__ == "__main__":
    main()
