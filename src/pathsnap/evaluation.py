"""Route evaluation shared by the development scripts."""

from collections import Counter

import numpy as np

from .hmm import classifier_route, decode


def sequence_f1(segments, reference):
    """Arc-overlap F1 with repeat counts. This score ignores traversal order."""
    predicted = Counter(arc for segment in segments for arc in segment)
    actual = Counter(int(arc) for arc in reference)
    return 2 * sum((predicted & actual).values()) / (sum(predicted.values()) + sum(actual.values()))


def evaluate_trip(trip, candidates, graph, reference, settings):
    raw = classifier_route(trip, candidates, graph)
    matched = decode(trip, candidates, g=graph, **settings)
    result = {"observations": len(matched.observations)}
    for name, route in (("classifier", raw), ("hmm", matched)):
        result[f"{name}_top1"] = float(np.mean(route.arcs == reference[route.observations]))
        result[f"{name}_continuity"] = route.continuity
        result[f"{name}_route_f1"] = sequence_f1(route.segments, trip.route)
    return result
