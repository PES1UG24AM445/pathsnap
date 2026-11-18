"""Directed, candidate-state Viterbi decoding and route reconstruction."""

from dataclasses import dataclass
import heapq

import numpy as np


@dataclass
class DecodeResult:
    observations: np.ndarray
    arcs: np.ndarray
    segments: list[list[int]]
    continuity: float


class RoadGraph:
    def __init__(self, trip):
        self.arcs = trip.arcs
        start, end = self.arcs.T
        self.lengths = np.linalg.norm(trip.nodes_xy[end] - trip.nodes_xy[start], axis=1)
        self.outgoing = [[] for _ in trip.nodes_xy]
        for arc, node in enumerate(start):
            self.outgoing[node].append(arc)

    def search(self, start, targets, limit):
        """Find bounded shortest paths to the requested nodes, retaining arc IDs."""
        distances, previous = {start: 0.0}, {}
        pending = [(0.0, start)]
        targets = set(targets)
        while pending and targets:
            distance, node = heapq.heappop(pending)
            if distance != distances[node]:
                continue
            targets.discard(node)
            for arc in self.outgoing[node]:
                end = int(self.arcs[arc, 1])
                total = distance + self.lengths[arc]
                if total <= limit and total < distances.get(end, float("inf")):
                    distances[end] = total
                    previous[end] = arc
                    heapq.heappush(pending, (total, end))
        return distances, previous

    def bridge(self, start, end, limit):
        distances, previous = self.search(start, [end], limit)
        if end not in distances:
            return None
        edges = []
        while end != start:
            arc = previous[end]
            edges.append(arc)
            end = int(self.arcs[arc, 0])
        return edges[::-1]


def step(trip, previous, current):
    gap = max(float(trip.t[current] - trip.t[previous]), 1.0)
    distance = float(np.linalg.norm(trip.xy[current] - trip.xy[previous]))
    return gap, distance, max(3 * distance + 300, min(35 * gap, 2000))


def network_distances(graph, earlier, later, a, b, limit):
    starts = graph.arcs[later, 0]
    ends = graph.arcs[earlier, 1]
    sources = {int(node): graph.search(int(node), starts, limit)[0] for node in set(ends)}
    network = np.array([[sources[int(i)].get(int(j), np.inf) for j in starts] for i in ends])
    distance = (graph.lengths[earlier] - a)[:, None] + network + b[None, :]
    along_delta = b[None, :] - a[:, None]
    # Small backward movement on a directed arc can be projection jitter.
    stay = (earlier[:, None] == later[None, :]) & (along_delta >= -10)
    return np.where(stay, np.maximum(along_delta, 0), distance)


def reconstruct(graph, trip, observations, arcs, along):
    segments = [[int(arcs[0])]]
    connected = 0
    for i in range(1, len(arcs)):
        previous, current = map(int, arcs[i - 1:i + 1])
        if previous == current and along[i] >= along[i - 1] - 10:
            connected += 1
            continue
        _, _, limit = step(trip, observations[i - 1], observations[i])
        start, end = int(graph.arcs[previous, 1]), int(graph.arcs[current, 0])
        bridge = graph.bridge(start, end, limit)
        if bridge is None:
            segments.append([current])
        else:
            segments[-1].extend([*bridge, current])
            connected += 1
    return segments, connected / max(1, len(arcs) - 1) if len(arcs) > 1 else 1.0


def build_graph(trip):
    return RoadGraph(trip)


def ordered_candidates(cand, graph):
    candidates = cand.sort_values("obs", kind="stable")
    obs = candidates["obs"].to_numpy(dtype=int)
    if not len(obs):
        raise ValueError("Viterbi needs candidates for at least one observation.")
    arcs = candidates["arc"].to_numpy(dtype=int)
    along = np.clip(candidates["along"].to_numpy(), 0, graph.lengths[arcs])
    starts = np.r_[0, np.flatnonzero(np.diff(obs)) + 1]
    ends = np.r_[starts[1:], len(obs)]
    return candidates, arcs, along, obs[starts], starts, ends


def _transition_costs(trip, graph, arcs, along, observations, starts, ends):
    for i in range(1, len(starts)):
        previous = slice(starts[i - 1], ends[i - 1])
        current = slice(starts[i], ends[i])
        gap, distance, limit = step(trip, observations[i - 1], observations[i])
        network = network_distances(graph, arcs[previous], arcs[current], along[previous], along[current], limit)
        yield np.abs(network - distance) / np.sqrt(gap)


def prepare_transition_costs(trip, candidates, graph):
    """Cache distance-based terms that stay fixed across decoder tuning trials."""
    _, arcs, along, observations, starts, ends = ordered_candidates(candidates, graph)
    return list(_transition_costs(trip, graph, arcs, along, observations, starts, ends))


def decode(trip, cand, g=None, beta=30.0, w=1.0, jump=-30.0, *, transition_costs=None):
    """Return selected arcs and directed routes, keeping disconnected pieces separate."""
    graph = g if g is not None else build_graph(trip)
    candidates, arcs, along, observations, starts, ends = ordered_candidates(cand, graph)
    emission = w * np.log(np.clip(candidates["p"].to_numpy(), 1e-6, 1))
    if transition_costs is None:
        transition_costs = _transition_costs(trip, graph, arcs, along, observations, starts, ends)
    scores = emission[starts[0]:ends[0]].copy()
    back = []
    for i, cost in enumerate(transition_costs, start=1):
        current = slice(starts[i], ends[i])
        transition = np.where(np.isfinite(cost), -cost / beta, jump)
        combined = scores[:, None] + transition
        choice = combined.argmax(axis=0)
        back.append(choice)
        scores = emission[current] + combined[choice, np.arange(len(choice))]

    index = int(scores.argmax())
    selected = np.empty(len(starts), dtype=int)
    for i in range(len(starts) - 1, -1, -1):
        selected[i] = starts[i] + index
        if i:
            index = back[i - 1][index]
    segments, continuity = reconstruct(graph, trip, observations, arcs[selected], along[selected])
    return DecodeResult(observations, arcs[selected], segments, continuity)


def classifier_route(trip, candidates, graph):
    indices = candidates.groupby("obs")["p"].idxmax()
    selected = candidates.loc[indices].sort_values("obs")
    obs = selected["obs"].to_numpy()
    arcs = selected["arc"].to_numpy()
    segments, continuity = reconstruct(graph, trip, obs, arcs, selected["along"].to_numpy())
    return DecodeResult(obs, arcs, segments, continuity)


def route_coords(trip, segments):
    return [
        [trip.nodes_ll[node, ::-1].tolist() for node in
         [trip.arcs[segment[0], 0], *trip.arcs[segment, 1]]]
        for segment in segments
    ]
