"""Regression checks for directed links and access to reserved trip files."""

from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from pathsnap.hmm import build_graph, decode, prepare_transition_costs, route_coords
from pathsnap.io import find_trips


def test_reconstructs_intermediate_arcs_and_breaks():
    xy = np.array([[0., 0.], [10., 0.], [20., 0.], [30., 0.], [40., 0.], [50., 0.]])
    trip = SimpleNamespace(
        xy=xy[[0, 3, 4]], t=np.arange(3), nodes_xy=xy, nodes_ll=xy,
        arcs=np.array([[0, 1], [1, 2], [2, 3], [4, 5]]),
    )
    candidates = pd.DataFrame({"obs": [0, 1, 2], "arc": [0, 2, 3],
                               "along": [0., 10., 0.], "p": [0.8, 0.9, 0.9]})
    result = decode(trip, candidates)
    assert result.arcs.tolist() == [0, 2, 3]
    assert result.segments == [[0, 1, 2], [3]]
    assert result.continuity == 0.5
    assert [len(line) for line in route_coords(trip, result.segments)] == [4, 2]
    graph = build_graph(trip)
    costs = prepare_transition_costs(trip, candidates, graph)
    cached = decode(trip, candidates, g=graph, transition_costs=costs)
    np.testing.assert_array_equal(cached.arcs, result.arcs)
    assert cached.segments == result.segments


def test_development_inventory_does_not_discover_evaluator_references(tmp_path: Path):
    folder = tmp_path / "evaluator"
    folder.mkdir()
    for suffix in (".track", ".nodes", ".arcs", ".route"):
        (folder / ("trip" + suffix)).touch()
    assert "route" not in find_trips(tmp_path)[0]
    assert find_trips(folder, partition="serving")[0]["route"] == folder / "trip.route"


def test_unknown_partition_raises_instead_of_training(tmp_path: Path):
    folder = tmp_path / "unknown"
    folder.mkdir()
    (folder / "trip.track").touch()
    with pytest.raises(ValueError, match="Cannot determine partition"):
        find_trips(tmp_path)
