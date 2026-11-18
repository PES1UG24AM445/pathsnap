
"""Folium visualizations for PathSnap map-matching comparisons."""

import folium

from pathsnap.hmm import route_coords


def make_map(trip, classifier, matched, output_path):
    """Generate a comparison map for one student trip."""

    gps = trip.lonlat[:, ::-1]

    map_ = folium.Map(
        location=gps.mean(axis=0).tolist(),
        zoom_start=14,
        tiles="OpenStreetMap"
    )

    # Road network
    road = folium.FeatureGroup(name="Road network", show=False)

    for start, end in trip.arcs:
        coordinates = trip.nodes_ll[[start, end], ::-1].tolist()

        folium.PolyLine(
            coordinates,
            color="#8a929b",
            weight=1
        ).add_to(road)

    road.add_to(map_)

    # Noisy GPS observations
    points = folium.FeatureGroup(name="Noisy GPS")

    for point in gps[::max(1, len(gps) // 1500)]:
        folium.CircleMarker(
            location=point.tolist(),
            radius=2,
            color="#2459b5",
            fill=True
        ).add_to(points)

    points.add_to(map_)

    # Reference route
    reference_segments = [[]]

    for arc in trip.route:
        current = reference_segments[-1]

        if current and trip.arcs[current[-1], 1] != trip.arcs[arc, 0]:
            reference_segments.append([])

        reference_segments[-1].append(int(arc))

    # Route comparison layers
    layers = [
        ("Reference", reference_segments, "#23814b", True),
        ("Classifier", classifier.segments, "#a86100", False),
        ("HMM/Viterbi", matched.segments, "#ce3344", True),
    ]

    for name, segments, color, visible in layers:
        layer = folium.FeatureGroup(name=name, show=visible)

        for line in route_coords(trip, segments):
            folium.PolyLine(
                line,
                color=color,
                weight=4,
                tooltip=name
            ).add_to(layer)

        layer.add_to(map_)

    # Fit map to GPS observations
    map_.fit_bounds([
        gps.min(axis=0).tolist(),
        gps.max(axis=0).tolist()
    ])

    folium.LayerControl().add_to(map_)

    map_.save(str(output_path))

    return map_
