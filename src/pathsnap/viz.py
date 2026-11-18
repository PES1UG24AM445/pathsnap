"""Interactive route comparison for permitted development trips."""

from html import escape

import folium

from .hmm import route_coords


def make_map(trip, matched, *, classifier=None, reference=None, title=None):
    """Render GPS, reconstructed geometry, and optional offline comparison layers."""
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
    for name, segments, color in (
        ("Reference", reference, "#23814b"),
        ("Classifier", classifier, "#a86100"),
        ("HMM/Viterbi", matched, "#ce3344"),
    ):
        if not segments:
            continue
        layer = folium.FeatureGroup(name=name, show=name != "Classifier")
        for line in route_coords(trip, segments):
            folium.PolyLine(line, color=color, weight=3, tooltip=name).add_to(layer)
        layer.add_to(map_)
    map_.fit_bounds([gps.min(axis=0).tolist(), gps.max(axis=0).tolist()])
    folium.LayerControl().add_to(map_)
    if title:
        map_.get_root().html.add_child(folium.Element(
            '<div style="position:fixed;z-index:1000;top:12px;left:52px;'
            'padding:8px 12px;background:white;border:1px solid #889">'
            f'{escape(title)}</div>'
        ))
    return map_
