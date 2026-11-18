import folium

def make_map(trip_info, pred_path=None):
m = folium.Map(location=[12.9716, 77.5946], zoom_start=14)
if pred_path:
    folium.PolyLine(pred_path, color="blue", weight=4, opacity=0.8).add_to(m)
return m
