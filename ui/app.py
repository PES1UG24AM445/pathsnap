"""PathSnap: Distributed Map-Matching User Interface.

Owned by: Sanjit (feat/sanjit)
Deployed on: Laptop 2 (UI client)
Target Backend: Laptop 1 (FastAPI Server on http://<LAPTOP1_IP>:8000)

Architecture & Integrity Safeguards:
- NEVER imports the `pathsnap` Python package.
- NEVER has access to raw road networks, model weights, or ground-truth labels.
- Only communicates with the backend via standard REST endpoints (/health, /trips, /predict).
- Displays live Folium interactive map and server-side computed evaluation metrics.
"""

import os
import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

# Page configuration
st.set_page_config(
    page_title="PathSnap | Map-Matching",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for high-clarity evaluation presentation
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #1f77b4;
    }
    .status-ok { color: #28a745; font-weight: bold; }
    .status-err { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🗺️ PathSnap: Distributed GPS Map-Matching")
st.caption("PES University | UE24CS352A Machine Learning Hackathon | Evaluator Interface")

# Sidebar Configuration
st.sidebar.header("⚙️ API Configuration")
default_api = os.environ.get("API_URL", "http://127.0.0.1:8000")
api_url = st.sidebar.text_input("Backend API URL", value=default_api, help="Laptop 1 IP address, e.g. http://192.168.1.5:8000")

# Check API Health
api_healthy = False
trips_list = []

try:
    health_resp = requests.get(f"{api_url}/health", timeout=5)
    if health_resp.status_code == 200:
        health_data = health_resp.json()
        api_healthy = True
        st.sidebar.markdown(f"**Server Status:** <span class='status-ok'>Connected</span>", unsafe_allow_html=True)
        st.sidebar.info(f"Available Trips on Server: **{health_data.get('trips', 'N/A')}**")
    else:
        st.sidebar.markdown(f"**Server Status:** <span class='status-err'>HTTP {health_resp.status_code}</span>", unsafe_allow_html=True)
except requests.exceptions.RequestException as e:
    st.sidebar.markdown(f"**Server Status:** <span class='status-err'>Unreachable</span>", unsafe_allow_html=True)
    st.sidebar.warning(f"Could not reach backend at `{api_url}`. Ensure Laptop 1 has uvicorn running and firewall port 8000 allowed.")

# Fetch Trip IDs
if api_healthy:
    try:
        trips_resp = requests.get(f"{api_url}/trips", timeout=10)
        if trips_resp.status_code == 200:
            trips_list = trips_resp.json()
    except Exception as e:
        st.error(f"Failed to fetch trips list: {e}")

if not trips_list:
    st.info("⚠️ Waiting for API connection. Please start the FastAPI backend on Laptop 1 and check network connectivity.")
    st.stop()

# Trip Selection & Execution Controls
st.subheader("1. Select Evaluation Trip")
col_sel, col_btn = st.columns([3, 1])

with col_sel:
    selected_tid = st.selectbox("Choose a reserved evaluation trip:", trips_list, index=0)

with col_btn:
    st.write("") # Spacer
    st.write("")
    run_btn = st.button("🚀 Reconstruct Route", type="primary", use_container_width=True)

if run_btn:
    with st.spinner(f"Sending inference request for trip `{selected_tid}` to Laptop 1..."):
        try:
            resp = requests.post(
                f"{api_url}/predict",
                json={"trip_id": selected_tid},
                timeout=900  # Up to 15 minutes timeout for large evaluation graphs
            )
            if resp.status_code == 200:
                st.session_state["result"] = resp.json()
                st.session_state["current_trip"] = selected_tid
                st.success(f"Trip `{selected_tid}` successfully matched in {resp.json().get('seconds', 0)} seconds!")
            else:
                st.error(f"API Error {resp.status_code}: {resp.text}")
        except requests.exceptions.Timeout:
            st.error("Request timed out. The server-side Viterbi Dijkstra calculation took too long.")
        except requests.exceptions.RequestException as e:
            st.error(f"Network error during prediction: {e}")

# Display Results
res = st.session_state.get("result")
if res and st.session_state.get("current_trip") == selected_tid:
    st.subheader("2. Performance & Verification Metrics")
    
    # Delta Calculation
    acc_hmm = res.get("accuracy_hmm", 0.0)
    acc_clf = res.get("accuracy_classifier_only", 0.0)
    acc_delta = acc_hmm - acc_clf

    cont_hmm = res.get("continuity_hmm", 0.0)
    cont_clf = res.get("continuity_classifier_only", 0.0)
    cont_delta = cont_hmm - cont_clf

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy (HMM + Classifier)", f"{acc_hmm:.3f}", delta=f"{acc_delta:+.3f}")
    m2.metric("Accuracy (Classifier Only)", f"{acc_clf:.3f}")
    m3.metric("Route Continuity (HMM)", f"{cont_hmm:.3f}", delta=f"{cont_delta:+.3f}")
    m4.metric("GPS Observations", f"{res.get('n_observations', 0):,}")
    m5.metric("Latency", f"{res.get('seconds', 0.0):.2f} s")

    # Map Visualization
    st.subheader("3. Trajectory & Reconstructed Route Map")
    gps_pts = res.get("gps", [])
    segments = res.get("predicted_route_segments") or [res.get("predicted_route", [])]

    if gps_pts:
        mid_idx = len(gps_pts) // 2
        center_lat, center_lon = gps_pts[mid_idx][0], gps_pts[mid_idx][1]
        m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

        # GPS Observations Layer
        fg_gps = folium.FeatureGroup(name="GPS Observations (Blue Dots)")
        for lat, lon in gps_pts[::2]:  # Subsample by 2 for smooth rendering
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color="#1f77b4",
                fill=True,
                fill_color="#1f77b4",
                fill_opacity=0.8
            ).add_to(fg_gps)
        fg_gps.add_to(m)

        # Reconstructed Route Layer
        if any(segments):
            fg_route = folium.FeatureGroup(name="Reconstructed Route (Red Polyline)")
            for segment in segments:
                if segment:
                    folium.PolyLine(
                        locations=segment, color="#d62728", weight=4, opacity=0.85
                    ).add_to(fg_route)
            fg_route.add_to(m)

        # Bounding box auto-fit
        lats = [p[0] for p in gps_pts]
        lons = [p[1] for p in gps_pts]
        m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

        # Interactive Layer Controls
        folium.LayerControl(collapsed=False).add_to(m)

        # Map display in Streamlit
        st_folium(m, width=1200, height=550, returned_objects=[])
    else:
        st.warning("No GPS coordinates returned by API.")

st.markdown("---")
st.markdown("<small style='color: grey;'>PathSnap Hackathon Deployment | Sanjit's Component: UI & Evaluator Dashboard</small>", unsafe_allow_html=True)
