import joblib
from models.model import train_gbm, UNet
from scripts.data_ingestion import load_weather_data, load_terrain_data, merge_data
import streamlit as st
from streamlit_keplergl import keplergl_static
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import sys
import os
import subprocess
import scripts.fetch_live_weather as live_weather

# Add project root to path
sys.path.append(os.path.dirname(__file__))


st.title("AI-Based Flood Prediction")

st.sidebar.header("Data Options")
use_sample_data = st.sidebar.checkbox("Use Sample Data")

if use_sample_data:
    if st.sidebar.button("Generate Sample Data"):
        with st.spinner("Generating sample data..."):
            subprocess.run([sys.executable, "scripts/generate_sample_data.py"])
        st.sidebar.success("Sample data generated!")

    # Load sample data
    sample_weather = "data/sample_weather.csv"
    sample_terrain = "data/sample_terrain.tif"
    sample_flood = "data/sample_flood.csv"
    sample_flood_mask = "data/sample_flood_mask.tif"

    if os.path.exists(sample_weather):
        weather_df = load_weather_data(sample_weather)
        st.sidebar.write("Sample weather data loaded.")
    else:
        weather_df = None

    if os.path.exists(sample_terrain):
        terrain_data, terrain_profile = load_terrain_data(sample_terrain)
        st.sidebar.write("Sample terrain data loaded.")
    else:
        terrain_data = None
        terrain_profile = None

    if os.path.exists(sample_flood):
        flood_df = pd.read_csv(sample_flood)
        st.sidebar.write("Sample flood data loaded.")
    else:
        flood_df = None

    if os.path.exists(sample_flood_mask):
        flood_mask_data, flood_mask_profile = load_terrain_data(sample_flood_mask)
        st.sidebar.write("Sample flood mask loaded.")
    else:
        flood_mask_data = None
        flood_mask_profile = None
else:
    uploaded_weather = st.sidebar.file_uploader(
        "Upload Weather Data (CSV)", type=["csv"])
    uploaded_terrain = st.sidebar.file_uploader(
        "Upload Terrain Data (GeoTIFF)", type=["tif", "tiff"])
    uploaded_flood = st.sidebar.file_uploader(
        "Upload Historical Flood Data (CSV)", type=["csv"])
    uploaded_flood_mask = st.sidebar.file_uploader(
        "Upload Flood Mask (GeoTIFF)", type=["tif", "tiff"])

    weather_df = load_weather_data(uploaded_weather) if uploaded_weather else None
    if uploaded_terrain:
        terrain_data, terrain_profile = load_terrain_data(uploaded_terrain)
    else:
        terrain_data, terrain_profile = None, None
    flood_df = pd.read_csv(uploaded_flood) if uploaded_flood else None
    if uploaded_flood_mask:
        flood_mask_data, flood_mask_profile = load_terrain_data(uploaded_flood_mask)
    else:
        flood_mask_data, flood_mask_profile = None, None

# Live Weather - Process this FIRST
st.sidebar.header("Live Weather")
live_api_key = st.sidebar.text_input("OpenWeatherMap API Key", "")
if st.sidebar.button("Fetch Live Weather for Astana") and live_api_key:
    with st.spinner("Fetching live weather data..."):
        try:
            weather = live_weather.fetch_openweathermap(
                live_api_key, city="Astana, KZ")
            st.session_state['live_weather'] = weather
            st.sidebar.success(
                f"Fetched: Rainfall={weather['rainfall']} mm, Temp={weather['temperature']} °C")
        except Exception as e:
            st.sidebar.error(f"Error fetching weather: {e}")

st.sidebar.header("Simulation Controls")
# Initialize with slider defaults
rainfall = st.sidebar.slider("Rainfall (mm)", 0, 300, 50)
temperature = st.sidebar.slider("Temperature (°C)", -10, 40, 20)

# Override logic: allow switching between live weather and simulation sliders
if 'live_weather' in st.session_state:
    weather = st.session_state['live_weather']
    # Initialize state for toggle persistence
    if 'use_live_weather' not in st.session_state:
        st.session_state['use_live_weather'] = True
    # Toggle to enable/disable applying live weather to model inputs
    st.session_state['use_live_weather'] = st.sidebar.checkbox(
        "Use Live Weather for Prediction",
        value=st.session_state['use_live_weather']
    )
    st.sidebar.info(
        f"Live Weather: Rainfall={weather['rainfall']} mm, Temp={weather['temperature']} °C @ {weather['timestamp']}"
    )
    # Optional clear button
    if st.sidebar.button("Clear Live Weather"):
        st.session_state.pop('live_weather', None)
        st.session_state['use_live_weather'] = False
        st.rerun()
    # Apply live weather only if toggle enabled
    if st.session_state['use_live_weather']:
        rainfall = weather['rainfall']
        temperature = weather['temperature']

# Load models (placeholders - train or load pre-trained models)


@st.cache_resource
def load_models():
    # For demo, train a dummy GBM if no saved model
    # In production, load from models/ directory
    try:
        gbm = joblib.load('models/gbm_model.pkl')
    except:
        # Dummy training - replace with real data
        # Use 2 features to match (rainfall, temperature)
        X_dummy = np.random.rand(100, 2)
        y_dummy = np.random.randint(0, 2, 100)
        gbm = train_gbm(X_dummy, y_dummy)
    unet = UNet()
    return gbm, unet


def generate_curated_flood_areas(base_risk: float, total_points: int = 25, jitter_m=250):
    base_risk = float(max(base_risk, 0.0))
    anchors = [
        (51.1694, 71.4491, 1.00),
        (51.1589, 71.4320, 0.90),
        (51.1750, 71.4650, 0.80),
        (51.1620, 71.4580, 0.70),
        (51.1720, 71.4380, 0.60),
    ]
    areas = []
    rng = np.random.default_rng(seed=42)

    def jitter_deg(meters, lat_ref):
        d_lat = meters / 111320.0
        d_lon = meters / (111320.0 * np.cos(np.radians(lat_ref)))
        return d_lat, d_lon

    for lat, lon, scale in anchors:
        areas.append({'lat': lat, 'lon': lon, 'risk': min(
            base_risk * scale, 1.0), 'anchor': True})

    remaining = max(0, total_points - len(anchors))
    if remaining > 0:
        scales = np.array([a[2] for a in anchors])
        probs = scales / scales.sum()
        anchor_indices = rng.choice(len(anchors), size=remaining, p=probs)
        for idx in anchor_indices:
            a_lat, a_lon, a_scale = anchors[idx]
            radius_m = rng.uniform(20, jitter_m)
            angle = rng.uniform(0, 2 * np.pi)
            j_lat_m = radius_m * np.cos(angle)
            j_lon_m = radius_m * np.sin(angle)
            d_lat, _ = jitter_deg(j_lat_m, a_lat)
            _, d_lon = jitter_deg(j_lon_m, a_lat)
            new_lat = a_lat + d_lat
            new_lon = a_lon + d_lon
            dist_scale = 1 - (radius_m / (jitter_m * 1.2))
            risk = min(base_risk * a_scale * dist_scale, 1.0)
            areas.append({'lat': new_lat, 'lon': new_lon,
                         'risk': risk, 'anchor': False})

    return areas


gbm, unet = load_models()

# Compute risk probability using current (slider or live) values
features = np.array([[rainfall, temperature]])
try:
    risk_prob = gbm.predict_proba(features)[0][1]
except Exception as e:
    st.warning(f"Model feature mismatch: {e}. Using heuristic risk estimate.")
    risk_prob = min(1.0, (rainfall / 300.0) *
                    (1.0 + max(0, (25 - abs(temperature - 15)) / 50)))

st.write("## Flood Risk Prediction")
st.write(f"Rainfall: {rainfall} mm, Temperature: {temperature} °C")
st.write(f"Predicted Flood Risk Probability: {risk_prob:.2f}")

if risk_prob > 0.6:
    st.warning("Elevated flood risk detected.")
elif risk_prob > 0.35:
    st.info("Moderate flood risk.")
else:
    st.success("Low flood risk.")

# Show raw data heads if present
if weather_df is not None and flood_df is not None:
    with st.expander("Sample Historical Data"):
        st.write(weather_df.head())
        st.write(flood_df.head())
elif weather_df is None and flood_df is None:
    st.info(
        "Load sample data or upload weather/flood data to see historical information.")

# Determine if we can run terrain/U-Net mode
terrain_mode = terrain_data is not None and weather_df is not None and flood_df is not None

# Sidebar control for number of points
desired_points = st.sidebar.slider(
    "Number of map points", 5, 150, 40, help="Controls how many flood risk points are generated")
jitter_distance = st.sidebar.selectbox("Point spread (radius)", [
                                       150, 250, 400, 600], index=1, help="Approximate max spread in meters around anchor areas")

mask_prob = None
if terrain_mode:
    st.write("Running U-Net terrain inference (demo)...")
    import torch
    terrain_tensor = torch.tensor(terrain_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    unet.eval()
    with torch.no_grad():
        mask_prob = torch.sigmoid(unet(terrain_tensor)).squeeze().cpu().numpy()
    flood_areas = generate_curated_flood_areas(risk_prob, total_points=desired_points, jitter_m=jitter_distance)
else:
    flood_areas = generate_curated_flood_areas(risk_prob, total_points=desired_points, jitter_m=jitter_distance)

# Persist in session state so a transient low-risk prediction doesn't blank the map unexpectedly
st.session_state['flood_areas'] = flood_areas

st.write("## Map Visualization")
m = folium.Map(location=[51.1694, 71.4491], zoom_start=10)

# Color gradient helper (simple two-stop between yellow and red)


def risk_to_color(r):
    # r in [0,1]; interpolate between yellow (#ffff00) and red (#ff0000)
    r = max(0.0, min(1.0, r))
    g_component = int(255 * (1 - r))  # from 255 -> 0
    return f"#{255:02x}{g_component:02x}00"


for area in flood_areas:
    color = risk_to_color(area['risk'])
    radius = 6 + area['risk'] * 6
    if area.get('anchor'):
        radius += 2  # make anchors slightly larger
    folium.CircleMarker(
        location=[area['lat'], area['lon']],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fillOpacity=0.65,
        popup=f"Flood Risk: {area['risk']:.2f}{' (anchor)' if area.get('anchor') else ''}"
    ).add_to(m)

from folium.raster_layers import ImageOverlay
from PIL import Image
import io, base64

def _profile_bounds(profile, height, width):
    t = profile["transform"]
    west = t.c
    north = t.f
    east = west + width * t.a
    south = north + height * t.e
    return south, west, north, east

st.sidebar.header("Map Layers")
show_unet_overlay = st.sidebar.checkbox("Show U-Net mask overlay", value=False, disabled=terrain_data is None)
unet_threshold = st.sidebar.slider("U-Net mask threshold", 0.1, 0.95, 0.8, 0.05, disabled=not show_unet_overlay)
overlay_opacity = st.sidebar.slider("Overlay opacity", 0.0, 1.0, 0.4, 0.05)
show_uploaded_mask = st.sidebar.checkbox("Show uploaded flood mask", value=False, disabled=flood_mask_data is None)

if show_unet_overlay and (mask_prob is not None) and (terrain_profile is not None):
    import numpy as np
    h, w = mask_prob.shape
    prob = np.clip(mask_prob, 0, 1)
    # Gradient alpha above threshold for smoother visualization
    span = max(1e-3, 1.0 - unet_threshold)
    alpha_f = np.clip((prob - unet_threshold) / span, 0, 1)
    alpha = (alpha_f * 220).astype(np.uint8)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 3] = alpha
    south, west, north, east = _profile_bounds(terrain_profile, h, w)
    img = Image.fromarray(rgba, mode='RGBA')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('ascii')
    ImageOverlay(image=data_url, bounds=[[south, west], [north, east]], name="U-Net Mask", opacity=overlay_opacity, interactive=False).add_to(m)

if show_uploaded_mask and (flood_mask_data is not None):
    import numpy as np
    h, w = flood_mask_data.shape
    mask = (flood_mask_data > 0).astype(np.uint8) * 200
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 1] = 255
    rgba[..., 3] = mask
    prof = flood_mask_profile if 'flood_mask_profile' in locals() and flood_mask_profile is not None else terrain_profile
    if prof is not None:
        south, west, north, east = _profile_bounds(prof, h, w)
        img = Image.fromarray(rgba, mode='RGBA')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('ascii')
    ImageOverlay(image=data_url, bounds=[[south, west], [north, east]], name="Flood Mask", opacity=overlay_opacity, interactive=False).add_to(m)

# Add a simple legend
legend_html = '''\
 <style>
 .legend-box {position: fixed; bottom: 50px; left: 50px; z-index: 9999; background: rgba(255,255,255,0.95); padding: 8px 10px; border:1px solid #444; font-size:12px; color:#1f2937; line-height:1.3; box-shadow:0 1px 4px rgba(0,0,0,0.2); border-radius:4px;}
 .legend-title {font-weight:700; color:#111827; margin-bottom:4px; display:block;}
 .legend-row {display:flex; align-items:center; margin:2px 0;}
 .legend-swatch {width:12px; height:12px; margin-right:6px; border:1px solid rgba(0,0,0,0.2);} 
 </style>
 <div class="legend-box">
     <span class="legend-title">Risk Legend</span>
     <div class="legend-row"><span class="legend-swatch" style="background:#ffff00;"></span><span>Low</span></div>
     <div class="legend-row"><span class="legend-swatch" style="background:#ffbf00;"></span><span>Moderate</span></div>
     <div class="legend-row"><span class="legend-swatch" style="background:#ff6000;"></span><span>High</span></div>
     <div class="legend-row"><span class="legend-swatch" style="background:#ff0000;"></span><span>Severe</span></div>
 </div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, width=700)

# Debug panel to help diagnose disappearing points
with st.expander("Debug Info"):
    st.write({
        'risk_prob': risk_prob,
        'terrain_mode': terrain_mode,
        'num_flood_areas': len(flood_areas),
        'live_weather_set': 'live_weather' in st.session_state
    })

# Optional: Kepler.gl for advanced visualization
if st.checkbox("Use Kepler.gl for advanced map"):
    try:
        from keplergl import KeplerGl
        # Use the same flood areas as Folium map for consistency
        data = pd.DataFrame([
            {
                'lat': area['lat'],
                'lon': area['lon'],
                'risk': area['risk'],
                'anchor': int(area.get('anchor', False))
            } for area in flood_areas
        ])
        map_ = KeplerGl(height=400)
        map_.add_data(data, "Flood Risk")
        config = {
            "version": "v1",
            "config": {
                "visState": {
                    "layers": [
                        {
                            "id": "flood-risk-layer",
                            "type": "point",
                            "config": {
                                "dataId": "Flood Risk",
                                "label": "Flood Risk Points",
                                "color": [255, 0, 0],
                                "columns": {
                                    "lat": "lat",
                                    "lng": "lon",
                                    "altitude": None
                                },
                                "isVisible": True,
                                "visConfig": {
                                    "radius": 20,
                                    "colorRange": {
                                        "name": "ColorBrewer YlOrRd-6",
                                        "type": "sequential",
                                        "category": "ColorBrewer",
                                        "colors": ["#ffffb2", "#fed976", "#feb24c", "#fd8d3c", "#f03b20", "#bd0026"]
                                    },
                                    "colorField": {"name": "risk", "type": "real"},
                                    "sizeField": {"name": "risk", "type": "real"},
                                    "sizeRange": [2, 25]
                                }
                            }
                        }
                    ]
                },
                "mapState": {
                    "latitude": 51.1694,
                    "longitude": 71.4491,
                    "zoom": 10
                }
            }
        }
        map_.config = config
        keplergl_static(map_, height=400)
    except ImportError:
        st.warning("Kepler.gl not installed. Install with: pip install keplergl")
