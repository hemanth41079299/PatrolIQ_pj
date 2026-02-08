import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# Geo Hotspots page (UPDATED: fixes MessageSizeError + faster)
# ---------------------------------------------------------

st.set_page_config(layout="wide")
st.title("🌍 Geographic Crime Hotspots")

DATA_PATH = "data/chicago_crime_500k_features.csv"

# 1) Load once + cache
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Ensure numeric + clean types (safe-guards)
    for c in ["latitude", "longitude", "year"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "primary_type" in df.columns:
        df["primary_type"] = df["primary_type"].astype(str)

    return df

df = load_data(DATA_PATH)

# 2) Sidebar / Filters
col1, col2, col3 = st.columns(3)

with col1:
    years = sorted(df["year"].dropna().unique().astype(int))
    year = st.selectbox("Year", years, index=len(years)-1 if years else 0)

with col2:
    crime_types = sorted(df["primary_type"].dropna().unique())
    crime_type = st.selectbox("Primary Type", ["ALL"] + crime_types)

with col3:
    k = st.slider("KMeans clusters (K)", 5, 15, 10)

# 3) Filter data
dff = df[df["year"] == year].copy()

if crime_type != "ALL":
    dff = dff[dff["primary_type"] == crime_type].copy()

dff = dff.dropna(subset=["latitude", "longitude"])

st.write(f"Rows (filtered): **{len(dff):,}**")

if len(dff) < k:
    st.warning("Not enough rows for the selected K. Reduce K or change filters.")
    st.stop()

# ---------------------------------------------------------
# 4) CRITICAL FIX: reduce data sent to browser (MessageSizeError)
#    - Keep full dff for KMeans (optional)
#    - Use df_plot for visualization only
# ---------------------------------------------------------
MAX_PLOT_POINTS = 50_000  # safe default (reduce if still heavy)

if len(dff) > MAX_PLOT_POINTS:
    df_plot = dff.sample(MAX_PLOT_POINTS, random_state=42).copy()
else:
    df_plot = dff.copy()

st.caption(f"Rows (plotted): **{len(df_plot):,}** (sampled for performance)")

# ---------------------------------------------------------
# 5) KMeans for "live" cluster centers
#    - run on df_plot for speed (recommended)
#    - if you want "true" clusters from full filtered data, set USE_FULL_FOR_KMEANS=True
# ---------------------------------------------------------
USE_FULL_FOR_KMEANS = False

fit_df = dff if USE_FULL_FOR_KMEANS else df_plot

X = fit_df[["latitude", "longitude"]].to_numpy(dtype=np.float64)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

km = KMeans(n_clusters=int(k), random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)

fit_df = fit_df.copy()
fit_df["geo_cluster_live"] = labels

# cluster centers back to lat/lon
centers = scaler.inverse_transform(km.cluster_centers_)
centers_df = pd.DataFrame(centers, columns=["latitude", "longitude"])
centers_df["cluster"] = np.arange(int(k))

# ---------------------------------------------------------
# 6) PyDeck Layers
#    Heatmap uses df_plot (never full huge df)
#    Centers uses centers_df
# ---------------------------------------------------------

# Heatmap
heat_layer = pdk.Layer(
    "HeatmapLayer",
    data=df_plot,
    get_position='[longitude, latitude]',
    radiusPixels=60,
    opacity=0.85
)

# Cluster center markers
center_layer = pdk.Layer(
    "ScatterplotLayer",
    data=centers_df,
    get_position='[longitude, latitude]',
    get_radius=160,          # meters (approx visual size)
    pickable=True,
    auto_highlight=True
)

# Optional: show points colored by cluster (heavy; keep small if you enable)
SHOW_CLUSTER_POINTS = st.toggle("Show clustered points (sample)", value=False)

cluster_points_layer = None
if SHOW_CLUSTER_POINTS:
    # use only a smaller sample to avoid heavy rendering
    pts = df_plot.sample(min(20_000, len(df_plot)), random_state=42).copy()

    # Assign cluster labels for these points using the fitted scaler+km
    X_pts = pts[["latitude", "longitude"]].to_numpy(dtype=np.float64)
    X_pts_scaled = scaler.transform(X_pts)
    pts["geo_cluster_live"] = km.predict(X_pts_scaled).astype(int)

    cluster_points_layer = pdk.Layer(
        "ScatterplotLayer",
        data=pts,
        get_position='[longitude, latitude]',
        get_radius=35,
        get_fill_color='[geo_cluster_live * 20, 140, 200, 140]',
        pickable=True
    )

# View state centered on plotted data
view_state = pdk.ViewState(
    latitude=float(df_plot["latitude"].mean()),
    longitude=float(df_plot["longitude"].mean()),
    zoom=10
)

layers = [heat_layer, center_layer]
if cluster_points_layer is not None:
    layers.append(cluster_points_layer)

deck = pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    tooltip={"text": "Cluster: {cluster}"} if not SHOW_CLUSTER_POINTS else {"text": "Cluster: {geo_cluster_live}"}
)

st.pydeck_chart(deck)

# ---------------------------------------------------------
# 7) Small summary table (useful)
# ---------------------------------------------------------
st.subheader("Cluster Centers (Live)")
st.dataframe(
    centers_df.sort_values("cluster").reset_index(drop=True),
    use_container_width=True
)
