import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils import load_data

st.set_page_config(layout="wide")
st.title("🌍 Geographic Crime Hotspots")

df = load_data()

# Filters
col1, col2, col3 = st.columns(3)

with col1:
    years = sorted(df["year"].dropna().unique().astype(int))
    year = st.selectbox("Year", years, index=len(years) - 1)

with col2:
    crime_types = sorted(df["primary_type"].dropna().unique())
    crime_type = st.selectbox("Primary Type", ["ALL"] + crime_types)

with col3:
    k = st.slider("KMeans clusters (K)", 5, 15, 10)

# Filter df
dff = df[df["year"] == year].copy()
if crime_type != "ALL":
    dff = dff[dff["primary_type"] == crime_type].copy()

dff = dff.dropna(subset=["latitude", "longitude"])
st.write(f"Rows (filtered): **{len(dff):,}**")

if len(dff) < int(k):
    st.warning("Not enough rows for the selected K. Reduce K or change filters.")
    st.stop()

# ✅ MessageSizeError Fix: sample for plotting
MAX_PLOT_POINTS = 50_000
df_plot = dff.sample(MAX_PLOT_POINTS, random_state=42) if len(dff) > MAX_PLOT_POINTS else dff.copy()
st.caption(f"Rows (plotted): **{len(df_plot):,}** (sampled for performance)")

# KMeans on sample (fast + stable)
X = df_plot[["latitude", "longitude"]].to_numpy(dtype=np.float64)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

km = KMeans(n_clusters=int(k), random_state=42, n_init=10)
km.fit(X_scaled)

# centers back to real coords
centers = scaler.inverse_transform(km.cluster_centers_)
centers_df = pd.DataFrame(centers, columns=["latitude", "longitude"])
centers_df["cluster"] = np.arange(int(k))

# Heatmap layer
heat_layer = pdk.Layer(
    "HeatmapLayer",
    data=df_plot,
    get_position="[longitude, latitude]",
    radiusPixels=60,
    opacity=0.85,
)

# Cluster center markers
center_layer = pdk.Layer(
    "ScatterplotLayer",
    data=centers_df,
    get_position="[longitude, latitude]",
    get_radius=160,
    pickable=True,
    auto_highlight=True,
)

# optional points layer
SHOW_CLUSTER_POINTS = st.toggle("Show clustered points (sample)", value=False)
cluster_points_layer = None

if SHOW_CLUSTER_POINTS:
    pts = df_plot.sample(min(20_000, len(df_plot)), random_state=42).copy()
    X_pts = pts[["latitude", "longitude"]].to_numpy(dtype=np.float64)
    pts["geo_cluster"] = km.predict(scaler.transform(X_pts)).astype(int)

    cluster_points_layer = pdk.Layer(
        "ScatterplotLayer",
        data=pts,
        get_position="[longitude, latitude]",
        get_radius=35,
        get_fill_color="[geo_cluster * 20, 140, 200, 140]",
        pickable=True,
    )

view_state = pdk.ViewState(
    latitude=float(df_plot["latitude"].mean()),
    longitude=float(df_plot["longitude"].mean()),
    zoom=10,
)

layers = [heat_layer, center_layer]
if cluster_points_layer is not None:
    layers.append(cluster_points_layer)

deck = pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    tooltip={"text": "Cluster: {cluster}"} if not SHOW_CLUSTER_POINTS else {"text": "Cluster: {geo_cluster}"},
)

st.pydeck_chart(deck)

st.subheader("Cluster Centers")
st.dataframe(centers_df.sort_values("cluster"), use_container_width=True)
