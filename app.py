import streamlit as st

st.set_page_config(
    page_title="PatrolIQ - Chicago Crime",
    layout="wide"
)

st.title("🚓 PatrolIQ - Chicago Crime Intelligence (500K)")
st.write("""
This app provides:
- Geographic hotspot clustering (KMeans + cluster boundaries)
- Temporal crime pattern dashboards
- PCA + t-SNE dimensionality reduction visualizations
- MLflow experiment tracking and model monitoring
""")

st.info("Use the left sidebar to open pages: Geo, Temporal, Dimensionality Reduction, MLflow.")
