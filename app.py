import streamlit as st

st.set_page_config(page_title="PatrolIQ", layout="wide")

st.title("🚓 PatrolIQ - Chicago Crime Intelligence (500K)")
st.write(
    """
This app provides:
- Geographic hotspot clustering (KMeans + cluster centers)
- Temporal crime pattern dashboards
- Dimensionality reduction visualizations (PCA + t-SNE)
- MLflow experiment tracking + model monitoring

Use the **left sidebar** to open pages.
"""
)

st.info(
    "✅ If you're on Streamlit Cloud, make sure you added `DATA_URL` in **Secrets** "
    "so the app can download the dataset."
)
