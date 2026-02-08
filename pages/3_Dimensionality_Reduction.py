import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from utils import load_data  # ✅ use centralized loader (local + cloud)

st.set_page_config(layout="wide")
st.title("📉 Dimensionality Reduction (PCA + t-SNE)")

# ✅ Load data (works locally + Streamlit Cloud)
df = load_data()

# ---- Features
pca_features = [
    "latitude", "longitude", "hour", "day_of_week", "month",
    "arrest", "domestic", "beat", "district", "ward",
    "community_area", "year"
]

# ---- Validate columns (prevents KeyError)
missing = [c for c in pca_features if c not in df.columns]
if missing:
    st.error(f"Dataset is missing these required columns: {missing}")
    st.stop()

# ---- Prep dataframe
dfp = df[pca_features].copy()

# Convert bools to ints (safe for scaling)
for c in ["arrest", "domestic"]:
    dfp[c] = dfp[c].astype(int)

dfp = dfp.dropna()

st.write(f"Rows available after cleaning: **{len(dfp):,}**")

if len(dfp) < 1000:
    st.warning("Not enough rows after cleaning to run PCA/t-SNE. Check your dataset.")
    st.stop()

# ---- Controls
col1, col2, col3 = st.columns(3)

with col1:
    sample_size = st.slider(
        "Sample size for t-SNE (performance)",
        2000,
        min(12000, len(dfp)),
        min(8000, len(dfp)),
        step=1000,
    )

with col2:
    perplexity = st.slider(
        "t-SNE perplexity",
        5,
        50,
        30,
        step=5,
    )

with col3:
    learning_rate = st.slider(
        "t-SNE learning rate",
        10,
        500,
        200,
        step=10,
    )

# t-SNE constraint: perplexity must be < n_samples
max_valid_perp = max(5, min(50, (sample_size - 1) // 3))
if perplexity >= sample_size:
    st.warning("Perplexity must be < sample size. Lower perplexity or increase sample size.")
    st.stop()
if perplexity > max_valid_perp:
    st.info(f"Tip: With sample size {sample_size}, keep perplexity ≤ {max_valid_perp} for stability.")

# ---- Scale
X = dfp.values.astype(np.float64)
X_scaled = StandardScaler().fit_transform(X)

# ---- PCA
pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_scaled)

st.write("### PCA explained variance")
st.write(pca.explained_variance_ratio_, "Sum:", float(pca.explained_variance_ratio_.sum()))

loading_df = pd.DataFrame(pca.components_.T, index=pca_features, columns=["PC1", "PC2", "PC3"])
loading_df["importance"] = loading_df[["PC1", "PC2", "PC3"]].abs().sum(axis=1)

st.write("### PCA Feature Loadings (Top 15)")
st.dataframe(loading_df.sort_values("importance", ascending=False).head(15), width="stretch")  # ✅ cloud-safe

# Optional PCA scatter (fast)
with st.expander("Show PCA scatter (fast)", expanded=False):
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], s=5, alpha=0.6)
    plt.title("PCA (PC1 vs PC2)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    st.pyplot(fig)

# ---- t-SNE
st.write("### t-SNE Visualization")

rng = np.random.RandomState(42)
idx = rng.choice(X_scaled.shape[0], sample_size, replace=False)
X_tsne_in = X_scaled[idx]

tsne = TSNE(
    n_components=2,
    perplexity=perplexity,
    learning_rate=learning_rate,
    init="pca",
    random_state=42,
)

X_tsne = tsne.fit_transform(X_tsne_in)

fig = plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=5, alpha=0.6)
plt.title("t-SNE (sample)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
st.pyplot(fig)
