import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📉 Dimensionality Reduction (PCA + t-SNE)")

@st.cache_data
def load_data():
    return pd.read_csv("data/chicago_crime_500k_features.csv")

df = load_data()

pca_features = ["latitude","longitude","hour","day_of_week","month","arrest","domestic","beat","district","ward","community_area","year"]
dfp = df[pca_features].dropna().copy()

sample_size = st.slider("Sample size for t-SNE (performance)", 2000, 12000, 8000, step=1000)

X = dfp.values
X_scaled = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_scaled)

st.write("### PCA explained variance")
st.write(pca.explained_variance_ratio_, "Sum:", float(pca.explained_variance_ratio_.sum()))

# PCA loadings
loading_df = pd.DataFrame(pca.components_.T, index=pca_features, columns=["PC1","PC2","PC3"])
loading_df["importance"] = loading_df.abs().sum(axis=1)
st.write("### PCA Feature Loadings (Top 15)")
st.dataframe(loading_df.sort_values("importance", ascending=False).head(15))

# t-SNE
st.write("### t-SNE Visualization")
rng = np.random.RandomState(42)
idx = rng.choice(X_scaled.shape[0], sample_size, replace=False)
X_tsne_in = X_scaled[idx]

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
X_tsne = tsne.fit_transform(X_tsne_in)

fig = plt.figure(figsize=(8,6))
plt.scatter(X_tsne[:,0], X_tsne[:,1], s=5, alpha=0.6)
plt.title("t-SNE (sample)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
st.pyplot(fig)
