import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_data

st.set_page_config(layout="wide")
st.title("⏱️ Temporal Pattern Analysis")

df = load_data()

col1, col2 = st.columns(2)

with col1:
    years = sorted(df["year"].dropna().unique().astype(int))
    year = st.selectbox("Year", years, index=len(years) - 1)

with col2:
    crime_types = sorted(df["primary_type"].dropna().unique())
    crime_type = st.selectbox("Primary Type", ["ALL"] + crime_types)

dff = df[df["year"] == year].copy()
if crime_type != "ALL":
    dff = dff[dff["primary_type"] == crime_type].copy()

# Hour analysis
if "hour" not in dff.columns:
    st.error("Dataset missing 'hour' column. Add it during feature engineering.")
    st.stop()

hour_counts = dff["hour"].dropna().astype(int).value_counts().sort_index()

fig = plt.figure()
plt.plot(hour_counts.index, hour_counts.values, marker="o")
plt.title("Crimes by Hour")
plt.xlabel("Hour")
plt.ylabel("Count")
st.pyplot(fig)
