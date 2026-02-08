import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import load_data

st.set_page_config(layout="wide")
st.title("⏱️ Temporal Pattern Analysis")

df = load_data()

# ----------------------------
# Filters
# ----------------------------
col1, col2, col3 = st.columns([1, 2, 2])

with col1:
    years = sorted(df["year"].dropna().unique().astype(int))
    if not years:
        st.error("No 'year' values found in dataset.")
        st.stop()
    year = st.selectbox("Year", years, index=len(years) - 1)

with col2:
    crime_types = sorted(df["primary_type"].dropna().unique()) if "primary_type" in df.columns else []
    crime_type = st.selectbox("Primary Type", ["ALL"] + crime_types) if crime_types else "ALL"

with col3:
    max_rows = st.slider("Max rows (performance)", 50_000, 500_000, 250_000, step=50_000)

dff = df[df["year"] == year].copy()
if crime_type != "ALL" and "primary_type" in dff.columns:
    dff = dff[dff["primary_type"] == crime_type].copy()

# Performance guard (Cloud-safe)
if len(dff) > max_rows:
    dff = dff.sample(max_rows, random_state=42)

st.caption(f"Rows used: **{len(dff):,}**")

# Required columns check
needed = ["hour", "day_of_week", "month"]
missing = [c for c in needed if c not in dff.columns]
if missing:
    st.error(f"Dataset missing columns: {missing}. Add them during feature engineering.")
    st.stop()

# Clean types
dff["hour"] = pd.to_numeric(dff["hour"], errors="coerce")
dff["day_of_week"] = pd.to_numeric(dff["day_of_week"], errors="coerce")
dff["month"] = pd.to_numeric(dff["month"], errors="coerce")

# ----------------------------
# Charts layout
# ----------------------------
c1, c2 = st.columns(2)

# ---- Crimes by Hour
with c1:
    st.subheader("Crimes by Hour")
    hour_counts = dff["hour"].dropna().astype(int).value_counts().sort_index()
    # Ensure full 0-23 shown even if empty
    hour_counts = hour_counts.reindex(range(24), fill_value=0)

    fig = plt.figure()
    plt.plot(hour_counts.index, hour_counts.values, marker="o")
    plt.xlabel("Hour (0–23)")
    plt.ylabel("Count")
    st.pyplot(fig)

# ---- Crimes by Day of Week
with c2:
    st.subheader("Crimes by Day of Week")
    dow_counts = dff["day_of_week"].dropna().astype(int).value_counts().sort_index()

    # If your day_of_week is 0-6 or 1-7, label accordingly:
    # We'll normalize to 0-6 if it looks like 1-7.
    if dow_counts.index.min() == 1 and dow_counts.index.max() == 7:
        labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow_counts = dow_counts.reindex(range(1, 8), fill_value=0)
        x = labels
        y = dow_counts.values
    else:
        labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow_counts = dow_counts.reindex(range(7), fill_value=0)
        x = labels
        y = dow_counts.values

    fig = plt.figure()
    plt.bar(x, y)
    plt.xlabel("Day")
    plt.ylabel("Count")
    st.pyplot(fig)

# ---- Crimes by Month
st.subheader("Crimes by Month")
month_counts = dff["month"].dropna().astype(int).value_counts().sort_index()
month_counts = month_counts.reindex(range(1, 13), fill_value=0)

fig = plt.figure()
plt.plot(month_counts.index, month_counts.values, marker="o")
plt.xlabel("Month (1–12)")
plt.ylabel("Count")
st.pyplot(fig)

# ---- Heatmap: Hour x Day of Week
st.subheader("Heatmap: Hour × Day of Week")

# Normalize DOW to 0-6
dow = dff["day_of_week"].dropna().astype(int)
if dow.min() == 1 and dow.max() == 7:
    dff["_dow0"] = dff["day_of_week"].astype("Int64") - 1
else:
    dff["_dow0"] = dff["day_of_week"].astype("Int64")

dff["_hour"] = dff["hour"].astype("Int64")

hm = (
    dff.dropna(subset=["_dow0", "_hour"])
       .groupby(["_dow0", "_hour"])
       .size()
       .unstack(fill_value=0)
)

# Ensure full grid 7x24
hm = hm.reindex(index=range(7), columns=range(24), fill_value=0)

fig = plt.figure(figsize=(12, 4))
plt.imshow(hm.values, aspect="auto")
plt.yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
plt.xticks(range(0, 24, 1), range(24), rotation=0)
plt.xlabel("Hour")
plt.ylabel("Day of Week")
plt.title("Crime Intensity")
plt.colorbar(label="Count")
st.pyplot(fig)

# ---- Optional: Arrest/Domestic split
opt1, opt2 = st.columns(2)

with opt1:
    if "arrest" in dff.columns:
        st.subheader("Arrest Split")
        a = dff["arrest"].astype(str).value_counts()
        fig = plt.figure()
        plt.bar(a.index, a.values)
        plt.xlabel("Arrest")
        plt.ylabel("Count")
        st.pyplot(fig)
    else:
        st.info("No 'arrest' column found.")

with opt2:
    if "domestic" in dff.columns:
        st.subheader("Domestic Split")
        d = dff["domestic"].astype(str).value_counts()
        fig = plt.figure()
        plt.bar(d.index, d.values)
        plt.xlabel("Domestic")
        plt.ylabel("Count")
        st.pyplot(fig)
    else:
        st.info("No 'domestic' column found.")
