import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("⏱️ Temporal Pattern Analysis")

@st.cache_data
def load_data():
    return pd.read_csv("data/chicago_crime_500k_features.csv")

df = load_data()

col1, col2 = st.columns(2)
with col1:
    year = st.selectbox("Year", sorted(df["year"].dropna().unique()))
with col2:
    crime_type = st.selectbox("Primary Type", ["ALL"] + sorted(df["primary_type"].dropna().unique()))

dff = df[df["year"] == year].copy()
if crime_type != "ALL":
    dff = dff[dff["primary_type"] == crime_type].copy()

# Hour chart
hourly = dff.groupby("hour").size().reindex(range(24), fill_value=0)

fig = plt.figure(figsize=(10,4))
plt.plot(hourly.index, hourly.values, marker="o")
plt.title("Crimes by Hour")
plt.xlabel("Hour")
plt.ylabel("Count")
st.pyplot(fig)

# Day of week chart
dow = dff.groupby("day_of_week").size().reindex(range(7), fill_value=0)
labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

fig2 = plt.figure(figsize=(10,4))
plt.bar(labels, dow.values)
plt.title("Crimes by Day of Week")
plt.xlabel("Day")
plt.ylabel("Count")
st.pyplot(fig2)

# Month chart
month = dff.groupby("month").size().reindex(range(1,13), fill_value=0)

fig3 = plt.figure(figsize=(10,4))
plt.plot(month.index, month.values, marker="o")
plt.title("Crimes by Month")
plt.xlabel("Month")
plt.ylabel("Count")
st.pyplot(fig3)
