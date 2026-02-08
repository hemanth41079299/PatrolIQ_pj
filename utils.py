import os
from pathlib import Path
import urllib.request
import pandas as pd
import streamlit as st

LOCAL_DATA_PATH = "data/chicago_crime_500k_features.csv"
TMP_DATA_PATH = Path("/tmp/chicago_crime_500k_features.csv")

@st.cache_data(show_spinner=True)
def load_data():
    # 1️⃣ Local (works on laptop)
    if os.path.exists(LOCAL_DATA_PATH):
        return clean_df(pd.read_csv(LOCAL_DATA_PATH))

    # 2️⃣ Cloud (Google Drive)
    data_url = st.secrets.get("DATA_URL", "")

    if not data_url:
        st.error("DATA_URL not found in Streamlit secrets")
        st.stop()

    if not TMP_DATA_PATH.exists():
        urllib.request.urlretrieve(data_url, TMP_DATA_PATH)

    return clean_df(pd.read_csv(TMP_DATA_PATH))


def clean_df(df):
    for col in ["latitude", "longitude", "year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "primary_type" in df.columns:
        df["primary_type"] = df["primary_type"].astype(str)

    return df
