import os
from pathlib import Path
import urllib.request

import pandas as pd
import streamlit as st

LOCAL_DATA_PATH = "data/chicago_crime_500k_features.csv"
TMP_DATA_PATH = Path("/tmp/chicago_crime_500k_features.csv")

REQUIRED_COLS = ["latitude", "longitude", "year", "primary_type"]


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # convert safely to numeric
    for c in ["latitude", "longitude", "year", "month", "hour", "day", "week"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # string cols
    if "primary_type" in df.columns:
        df["primary_type"] = df["primary_type"].astype(str)

    return df


@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    """
    Loads dataset:
    - Local: ./data/chicago_crime_500k_features.csv
    - Streamlit Cloud: downloads from st.secrets['DATA_URL'] and stores in /tmp
    """
    # A) Local exists
    if os.path.exists(LOCAL_DATA_PATH):
        df = pd.read_csv(LOCAL_DATA_PATH)
        df = clean_df(df)
        return df

    # B) Cloud: download from secrets
    data_url = st.secrets.get("DATA_URL", "")

    if not data_url:
        raise FileNotFoundError(
            f"Dataset not found at '{LOCAL_DATA_PATH}'.\n"
            "And DATA_URL is not set in Streamlit Secrets.\n\n"
            "Fix:\n"
            "Streamlit Cloud → App → Settings → Secrets\n"
            'Add:\nDATA_URL = "DIRECT_DOWNLOAD_LINK"\n'
        )

    # Download once
    if not TMP_DATA_PATH.exists():
        urllib.request.urlretrieve(data_url, TMP_DATA_PATH)

    df = pd.read_csv(TMP_DATA_PATH)
    df = clean_df(df)

    # Basic validation
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    return df
