import os
from pathlib import Path

import pandas as pd
import streamlit as st

LOCAL_DATA_PATH = Path("data/chicago_crime_500k_features.csv")
TMP_DATA_PATH = Path("/tmp/chicago_crime_500k_features.csv")


@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    # 1) Local first (works in your laptop)
    if LOCAL_DATA_PATH.exists():
        df = pd.read_csv(LOCAL_DATA_PATH)
        return clean_df(df)

    # 2) Cloud: get DATA_URL
    data_url = None
    try:
        data_url = st.secrets.get("DATA_URL", None)
    except Exception:
        data_url = None

    if not data_url:
        data_url = os.getenv("DATA_URL")

    if not data_url:
        raise FileNotFoundError(
            "Dataset not found locally and DATA_URL is not set.\n"
            "Add DATA_URL in Streamlit Cloud → Settings → Secrets."
        )

    # 3) Download once into /tmp
    if not TMP_DATA_PATH.exists():
        download_file(data_url, TMP_DATA_PATH)

    # 4) Read CSV
    df = pd.read_csv(TMP_DATA_PATH)

    # 5) If Google Drive returned HTML (bad download), fail with clear message
    if len(df.columns) <= 1 or any("html" in str(c).lower() for c in df.columns):
        raise ValueError(
            "Downloaded file is not a valid CSV (looks like an HTML page from Google Drive).\n"
            "Fix: Use gdown download + ensure DATA_URL points to a Google Drive FILE link."
        )

    return clean_df(df)


def download_file(url: str, out_path: Path) -> None:
    # ✅ Google Drive support
    if "drive.google.com" in url:
        import gdown
        gdown.download(url, str(out_path), quiet=False, fuzzy=True)
        return

    # Normal direct URL
    import urllib.request
    urllib.request.urlretrieve(url, out_path)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # standardize columns
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # generate year/month/hour/day_of_week if missing
    if "year" not in df.columns:
        for date_col in ["date", "date_time", "datetime"]:
            if date_col in df.columns:
                dt = pd.to_datetime(df[date_col], errors="coerce")
                df["year"] = dt.dt.year
                df["month"] = dt.dt.month
                df["hour"] = dt.dt.hour
                df["day_of_week"] = dt.dt.dayofweek
                break

    return df
