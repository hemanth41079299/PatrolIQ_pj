import pandas as pd
import streamlit as st

@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    return df
