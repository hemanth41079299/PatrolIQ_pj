import os
import streamlit as st
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

st.set_page_config(layout="wide")
st.title("🧪 MLflow Monitoring")

st.write("Shows latest MLflow runs + key clustering metrics.")

# ✅ Cloud-safe defaults:
# - Streamlit Cloud won't have your local ./mlruns folder
# - Use a remote tracking server OR disable this page gracefully in cloud
DEFAULT_LOCAL_URI = "file:./mlruns"
DEFAULT_REMOTE_URI = st.secrets.get("MLFLOW_TRACKING_URI", "") if hasattr(st, "secrets") else ""

# Detect if running on Streamlit Cloud (best-effort)
IS_CLOUD = bool(os.environ.get("STREAMLIT_CLOUD")) or bool(os.environ.get("STREAMLIT_SERVER_ADDRESS"))

# ---- Inputs
tracking_uri = st.text_input(
    "MLflow Tracking URI",
    value=DEFAULT_REMOTE_URI if DEFAULT_REMOTE_URI else DEFAULT_LOCAL_URI,
    help="Local: file:./mlruns | Remote: http://<host>:5000",
)

exp_name = st.text_input("Experiment Name", value="PatrolIQ_ChicagoCrime_500k")

# ---- Guard: Cloud can't see local filesystem mlruns
if IS_CLOUD and tracking_uri.strip().startswith("file:"):
    st.warning(
        "You're running on Streamlit Cloud. `file:./mlruns` won't exist there.\n\n"
        "✅ Fix options:\n"
        "1) Use a remote MLflow tracking server (recommended)\n"
        "2) Or hide/disable this MLflow page in Cloud\n\n"
        "Set `MLFLOW_TRACKING_URI` in Streamlit Secrets to a remote URI."
    )
    st.stop()

# ---- Connect
try:
    mlflow.set_tracking_uri(tracking_uri.strip())
except Exception as e:
    st.error(f"Failed to set tracking URI: {e}")
    st.stop()

# ---- Find experiment
try:
    exp = mlflow.get_experiment_by_name(exp_name.strip())
except Exception as e:
    st.error(f"Error while reading experiment: {e}")
    st.stop()

if not exp:
    st.error(
        "Experiment not found.\n\n"
        "If you're using local tracking, make sure you ran training in the same project folder.\n"
        "If you're using remote tracking, confirm the experiment exists on that server."
    )
    st.stop()

# ---- Fetch runs
client = MlflowClient()

try:
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=100,
    )
except Exception as e:
    st.error(f"Could not fetch runs from MLflow: {e}")
    st.stop()

if not runs:
    st.info("No runs found yet for this experiment.")
    st.stop()

# ---- Build table
rows = []
for r in runs:
    rows.append(
        {
            "run_name": r.data.tags.get("mlflow.runName"),
            "task": r.data.tags.get("task"),
            "algorithm": r.data.tags.get("algorithm"),
            "silhouette": r.data.metrics.get("silhouette_sample"),
            "davies_bouldin": r.data.metrics.get("davies_bouldin_sample"),
            "start_time": pd.to_datetime(r.info.start_time, unit="ms", utc=True).tz_convert(None),
            "run_id": r.info.run_id,
        }
    )

df_runs = pd.DataFrame(rows)

# ---- Summary (best run)
metric_choice = st.selectbox(
    "Rank runs by",
    ["silhouette (higher is better)", "davies_bouldin (lower is better)"],
    index=0,
)

if "silhouette" in metric_choice:
    best = df_runs.dropna(subset=["silhouette"]).sort_values("silhouette", ascending=False).head(1)
else:
    best = df_runs.dropna(subset=["davies_bouldin"]).sort_values("davies_bouldin", ascending=True).head(1)

st.subheader("Best run (based on selection)")
if len(best) == 0:
    st.info("No valid metrics found to rank runs (metrics missing).")
else:
    st.dataframe(best, width="stretch")

st.subheader("Latest runs")
st.dataframe(df_runs, width="stretch")

st.info(
    "✅ Local usage: run your training code so MLflow writes to `./mlruns`.\n"
    "✅ Cloud usage: use a remote MLflow server and set `MLFLOW_TRACKING_URI` in Streamlit Secrets."
)
