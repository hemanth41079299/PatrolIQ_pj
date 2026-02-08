import os
import streamlit as st
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

st.set_page_config(layout="wide")
st.title("🧪 MLflow Monitoring")
st.write("Shows latest MLflow runs + key clustering metrics.")

# Detect Streamlit Cloud
IS_CLOUD = os.getenv("STREAMLIT_CLOUD", "") != "" or os.path.exists("/mount/src")

if IS_CLOUD:
    st.warning(
        "You're running on **Streamlit Cloud**.\n\n"
        "✅ Your local `./mlruns` folder is not available here, so MLflow experiments won't be found.\n\n"
        "**What you can do:**\n"
        "1) Use a remote MLflow server (best)\n"
        "2) Or keep MLflow monitoring for local-only runs"
    )
    st.stop()

# ---- Local mode below ----
tracking_uri = st.text_input("MLflow Tracking URI", value="file:./mlruns")
mlflow.set_tracking_uri(tracking_uri)

client = MlflowClient()

# Show experiments that exist (helps debugging)
exps = client.search_experiments()
exp_names = [e.name for e in exps]
st.caption(f"Found {len(exp_names)} experiments in this tracking URI.")
if exp_names:
    st.write("Available experiments:", exp_names)

exp_name = st.text_input("Experiment Name", value="PatrolIQ_ChicagoCrime_500k")
exp = mlflow.get_experiment_by_name(exp_name)

if not exp:
    st.error(
        f"Experiment not found: `{exp_name}`\n\n"
        "Tip: Copy the correct name from the list above, or ensure your training ran using this tracking URI."
    )
    st.stop()

runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    order_by=["attributes.start_time DESC"],
    max_results=50
)

rows = []
for r in runs:
    rows.append({
        "run_name": r.data.tags.get("mlflow.runName"),
        "task": r.data.tags.get("task"),
        "algorithm": r.data.tags.get("algorithm"),
        "silhouette": r.data.metrics.get("silhouette_sample"),
        "dbi": r.data.metrics.get("davies_bouldin_sample"),
        "start_time": r.info.start_time,
        "run_id": r.info.run_id
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.info(
    "✅ Local tip: Run your training script in the SAME project folder so `./mlruns` gets created.\n"
    "Remote tip: If using a server, set tracking URI like `http://127.0.0.1:5000`."
)
