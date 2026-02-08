import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

st.set_page_config(layout="wide")
st.title("🧪 MLflow Monitoring")

st.write("This page reads MLflow runs and shows best metrics.")

tracking_uri = st.text_input("MLflow Tracking URI", value="file:./mlruns")
mlflow.set_tracking_uri(tracking_uri)

exp_name = st.text_input("Experiment Name", value="PatrolIQ_ChicagoCrime_500k")
exp = mlflow.get_experiment_by_name(exp_name)

if not exp:
    st.error("Experiment not found. Check URI + name.")
    st.stop()

client = MlflowClient()
runs = client.search_runs(experiment_ids=[exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=50)

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

st.dataframe(pd.DataFrame(rows))
st.info("Tip: If you're using local mlflow server, paste its URI above (e.g., http://127.0.0.1:5000)")
