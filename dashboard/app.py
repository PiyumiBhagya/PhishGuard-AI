"""PhishGuard monitoring dashboard.

Run from the repo root:   streamlit run dashboard/app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import engine  # noqa: E402
from src.drift import (  # noqa: E402
    PSI_THRESHOLD,
    REFERENCE_PATH,
    SHARE_THRESHOLD,
    baseline_recall,
    compute_drift,
    features_of,
    performance,
)
from src.monitor import bucketed_metrics  # noqa: E402

st.set_page_config(page_title="PhishGuard monitor", layout="wide")

QUERY = text(
    "SELECT created_at, url, features_json, probability, verdict, model_version, "
    "latency_ms, true_label FROM predictions ORDER BY created_at DESC LIMIT :n"
)


@st.cache_data(ttl=10)
def load_history(limit: int) -> pd.DataFrame:
    """Predictions from the log database, oldest first."""
    df = pd.read_sql(QUERY, engine, params={"n": limit})
    return df.iloc[::-1].reset_index(drop=True)


@st.cache_data(ttl=10)
def load_reference() -> pd.DataFrame:
    return pd.read_csv(REFERENCE_PATH, keep_default_na=False)


def champion_info() -> dict:
    path = Path("models/champion_metrics.json")
    return json.loads(path.read_text()) if path.exists() else {}


st.title("PhishGuard monitoring")
st.caption("Live view of the prediction log: traffic, model quality and data drift.")

with st.sidebar:
    st.header("Settings")
    window = st.slider("Drift window (last N predictions)", 100, 1000, 250, 50)
    psi_threshold = st.slider("PSI threshold per feature", 0.05, 0.50, PSI_THRESHOLD, 0.05)
    share_threshold = st.slider("Alarm when this share of features drifted", 0.10, 0.90,
                                SHARE_THRESHOLD, 0.05)
    bucket = st.slider("Chart bucket (predictions per point)", 10, 100, 50, 10)
    if st.button("Refresh now"):
        st.cache_data.clear()
        st.rerun()

history = load_history(5000)
if history.empty:
    st.info("No predictions logged yet. Send traffic: python -m simulator.stream")
    st.stop()

recent = history.tail(window)
perf = performance(recent)
baseline = baseline_recall()
info = champion_info()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Model version", info.get("model_version", "unknown"))
c2.metric("Predictions logged", f"{len(history):,}")
c3.metric("Labelled", f"{int(history['true_label'].notna().sum()):,}")
c4.metric("Flagged phishing (window)", f"{(recent['probability'] >= 0.5).mean():.0%}")
if perf.get("n_labeled", 0) > 0:
    delta = f"{perf['recall'] - baseline:+.2f} vs baseline" if baseline else None
    c5.metric("Recall (window)", f"{perf['recall']:.2f}", delta)
else:
    c5.metric("Recall (window)", "n/a")

st.divider()
st.subheader(f"Data drift (last {len(recent)} predictions vs training reference)")

if len(recent) < 100:
    st.warning("Need at least 100 logged predictions to check drift.")
elif not REFERENCE_PATH.exists():
    st.warning("models/reference.csv is missing. Run: python -m src.drift --make-reference")
else:
    drift = compute_drift(load_reference(), features_of(recent), psi_threshold, share_threshold)
    summary = (f"{len(drift['drifted_features'])} of {drift['n_features']} features drifted "
               f"({drift['share_drifted']:.0%}, alarm at {share_threshold:.0%})")
    if drift["drift_detected"]:
        st.error("DRIFT DETECTED: " + summary)
    else:
        st.success("No drift: " + summary)

    if (baseline is not None and perf.get("n_labeled", 0) >= 50
            and perf["recall"] < baseline - 0.15):
        st.error(f"PERFORMANCE ALERT: recall {perf['recall']:.2f} is far below the "
                 f"training baseline {baseline:.2f}")

    left, right = st.columns([3, 2])
    with left:
        st.caption(f"PSI per feature (above {psi_threshold:.2f} counts as drifted)")
        st.bar_chart(pd.Series(drift["psi"]).sort_values(ascending=False))
    with right:
        st.caption("Verdict mix in the window")
        st.bar_chart(recent["verdict"].value_counts(normalize=True))

st.divider()
st.subheader(f"Trends ({bucket} predictions per point, oldest on the left)")
metrics = bucketed_metrics(history, bucket)
a, b = st.columns(2)
with a:
    st.caption("Recall and accuracy on labelled rows")
    st.line_chart(metrics[["recall", "accuracy"]])
with b:
    st.caption("Share of requests flagged as phishing")
    st.line_chart(metrics[["flagged_share"]])
st.caption("Average latency per point (ms)")
st.line_chart(metrics[["avg_latency_ms"]])

st.divider()
st.subheader("Latest predictions")
latest = history.tail(20).iloc[::-1][
    ["created_at", "url", "probability", "verdict", "model_version", "true_label"]
]
st.dataframe(latest, hide_index=True)