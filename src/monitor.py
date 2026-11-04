"""Helpers for the monitoring dashboard (kept out of the UI file so they can be tested)."""
import numpy as np
import pandas as pd

COLUMNS = ["flagged_share", "recall", "accuracy", "avg_latency_ms"]


def bucketed_metrics(df: pd.DataFrame, size: int = 50) -> pd.DataFrame:
    """Metrics per bucket of `size` consecutive predictions (oldest first).

    Needs the columns: probability, true_label (NaN when unknown), latency_ms.
    """
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)

    d = df.reset_index(drop=True).copy()
    d["batch"] = d.index // size + 1
    d["flagged"] = d["probability"] >= 0.5

    rows = []
    for batch, g in d.groupby("batch"):
        positives = g[g["true_label"] == 1]
        labelled = g.dropna(subset=["true_label"])
        rows.append({
            "batch": batch,
            "flagged_share": float(g["flagged"].mean()),
            "recall": float(positives["flagged"].mean()) if len(positives) else np.nan,
            "accuracy": float((labelled["flagged"] == (labelled["true_label"] == 1)).mean())
            if len(labelled) else np.nan,
            "avg_latency_ms": float(g["latency_ms"].mean()),
        })
    return pd.DataFrame(rows).set_index("batch")