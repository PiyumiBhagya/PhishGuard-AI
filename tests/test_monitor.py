import numpy as np
import pandas as pd

from src.monitor import bucketed_metrics


def _df():
    return pd.DataFrame({
        "probability": [0.9, 0.8, 0.2, 0.1, 0.1, 0.2, 0.3, 0.9],
        "true_label": [1, 0, 1, 0, 1, 1, np.nan, 0],
        "latency_ms": [10, 20, 30, 40, 10, 10, 10, 10],
    })


def test_buckets_are_consecutive_groups():
    out = bucketed_metrics(_df(), size=4)
    assert list(out.index) == [1, 2]


def test_recall_and_flagged_share():
    out = bucketed_metrics(_df(), size=4)
    assert out.loc[1, "recall"] == 0.5          # 2 phishing rows, 1 flagged
    assert out.loc[1, "flagged_share"] == 0.5   # 0.9 and 0.8 flagged
    assert out.loc[2, "recall"] == 0.0          # both phishing rows missed


def test_accuracy_ignores_unlabelled_rows():
    out = bucketed_metrics(_df(), size=4)
    # batch 2 has 3 labelled rows and all 3 are wrong; the unlabelled row must not count
    assert out.loc[2, "accuracy"] == 0.0


def test_empty_frame():
    assert bucketed_metrics(pd.DataFrame(columns=["probability", "true_label", "latency_ms"])).empty