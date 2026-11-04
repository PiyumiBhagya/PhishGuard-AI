import numpy as np
import pandas as pd

from src.drift import compute_drift, performance, psi


def _frame(rng, shift=0.0, n=2000):
    return pd.DataFrame({
        "length": rng.normal(20 + shift, 4, n),
        "digits": rng.poisson(1 + shift / 4, n),
        "tld": rng.choice(["com", "org", "net", "ru"], n, p=[0.6, 0.2, 0.15, 0.05]),
    })


def test_psi_is_small_for_same_distribution():
    rng = np.random.default_rng(0)
    a, b = _frame(rng), _frame(rng)
    assert psi(a["length"], b["length"]) < 0.1
    assert psi(a["tld"], b["tld"]) < 0.1


def test_psi_is_large_for_shifted_distribution():
    rng = np.random.default_rng(1)
    a, b = _frame(rng), _frame(rng, shift=12)
    assert psi(a["length"], b["length"]) > 0.2


def test_psi_detects_new_categories():
    ref = pd.Series(["com"] * 90 + ["org"] * 10)
    cur = pd.Series(["xyz"] * 80 + ["com"] * 20)
    assert psi(ref, cur) > 0.2


def test_compute_drift_flags_only_when_enough_features_shift():
    rng = np.random.default_rng(2)
    ref = _frame(rng)
    same = compute_drift(ref, _frame(rng))
    shifted = compute_drift(ref, _frame(rng, shift=12))
    assert same["drift_detected"] is False
    assert shifted["drift_detected"] is True
    assert "length" in shifted["drifted_features"]


def test_performance_metrics():
    df = pd.DataFrame({
        "probability": [0.9, 0.8, 0.2, 0.1, 0.7],
        "true_label": [1, 1, 1, 0, None],
    })
    perf = performance(df)
    assert perf["n_labeled"] == 4
    assert perf["precision"] == 1.0
    assert abs(perf["recall"] - 0.667) < 0.01


def test_performance_without_labels():
    df = pd.DataFrame({"probability": [0.9], "true_label": [None]})
    assert performance(df) == {"n_labeled": 0}