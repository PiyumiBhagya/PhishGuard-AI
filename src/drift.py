"""Data-drift and performance monitoring.

Compares recent live traffic (features logged by the API) with the training
reference, using the Population Stability Index (PSI) per feature.

    python -m src.drift --make-reference     # once, and after every retrain
    python -m src.drift --window 250         # check the last 250 predictions
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

MODELS_DIR = Path("models")
REFERENCE_PATH = MODELS_DIR / "reference.csv"
REPORT_DIR = Path("reports")

PSI_THRESHOLD = 0.25     # common rule: < 0.1 no shift, 0.1-0.25 moderate, > 0.25 major shift
SHARE_THRESHOLD = 0.4    # drift alarm when this share of the features has a major shift
MIN_ROWS = 100           # too few rows = no reliable verdict
EPS = 1e-4
MAX_CATEGORIES = 15


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between two samples of one feature."""
    ref = reference.dropna()
    cur = current.dropna()
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    if not pd.api.types.is_numeric_dtype(ref) or ref.nunique() <= MAX_CATEGORIES:
        # categorical (or few distinct values): compare the category shares
        top = ref.value_counts().index[:MAX_CATEGORIES]
        ref_c = ref.where(ref.isin(top), "__other__")
        cur_c = cur.where(cur.isin(top), "__other__")
        cats = pd.Index(ref_c.unique()).union(pd.Index(cur_c.unique()))
        r = ref_c.value_counts(normalize=True).reindex(cats, fill_value=0.0)
        c = cur_c.value_counts(normalize=True).reindex(cats, fill_value=0.0)
    else:
        # numeric: bins from the reference quantiles
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        r = pd.Series(np.histogram(ref, edges)[0] / len(ref))
        c = pd.Series(np.histogram(cur, edges)[0] / len(cur))

    r = r.clip(lower=EPS).to_numpy(dtype=float)
    c = c.clip(lower=EPS).to_numpy(dtype=float)
    return float(((c - r) * np.log(c / r)).sum())


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame,
                  psi_threshold: float = PSI_THRESHOLD,
                  share_threshold: float = SHARE_THRESHOLD) -> dict:
    cols = [c for c in reference.columns if c in current.columns]
    scores = {c: psi(reference[c], current[c]) for c in cols}
    drifted = sorted((c for c, s in scores.items() if s > psi_threshold),
                     key=lambda c: -scores[c])
    share = len(drifted) / max(len(cols), 1)
    return {
        "drift_detected": bool(share >= share_threshold),
        "share_drifted": round(share, 3),
        "n_features": len(cols),
        "psi_threshold": psi_threshold,
        "share_threshold": share_threshold,
        "drifted_features": drifted,
        "psi": {c: round(s, 4) for c, s in sorted(scores.items(), key=lambda kv: -kv[1])},
    }


def performance(df: pd.DataFrame) -> dict:
    """Precision / recall / F1 on the rows that already have a true label."""
    lab = df.dropna(subset=["true_label"])
    if lab.empty:
        return {"n_labeled": 0}
    y = lab["true_label"].astype(int).to_numpy()
    p = (lab["probability"].to_numpy() >= 0.5).astype(int)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"n_labeled": int(len(lab)), "precision": round(precision, 3),
            "recall": round(recall, 3), "f1": round(f1, 3)}


def load_recent(window: int) -> pd.DataFrame:
    """The last `window` predictions from the log database, newest first."""
    from sqlalchemy import text

    from app.db import engine

    query = text(
        "SELECT created_at, features_json, probability, verdict, model_version, true_label "
        "FROM predictions ORDER BY created_at DESC LIMIT :n"
    )
    return pd.read_sql(query, engine, params={"n": window})


def features_of(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([json.loads(s) for s in df["features_json"]])


def baseline_recall() -> float | None:
    path = MODELS_DIR / "champion_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("metrics", {}).get("test_recall")


def build_reference(n: int = 5000, seed: int = 42) -> Path:
    """Save a sample of TRAINING features, computed with the champion's feature set."""
    import joblib

    from src.data import load_clean, split
    from src.features import FEATURE_FUNCTIONS

    bundle = joblib.load(MODELS_DIR / "champion.joblib")
    fn = FEATURE_FUNCTIONS[bundle["feature_set"]]
    train, _, _ = split(load_clean())
    sample = train.sample(min(n, len(train)), random_state=seed)
    feats = pd.DataFrame([fn(u) for u in sample["URL"]])[bundle["feature_columns"]]
    MODELS_DIR.mkdir(exist_ok=True)
    feats.to_csv(REFERENCE_PATH, index=False)
    return REFERENCE_PATH

def run(window: int = 250, psi_threshold: float = PSI_THRESHOLD,
        share_threshold: float = SHARE_THRESHOLD) -> dict:
    if not REFERENCE_PATH.exists():
        raise SystemExit("models/reference.csv is missing. Run: python -m src.drift --make-reference")
    reference = pd.read_csv(REFERENCE_PATH, keep_default_na=False)

    recent = load_recent(window)
    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "window": window,
        "n_rows": int(len(recent)),
    }
    if len(recent) < MIN_ROWS:
        result.update(drift_detected=False,
                      note=f"only {len(recent)} predictions logged, need {MIN_ROWS}")
        return result

    current = features_of(recent)
    result.update(compute_drift(reference, current, psi_threshold, share_threshold))
    result["model_version"] = str(recent["model_version"].iloc[0])
    result["verdict_mix"] = recent["verdict"].value_counts(normalize=True).round(3).to_dict()
    perf = performance(recent)
    result["performance"] = perf
    base = baseline_recall()
    result["baseline_recall"] = base
    result["performance_alert"] = bool(
        base is not None and perf.get("n_labeled", 0) >= 50 and perf["recall"] < base - 0.15
    )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-reference", action="store_true")
    ap.add_argument("--window", type=int, default=250)
    ap.add_argument("--psi-threshold", type=float, default=PSI_THRESHOLD)
    ap.add_argument("--share-threshold", type=float, default=SHARE_THRESHOLD)
    args = ap.parse_args()

    if args.make_reference:
        print("saved", build_reference())
        return

    result = run(args.window, args.psi_threshold, args.share_threshold)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "drift.json").write_text(json.dumps(result, indent=2))

    print(f"rows checked: {result['n_rows']} (window {result['window']})")
    if "note" in result:
        print(result["note"])
        return
    print(f"features drifted: {len(result['drifted_features'])}/{result['n_features']} "
          f"({result['share_drifted']:.0%}), alarm at {result['share_threshold']:.0%}")
    print("PSI per feature (> %.2f = drifted):" % result["psi_threshold"])
    for name, score in result["psi"].items():
        print(f"  {name:22s} {score:8.3f} {'<-- drifted' if score > result['psi_threshold'] else ''}")
    print("verdict mix:", result["verdict_mix"])
    print("performance on labelled rows:", result["performance"],
          "| baseline recall:", result["baseline_recall"])
    print("\nDRIFT DETECTED" if result["drift_detected"] else "\nno drift")
    if result["performance_alert"]:
        print("PERFORMANCE ALERT: recall dropped well below the baseline")


if __name__ == "__main__":
    main()