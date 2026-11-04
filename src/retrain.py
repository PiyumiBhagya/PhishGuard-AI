"""Retrain on logged, labelled traffic and promote the challenger only if it is better.

    python -m src.retrain             # train a challenger, compare it with the champion
    python -m src.retrain --dry-run   # same, but never replace the champion
    python -m src.retrain --if-drift  # only when src.drift reports drift or a performance alert
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.data import load_clean, registered_domain, split
from src.drift import features_of
from src.drift import run as drift_run
from src.features import FEATURE_FUNCTIONS

MODELS_DIR = Path("models")
CHAMPION_PATH = MODELS_DIR / "champion.joblib"
METRICS_PATH = MODELS_DIR / "champion_metrics.json"
HISTORY_PATH = MODELS_DIR / "history.json"

RECENT_WEIGHT = 20.0   # one logged example counts as much as this many original examples
MIN_LABELLED = 150     # fewer labelled predictions than this: do not retrain
EVAL_SHARE = 0.4       # share of the logged data kept aside to judge the challenger
MAX_REGRESSION = 0.03  # challenger may lose at most this much F1 on the original holdout
MIN_GAIN = 0.02        # challenger must gain at least this much F1 on recent traffic
SEED = 42


def next_version(current: str) -> str:
    try:
        return f"v{int(str(current).lstrip('v')) + 1}"
    except ValueError:
        return "v2"


def should_promote(champion_holdout_f1: float, challenger_holdout_f1: float,
                   champion_recent_f1: float, challenger_recent_f1: float,
                   max_regression: float = MAX_REGRESSION,
                   min_gain: float = MIN_GAIN) -> tuple[bool, str]:
    """Promotion rule: no real regression on the original holdout AND a clear gain on recent data."""
    if challenger_holdout_f1 < champion_holdout_f1 - max_regression:
        return False, (f"rejected: holdout F1 fell from {champion_holdout_f1:.3f} to "
                       f"{challenger_holdout_f1:.3f} (allowed drop {max_regression})")
    if challenger_recent_f1 < champion_recent_f1 + min_gain:
        return False, (f"rejected: recent-traffic F1 {challenger_recent_f1:.3f} is not at least "
                       f"{min_gain} above the champion's {champion_recent_f1:.3f}")
    return True, (f"promoted: recent-traffic F1 {champion_recent_f1:.3f} -> {challenger_recent_f1:.3f}, "
                  f"holdout F1 {champion_holdout_f1:.3f} -> {challenger_holdout_f1:.3f}")


def load_labelled_logs() -> pd.DataFrame:
    """Logged predictions that already have a true label, oldest first."""
    from sqlalchemy import text

    from app.db import engine

    query = text(
        "SELECT created_at, url, features_json, true_label FROM predictions "
        "WHERE true_label IS NOT NULL ORDER BY created_at"
    )
    return pd.read_sql(query, engine)


def split_logs(logs: pd.DataFrame, eval_share: float = EVAL_SHARE, seed: int = SEED):
    """Train / eval split of the logged data. Same registered domain never lands in both parts."""
    groups = logs["url"].map(registered_domain)
    splitter = GroupShuffleSplit(n_splits=1, test_size=eval_share, random_state=seed)
    train_idx, eval_idx = next(splitter.split(logs, groups=groups))
    return logs.iloc[train_idx], logs.iloc[eval_idx]


def feature_frame(bundle: dict, urls) -> pd.DataFrame:
    fn = FEATURE_FUNCTIONS[bundle["feature_set"]]
    return pd.DataFrame([fn(u) for u in urls])[bundle["feature_columns"]]


def _write_history(entry: dict) -> None:
    history = json.loads(HISTORY_PATH.read_text()) if HISTORY_PATH.exists() else []
    history.append(entry)
    HISTORY_PATH.write_text(json.dumps(history, indent=2))


def retrain(dry_run: bool = False) -> dict:
    import joblib
    import mlflow

    from src.train import evaluate, make_pipeline  # heavy imports only when really retraining

    bundle = joblib.load(CHAMPION_PATH)
    cols = bundle["feature_columns"]

    logs = load_labelled_logs()
    if len(logs) < MIN_LABELLED:
        return {"promoted": False,
                "reason": f"only {len(logs)} labelled predictions logged, need {MIN_LABELLED}"}

    logs_train, logs_eval = split_logs(logs)
    y_recent = logs_eval["true_label"].astype(int)
    if y_recent.nunique() < 2:
        return {"promoted": False, "reason": "the recent evaluation set has only one class"}

    print("original data: building features (about a minute) ...")
    train, val, _ = split(load_clean())
    X_orig = feature_frame(bundle, train["URL"])
    X_log = features_of(logs_train)[cols]
    X_train = pd.concat([X_orig, X_log], ignore_index=True)
    y_train = pd.concat([train["is_phishing"].reset_index(drop=True),
                         logs_train["true_label"].astype(int).reset_index(drop=True)],
                        ignore_index=True)
    weights = np.r_[np.ones(len(X_orig)), np.full(len(X_log), RECENT_WEIGHT)]

    print(f"training the challenger on {len(X_orig):,} original + {len(X_log):,} logged rows ...")
    challenger = make_pipeline()
    challenger.fit(X_train, y_train, clf__sample_weight=weights)

    X_val = feature_frame(bundle, val["URL"])
    y_val = val["is_phishing"]
    X_recent = features_of(logs_eval)[cols]

    scores = {}
    for name, model in (("champion", bundle["pipeline"]), ("challenger", challenger)):
        scores[name] = {**evaluate(model, X_val, y_val, "holdout"),
                        **evaluate(model, X_recent, y_recent, "recent")}

    promote, reason = should_promote(
        scores["champion"]["holdout_f1"], scores["challenger"]["holdout_f1"],
        scores["champion"]["recent_f1"], scores["challenger"]["recent_f1"],
    )

    current = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    old_version = current.get("model_version", "v1")
    new_version = next_version(old_version)
    now = datetime.now(timezone.utc).isoformat()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("phishguard")
    with mlflow.start_run(run_name=f"retrain-{new_version}"):
        mlflow.log_params({"n_original": len(X_orig), "n_logged_train": len(X_log),
                           "n_logged_eval": len(X_recent), "recent_weight": RECENT_WEIGHT,
                           "feature_set": bundle["feature_set"]})
        for name in scores:
            mlflow.log_metrics({f"{name}_{k}": v for k, v in scores[name].items()})
        mlflow.set_tag("promoted", str(promote and not dry_run))
        mlflow.set_tag("reason", reason)

    if promote and not dry_run:
        MODELS_DIR.mkdir(exist_ok=True)
        new_bundle = {"pipeline": challenger, "feature_set": bundle["feature_set"],
                      "feature_columns": cols}
        tmp = CHAMPION_PATH.with_suffix(".tmp")
        joblib.dump(new_bundle, tmp)
        os.replace(tmp, CHAMPION_PATH)  # the API never sees a half-written file
        metrics = dict(scores["challenger"])
        # drift.baseline_recall() reads "test_recall": here it is the recall on the fixed holdout
        metrics["test_recall"] = scores["challenger"]["holdout_recall"]
        METRICS_PATH.write_text(json.dumps({
            "model_version": new_version,
            "feature_set": bundle["feature_set"],
            "trained_at": now,
            "trained_on": {"original_rows": len(X_orig), "logged_rows": len(X_log)},
            "metrics": metrics,
        }, indent=2))

    entry = {"time": now, "from_version": old_version,
             "to_version": new_version if promote and not dry_run else old_version,
             "promoted": bool(promote and not dry_run), "dry_run": dry_run, "reason": reason,
             "champion": scores["champion"], "challenger": scores["challenger"]}
    _write_history(entry)
    return entry


def print_report(result: dict) -> None:
    if "champion" not in result:
        print("not retrained:", result["reason"])
        return
    keys = ["precision", "recall", "f1"]
    print("\n                        champion  challenger")
    for part in ("holdout", "recent"):
        for k in keys:
            c = result["champion"][f"{part}_{k}"]
            n = result["challenger"][f"{part}_{k}"]
            print(f"  {part:8s} {k:10s}     {c:7.3f}     {n:7.3f}")
    print("\n" + result["reason"])
    if result["promoted"]:
        print(f"NEW CHAMPION: {result['to_version']} (models/champion.joblib updated)")
    elif result.get("dry_run"):
        print("dry run: the champion was not changed")
    else:
        print(f"champion stays {result['from_version']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="compare only, never promote")
    ap.add_argument("--if-drift", action="store_true",
                    help="retrain only when drift or a performance alert is detected")
    args = ap.parse_args()

    if args.if_drift:
        status = drift_run()
        if not (status.get("drift_detected") or status.get("performance_alert")):
            print("no drift and no performance alert: not retraining")
            return
        print("drift or performance alert detected: retraining")

    print_report(retrain(dry_run=args.dry_run))


if __name__ == "__main__":
    main()