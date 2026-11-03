"""Train PhishGuard models and track the experiments with MLflow."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data import RAW_PATH, load_clean, split
from src.features import extract_features, extract_host_features

FEATURE_SETS = {"full_url": extract_features, "host_only": extract_host_features}
PRODUCTION_SET = "host_only"  # the model the API will serve
MODELS_DIR = Path("models")
TMP_DIR = Path("data/processed")  # git-ignored scratch folder

LGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

# My own made-up examples (second value: 1 = phishing, 0 = legitimate).
SANITY = [
    ("https://github.com/PiyumiBhagya/PhishGuard-AI", 0),
    ("https://en.wikipedia.org/wiki/Phishing", 0),
    ("https://www.sliit.lk/", 0),
    ("https://stackoverflow.com/questions/1", 0),
    ("https://docs.python.org/3/library/re.html", 0),
    ("https://learn.microsoft.com/en-us/azure/", 0),
    ("http://paypal-secure-login.verify-account.weeblysite.com/signin", 1),
    ("http://192.168.10.5/bank/login.php", 1),
    ("https://appleid-confirm-0utlook.firebaseapp.com/", 1),
    ("http://secure-update-account.top/login?user=1", 1),
    ("http://xn--paypa1-9ve.com/verify", 1),
]


def build_features(urls, fn) -> pd.DataFrame:
    return pd.DataFrame([fn(u) for u in list(urls)])


def make_pipeline() -> Pipeline:
    prep = ColumnTransformer(
        [(
            "tld",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=100,
                sparse_output=False,
            ),
            ["tld"],
        )],
        remainder="passthrough",
    )
    return Pipeline([("prep", prep), ("clf", LGBMClassifier(**LGBM_PARAMS))])


def evaluate(pipe, X, y, prefix: str) -> dict:
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
    }
    return {f"{prefix}_{k}": float(v) for k, v in out.items()}


def sanity_check(pipe, fn) -> float:
    urls = [u for u, _ in SANITY]
    proba = pipe.predict_proba(build_features(urls, fn))[:, 1]
    hits = 0
    for (url, label), p in zip(SANITY, proba):
        ok = int(p >= 0.5) == label
        hits += ok
        print(f"   {'OK ' if ok else 'BAD'} p_phishing={p:.2f}  {url[:70]}")
    return hits / len(SANITY)


def run(name, fn, parts, save_champion: bool) -> dict:
    train, val, test = parts
    print(f"\n=== {name}: extracting features ===")
    t0 = time.time()
    X_train = build_features(train["URL"], fn)
    X_val = build_features(val["URL"], fn)
    X_test = build_features(test["URL"], fn)
    print(f"   features ready in {time.time() - t0:.0f}s ({X_train.shape[1]} columns)")

    pipe = make_pipeline()
    with mlflow.start_run(run_name=name):
        mlflow.log_params({
            "feature_set": name,
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "n_features": X_train.shape[1],
            **{f"lgbm_{k}": v for k, v in LGBM_PARAMS.items()},
        })
        pipe.fit(X_train, train["is_phishing"])

        metrics = {
            **evaluate(pipe, X_val, val["is_phishing"], "val"),
            **evaluate(pipe, X_test, test["is_phishing"], "test"),
        }
        print("   sanity check on made-up real-looking URLs:")
        metrics["sanity_acc"] = sanity_check(pipe, fn)
        mlflow.log_metrics(metrics)

        TMP_DIR.mkdir(parents=True, exist_ok=True)
        bundle = {
            "pipeline": pipe,
            "feature_set": name,
            "feature_columns": list(X_train.columns),
        }
        path = TMP_DIR / f"{name}.joblib"
        joblib.dump(bundle, path)
        mlflow.log_artifact(str(path))

        if save_champion and name == PRODUCTION_SET:
            MODELS_DIR.mkdir(exist_ok=True)
            joblib.dump(bundle, MODELS_DIR / "champion.joblib")
            info = {
                "model_version": "v1",
                "feature_set": name,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics,
            }
            (MODELS_DIR / "champion_metrics.json").write_text(json.dumps(info, indent=2))
            print("   saved models/champion.joblib")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true",
                    help="quick smoke test on data/sample.csv (does not save a champion)")
    args = ap.parse_args()

    df = load_clean("data/sample.csv" if args.sample else RAW_PATH)
    parts = split(df)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("phishguard")

    results = {n: run(n, fn, parts, not args.sample) for n, fn in FEATURE_SETS.items()}

    cols = ["test_precision", "test_recall", "test_f1", "test_pr_auc", "sanity_acc"]
    print("\n=== SUMMARY ===")
    print(pd.DataFrame(results).T[cols].round(3))


if __name__ == "__main__":
    main()