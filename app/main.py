"""PhishGuard API: URL in, phishing risk out. Every prediction is logged."""
import json
import logging
import os
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.db import init_db, log_prediction, set_feedback, stats
from src.features import FEATURE_FUNCTIONS

logger = logging.getLogger("phishguard")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/champion.joblib"))
METRICS_PATH = MODEL_PATH.with_name("champion_metrics.json")
RISK_LOW = float(os.getenv("RISK_LOW", "0.4"))
RISK_HIGH = float(os.getenv("RISK_HIGH", "0.7"))
MAX_URL_LENGTH = 2048

app = FastAPI(title="PhishGuard API", version="0.2.0")

_bundle = None


def get_bundle() -> dict:
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def get_model_info() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {"model_version": "unknown"}


def risk_verdict(p: float) -> str:
    if p >= RISK_HIGH:
        return "phishing"
    if p >= RISK_LOW:
        return "suspicious"
    return "legitimate"


def score(urls: list[str]) -> tuple[list[float], list[dict]]:
    """Return (probabilities, feature dicts) for the given URLs."""
    bundle = get_bundle()
    feature_fn = FEATURE_FUNCTIONS[bundle["feature_set"]]
    feats = [feature_fn(u) for u in urls]
    X = pd.DataFrame(feats)[bundle["feature_columns"]]
    probs = [float(p) for p in bundle["pipeline"].predict_proba(X)[:, 1]]
    return probs, feats


def warm_up() -> None:
    """Load the model and run one dummy prediction so the first real request is fast."""
    try:
        score(["https://example.com"])
    except Exception:
        logger.exception("warm-up failed")


class PredictRequest(BaseModel):
    url: str = Field(min_length=1, max_length=MAX_URL_LENGTH)


class BatchRequest(BaseModel):
    urls: list[str] = Field(min_length=1, max_length=100)


class FeedbackRequest(BaseModel):
    request_id: str
    true_label: int = Field(ge=0, le=1)  # 1 = phishing, 0 = legitimate


def respond(url: str, features: dict, p: float, started: float) -> dict:
    result = {
        "request_id": str(uuid.uuid4()),
        "url": url,
        "probability": round(p, 4),
        "verdict": risk_verdict(p),
        "model_version": get_model_info().get("model_version", "unknown"),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    try:  # a logging problem must never break a prediction
        log_prediction(
            request_id=result["request_id"], url=url, features=features,
            probability=result["probability"], verdict=result["verdict"],
            model_version=result["model_version"], latency_ms=result["latency_ms"],
        )
    except Exception:
        logger.exception("could not log prediction")
    return result


@app.get("/health")
def health():
    return {"status": "ok", "model_file_found": MODEL_PATH.exists()}


@app.get("/model-info")
def model_info():
    return {
        **get_model_info(),
        "thresholds": {"legitimate_below": RISK_LOW, "phishing_from": RISK_HIGH},
    }


@app.post("/predict")
def predict(req: PredictRequest):
    started = time.perf_counter()
    probs, feats = score([req.url])
    return respond(req.url, feats[0], probs[0], started)


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    if any(not u.strip() or len(u) > MAX_URL_LENGTH for u in req.urls):
        raise HTTPException(status_code=422, detail="Every URL must be 1-2048 characters.")
    started = time.perf_counter()
    probs, feats = score(req.urls)
    return {"results": [respond(u, f, p, started) for u, f, p in zip(req.urls, feats, probs)]}


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if not set_feedback(req.request_id, req.true_label):
        raise HTTPException(status_code=404, detail="Unknown request_id.")
    return {"status": "ok"}


@app.get("/stats")
def get_stats():
    return stats()


init_db()
warm_up()