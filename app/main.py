"""PhishGuard API: URL in, phishing risk out."""
import json
import os
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import FEATURE_FUNCTIONS

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/champion.joblib"))
METRICS_PATH = MODEL_PATH.with_name("champion_metrics.json")
RISK_LOW = float(os.getenv("RISK_LOW", "0.4"))
RISK_HIGH = float(os.getenv("RISK_HIGH", "0.7"))
MAX_URL_LENGTH = 2048

app = FastAPI(title="PhishGuard API", version="0.1.0")

_bundle = None  # loaded once, on first use


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


def score(urls: list[str]) -> list[float]:
    bundle = get_bundle()
    features = FEATURE_FUNCTIONS[bundle["feature_set"]]
    X = pd.DataFrame([features(u) for u in urls])[bundle["feature_columns"]]
    return [float(p) for p in bundle["pipeline"].predict_proba(X)[:, 1]]


class PredictRequest(BaseModel):
    url: str = Field(min_length=1, max_length=MAX_URL_LENGTH)


class BatchRequest(BaseModel):
    urls: list[str] = Field(min_length=1, max_length=100)


def build_result(url: str, p: float, started: float) -> dict:
    return {
        "request_id": str(uuid.uuid4()),
        "url": url,
        "probability": round(p, 4),
        "verdict": risk_verdict(p),
        "model_version": get_model_info().get("model_version", "unknown"),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


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
    p = score([req.url])[0]
    return build_result(req.url, p, started)


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    if any(not u.strip() or len(u) > MAX_URL_LENGTH for u in req.urls):
        raise HTTPException(status_code=422, detail="Every URL must be 1-2048 characters.")
    started = time.perf_counter()
    probs = score(req.urls)
    return {"results": [build_result(u, p, started) for u, p in zip(req.urls, probs)]}