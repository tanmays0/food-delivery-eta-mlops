from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "production_eta.joblib"
METRICS_PATH = ROOT / "outputs" / "training_metrics.json"
STATIC_DIR = ROOT / "static"

app = FastAPI(
    title="Food Delivery ETA Predictor",
    description="Real-time ETA inference microservice (FastAPI + XGBoost)",
    version="1.0.0",
)

_bundle: dict[str, Any] | None = None
_request_count = 0
_latencies_ms: list[float] = []

PREDICT_REQUESTS = Counter(
    "eta_predict_requests_total",
    "Total /predict requests",
    ["status", "fallback"],
)
PREDICT_LATENCY = Histogram(
    "eta_predict_latency_seconds",
    "Prediction latency in seconds",
    buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
ETA_VALUE = Histogram(
    "eta_predicted_minutes",
    "Predicted ETA distribution in minutes",
    buckets=(10, 20, 30, 40, 50, 60, 75, 90, 120),
)
MODEL_LOADED = Gauge("eta_model_loaded", "1 if production model is loaded")


class ETARequest(BaseModel):
    distance_km: float = Field(..., ge=0.1, le=30, examples=[4.5])
    prep_time_p50_min: float = Field(..., ge=1, le=90, examples=[18.0])
    courier_load: float = Field(..., ge=0, le=10, examples=[1])
    zone_order_rate: float = Field(..., ge=0, le=200, examples=[42.0])
    hour_of_day: float = Field(..., ge=0, le=23, examples=[19])
    day_of_week: float = Field(..., ge=0, le=6, examples=[5])
    is_rain: float = Field(..., ge=0, le=1, examples=[0])
    traffic_index: float = Field(..., ge=0.1, le=3.0, examples=[1.3])
    time_since_accept_min: float = Field(..., ge=0, le=60, examples=[3.0])


class ETAResponse(BaseModel):
    eta_minutes: float
    model_name: str
    model_version: str
    fallback: bool
    latency_ms: float


def load_model() -> dict[str, Any]:
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing {MODEL_PATH}. Run: python src/train.py"
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def heuristic_eta(payload: ETARequest) -> float:
    travel = payload.distance_km * (4.0 + 1.5 * payload.traffic_index)
    if payload.is_rain >= 0.5:
        travel *= 1.15
    return float(
        round(
            payload.prep_time_p50_min
            + 3.0
            + 1.5 * payload.courier_load
            + travel,
            2,
        )
    )


@app.on_event("startup")
def startup() -> None:
    try:
        load_model()
        MODEL_LOADED.set(1)
    except FileNotFoundError:
        MODEL_LOADED.set(0)


@app.get("/health")
def health() -> dict[str, Any]:
    ready = MODEL_PATH.exists()
    return {"status": "ok" if ready else "model_missing", "model_loaded": ready}


@app.get("/stats")
def stats() -> dict[str, Any]:
    train_metrics = {}
    if METRICS_PATH.exists():
        train_metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    p95 = float(np.percentile(_latencies_ms, 95)) if _latencies_ms else None
    return {
        "requests_served": _request_count,
        "latency_ms_p95": p95,
        "latency_ms_mean": float(np.mean(_latencies_ms)) if _latencies_ms else None,
        "training": train_metrics,
    }


@app.get("/metrics")
def prometheus_metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=ETAResponse)
def predict(payload: ETARequest) -> ETAResponse:
    global _request_count
    started = time.perf_counter()
    fallback = False
    status = "ok"
    try:
        bundle = load_model()
        cols = bundle["feature_columns"]
        row = np.array([[getattr(payload, c) for c in cols]], dtype=float)
        eta = float(bundle["model"].predict(row)[0])
        model_name = str(bundle["model_name"])
        model_version = "production-v1"
    except Exception:
        eta = heuristic_eta(payload)
        model_name = "heuristic_fallback"
        model_version = "fallback-v1"
        fallback = True
        status = "fallback"

    latency_s = time.perf_counter() - started
    latency_ms = latency_s * 1000.0
    _request_count += 1
    _latencies_ms.append(latency_ms)
    if len(_latencies_ms) > 5000:
        del _latencies_ms[:-2000]

    PREDICT_REQUESTS.labels(status=status, fallback=str(fallback).lower()).inc()
    PREDICT_LATENCY.observe(latency_s)
    ETA_VALUE.observe(eta)

    return ETAResponse(
        eta_minutes=round(eta, 2),
        model_name=model_name,
        model_version=model_version,
        fallback=fallback,
        latency_ms=round(latency_ms, 3),
    )


@app.get("/")
def index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Demo UI missing")
    return FileResponse(index_path)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
