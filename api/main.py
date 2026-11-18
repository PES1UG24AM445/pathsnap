"""Laptop 1 interface for on-demand prediction and server-side scoring."""

from contextlib import asynccontextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
import time

from fastapi import FastAPI, HTTPException, Request
import joblib
import numpy as np
from pydantic import BaseModel

from pathsnap.config import ART
from pathsnap.features import FEATURES, build_table
from pathsnap.hmm import build_graph, classifier_route, decode, route_coords
from pathsnap.io import find_trips, load_trip
from pathsnap.labels import reference_arcs

log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app):
    directory = os.environ.get("EVAL_DIR")
    if not directory:
        raise ValueError("Set EVAL_DIR to the authorized ten-trip serving folder.")
    folder = Path(directory)
    infos = find_trips(folder, partition="serving")
    if len(infos) != 10 or len({info["tid"] for info in infos}) != 10:
        raise ValueError("EVAL_DIR must point to one complete ten-trip serving partition.")
    if any(info["route"] is None for info in infos):
        raise ValueError("Every serving trip needs a server-side reference route for scoring.")
    app.state.trips = {info["tid"]: info for info in infos}
    artifact = ART / "final_model.joblib"
    with artifact.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = json.loads((ART / "final_model_manifest.json").read_text(encoding="utf-8"))
    bound = (ART / "hmm_model_sha256.txt").read_text(encoding="utf-8").strip()
    if manifest["sha256"] != digest or bound != digest or manifest["feature_order"] != FEATURES:
        raise ValueError("Model, decoder settings, and feature contract must match the same export.")
    app.state.model = joblib.load(artifact)
    app.state.settings = json.loads((ART / "hmm_config.json").read_text(encoding="utf-8"))
    log.info("Serving %d trips", len(infos))
    yield


app = FastAPI(title="PathSnap", lifespan=lifespan)


class PredictionRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "trips": len(request.app.state.trips)}


@app.get("/trips")
def trips(request: Request):
    return sorted(request.app.state.trips)


@app.post("/predict")
def predict(body: PredictionRequest, request: Request):
    state = request.app.state
    info = state.trips.get(body.trip_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Unknown trip ID")

    started = time.perf_counter()
    log.info("Received prediction request for %s", body.trip_id)
    trip = load_trip(info, with_route=False)
    candidates = build_table(trip, with_labels=False)
    candidates["p"] = state.model.predict_proba(candidates[FEATURES])[:, 1]
    graph = build_graph(trip)
    raw = classifier_route(trip, candidates, graph)
    matched = decode(trip, candidates, g=graph, **state.settings)

    # References are loaded only after both predictions have completed.
    reference = reference_arcs(load_trip(info, with_route=True))
    accuracy = float(np.mean(matched.arcs == reference[matched.observations]))
    raw_accuracy = float(np.mean(raw.arcs == reference[raw.observations]))
    elapsed = time.perf_counter() - started
    lines = route_coords(trip, matched.segments)
    log.info("Finished %s: accuracy %.4f, continuity %.4f, %.2fs",
             body.trip_id, accuracy, matched.continuity, elapsed)
    return {
        "trip_id": body.trip_id,
        "n_observations": len(trip.xy),
        "accuracy_hmm": accuracy,
        "accuracy_classifier_only": raw_accuracy,
        "continuity_hmm": matched.continuity,
        "continuity_classifier_only": raw.continuity,
        "seconds": round(elapsed, 3),
        "gps": trip.lonlat[:, ::-1].tolist(),
        "predicted_route": [point for line in lines for point in line],
        "predicted_route_segments": lines,
    }
