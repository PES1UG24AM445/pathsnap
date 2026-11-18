"""Run from Laptop 2 against Laptop 1 serving the ten student-test trips."""

import argparse
import csv
import math
from pathlib import Path
import time

import requests


def check_response(trip_id, result):
    if result["trip_id"] != trip_id or result["n_observations"] < 1:
        raise ValueError("Trip identity or observation count is invalid")
    for field in ("accuracy_hmm", "accuracy_classifier_only",
                  "continuity_hmm", "continuity_classifier_only"):
        if not 0 <= result[field] <= 1:
            raise ValueError(f"{field} is outside [0, 1]")
    if len(result["gps"]) != result["n_observations"]:
        raise ValueError("GPS payload length differs from observation count")
    for field in ("gps", "predicted_route"):
        if not result[field] or any(
            len(point) != 2 or not (-90 <= point[0] <= 90 and -180 <= point[1] <= 180)
            for point in result[field]
        ):
            raise ValueError(f"{field} contains missing or invalid coordinates")
    segments = result.get("predicted_route_segments")
    if segments is not None:
        if (not segments or any(len(segment) < 2 for segment in segments)
                or [point for segment in segments for point in segment] != result["predicted_route"]):
            raise ValueError("Route segments differ from the flat route payload")
    if not math.isfinite(result["seconds"]) or result["seconds"] < 0:
        raise ValueError("Invalid inference timing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("api_url")
    parser.add_argument("--student", action="store_true", required=True,
                        help="Confirm Laptop 1 is serving student-test data")
    parser.add_argument("--timeout", type=float, default=900,
                        help="Read timeout in seconds, matching the Build Guide UI by default")
    parser.add_argument("--output", type=Path, default=Path("results/rehearsal.csv"))
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a positive, finite number.")
    base = args.api_url.rstrip("/")
    results = []
    with requests.Session() as session:
        health = session.get(f"{base}/health", timeout=10)
        health.raise_for_status()
        health_data = health.json()
        if (not isinstance(health_data, dict)
                or health_data.get("status") != "ok" or health_data.get("trips") != 10):
            raise ValueError("Expected a healthy API with ten student trips")
        listing = session.get(f"{base}/trips", timeout=10)
        listing.raise_for_status()
        trip_ids = listing.json()
        if (not isinstance(trip_ids, list) or len(trip_ids) != 10
                or not all(isinstance(trip_id, str) and trip_id.strip() for trip_id in trip_ids)
                or len(set(trip_ids)) != 10):
            raise ValueError("Expected ten unique trip IDs")
        for trip_id in trip_ids:
            started = time.perf_counter()
            try:
                response = session.post(
                    f"{base}/predict", json={"trip_id": trip_id}, timeout=(5, args.timeout)
                )
                response.raise_for_status()
                result = response.json()
                check_response(trip_id, result)
                server_seconds, status = result["seconds"], "pass"
            except (requests.RequestException, ValueError, KeyError, TypeError) as error:
                server_seconds, status = None, f"FAIL: {error}"
            elapsed = round(time.perf_counter() - started, 3)
            results.append((trip_id, status, elapsed, server_seconds))
            print(trip_id, status, elapsed, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trip_id", "status", "round_trip_seconds", "server_seconds"])
        writer.writerows(results)
    if any(status != "pass" for _, status, _, _ in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
