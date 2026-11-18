"""Count trips without reading the reserved evaluator observations."""

from collections import Counter

from pathsnap.io import find_trips

trips = find_trips()
if not trips:
    raise SystemExit("No .track files found. See README.md for the data layout.")
for part, count in sorted(Counter(t["part"] for t in trips).items()):
    print(f"{part}: {count} trips")
for part in ("train", "student"):
    missing = sum(info["route"] is None for info in trips if info["part"] == part)
    print(f"{part}: {missing} missing reference routes")
