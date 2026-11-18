"""Smoke-test the copied PathSnap model and decoder settings on Laptop 1."""

import hashlib
from importlib.metadata import PackageNotFoundError, version
import inspect
import json
import math
import sys

import joblib
import numpy as np
import pandas as pd

from pathsnap.config import ART
from pathsnap.features import FEATURES
from pathsnap.hmm import decode


def main():
    manifest = json.loads((ART / "final_model_manifest.json").read_text(encoding="utf-8"))
    name = (ART / "final_model_name.txt").read_text(encoding="utf-8").strip()
    if name != manifest["model"] or list(FEATURES) != manifest["feature_order"]:
        raise ValueError("Model name or feature order differs from the export manifest.")

    joblib_path = ART / "final_model.joblib"
    with joblib_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError("Model checksum differs from the export manifest.")

    python = ".".join(map(str, sys.version_info[:2]))
    if "python" in manifest and python != manifest["python"]:
        print(f"Python version differs from export ({python} vs {manifest['python']}).")
    for package, exported in manifest.get("model_package_versions", {}).items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = "missing"
        if installed != exported:
            print(f"{package} differs from export ({installed} vs {exported}).")

    settings = json.loads((ART / "hmm_config.json").read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("HMM settings must be a JSON object.")
    inspect.signature(decode).bind(None, None, g=None, **settings)
    for key in ("beta", "w", "jump"):
        value = settings.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"HMM setting {key} must be a finite number.")
    if settings["beta"] <= 0 or settings["w"] < 0 or settings["jump"] > 0:
        raise ValueError("Invalid HMM distance scale, emission weight, or jump penalty.")

    # Use only an artifact exported by the team. The smoke input contains no trip data.
    model = joblib.load(joblib_path)
    if list(model.classes_) != [0, 1] or list(model.feature_names_in_) != list(FEATURES):
        raise ValueError("The loaded estimator does not match the prediction contract.")
    sample = pd.DataFrame([[0.0] * len(FEATURES)], columns=FEATURES)
    probabilities = model.predict_proba(sample)
    if (probabilities.shape != (1, 2)
            or not np.isfinite(probabilities).all()
            or not ((probabilities >= 0) & (probabilities <= 1)).all()
            or not np.allclose(probabilities.sum(axis=1), 1)):
        raise ValueError("The model did not return valid binary probabilities.")
    print(f"Server artifact check passed for {name}.")


if __name__ == "__main__":
    main()
