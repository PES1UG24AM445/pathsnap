"""Export the CV-selected model as a plain predict_proba-compatible joblib."""

import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd

from pathsnap.config import ART, RES
from pathsnap.experiment import load
from pathsnap.features import FEATURES
from pathsnap.model_selection import best_baseline, read_result, validate_table


def main():
    baseline = best_baseline()
    search = json.loads((RES / "best_params.json").read_text(encoding="utf-8"))
    expected = f"{baseline['model']}_tuned"
    if search["baseline"] != baseline["model"] or search["tuned"] != expected:
        raise ValueError("Baseline winner changed. Rerun Bayesian tuning before export.")
    tuned = read_result(expected)
    if json.loads(tuned["config"]) != search["full_config"]:
        raise ValueError("Tuned result and search configurations differ. Rerun tuning.")
    if not np.isclose(tuned["cv_top1"], search["best_cv_top1"], atol=1e-12, rtol=0):
        raise ValueError("Tuned CV score differs from the search record. Rerun tuning.")
    winner = max((baseline, tuned), key=lambda row: (row["cv_top1"], row["cv_macro_f1"]))
    name = winner["model"]
    source = ART / "models" / f"{name}.joblib"
    model = joblib.load(source)
    student = load("student")
    validate_table(student, FEATURES, "student")
    predictions = pd.read_parquet(RES / "preds" / f"{name}_student.parquet")
    sample = student.sample(min(len(student), 1024), random_state=0)
    keys = ["tid", "obs", "arc"]
    matched = sample.merge(predictions[keys + ["p"]], on=keys, how="left", validate="one_to_one")
    if list(model.classes_) != [0, 1]:
        raise ValueError("Model must expose class order [0, 1] for the FastAPI contract.")
    if list(model.feature_names_in_) != list(FEATURES):
        raise ValueError("Model feature names or order differ from the FastAPI contract.")
    probabilities = model.predict_proba(matched[FEATURES])
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities[:, 1], matched["p"], atol=1e-8, rtol=1e-7
    ):
        raise ValueError("Artifact predictions differ from saved student predictions.")

    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    dependencies = ["joblib", "numpy", "pandas", "scikit-learn", "scipy"]
    if baseline["model"] == "lightgbm":
        dependencies.append("lightgbm")
    metadata = {
        "model": name, "baseline": baseline["model"], "tuned_candidate": expected,
        "selection": "highest grouped-CV Top-1, then Macro F1, then baseline",
        "feature_order": list(FEATURES), "class_order": [0, 1],
        "sha256": digest, "bytes": source.stat().st_size,
        "python": ".".join(map(str, sys.version_info[:2])),
        "model_package_versions": {package: version(package) for package in dependencies},
    }
    previous_name = ART / "final_model_name.txt"
    decoder_config = ART / "hmm_config.json"
    if decoder_config.is_file():
        prior_model = ART / "final_model.joblib"
        prior_hash = None
        if prior_model.is_file():
            with prior_model.open("rb") as stream:
                prior_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if prior_hash != digest:
            print("Decoder configuration belongs to a previous export or is unverified. Regenerate it.")

    with TemporaryDirectory(dir=ART) as folder:
        staged = Path(folder)
        shutil.copyfile(source, staged / "final_model.joblib")
        (staged / previous_name.name).write_text(name + "\n", encoding="utf-8")
        (staged / "final_model_manifest.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        for filename in ("final_model.joblib", previous_name.name, "final_model_manifest.json"):
            os.replace(staged / filename, ART / filename)
    if metadata["bytes"] >= 100 * 1024 * 1024:
        print("Artifact exceeds GitHub's ordinary file limit. Use LFS or a release asset.")
    print(f"Exported {name}. Use this exact name for decoder evaluation and maps.")


if __name__ == "__main__":
    main()
