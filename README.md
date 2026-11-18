# PathSnap

PathSnap matches noisy GPS trajectories to directed road arcs. Candidate generation and features feed a classifier, then a Viterbi decoder uses GPS displacement and shortest-path distance to select a connected route where possible. When no bounded directed connection is available, the output marks a separate route segment.

## Setup and data

Use Python 3.11 or 3.12 and install into a virtual environment.

```sh
python -m pip install -r requirements.txt
python -m pip install -e .
```

On macOS, LightGBM may need `libomp`. Keep versions consistent between model training and serving.

Place the authorized raw files under `data/raw/train` (80 trips) and `data/raw/student` (10 trips). Each trip needs `.track`, `.nodes`, `.arcs`, and `.route`. Files sharing a basename can share a folder, or each trip can use its own folder with one file of each type. The supplied archive uses `student_test`, which is also supported. Keep evaluation data and raw trip files out of Git. Do not use reserved evaluation data for development.

## Reproduce the training results

Run from the repository root.

```sh
python scripts/01_inventory.py
python scripts/02_eda.py
python scripts/03_build_features.py
python scripts/04_train_model.py logreg
python scripts/04_train_model.py knn
python scripts/04_train_model.py rf
python scripts/04_train_harsha.py
python scripts/05_compare.py
python scripts/06_curves.py
python scripts/07_tune.py --trials 6
python scripts/08_finalize.py
python scripts/09_decode_eval.py
python scripts/10_maps.py
python scripts/13_model_diagnostics.py
```

Use the 80 training trips for grouped cross-validation and tuning. The classifier and decoder comparisons use the 10 student-test trips as development evidence. Use the same inputs and feature definitions for all five runs. The random forest draws 20% of fitting rows per tree and the MLP uses batches of 1,024 rows to keep training practical on a laptop. The six-trial search includes five startup trials and one TPE-guided trial. Search fits use at most 350,000 fitting rows per fold and score complete validation trips. The selected configuration reruns CV with the baseline fitting policy before comparison and export. A larger search can resume from the same study.

The final model and HMM settings are bound by the model SHA-256. If you replace or re-export the model, rerun decoder evaluation before serving. The decoder reports Top-1, bounded directed-route continuity, and arc-overlap F1. Its route continuity measures successful transitions after allowing intermediate arcs, while the classifier comparison table's continuity measures direct adjacency between independently selected arcs.

## Deploy on two laptops

On Laptop 1, copy the final model, model manifest and name, HMM settings, and HMM model checksum into `artifacts/`. Use the team's own trusted model export. Set `EVAL_DIR` to the student-test directory for rehearsal.

```sh
python scripts/12_check_server.py
export EVAL_DIR=data/raw/student
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

On Laptop 2, set the API URL to Laptop 1's LAN IP and run the UI and batch rehearsal.

```sh
export API_URL=http://LAPTOP1_IP:8000
python -m streamlit run ui/app.py
# In a second terminal
python scripts/11_rehearse.py http://LAPTOP1_IP:8000 --student
```

On PowerShell use `$env:EVAL_DIR = 'data/raw/student'` and `$env:API_URL = 'http://LAPTOP1_IP:8000'`. Verify all ten student trips through the UI and capture the UI and receiving-terminal evidence. For official evaluation, switch Laptop 1 to the authorized evaluator directory and restart the API. Ground truth stays on Laptop 1 and is loaded only to score completed predictions.

## Evidence for the final PDF

After the local rehearsal has written `results/rehearsal.csv`, run `python scripts/14_report.py` on the same checkout that holds the generated metrics. It compiles the measured results into `report/official_development_results.md`. The physical two-laptop screenshots and evaluation evidence still need to come from the live rehearsal.

- **Dataset and EDA:** `results/eda_inventory.csv`, `results/figs/eda_hist.png`, `results/figs/eda_trip.png`, `results/candidate_stats.csv` and a map.
- **Models:** `results/model_comparison.csv`, per-model results and comparison figures. Each owner explains parameters, why the model suits this task, and the measured results and limitations.
- **Tuning:** `results/optuna_trials.csv`, `results/best_params.json`, `results/tuned_vs_untuned.csv` and the student diagnostic chart.
- **Reconstruction:** `results/hmm_grid.csv`, `results/hmm_vs_classifier.csv`, per-trip comparison, and representative Folium maps generated in `results/maps/`. Use eligible student-test imagery and follow the organizer's data-sharing rules.
- **API and UI:** The working UI screenshot and corresponding Laptop 1 request/response log.

Populate the single submission PDF after the real experiment and rehearsal. Keep the final model, settings, code, and required reproducibility artifacts in the public submission as permitted by the organizers. Document AI assistance according to the event policy.
