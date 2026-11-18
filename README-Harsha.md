# PathSnap training and deployment handoff

Merge the included src/ and scripts/ directories into the team's PathSnap repository. This package uses the Build Guide's config and FEATURES modules, Rishi's processed candidate tables, and Sanjit's experiment harness and three baseline modules. Use Python 3.11+ and install the team's dependencies and editable package from the repo root. Match package versions on the training and serving machines. On macOS, LightGBM may require Homebrew libomp.

## Training and export

1. Rishi verifies the explicit partition roots, exact trip-file pairing, and reference-alignment traceback before building the processed tables. Check provenance before training. Counts and disjoint IDs alone cannot establish where rows came from.
2. Run Harsha's baselines and have Sanjit run LogReg, KNN, and RandomForest using the same harness, features, and splits. Rerun all five if the feature snapshot or grouped folds change.
3. Generate the loss curves. LightGBM records loss by boosting round. The MLP plot comes from a separate epoch-by-epoch diagnostic fit on the same held-out training-trip split as LightGBM. Its scaler only sees the fitting trips. The MLP baseline uses outer grouped CV and disables scikit-learn's row-wise early-stopping split.
4. Tune the best of the five baselines by grouped-CV observation Top-1, then candidate-row Macro F1. The search reuses each team's baseline estimator and preprocessing. If LightGBM wins, the search covers row and feature subsampling, tree capacity, learning rate, and L1/L2 regularization. The trial budget is a target for the number of completed trials. Repeat the command to resume the same study. A change to the training table, model or search source starts a fresh study. Five startup trials precede TPE. Each trial fits five folds, and the optional timeout applies between trials. Keep the database, CSV, and search metadata, including per-fold scores.
5. Export the baseline or tuned variant with the highest CV Top-1, then CV Macro F1. An exact tie retains the baseline. Export checks feature order, class order, and prediction parity on a sample of student candidate rows.

```sh
python scripts/04_train_harsha.py
python scripts/06_curves.py
python scripts/07_tune.py --trials 20
python scripts/08_finalize.py
python scripts/13_model_diagnostics.py
```

Use the exact name from artifacts/final_model_name.txt in decoder evaluation and map generation. Shailaja regenerates artifacts/hmm_config.json for that export. Keep final_model.joblib, final_model_name.txt, final_model_manifest.json, and the matching hmm_config.json together. The Build Guide API and UI exchange a flat predicted_route list of latitude/longitude pairs. If the team adds separate route segments to show gaps, update the API, UI, and rehearsal check together.

## Diagnose where the model gains or loses

- **Candidate coverage** is the fraction of observations where the reference arc is available. It sets the classifier's Top-1 ceiling. Many misses point to candidate generation. Many wrong choices with the reference present point to feature or ranking improvements.
- **Top-3 and mean reciprocal rank** show whether the correct arc remains near the top when Top-1 fails. Share these and the low-margin or GPS-gap slices with Shailaja to pick student trips for decoder error analysis. A score margin is descriptive, and the diagnostics do not change decisions or inference.
- **Paired trip comparison** reports how many student-test trips improved or regressed and estimates the mean per-trip Top-1 change with a paired trip-bootstrap interval. With ten trips, treat the interval as exploratory. Training out-of-fold results were used in model selection.

Read results/diagnostics/model_diagnostics.json for the summary and per_trip.csv and error_slices.csv in the same folder for details. Use student_error_breakdown.png from the real run to show how much error comes from missing candidates and how tuning affected each student-test trip.

## Two-laptop rehearsal

Put both laptops on the same network. Use the team repository on both laptops. Keep the model weights, decoder settings, and trip files on Laptop 1. The API uses the reference route only to score completed predictions and sends the prediction and aggregate scores to Laptop 2.

**Rishi / Laptop 1** runs the preflight in the same environment that will run FastAPI, then starts the server. Use the actual student-test directory in place of the example path.

```powershell
python scripts/12_check_server.py
$env:EVAL_DIR = 'data/raw/student'
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The preflight checks the copied model's checksum, feature and class order, package versions, and a smoke prediction. It also checks whether the HMM settings fit the decoder's API. Confirm with Shailaja that the settings were generated for this exact model.

**Sanjit / Laptop 2** points the UI at Laptop 1's LAN address, then checks that selecting a trip displays the matching route and scores. Check map tiles on the demo network.

```powershell
$env:API_URL = 'http://LAPTOP1_IP:8000'
python -m streamlit run ui/app.py
```

In Bash or zsh use `export EVAL_DIR='data/raw/student'` or `export API_URL='http://LAPTOP1_IP:8000'` in place of the PowerShell assignments. If Laptop 2 cannot reach the API, verify the LAN address, port 8000, and the receiving laptop's firewall.

**Harsha / Laptop 2** runs the batch check, then exercises all ten student-test trips in the UI with Sanjit. Review the slowest runs in the rehearsal CSV. Save a working UI screenshot and the corresponding request and response logs from Laptop 1's terminal for the report.

```sh
python scripts/11_rehearse.py http://LAPTOP1_IP:8000 --student
```

The batch check writes results/rehearsal.csv and fails if any response is invalid. The student flag confirms operator intent. Verify the directory on Laptop 1 before running it. The batch check exercises the API only. Reserve the evaluator trips for final evaluation.

On evaluation day, point EVAL_DIR at the authorized evaluator directory on Laptop 1 and restart FastAPI. The evaluator selects a trip on Laptop 2, the server runs inference and scoring on demand, and the UI displays the predicted route and scores. Keep the receiving terminal visible.

## Report evidence

Use the baseline JSON results, loss plots and CSVs, best_params.json, optuna_trials.csv, and tuned_vs_untuned.csv. Explain the objectives, training versus validation results, tuning ranges, trial count and cost, and why the export won. Top-1 is observation-level. The guide's binary Macro F1 and log loss are candidate-row metrics. Include only results from real team runs and disclose AI assistance according to the event policy.
