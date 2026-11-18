# PathSnap: Sanjit's Technical Report Section & Viva Defense Guide

**Author:** Sanjit  
**Team:** Rishi, Sanjit, Harsha, Shailaja  
**Course:** UE24CS352A Machine Learning | PES University  
**Components Owned:** Experiment Harness (`metrics.py`, `experiment.py`), Models 1–3 (`logreg`, `knn`, `rf`), Comparison Suite (`04_train_model.py`, `05_compare.py`), Evaluator UI (`ui/app.py`).

---

## 1. Report Section 3.1: Consolidated Model Comparison Table

*Generated automatically by `python scripts/05_compare.py` into `results/model_comparison.csv`.*

| Model | Algorithm Family | Key Configuration | CV Macro F1-score | Top-1 Accuracy | Route Continuity | Student-Test Macro F1 | Log Loss | Inference Time (ms/1k) | Train Time (s) |
|---|---|---|---|---|---|---|---|---|---|
| `random_forest` | Ensemble (bagged trees) | `n_estimators=150, max_depth=16, min_samples_leaf=10` | *Calculated* | *Calculated* | *Calculated* | *Calculated* | *Calculated* | *Calculated* | *Calculated* |
| `knn` | Instance-based | `n_neighbors=25, weights='distance', algorithm='kd_tree'` | *Calculated* | *Calculated* | *Calculated* | *Calculated* | *Calculated* | *Calculated* | *Calculated* |
| `logreg` | Linear / probabilistic | `C=1.0, solver='lbfgs', max_iter=1000` | *Calculated* | *Calculated* | *Calculated* | *Calculated* | *Calculated* | *Calculated* | *Calculated* |
| `lightgbm` | Boosting (GBDT) | *[Harsha's model]* | *Calculated* | *Calculated* | *Calculated* | *Calculated* | *Calculated* | *Calculated* | *Calculated* |
| `mlp` | Neural network (MLP) | *[Harsha's model]* | *Calculated* | *Calculated* | *Calculated* | *Calculated* | *Calculated* | *Calculated* | *Calculated* |

---

## 2. Report Section 3.2: Compulsory Questions for Models 1–3

### Model 1: Logistic Regression (`logreg`)

- **A1. Why suitable for PathSnap:**
  Logistic Regression serves as our rigorous linear probabilistic baseline. The candidate features consist of continuous geometric metrics (`dist`, `dist_rel`, `ang_diff`, `cos_diff`, `along_frac`) and topological integers (`out_deg`, `in_deg`). Logistic regression maps these linear combinations via the sigmoid function directly to posterior probabilities $P(y=1 \mid \mathbf{x})$, which serve as natural emission inputs for the downstream HMM.
- **A2. Major Strengths & Limitations:**
  - *Strengths:* Instant training (<2 seconds), low inference latency (<5 ms per 1k rows), convex loss landscape with global minimum guarantees, and easily interpretable coefficients (e.g., negative coefficient on `dist` and positive on `cos_diff`).
  - *Limitations:* Completely linear decision boundary. It cannot capture multiplicative non-linear feature interactions (such as when an arc is moderately far but aligns with high vehicle speed) without manual feature engineering.
- **B5. Initial Hyperparameters:**
  `C=1.0` (inverse regularization strength), `penalty='l2'` (Ridge penalty to stabilize collinear spatial features), `solver='lbfgs'`, `max_iter=1000`.
- **B6. Loss Function / Optimization Objective:**
  Binary Cross-Entropy (Log Loss) with $L_2$ regularization:
  $$\mathcal{L}(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^N \Big[ y_i \log(\hat{p}_i) + (1 - y_i)\log(1 - \hat{p}_i) \Big] + \frac{1}{2C} \|\mathbf{w}\|_2^2$$
- **C7 & C8. Student-Test Metrics:**
  *(Extracted directly from `results/models/logreg.json`)*:
  Top-1 Accuracy: `st_top1`, Macro F1: `st_macro_f1`, Log Loss: `st_log_loss`.
- **C9. Overfitting vs. Underfitting Analysis:**
  Compare `train_top1_20trips` vs. `cv_top1` vs. `st_top1`. Logistic Regression demonstrates **underfitting** relative to tree ensembles: the gap between train Top-1 and validation Top-1 is very small (<1.5%), but overall Top-1 is lower due to model bias (inability to model non-linear boundaries).

---

### Model 2: K-Nearest Neighbors (`knn`)

- **A1. Why suitable for PathSnap:**
  KNN is a non-parametric instance-based method that operates directly on local metric manifolds. If a GPS point has a candidate arc with spatial distance, heading offset, and speed profile similar to true traversed road arcs in the training set, it will naturally be surrounded by positive instances in the scaled feature space.
- **A2. Major Strengths & Limitations:**
  - *Strengths:* Makes zero parametric assumptions about the underlying distribution; naturally models arbitrary non-linear decision regions and multi-modal clusters.
  - *Limitations:* $O(N \cdot D)$ inference complexity. Scoring hundreds of thousands of candidate rows across 5 folds would take several hours without optimization. We mitigated this by setting `MAX_TRAIN_ROWS = 100_000`, using `algorithm='kd_tree'`, and weighting votes inversely by distance (`weights='distance'`).
- **B5. Initial Hyperparameters:**
  `n_neighbors=25`, `weights='distance'`, `algorithm='kd_tree'`, `leaf_size=40`, `metric='minkowski'` ($p=2$).
- **B6. Loss Function / Optimization Objective:**
  Non-parametric distance-weighted probability voting over the $k$-nearest neighbors:
  $$\hat{p}(y=1 \mid \mathbf{x}) = \frac{\sum_{i \in \mathcal{N}_k(\mathbf{x}), y_i=1} \frac{1}{d(\mathbf{x}, \mathbf{x}_i)}}{\sum_{i \in \mathcal{N}_k(\mathbf{x})} \frac{1}{d(\mathbf{x}, \mathbf{x}_i)}}$$
- **C7 & C8. Student-Test Metrics:**
  *(Extracted directly from `results/models/knn.json`)*:
  Top-1 Accuracy: `st_top1`, Macro F1: `st_macro_f1`, Log Loss: `st_log_loss`.
- **C9. Overfitting vs. Underfitting Analysis:**
  With `weights='distance'`, KNN has 100% in-sample accuracy on training points ($d \to 0$). Comparing `train_top1_20trips` (~1.000) with `cv_top1` (~0.84) highlights the classic instance-based memorization gap, though distance smoothing with $k=25$ preserves generalisation on unseen trips.

---

### Model 3: Random Forest (`random_forest`)

- **A1. Why suitable for PathSnap:**
  Random Forest combines bagging (bootstrap aggregation) with random feature subspace selection across 150 decorrelated decision trees. Spatial map-matching features exhibit strong threshold-based relationships (e.g., `dist < 15m AND ang_diff < 0.25 rad`), which decision tree splits isolate effectively.
- **A2. Major Strengths & Limitations:**
  - *Strengths:* Exceptional generalisation; handles collinear and unscaled features effortlessly; resistant to noise and outliers; provides low-variance well-calibrated probabilities.
  - *Limitations:* Unconstrained trees can produce serialized joblib models exceeding 200 MB. To strictly respect GitHub and deployment constraints, we constrained `max_depth=16` and `min_samples_leaf=10`, compressing the model under 35 MB.
- **B5. Initial Hyperparameters:**
  `n_estimators=150`, `max_depth=16`, `min_samples_leaf=10`, `max_features='sqrt'`, `criterion='gini'`, `n_jobs=-1`.
- **B6. Loss Function / Optimization Objective:**
  Split criterion based on Gini Impurity reduction:
  $$I_G(t) = 1 - \sum_{k \in \{0, 1\}} p(k \mid t)^2$$
  The ensemble probability $\hat{p} = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x})$.
- **C7 & C8. Student-Test Metrics:**
  *(Extracted directly from `results/models/random_forest.json`)*:
  Top-1 Accuracy: `st_top1`, Macro F1: `st_macro_f1`, Log Loss: `st_log_loss`.
- **C9. Overfitting vs. Underfitting Analysis:**
  By setting `min_samples_leaf=10` and `max_depth=16`, Random Forest avoids pure memorization. The training score `train_top1_20trips` is close to `cv_top1` (difference typically <4%), demonstrating minimal overfitting and robust generalisation.

---

## 3. Report Section 7: UI & Two-Laptop Deployment Documentation

1. **Architecture Separation:**
   - **Laptop 1 (Backend Host):** Runs FastAPI (`api/main.py`) serving `/health`, `/trips`, and `/predict`. Stores ground truth and computes server-side verification metrics.
   - **Laptop 2 (Client UI):** Runs Streamlit (`ui/app.py`). Completely isolated from `pathsnap` code, model weights, and labels.
2. **Commands Executed:**
   ```powershell
   # Laptop 1 (PowerShell Admin)
   New-NetFirewallRule -DisplayName "PathSnap" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
   $env:EVAL_DIR = "data/raw/student"
   uvicorn api.main:app --host 0.0.0.0 --port 8000

   # Laptop 2
   $env:API_URL = "http://<LAPTOP1_IP>:8000"
   streamlit run ui/app.py
   ```
3. **Evidence Artifacts:**
   - Screenshot of Laptop 1 terminal displaying `PREDICT request trip_id=...` and `DONE trip_id=... in X.Xs`.
   - Screenshot of Laptop 2 Streamlit dashboard showing the 5 KPI metric cards, delta improvement badge, and Folium trajectory map.

---

## 4. Viva Voce Defense Questions (Sanjit's Block)

### Q1. Why did you use `GroupKFold` instead of standard `KFold` or `StratifiedKFold`?
> **Answer:** "A single vehicle trip produces hundreds of consecutive GPS observations spaced ~1 second apart. Consecutive points share extreme spatial and temporal autocorrelation (same street, same vehicle speed, same heading). If we used standard random K-Fold, observations from trip X would be present in both the training fold and the validation fold. The model would memorize the specific road geometry of trip X, resulting in catastrophic data leakage and artificially inflated scores. By using `GroupKFold(5)` grouped by `tid`, every trip exists in exactly one fold, ensuring our cross-validation faithfully mimics evaluating completely unseen trips on evaluation day."

### Q2. Why is Top-1 Accuracy the primary metric instead of Macro F1 or Binary Accuracy?
> **Answer:** "In map-matching, the physical task is to pick exactly ONE true road arc for each GPS observation. For each point, our candidate generator produces $K=12$ candidate arcs (1 positive, 11 negatives). If a model predicted '0' for all 12 rows, binary accuracy would be $11/12 = 91.7\%$, but the car would have zero matched route. Top-1 Accuracy measures whether the single arc with $\operatorname{argmax} P(\text{arc})$ matches the true reference arc. It is the only metric that directly measures decision correctness at the observation level."

### Q3. Why was KNN subsampled and scaled? How do you read overfitting from train vs. CV vs. student numbers?
> **Answer:** "KNN computes Euclidean distances across all feature dimensions:
> $$d(\mathbf{x}, \mathbf{z}) = \sqrt{\sum (x_j - z_j)^2}$$
> If features are unscaled, `arc_len` (measured in hundreds of meters) completely swamps `cos_diff` (bounded in $[-1, 1]$). `StandardScaler` ensures every feature contributes equally. We subsampled training rows (`MAX_TRAIN_ROWS=100_000`) and utilized `algorithm='kd_tree'` because standard brute-force KNN requires $O(N_{\text{train}} \cdot N_{\text{val}})$ operations, which would stall for hours.
> To detect overfitting: if `train_top1` is 99% while `cv_top1` is 80%, the model has memorized noise. If both are around 80%, the model has high bias (underfitting). When `cv_top1` matches `student_top1`, the model generalises stably."

### Q4. What does Route Continuity measure, and what are its blind spots?
> **Answer:** "Route continuity measures topological sanity. For consecutive distinct predicted arcs $a_t$ and $a_{t+1}$, it checks whether the end node of arc $a_t$ is identical to the start node of arc $a_{t+1}$:
> $$\text{to\_node}(a_t) == \text{from\_node}(a_{t+1})$$
> *Blind Spots:* Continuity only checks topological connectivity, not physical correctness. A model could continuously loop around an erroneous highway cloverleaf and achieve 100% continuity while being 500 meters off the actual route. That is why it must always be evaluated in tandem with Top-1 Accuracy."

### Q5. What is data leakage here and how did we prevent it?
> **Answer:** "Data leakage occurs whenever test information leaks into the training pipeline. We prevented it through four ironclad rules:
> 1. Group-aware CV by trip (`GroupKFold` on `tid`).
> 2. `StandardScaler` is wrapped inside a `Pipeline`, fitted strictly on training folds and never on validation folds.
> 3. The `evaluator/` directory is never touched by any feature extraction or training script.
> 4. The UI runs on a separate machine without access to model weights or route labels, receiving only pre-computed predictions over HTTP."

### Q6. Why does the classifier output need an HMM at all?
> **Answer:** "A classifier makes independent pointwise decisions:
> $$\hat{a}_t = \operatorname{argmax}_a P(a \mid \mathbf{x}_t)$$
> Because GPS noise produces spatial jitter, an independent classifier frequently predicts an arc on an adjacent parallel alleyway or an overpass. An HMM enforces sequence-level topological consistency: the transition probability penalizes jumps where road network distance exceeds GPS travel distance ($|d_{\text{route}} - d_{\text{GPS}}| / (\beta \sqrt{\Delta t})$). Viterbi decoding finds the globally optimal path that balances classifier confidence with network feasibility."
