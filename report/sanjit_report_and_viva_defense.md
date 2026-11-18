# PathSnap: Sanjit's Technical Report Section & Viva Defense Guide

**Author:** Sanjit  
**Team:** Rishi, Sanjit, Harsha, Shailaja  
**Course:** UE24CS352A Machine Learning | PES University  
**Components Owned:** Experiment Harness (`metrics.py`, `experiment.py`), Models 1–3 (`logreg`, `knn`, `rf`), Comparison Suite (`04_train_model.py`, `05_compare.py`), Evaluator UI (`ui/app.py`).

---

## 1. Report Section 3.1: Consolidated Model Comparison Table

The table uses the merged five-baseline run. Regenerate it with `python scripts/05_compare.py --baselines`. The full CSV includes model configurations and timings.

| Model | CV Top-1 | CV Macro F1 | Student Top-1 | Student Macro F1 | Student arc adjacency | Log loss | Inference ms/1k rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| `lightgbm` | 75.12% | 0.8977 | 87.10% | 0.9258 | 72.95% | 0.0640 | 8.60 |
| `random_forest` | 75.08% | 0.8980 | 86.14% | 0.9212 | 73.53% | 0.0660 | 3.37 |
| `mlp` | 74.53% | 0.8932 | 86.53% | 0.9154 | 72.23% | 0.0718 | 0.56 |
| `logreg` | 73.16% | 0.8485 | 84.28% | 0.8869 | 71.38% | 0.1107 | 0.10 |
| `knn` | 72.62% | 0.8729 | 84.07% | 0.8956 | 69.36% | 0.1137 | 349.93 |

Top-1 is measured per GPS observation. Macro F1 and log loss measure binary candidate scores. Arc adjacency checks direct connections between consecutive distinct choices.

---

## 2. Report Section 3.2: Compulsory Questions for Models 1–3

### Model 1: Logistic Regression (`logreg`)

- **A1. Why suitable for PathSnap:**
  Logistic Regression serves as our rigorous linear probabilistic baseline. The candidate features consist of continuous geometric metrics (`dist`, `dist_rel`, `ang_diff`, `cos_diff`, `along_frac`) and topological integers (`out_deg`, `in_deg`). Logistic regression maps these linear combinations via the sigmoid function directly to posterior probabilities $P(y=1 \mid \mathbf{x})$, which serve as natural emission inputs for the downstream HMM.
- **A2. Major Strengths & Limitations:**
  - *Strengths:* Low fit and inference cost on the recorded run, a convex objective, and interpretable feature coefficients.
  - *Limitations:* Completely linear decision boundary. It cannot capture multiplicative non-linear feature interactions (such as when an arc is moderately far but aligns with high vehicle speed) without manual feature engineering.
- **B5. Initial Hyperparameters:**
  `C=1.0` (inverse regularization strength), `penalty='l2'` (Ridge penalty to stabilize collinear spatial features), `solver='lbfgs'`, `max_iter=1000`.
- **B6. Loss Function / Optimization Objective:**
  Binary Cross-Entropy (Log Loss) with $L_2$ regularization:
  $$\mathcal{L}(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^N \Big[ y_i \log(\hat{p}_i) + (1 - y_i)\log(1 - \hat{p}_i) \Big] + \frac{1}{2C} \|\mathbf{w}\|_2^2$$
- **C7 & C8. Student-Test Metrics:**
  The student-test run measured **84.28% Top-1**, **0.8869 Macro F1**, and **0.1107 log loss**. Scoring took 0.10 ms per 1,000 candidate rows.
- **C9. In-sample and validation comparison:**
  The diagnostic on the first 20 training trips was 76.98%, and mean grouped CV was 73.16%. The 3.83-point gap should be read alongside trip composition and candidate coverage because the in-sample subset and validation folds differ.

---

### Model 2: K-Nearest Neighbors (`knn`)

- **A1. Why suitable for PathSnap:**
  KNN is a non-parametric instance-based method that operates directly on local metric manifolds. If a GPS point has a candidate arc with spatial distance, heading offset, and speed profile similar to true traversed road arcs in the training set, it will naturally be surrounded by positive instances in the scaled feature space.
- **A2. Major Strengths & Limitations:**
  - *Strengths:* Makes zero parametric assumptions about the underlying distribution; naturally models arbitrary non-linear decision regions and multi-modal clusters.
  - *Limitations:* Neighbor queries were the slowest inference method on the recorded run. We capped fitting at 100,000 rows and used a KD-tree. Distance weighting gives closer neighbors more influence on the prediction.
- **B5. Initial Hyperparameters:**
  `n_neighbors=25`, `weights='distance'`, `algorithm='kd_tree'`, `leaf_size=40`, `metric='minkowski'` ($p=2$).
- **B6. Loss Function / Optimization Objective:**
  Non-parametric distance-weighted probability voting over the $k$-nearest neighbors:
  $$\hat{p}(y=1 \mid \mathbf{x}) = \frac{\sum_{i \in \mathcal{N}_k(\mathbf{x}), y_i=1} \frac{1}{d(\mathbf{x}, \mathbf{x}_i)}}{\sum_{i \in \mathcal{N}_k(\mathbf{x})} \frac{1}{d(\mathbf{x}, \mathbf{x}_i)}}$$
- **C7 & C8. Student-Test Metrics:**
  The student-test run measured **84.07% Top-1**, **0.8956 Macro F1**, and **0.1137 log loss**. Scoring took 349.93 ms per 1,000 candidate rows.
- **C9. In-sample and validation comparison:**
  The diagnostic on the first 20 training trips was 78.67%, and mean grouped CV was 72.62%. The 6.05-point gap should be read alongside trip composition and candidate coverage because the in-sample subset and validation folds differ.

---

### Model 3: Random Forest (`random_forest`)

- **A1. Why suitable for PathSnap:**
  Random Forest combines bagging (bootstrap aggregation) with random feature subspace selection across 150 decorrelated decision trees. Spatial map-matching features exhibit strong threshold-based relationships (e.g., `dist < 15m AND ang_diff < 0.25 rad`), which decision tree splits isolate effectively.
- **A2. Major Strengths & Limitations:**
  - *Strengths:* Tree averaging and randomized feature subsets can reduce variance. Tree splits learn interactions without feature scaling.
  - *Limitations:* Tree depth and bootstrap sample size affect fit time and model size. The official run limited each tree to 20% of the fitting rows.
- **B5. Initial Hyperparameters:**
  `n_estimators=150`, `max_depth=16`, `min_samples_leaf=10`, `max_features='sqrt'`, `criterion='gini'`, `max_samples=0.2`, `n_jobs=-1`.
- **B6. Loss Function / Optimization Objective:**
  Split criterion based on Gini Impurity reduction:
  $$I_G(t) = 1 - \sum_{k \in \{0, 1\}} p(k \mid t)^2$$
  The ensemble probability $\hat{p} = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x})$.
- **C7 & C8. Student-Test Metrics:**
  The student-test run measured **86.14% Top-1**, **0.9212 Macro F1**, and **0.0660 log loss**. Scoring took 3.37 ms per 1,000 candidate rows.
- **C9. In-sample and validation comparison:**
  The diagnostic on the first 20 training trips was 80.95%, and mean grouped CV was 75.08%. The 5.87-point gap should be read alongside trip composition and candidate coverage because the in-sample subset and validation folds differ.

---

## 3. Report Section 7: UI & Two-Laptop Deployment Documentation

1. **Architecture Separation:**
   - **Laptop 1 (Backend Host):** Runs FastAPI (`api/main.py`) serving `/health`, `/trips`, and `/predict`. Stores ground truth and computes server-side verification metrics.
   - **Laptop 2 (Client UI):** Runs Streamlit (`ui/app.py`). Completely isolated from `pathsnap` code, model weights, and labels.
2. **Two-laptop commands:**
   ```powershell
   # Laptop 1 (PowerShell)
   python scripts/12_check_server.py
   $env:EVAL_DIR = "data/raw/student_test"
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

   # Laptop 2
   $env:API_URL = "http://<LAPTOP1_IP>:8000"
   python -m streamlit run ui/app.py
   ```
3. **Evidence Artifacts:**
   - Local FastAPI and Streamlit tests verified trip selection, server-computed metrics, and the returned route using the official student-test data. All ten API requests passed.
   - The physical rehearsal still requires the Laptop 2 dashboard screenshot and the corresponding Laptop 1 request log. The [compiled PDF](report.pdf) records the local evidence.

---

## 4. Viva Voce Defense Questions (Sanjit's Block)

### Q1. Why did you use `GroupKFold` instead of standard `KFold` or `StratifiedKFold`?
> **Answer:** "A single vehicle trip produces hundreds of consecutive GPS observations spaced ~1 second apart. Consecutive points share extreme spatial and temporal autocorrelation (same street, same vehicle speed, same heading). If we used standard random K-Fold, observations from trip X would be present in both the training fold and the validation fold. The model would memorize the specific road geometry of trip X, resulting in catastrophic data leakage and artificially inflated scores. By using `GroupKFold(5)` grouped by `tid`, every trip exists in exactly one fold, ensuring our cross-validation faithfully mimics evaluating completely unseen trips on evaluation day."

### Q2. Why is Top-1 Accuracy the primary metric instead of Macro F1 or Binary Accuracy?
> **Answer:** "In map-matching, the physical task is to pick exactly ONE true road arc for each GPS observation. The generator returns up to 12 candidates. With one reference arc among 12, labeling every row negative would score 91.7% binary accuracy while giving no useful road ranking. Top-1 Accuracy measures whether the single arc with $\operatorname{argmax} P(\text{arc})$ matches the true reference arc. It is the only metric that directly measures decision correctness at the observation level."

### Q3. Why was KNN subsampled and scaled? How do you read overfitting from train vs. CV vs. student numbers?
> **Answer:** "KNN computes Euclidean distances across all feature dimensions:
> $$d(\mathbf{x}, \mathbf{z}) = \sqrt{\sum (x_j - z_j)^2}$$
> If features are unscaled, `arc_len` (measured in hundreds of meters) completely swamps `cos_diff` (bounded in $[-1, 1]$). `StandardScaler` puts features on a comparable numerical scale. We subsampled training rows (`MAX_TRAIN_ROWS=100_000`) and utilized `algorithm='kd_tree'` because standard brute-force KNN requires $O(N_{\text{train}} \cdot N_{\text{val}})$ operations, which would stall for hours.
> Compare in-sample and grouped-validation scores, then check candidate coverage and fold composition. Those differences can affect scores even when the fitting policy stays the same."

### Q4. What does Route Continuity measure, and what are its blind spots?
> **Answer:** "Route continuity measures topological sanity. For consecutive distinct predicted arcs $a_t$ and $a_{t+1}$, it checks whether the end node of arc $a_t$ is identical to the start node of arc $a_{t+1}$:
> $$\text{to\_node}(a_t) == \text{from\_node}(a_{t+1})$$
> *Blind Spots:* Continuity only checks topological connectivity, not physical correctness. A model could continuously loop around an erroneous highway cloverleaf and achieve 100% continuity while being 500 meters off the actual route. That is why it must always be evaluated in tandem with Top-1 Accuracy."

### Q5. What is data leakage here and how did we prevent it?
> **Answer:** "Data leakage occurs whenever test information leaks into the training pipeline. We prevented it through four ironclad rules:
> 1. Group-aware CV by trip (`GroupKFold` on `tid`).
> 2. `StandardScaler` is wrapped inside a `Pipeline`, fitted strictly on training folds and never on validation folds.
> 3. The `evaluator/` directory is never touched by any feature extraction or training script.
> 4. The UI is designed to run separately from the API and receives request-time predictions and aggregate scores over HTTP."

### Q6. Why does the classifier output need an HMM at all?
> **Answer:** "A classifier makes independent pointwise decisions:
> $$\hat{a}_t = \operatorname{argmax}_a P(a \mid \mathbf{x}_t)$$
> Because GPS noise produces spatial jitter, an independent classifier frequently predicts an arc on an adjacent parallel alleyway or an overpass. An HMM enforces sequence-level topological consistency: the transition probability penalizes jumps where road network distance exceeds GPS travel distance ($|d_{\text{route}} - d_{\text{GPS}}| / (\beta \sqrt{\Delta t})$). Viterbi decoding finds the globally optimal path that balances classifier confidence with network feasibility."
