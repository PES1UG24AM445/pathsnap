"""Random Forest Classifier for PathSnap.

Model Family: Ensemble (bagged trees)
Owned by: Sanjit (feat/sanjit)

Report Section 3.2 Notes:
- A1 (Why suitable): Ensemble of decorrelated decision trees using bootstrap aggregation (bagging) and feature
  subsampling. Ideal for tabular tabular spatial features; captures complex non-linear feature interactions
  (e.g., small distance + low angular difference + matching heading) without feature transformation.
- A2 (Strengths & Limits): High generalisation power, robust against outliers, provides out-of-bag variance reduction.
  Limitation: Slower inference than linear models, can grow large if unconstrained. Constrained with max_depth=16
  and min_samples_leaf=10 to guarantee compressed joblib stays strictly under 50 MB (compliant with GitHub limits).
- B5 (Initial Hyperparameters): n_estimators=150, max_depth=16, min_samples_leaf=10, max_features='sqrt',
  criterion='gini'.
- B6 (Objective / Loss Function): Gini impurity minimization across orthogonal splits:
    I_G(t) = 1 - sum_{k=0,1} p(k | t)^2
    Probability estimate = fraction of trees voting positive.
"""

from sklearn.ensemble import RandomForestClassifier

from ..config import SEED

NAME = 'random_forest'
FAMILY = 'Ensemble (bagged trees)'
CONFIG = {
    'n_estimators': 150,
    'max_depth': 16,
    'min_samples_leaf': 10,
    'max_features': 'sqrt',
    'criterion': 'gini'
}


def make() -> RandomForestClassifier:
    """Instantiate a bounded Random Forest Classifier optimized for speed, memory, and generalisation."""
    return RandomForestClassifier(
        **CONFIG,
        n_jobs=-1,
        random_state=SEED
    )
