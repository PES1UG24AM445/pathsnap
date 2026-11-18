"""Logistic Regression Baseline for PathSnap.

Model Family: Linear and probabilistic
Owned by: Sanjit (feat/sanjit)

Report Section 3.2 Notes:
- A1 (Why suitable): Fast, convex optimization producing well-calibrated posterior probabilities P(arc|features).
  Serves as an essential linear baseline against which tree-based non-linear combinations are measured.
- A2 (Strengths & Limits): Extremely fast training (<2s) and low inference latency.
  Limitation: Cannot model non-linear interactions between orthogonal features (e.g., distance * angle)
  without explicit polynomial feature generation.
- B5 (Initial Hyperparameters): C=1.0 (inverse regularization strength), max_iter=1000, solver='lbfgs'.
- B6 (Objective / Loss Function): Binary cross-entropy (log-loss) with L2 regularization penalty:
    L(w) = - sum(y_i * log(p_i) + (1 - y_i) * log(1 - p_i)) + (1 / 2C) * ||w||_2^2
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import SEED

NAME = 'logreg'
FAMILY = 'Linear / probabilistic'
CONFIG = {
    'C': 1.0,
    'max_iter': 1000,
    'solver': 'lbfgs'
}


def make() -> Pipeline:
    """Create a leakage-free pipeline with standard scaling followed by Logistic Regression."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(**CONFIG, random_state=SEED)
    )
