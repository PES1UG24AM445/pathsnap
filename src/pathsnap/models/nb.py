"""Gaussian Naive Bayes (Optional 6th Buffer Model) for PathSnap.

Model Family: Probabilistic
Author / Owner: Sanjit (feat/sanjit)

Report Section 3.2 Notes:
- A1 (Why suitable): Classic generative probabilistic benchmark. Uses Bayes' theorem with feature conditional
  independence assumptions to rapidly compute posterior arc probabilities:
    P(y = 1 | x) proportional to P(y = 1) * prod_{j=1}^D P(x_j | y = 1)
- A2 (Strengths & Limits): Extremely fast O(N * D) closed-form parameter estimation with zero hyperparameter tuning.
  Limitation: Conditional independence assumption is violated by correlated spatial features (e.g., dist and dist_rel).
- B5 (Initial Hyperparameters): var_smoothing=1e-9 (additive smoothing to prevent zero variance division).
- B6 (Objective / Loss Function): Maximum Likelihood Estimation of Gaussian mean (mu) and variance (sigma^2) per class:
    L(mu, sigma) = sum log N(x_ij; mu_kj, sigma_kj^2)
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

NAME = 'naive_bayes'
FAMILY = 'Probabilistic'
CONFIG = {
    'var_smoothing': 1e-9
}


def make() -> Pipeline:
    """Create a pipeline with standard scaling followed by Gaussian Naive Bayes."""
    return make_pipeline(
        StandardScaler(),
        GaussianNB(**CONFIG)
    )
