"""K-Nearest Neighbors Classifier for PathSnap.

Model Family: Instance-based
Owned by: Sanjit (feat/sanjit)

Report Section 3.2 Notes:
- A1 (Why suitable): Non-parametric local density estimation. In spatial matching, points that share
  nearly identical spatial offsets, angle deviations, and road speed limits naturally map to correct arcs.
- A2 (Strengths & Limits): Captures local non-linear manifolds without strong parametric assumptions.
  Limitation: Extremely high computational cost at inference time O(N_train * D), sensitive to curse of
  dimensionality and uninformative features. Required subsampling (MAX_TRAIN_ROWS=100,000) and feature scaling.
- B5 (Initial Hyperparameters): n_neighbors=25, weights='distance' (inversely proportional weighting),
  algorithm='kd_tree', leaf_size=40, metric='minkowski' (p=2 Euclidean).
- B6 (Objective / Loss Function): Non-parametric distance-weighted majority voting:
    p(y = 1 | x) = sum_{i in N_k(x), y_i=1} (1 / d(x, x_i)) / sum_{i in N_k(x)} (1 / d(x, x_i))
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

NAME = 'knn'
FAMILY = 'Instance-based'
CONFIG = {
    'n_neighbors': 25,
    'weights': 'distance',
    'algorithm': 'kd_tree',
    'leaf_size': 40
}
# Subsampled training rows to guarantee tractable CV and training runtimes on hundreds of thousands of candidate rows
MAX_TRAIN_ROWS = 100_000


def make() -> Pipeline:
    """Create a leakage-free pipeline with standard scaling followed by distance-weighted KNN.
    
    StandardScaler is mandatory here because unscaled features (e.g. arc_len ~ 500m vs cos_diff in [-1, 1])
    would otherwise distort Euclidean distance calculations.
    """
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(**CONFIG, n_jobs=-1)
    )
