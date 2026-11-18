"""Scaled MLP candidate classifier. Validation is handled by the grouped harness."""

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import SEED

NAME = "mlp"
FAMILY = "Multilayer perceptron"
MAX_TRAIN_ROWS = 400_000
CONFIG = {
    "hidden_layer_sizes": (64, 32),
    "alpha": 0.001,
    "learning_rate_init": 0.001,
    "max_iter": 50,
    "batch_size": 1024,
    "early_stopping": False,
}


def make(**overrides):
    return make_pipeline(
        StandardScaler(), MLPClassifier(**(CONFIG | overrides), random_state=SEED)
    )
