"""AthPathFinder: hierarchical gene-set annotation for Arabidopsis metabolism.

Public inference::

    from athpathfinder import predict
    predict(["AT1G01040", "AT1G01050", ...])
"""

from .algorithms.two_stage import threshold_by_balanced_accuracy
from .predict import AthPathFinder, Prediction, load_publication_model, predict
from .problem import FAMILIES, NON_METABOLIC, SEED, Y

__all__ = [
    "AthPathFinder",
    "FAMILIES",
    "NON_METABOLIC",
    "Prediction",
    "SEED",
    "Y",
    "load_publication_model",
    "predict",
    "threshold_by_balanced_accuracy",
]
