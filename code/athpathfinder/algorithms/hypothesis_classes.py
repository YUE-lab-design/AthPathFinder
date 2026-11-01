"""Fixed LightGBM estimator used by the deployed pipeline and flat baseline."""
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
try:
    from lightgbm import LGBMClassifier
except ModuleNotFoundError:
    LGBMClassifier=None
from ..problem import SEED

def tree(clf) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])

def deployed_lightgbm():
    """LightGBM PATHWAY-LIKE used by Algorithm 1. Does not import the 13-way library."""
    if LGBMClassifier is None:
        raise ModuleNotFoundError(
            "Install lightgbm from requirements.txt (and OpenMP / libomp on macOS)."
        )
    return tree(LGBMClassifier(
        n_estimators=500, max_depth=5, learning_rate=.03, subsample=.8,
        colsample_bytree=.7, reg_alpha=.1, reg_lambda=1,
        class_weight="balanced", random_state=SEED, n_jobs=-1, verbosity=-1,
    ))

def deployed_stage1():
    return deployed_lightgbm()

def deployed_stage2():
    raise ValueError("The current family head is Jaccard 1-NN, not a supervised classifier")
