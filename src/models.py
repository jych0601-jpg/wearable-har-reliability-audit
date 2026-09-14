"""Fixed, CPU-friendly probabilistic classifier factory."""

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_NAMES = ("logistic", "random_forest", "extra_trees", "hist_gradient_boosting")


def build_model(name: str, seed: int, n_jobs: int = 1, pilot: bool = False):
    if name == "logistic":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(C=1.0, solver="lbfgs", max_iter=500 if pilot else 2000),
                ),
            ]
        )
    trees = 40 if pilot else 300
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=trees,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=n_jobs,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=trees,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=n_jobs,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=30 if pilot else 200,
            max_leaf_nodes=31,
            class_weight="balanced",
            random_state=seed,
        )
    raise ValueError(f"unknown model: {name}")
