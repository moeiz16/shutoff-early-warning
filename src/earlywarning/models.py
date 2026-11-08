"""The models compared. Few of them, and plain."""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def logistic(C: float = 1.0):
    # No class weighting: the threshold step needs probabilities that mean what they say.
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def boosted_trees(seed: int):
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=1.0,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=40,
        random_state=seed,
    )


def out_of_fold(model, X, y, n_folds: int, seed: int):
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return cross_val_predict(model, X, y, cv=folds, method="predict_proba")[:, 1]
