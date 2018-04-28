import numpy as np
import utils


def cross_validation(model, X, y, cv=5):
    splits = create_splits(len(X), cv)
    scores = []
    for s in splits:
        # model_copy = clone(model)
        train_index, test_index = [s == 0], [s == 1]
        model.fit(X[train_index], y[train_index])
        y_pred = model.predict(X[test_index])
        scores.append(rmse(y[test_index], y_pred[:, -1]))

    return scores


def create_splits(n_pts, n_folds):
    rand = np.random.rand(n_pts)
    splits = []

    for i in range(n_folds):
        i_min, i_max = (i / n_folds), ((i+1) / n_folds)
        indices = np.array([1 if (j > i_min and j < i_max) else 0 for j in rand])
        splits.append(indices)

    return splits


def rmse(u, v):
    if len(u) != len(v):
        print("[Error] rmse can't be computed for vectors of different length")
        return 0

    n = len(u)
    mse = sum((u-v)**2) / n
    return np.sqrt(mse)
