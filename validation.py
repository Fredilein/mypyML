import numpy as np
import utils

def cross_validation(inp_func, X, y, cv=5):
    loss = np.array([])
    splits = create_splits(len(X), cv)
    for s in splits:
        train_index, test_index = [s == 0], [s == 1]
        res = inp_func(X[train_index], y[train_index])
        print(res['weights'])
        plt = utils.saveplot(f_X, f_y, res['weights'])
    plt.savefig('cv_plot.png')


def create_folds(X, y, n_folds):
    xy = np.c_[X, y]
    folds = []
    rand = np.random.rand(len(xy))

    for i in range(n_folds):
        i_min, i_max = (i / n_folds), ((i+1) / n_folds)
        indices = np.array([1 if (j > i_min and j < i_max) else 0 for j in rand])
        folds.append(xy[indices == 1])
        print("len of folds[i]:", len(folds[i]))

    return folds


def create_splits(n_pts, n_folds):
    rand = np.random.rand(n_pts)
    splits = []

    for i in range(n_folds):
        i_min, i_max = (i / n_folds), ((i+1) / n_folds)
        indices = np.array([1 if (j > i_min and j < i_max) else 0 for j in rand])
        splits.append(indices)

    return splits
