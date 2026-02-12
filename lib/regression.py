"""Train and evaluate CoMFA regression models in parallel over multiple hyperparameters."""

import os
from itertools import combinations, product
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import KFold

AUTO_NUM_WORKERS = os.cpu_count() or 1
NUM_WORKERS = AUTO_NUM_WORKERS


def regression(X_train, X_test, y_train, y, method):
    """
    Perform regression using Ridge, Lasso, ElasticNet, or PLS.

    Args:
        X_train (np.ndarray): Training feature matrix.
        X_test (np.ndarray): Target feature matrix for prediction.
        y_train (np.ndarray): Training target values.
        y (np.ndarray): Full target values used for clipping prediction range.
        method (str): Regression method string.

    Returns:
        tuple[np.ndarray, np.ndarray]: (coefficients, clipped predictions)
    """
    if "Ridge" in method:
        alpha = float(method.split()[1])
        model = Ridge(alpha=alpha, fit_intercept=False, max_iter=10000)
        model.fit(X_train, y_train)
        coef = model.coef_
        predict = model.predict(X_test)
    elif "Lasso" in method:
        alpha = float(method.split()[1])
        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
        model.fit(X_train, y_train)
        coef = model.coef_
        predict = model.predict(X_test)
    elif "ElasticNet" in method:
        alpha, l1ratio = map(float, method.split()[1:3])
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1ratio,
            fit_intercept=False,
            max_iter=10000,
            warm_start=True,
        )
        model.fit(X_train, y_train)
        coef = model.coef_
        predict = model.predict(X_test)
    elif "PLS" in method:
        n_components = int(method.split()[1])
        model = PLSRegression(n_components=n_components, scale=False)
        model.fit(X_train, y_train)
        coef = model.coef_[0]
        predict = model.predict(X_test)[:, 0]
    else:
        raise ValueError(f"Unsupported method: {method}")

    predict = np.clip(predict, np.min(y), np.max(y))
    return coef, predict


def regression_parallel(args):
    """Run one method end-to-end: fit, predict full set, and compute LOOCV predictions."""
    X_train, X, y_train, y, method, labels = args
    coef, predict = regression(X_train, X, y_train, y, method)

    cvs = []
    sort_index = []
    kf = KFold(n_splits=len(y_train), shuffle=False)
    for train_index, test_index in kf.split(y_train):
        _, cv = regression(X_train[train_index], X_train[test_index], y_train[train_index], y, method)
        cvs.extend(cv)
        sort_index.extend(test_index)

    original_array = np.empty_like(cvs)
    original_array[sort_index] = cvs

    # Scaffold-based CV is intentionally left disabled in the current workflow.
    _ = labels
    original_array_scaf = np.array([])

    return method, coef, predict, original_array, original_array_scaf


def nan_rmse(x, y):
    """Calculate RMSE while ignoring NaN values."""
    return np.sqrt(np.nanmean((y - x) ** 2))


def nan_r2(x, y):
    """Calculate R2 while ignoring NaN values in `x`."""
    x, y = x[~np.isnan(x)], y[~np.isnan(x)]
    return 1 - np.sum((y - x) ** 2) / np.sum((y - np.mean(y)) ** 2)


def regression_(path, names):
    """
    Run model sweeps for one dataset and save predictions and coefficient tables.

    Args:
        path (str): Input pickle path.
        names (list[str]): Feature prefixes to include (e.g., ["electronic", "electrostatic"]).
    """
    print(path)
    df = pd.read_pickle(path)
    labels = df["label"]
    df_train = df[df["test"] == 0]

    trains = []
    train_tests = []
    stds = []
    for name in names:
        train = df_train.filter(like=f"{name}_fold").to_numpy()
        std = np.linalg.norm(train)
        train_test = df.filter(like=f"{name}_fold").to_numpy()
        train /= std
        train_test /= std
        train_tests.append(train_test)
        trains.append(train)
        stds.append(std)

    y_train, y = df_train["ΔΔG.expt."].values, df["ΔΔG.expt."].values

    methods = []
    for alpha in np.logspace(-20, -11, 10, base=2):
        methods.append(f"Lasso {alpha}")
    for alpha in np.logspace(-20, -11, 10, base=2):
        methods.append(f"Ridge {alpha}")
    for alpha, l1ratio in product(np.logspace(-20, -11, 10, base=2), [0.5]):
        methods.append(f"ElasticNet {alpha} {l1ratio}")
    for n_components in range(1, 10):
        methods.append(f"PLS {n_components}")

    grid = pd.DataFrame(index=[col.replace("electronic_fold ", "") for col in df.filter(like="electronic_fold ").columns])

    x_train_all = np.concatenate(trains, axis=1)
    x_all = np.concatenate(train_tests, axis=1)
    print(x_train_all.shape)

    with Pool(NUM_WORKERS) as pool:
        results = list(
            pool.imap_unordered(
                regression_parallel,
                [(x_train_all, x_all, y_train, y, method, labels) for method in methods],
            )
        )

    for result in results:
        method, coef, predict, original_array, original_array_scaf = result
        print(method)

        for name, std, coef_ in zip(names, stds, np.split(coef, len(names))):
            grid[f"{method} {name}_coef"] = coef_ / std

        df[f"{method} regression"] = np.where(df["test"] == 0, predict, np.nan)
        df[f"{method} prediction"] = np.where(df["test"] == 1, predict, np.nan)
        print(original_array.shape, original_array_scaf.shape)
        df.loc[df["test"] == 0, f"{method} cv"] = original_array

    feature_names = "_".join(names)
    df.to_pickle(path.replace(".pkl", f"_{feature_names}_regression.pkl"))
    grid.to_csv(path.replace(".pkl", f"_{feature_names}_regression.csv"))



def generate_combinations(elements):
    """Return all non-empty combinations of feature names."""
    result = []
    for r in range(1, len(elements) + 1):
        result.extend([list(c) for c in combinations(elements, r)])
    return result


if __name__ == "__main__":
    feature_sets = generate_combinations(["electronic", "electrostatic"])
    dataset_paths = [
        "dataset/CBS.pkl",
        "dataset/DIP.pkl",
        "dataset/alpine_borane.pkl",
    ]

    for feat, path in product(feature_sets, dataset_paths):
        regression_(path, feat)
