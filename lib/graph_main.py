"""Utilities for evaluating regression results, generating plots, and exporting cube data."""

import os
import re
import time
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools

DATASET_DIR = "results"
CALC_ROOT = os.path.join(os.path.expanduser("~"), "CoMFA_calc")
RESULTS_ROOT = os.path.join(os.path.expanduser("~"), "CoMFA_results")


def nan_rmse(x, y):
    """Return RMSE between prediction and reference arrays, skipping NaN prediction entries."""
    return np.sqrt(np.nanmean((y - x) ** 2))


def nan_r2(x, y):
    """Return R2 score after filtering out rows where prediction values are NaN."""
    x, y = x[~np.isnan(x)], y[~np.isnan(x)]
    return 1 - np.sum((y - x) ** 2) / np.sum((y - np.mean(y)) ** 2)


def evaluate_result(path):
    """
    Evaluate one regression result file and save summary metrics.

    It computes CV and regression RMSE/R2 values for each method and writes
    a `*_results.csv` file next to the input pickle.
    """
    start = time.time()
    df = pd.read_pickle(path)
    print(time.time() - start)

    df_results = pd.DataFrame(index=df.filter(like="cv").columns)
    df_results["cv_RMSE"] = df_results.index.map(
        lambda column: nan_rmse(df[column].values, df["ΔΔG.expt."].values)
    )
    df_results["cv_r2"] = df_results.index.map(
        lambda column: nan_r2(df[column].values, df["ΔΔG.expt."].values)
    )
    df_results["regression_RMSE"] = df.filter(like="regression").columns.map(
        lambda column: nan_rmse(df[column].values, df["ΔΔG.expt."].values)
    )
    df_results["regression_r2"] = df.filter(like="regression").columns.map(
        lambda column: nan_r2(df[column].values, df["ΔΔG.expt."].values)
    )

    df_results.to_csv(path.replace("_regression.pkl", "_results.csv"))
    best_cv_column = df_results["cv_RMSE"].idxmin()
    print(best_cv_column, np.log2(float(best_cv_column.split()[1])))
    return best_cv_column


def best_parameter(path):
    """
    Select the best CV setting, rebuild per-molecule contributions, and export report tables.

    The function adds `cv`, `prediction`, contribution terms, errors, and an
    Excel file with molecule images for manual inspection.
    """
    best_cv_column = pd.read_csv(path, index_col=0)["cv_RMSE"].idxmin()

    coef = pd.read_csv(path.replace("_results.csv", "_regression.csv"), index_col=0)
    coef = coef[
        [
            best_cv_column.replace("cv", "electronic_coef"),
            best_cv_column.replace("cv", "electrostatic_coef"),
        ]
    ]
    coef.columns = ["electronic_coef", "electrostatic_coef"]

    df = pd.read_pickle(path.replace("_results.csv", "_regression.pkl"))

    start = time.time()
    columns = (
        df.filter(like="electronic_unfold").columns.tolist()
        + df.filter(like="electrostatic_unfold").columns.tolist()
    )

    def calc_cont(column):
        x, y, z = map(int, re.findall(r"[+-]?\d+", column))
        coef_column = column.replace(f"_unfold {x} {y} {z}", "_coef")
        return df[column] * coef.at[f"{x} {abs(y)} {abs(z)}", coef_column] * np.sign(z)

    data = {col.replace("unfold", "cont"): calc_cont(col) for col in columns}
    data = pd.DataFrame(data=data)
    half = len(data.columns) // 2
    data["electronic_cont"] = data.iloc[:, :half].sum(axis=1)
    data["electrostatic_cont"] = data.iloc[:, half:].sum(axis=1)
    df = pd.concat([df, data], axis=1)
    print("time", time.time() - start)

    df["cv"] = df[best_cv_column]
    df["prediction"] = df[best_cv_column.replace("cv", "prediction")]
    df["er.prediction"] = 100 / (1 + np.exp(df["prediction"] / 1.99 / df["temperature"] / 0.001))
    df["er.cv"] = 100 / (1 + np.exp(df["cv"] / 1.99 / df["temperature"] / 0.001))
    df["regression"] = df[best_cv_column.replace("cv", "regression")]
    df["cv_error"] = df["cv"] - df["ΔΔG.expt."]
    df["prediction_error"] = df["prediction"] - df["ΔΔG.expt."]

    df_output = df[
        [
            "SMILES",
            "InChIKey",
            "ΔΔG.expt.",
            "electronic_cont",
            "electrostatic_cont",
            "regression",
            "prediction",
            "er.prediction",
            "er.cv",
            "cv",
            "prediction_error",
            "cv_error",
        ]
    ].fillna("NAN")
    PandasTools.AddMoleculeColumnToFrame(df_output, "SMILES")

    output_xlsx_path = path.replace("_results.csv", "_regression.xlsx")
    PandasTools.SaveXlsxFromFrame(df_output, output_xlsx_path, size=(100, 100))
    return df


def make_cube(df, path):
    """
    Export contribution cube files for each molecule in the dataframe.

    For each InChIKey, it writes `electronic.cube` and `electrostatic.cube`
    using the grid contribution values already computed in `df`.
    """
    grid = np.array([re.findall(r"[+-]?\d+", col) for col in df.filter(like="electronic_cont ").columns]).astype(int)
    min_coord = np.min(grid, axis=0).astype(int)
    max_coord = np.max(grid, axis=0).astype(int)
    span = max_coord - min_coord

    columns = ["ΔΔG.expt.", "temperature"]
    for x, y, z in product(
        range(min_coord[0], max_coord[0] + 1),
        range(min_coord[1], max_coord[1] + 1),
        range(min_coord[2], max_coord[2] + 1),
    ):
        if x != 0 and y != 0 and z != 0:
            columns.append(f"electronic_cont {x} {y} {z}")

    for x, y, z in product(
        range(min_coord[0], max_coord[0] + 1),
        range(min_coord[1], max_coord[1] + 1),
        range(min_coord[2], max_coord[2] + 1),
    ):
        if x != 0 and y != 0 and z != 0:
            columns.append(f"electrostatic_cont {x} {y} {z}")

    df = df.set_index("InChIKey").reindex(columns=columns, fill_value=0)

    step = 2
    min_text = " ".join(map(str, (min_coord + np.array([0.5, 0.5, 0.5])) * step))

    for inchikey, expt, temp, value in zip(
        df.index,
        df["ΔΔG.expt."],
        df["temperature"],
        df.iloc[:, 2:].values,
    ):
        dt_path = f"{CALC_ROOT}/{inchikey}/Dt0.cube"
        with open(dt_path, "r", encoding="utf-8") as f:
            f.readline()
            f.readline()
            n_atom, x, y, z = f.readline().split()
            n_atom = int(n_atom)
            f.readline()
            f.readline()
            f.readline()
            coord = [f.readline() for _ in range(n_atom)]
        coord = "".join(coord)

        electronic = "\n".join([" ".join(f"{x}" for x in value[i : i + 6]) for i in range(0, len(value) // 2, 6)])
        electrostatic = "\n".join(
            [" ".join(f"{x}" for x in value[i : i + 6]) for i in range(len(value) // 2, len(value), 6)]
        )

        contribution = np.sum(value[: len(value) // 2]), np.sum(value[len(value) // 2 :])
        pred = 100 / (1 + np.exp(sum(contribution) / 1.99 / temp / 0.001))

        os.makedirs(f"{path}/{inchikey}", exist_ok=True)
        with open(f"{path}/{inchikey}/electronic.cube", "w") as f:
            print(
                f"contribution Gaussian Cube File.\n"
                f"Property: Default # color electronic {contribution[0]:.2f} "
                f"predict {sum(contribution):.2f} expt {expt:.2f} pred {pred:.0f}\n"
                f"{n_atom} {min_text}\n"
                f"{span[0]} {step} 0 0\n"
                f"{span[1]} 0 {step} 0\n"
                f"{span[2]} 0 0 {step}\n"
                f"{coord}\n"
                f"{electronic}",
                file=f,
            )

        with open(f"{path}/{inchikey}/electrostatic.cube", "w") as f:
            print(
                f"contribution Gaussian Cube File.\n"
                f"Property: ALIE # color electrostatic {contribution[1]:.2f} "
                f"predict {sum(contribution):.2f} expt {expt:.2f} pred {pred:.0f}\n"
                f"{n_atom} {min_text}\n"
                f"{span[0]} {step} 0 0\n"
                f"{span[1]} 0 {step} 0\n"
                f"{span[2]} 0 0 {step}\n"
                f"{coord}\n"
                f"{electrostatic}",
                file=f,
            )


def graph_(df, path):
    """
    Create a publication-style scatter plot for model performance.

    The figure overlays regression, LOOCV, and test predictions against
    experimental `ΔΔG` values and writes it to disk.
    """
    plt.figure(figsize=(3, 3))
    plt.yticks([-4, 0, 4])
    plt.xticks([-4, 0, 4])
    plt.ylim(-4, 4)
    plt.xlim(-4, 4)

    plt.scatter(df["ΔΔG.expt."], df["regression"], c="black", linewidths=0, s=10, alpha=0.5)
    rmse = nan_rmse(df["regression"].values, df["ΔΔG.expt."].values)
    r2 = nan_r2(df["regression"].values, df["ΔΔG.expt."].values)
    plt.scatter(
        [],
        [],
        label="regression $r^2$ = " + f"{r2:.2f}" + "\n$\\mathrm{RMSE}$" + f" = {rmse:.2f} kcal/mol",
        c="black",
        linewidths=0,
        alpha=0.5,
        s=10,
    )

    rmse = nan_rmse(df["cv"].values, df["ΔΔG.expt."].values)
    r2 = nan_r2(df["cv"].values, df["ΔΔG.expt."].values)
    plt.scatter(
        [],
        [],
        label="LOOCV $r^2$ = " + f"{r2:.2f}" + "\n$\\mathrm{RMSE}$" + f" = {rmse:.2f} kcal/mol",
        c="dodgerblue",
        linewidths=0,
        alpha=0.6,
        s=10,
    )

    rmse = nan_rmse(df["prediction"].values, df["ΔΔG.expt."].values)
    r2 = nan_r2(df["prediction"].values, df["ΔΔG.expt."].values)
    plt.scatter(
        [],
        [],
        label="test $r^2$ = " + f"{r2:.2f}" + "\n$\\mathrm{RMSE}$" + f" = {rmse:.2f} kcal/mol",
        c="red",
        linewidths=0,
        alpha=0.8,
        s=10,
    )

    plt.scatter(df["ΔΔG.expt."], df["cv"], c="dodgerblue", linewidths=0, s=10, alpha=0.6)
    plt.scatter(df["ΔΔG.expt."], df["prediction"], c="red", linewidths=0, s=10, alpha=0.8)
    plt.xlabel("ΔΔ$\\mathit{G}^{‡}_{\\mathrm{expt}}$ [kcal/mol]")
    plt.ylabel("ΔΔ$\\mathit{G}^{‡}_{\\mathrm{predict}}$ [kcal/mol]")
    plt.legend(loc="lower right", fontsize=5, ncols=1)

    plt.text(
        -3.6,
        3.6,
        "$\\mathit{N}_{\\mathrm{test}}$"
        + f' = {len(df[df["test"] == 1])}\n'
        + "$\\mathit{N}_{\\mathrm{training}}$"
        + f' = {len(df[df["test"] == 0])}',
        fontsize=10,
        verticalalignment="top",
    )

    plt.tight_layout()
    plt.savefig(path.replace(".pkl", ".png"), dpi=500, transparent=True)


def bar():
    """
    Draw a combined LOOCV benchmark chart across datasets and model families.

    The left axis shows `r2`, the right axis shows RMSE, and each model
    family is visualized for CBS, DIP, and alpine borane datasets.
    """
    path = f"{DATASET_DIR}/"
    cbs = pd.read_csv(path + "cbs_electronic_electrostatic_results.csv", index_col=0)
    dip = pd.read_csv(path + "DIP_electronic_electrostatic_results.csv", index_col=0)
    alpine_borane = pd.read_csv(path + "alpine_borane_electronic_electrostatic_results.csv", index_col=0)

    dataset_labels = [r"$\mathit{(S)}$-CBS", r"$\mathit{(+)}$-DIP-Cl", r"$\mathit{(S)}$-alpine borane"]
    base_x = np.arange(3.0) * 4

    models = [
        (r"PLS [+-]?\d+ cv", "tab:red", "PLS"),
        (r"^Ridge .{0,} cv", "tab:orange", "Ridge"),
        (r"^ElasticNet .{0,} cv", "tab:green", "Elastic Net"),
        (r"^Lasso .{0,} cv", "tab:blue", "Lasso"),
    ]

    fig, ax1 = plt.subplots(figsize=(4.8, 3.2))
    ax2 = ax1.twinx()

    handles = []
    labels = []
    r2_array_max = np.array([cbs.max()["cv_r2"], dip.max()["cv_r2"], alpine_borane.max()["cv_r2"]])

    for model_idx, (regex, color, label) in enumerate(models):
        x_positions = base_x + model_idx * 0.9

        r2_array = np.array(
            [
                cbs.filter(regex=regex, axis=0).max()["cv_r2"],
                dip.filter(regex=regex, axis=0).max()["cv_r2"],
                alpine_borane.filter(regex=regex, axis=0).max()["cv_r2"],
            ]
        )

        rmse_array = np.array(
            [
                cbs.filter(regex=regex, axis=0).min()["cv_RMSE"],
                dip.filter(regex=regex, axis=0).min()["cv_RMSE"],
                alpine_borane.filter(regex=regex, axis=0).min()["cv_RMSE"],
            ]
        )

        face_colors = []
        for r2_val, r2_max in zip(r2_array, r2_array_max):
            if np.isclose(r2_val, r2_max):
                face_colors.append(color)
            else:
                face_colors.append("white")

        s = ax1.scatter(x_positions, r2_array, color=color, alpha=1, facecolor=face_colors)
        s = ax1.scatter(x_positions, r2_array, color=color, alpha=1, label=label + r" $r^2$", facecolor="none")
        b = ax2.bar(x_positions, rmse_array, color=color, alpha=1, width=0.4, label=label + " RMSE")

        handles.append(s)
        labels.append(label)
        handles.append(b)
        labels.append(label)

    ax1.set_ylabel(r"$r^2_{\mathrm{LOOCV}}$")
    ax1.set_yticks(np.arange(0, 1.1, 0.1))
    ax1.set_ylim(0.5, 0.9)
    ax1.tick_params(axis="y")

    ax2.set_ylabel("RMSE" + r"$_{\mathrm{LOOCV}}$" + " [kcal/mol]")
    ax2.set_ylim(0.5, 1)
    ax2.tick_params(axis="y")

    mid_x = base_x + 1.35
    ax1.set_xticks(mid_x)
    ax1.set_xticklabels(dataset_labels)
    ax1.axhline(0, color="black", linewidth=1.0)
    ax1.xaxis.set_ticks_position("none")

    plt.legend(handles=handles, ncol=4, bbox_to_anchor=(0.5, 1.02), loc="lower center", frameon=True, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path + "results_with_rmse.png", dpi=500, transparent=False)


if __name__ == "__main__":
    start = time.time()

    for cond in ["cbs", "DIP", "alpine_borane"]:
        evaluate_result(f"{DATASET_DIR}/{cond}_electronic_electrostatic_regression.pkl")

    df_cbs = best_parameter(f"{DATASET_DIR}/cbs_electronic_electrostatic_results.csv")
    df_dip = best_parameter(f"{DATASET_DIR}/DIP_electronic_electrostatic_results.csv")
    df_alp = best_parameter(f"{DATASET_DIR}/alpine_borane_electronic_electrostatic_results.csv")

    bar()

    make_cube(df_cbs, f"{RESULTS_ROOT}/CBS")
    make_cube(df_dip, f"{RESULTS_ROOT}/DIP")
    make_cube(df_alp, f"{RESULTS_ROOT}/alp")

    graph_(df_cbs, f"{DATASET_DIR}/regression_cbs.png")
    graph_(df_dip, f"{DATASET_DIR}/regression_dip.png")
    graph_(df_alp, f"{DATASET_DIR}/regression_alpine_borane.png")
    graph_(pd.concat([df_cbs, df_dip, df_alp]), f"{DATASET_DIR}/regression.png")

    print(f"Elapsed: {time.time() - start:.2f}s")
