"""Convert Gaussian/Psi4 outputs into CoMFA grid features in parallel."""

import glob
import os
from itertools import product
from multiprocessing import Pool

import cclib
import numpy as np
import pandas as pd

AUTO_NUM_WORKERS = os.cpu_count() or 1
NUM_WORKERS = AUTO_NUM_WORKERS
CALC_ROOT = os.path.join(os.path.expanduser("~"), "CoMFA_calc")


def calc_grid__(log, temperature):
    """
    Parse one conformer log/cube set and return raw grid values plus thermodynamic weight.

    Args:
        log (str): Path to a log file readable by cclib.
        temperature (float): Temperature in Kelvin.

    Returns:
        tuple[pd.DataFrame, float]:
            Grid dataframe with `x, y, z, electronic, electrostatic`
            and a Boltzmann-related energy term (`enthalpy + entropy * T`).
    """
    data = cclib.io.ccread(log)
    weight = data.enthalpy + data.entropy * temperature

    dt_path = log.replace("opt", "Dt").replace(".log", ".cube")
    esp_path = log.replace("opt", "ESP").replace(".log", ".cube")

    with open(dt_path, "r", encoding="utf-8") as f:
        f.readline()
        f.readline()
        n_atom, x, y, z = f.readline().split()
        n1, x1, y1, z1 = f.readline().split()
        n2, x2, y2, z2 = f.readline().split()
        n3, x3, y3, z3 = f.readline().split()

        n_atom = int(n_atom)
        orient = np.array([x, y, z]).astype(float)
        size = np.array([n1, n2, n3]).astype(int)
        axis = np.array([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]]).astype(float)
        coord = np.array(list(product(range(size[0]), range(size[1]), range(size[2])))) @ axis + orient

        for _ in range(n_atom):
            f.readline()
        dt = np.fromstring(f.read(), dtype=float, sep=" ").reshape(-1, 1)

    with open(esp_path, "r", encoding="utf-8") as f:
        for _ in range(6 + n_atom):
            f.readline()
        esp = np.fromstring(f.read(), dtype=float, sep=" ").reshape(-1, 1)

    df = pd.DataFrame(
        data=np.hstack((coord, dt, esp)),
        columns=["x", "y", "z", "electronic", "electrostatic"],
    )
    return df, weight


def normal(x, u, v):
    """Evaluate a normal-distribution kernel value used as a smooth weighting term."""
    return 1 / np.sqrt(2 * np.pi * v) * np.exp(-((x - u) ** 2) / (2 * v))


def calc_grid(path, temperature):
    """
    Aggregate all conformer grids in one molecule directory into final fold/unfold features.

    Args:
        path (str): Directory containing `opt*.log` files.
        temperature (float): Temperature in Kelvin.

    Returns:
        pd.Series: Flattened CoMFA feature vector with unfolded and folded terms.
    """
    grids = []

    for log in glob.glob(f"{path}/opt*.log"):
        try:
            df, weight = calc_grid__(log, temperature)
            print(f"PARSING SUCCESS {log}")
        except Exception as e:
            print(f"PARSING FAILURE {log}")
            print(e)
            continue

        df = df[(df["x"] <= 6) & (df["x"] >= -12)]
        df = df[(df["y"] <= 6) & (df["y"] >= -6)]
        df = df[(df["z"] <= 12) & (df["z"] >= -12)]
        df["electrostatic"] = df["electrostatic"] * normal(np.log(df["electronic"]), np.log(0.001), 1)
        df["electronic"] = normal(np.log(df["electronic"]), np.log(0.001), 1)
        df["binary"] = np.where(df["electronic"] < 1e-3, 0, 1)

        df[["x", "y", "z"]] /= 2
        df[["x", "y", "z"]] = np.where(
            df[["x", "y", "z"]] > 0,
            np.ceil(df[["x", "y", "z"]]),
            np.floor(df[["x", "y", "z"]]),
        ).astype(int)
        df = df.groupby(["x", "y", "z"], as_index=False)[["electronic", "electrostatic", "binary"]].sum()

        w = np.exp(-df["electronic"] / df["electronic"].std())
        w /= w.sum()
        df["electronic"] *= w

        w = np.exp(-abs(df["electrostatic"]) / df["electrostatic"].std())
        w /= w.sum()
        df["electrostatic"] *= w

        df["gibbs"] = weight
        grids.append(df.copy())

    def total_keepnoindex(d):
        weights = d.gibbs.values
        weights = np.array(weights) - np.min(weights)

        sweights = d.electronic.values
        if np.sqrt(np.average(sweights**2)) == 0:
            sweights = 1
        else:
            sweights = np.exp(-weights / 3.1668114e-6 / temperature)
            sweights /= np.sum(sweights)

        eweights = d.electrostatic.values
        if np.sqrt(np.average(eweights**2)) == 0:
            eweights = 1
        else:
            eweights = np.exp(-weights / 3.1668114e-6 / temperature)
            eweights /= np.sum(eweights)

        return pd.DataFrame(
            {
                "x": d.x.mean(),
                "y": d.y.mean(),
                "z": d.z.mean(),
                "electronic": (d.electronic * sweights).sum(),
                "electrostatic": (d.electrostatic * eweights).sum(),
                "binary": (d.binary * weights).sum(),
            },
            index=["hoge"],
        )

    grids = pd.concat(grids)
    wgrids = grids.groupby(["x", "y", "z"], as_index=False).apply(total_keepnoindex).astype(
        {"x": int, "y": int, "z": int}
    )

    electronic = pd.Series(
        {
            f"electronic_unfold {int(row.x)} {int(row.y)} {int(row.z)}": row.electronic
            for _, row in wgrids.iterrows()
        }
    )
    electrostatic = pd.Series(
        {
            f"electrostatic_unfold {int(row.x)} {int(row.y)} {int(row.z)}": row.electrostatic
            for _, row in wgrids.iterrows()
        }
    )

    wgrids.loc[wgrids["z"] < 0, ["electronic", "electrostatic", "binary"]] *= -1
    wgrids[["y", "z"]] = wgrids[["y", "z"]].abs()
    wgrids = wgrids.groupby(["x", "y", "z"], as_index=False)[["electronic", "electrostatic", "binary"]].sum()

    fold_electronic = pd.Series(
        {
            f"electronic_fold {int(row.x)} {int(row.y)} {int(row.z)}": row.electronic
            for _, row in wgrids.iterrows()
        }
    )
    fold_electrostatic = pd.Series(
        {
            f"electrostatic_fold {int(row.x)} {int(row.y)} {int(row.z)}": row.electrostatic
            for _, row in wgrids.iterrows()
        }
    )

    return pd.concat([electronic, electrostatic, fold_electronic, fold_electrostatic])


def process_row(row):
    """Process one dataframe row by mapping `InChIKey` and `temperature` to `calc_grid`."""
    return calc_grid(f"{CALC_ROOT}/{row['InChIKey']}", row["temperature"])


def calc_grid_(path):
    """
    Generate CoMFA grid features for all molecules in one dataset file.

    Args:
        path (str): Input Excel path containing at least `InChIKey` and `temperature`.
    """
    print(f"START PARSING {path}")
    df = pd.read_excel(path)

    with Pool(NUM_WORKERS) as pool:
        results = pool.map(process_row, [row for _, row in df.iterrows()])

    data = pd.DataFrame(results)
    df = pd.concat([df, data], axis=1).fillna(0)
    output_path = path.replace(".xlsx", ".pkl")
    df.to_pickle(output_path)


if __name__ == "__main__":
    calc_grid_("results/alpine_borane.xlsx")
    calc_grid_("results/CBS.xlsx")
    calc_grid_("results/DIP.xlsx")
