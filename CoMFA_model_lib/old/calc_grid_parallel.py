from itertools import product
import numpy as np
import pandas as pd
import glob
import cclib
from multiprocessing import Pool, cpu_count
from functools import partial


def calc_grid__(log, T):
    """
    Extracts and processes grid data from molecular calculation output files.
    """
    data = cclib.io.ccread(log)
    weight = data.enthalpy + data.entropy * T

    dt = log.replace('opt', 'Dt').replace('.log', '.cube')
    esp = log.replace('opt', 'ESP').replace('.log', '.cube')

    with open(dt, 'r', encoding='UTF-8') as f:
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
        dt = np.fromstring(f.read(), dtype=float, sep=' ').reshape(-1, 1)

    with open(esp, 'r', encoding='UTF-8') as f:
        for _ in range(6 + n_atom):
            f.readline()
        esp = np.fromstring(f.read(), dtype=float, sep=' ').reshape(-1, 1)

    df = pd.DataFrame(data=np.hstack((coord, dt, esp)), columns=["x", "y", "z", "steric", "electrostatic"])
    return df, weight


def process_log_file(log, T):
    try:
        df, weight = calc_grid__(log, T)
        print(f'PARCING SUCCESS {log}')

        df = df[df["steric"] > 1e-6]
        df["steric"] = df["steric"].where(df["steric"] < 1e-3, 0)
        df["electrostatic"] = df["steric"] * df["electrostatic"]

        df[["x", "y", "z"]] = np.where(df[["x", "y", "z"]] > 0, np.ceil(df[["x", "y", "z"]]), np.floor(df[["x", "y", "z"]])).astype(int)
        df = df.groupby(['x', 'y', 'z'], as_index=False)[["steric", "electrostatic"]].sum()

        fold_df = df.copy()
        fold_df.loc[fold_df['z'] < 0, ['steric', 'electrostatic']] *= -1
        fold_df[["y", "z"]] = fold_df[["y", "z"]].abs()
        fold_df = fold_df.groupby(['x', 'y', 'z'], as_index=False)[["steric", "electrostatic"]].sum()

        return df, fold_df, weight

    except Exception as e:
        print(f'PARCING FAILURE {log}')
        print(e)
        return None, None, None


def calc_grid_parallel(path, T):
    """
    Aggregates and processes grid data for steric and electrostatic potentials from multiple log files in parallel.
    """
    log_files = glob.glob(f'{path}/opt*.log')
    with Pool(max(len(log_files),5)) as pool:#cpu_count()
        results = pool.map(partial(process_log_file, T=T), log_files)

    grids, fold_grids, weights = [], [], []

    for df, fold_df, weight in results:
        if df is not None and fold_df is not None and weight is not None:
            grids.append(df)
            fold_grids.append(fold_df)
            weights.append(weight)

    weights = np.array(weights) - np.min(weights)
    weights = np.exp(-weights / (3.1668114e-6 * T))
    weights /= np.sum(weights)

    wgrids = []
    wfold_grids = []
    for weight, grid, fold_grid in zip(weights, grids, fold_grids):
        grid[["steric", "electrostatic"]] *= weight
        fold_grid[["steric", "electrostatic"]] *= weight
        wgrids.append(grid)
        wfold_grids.append(fold_grid)

    wgrids = pd.concat(wgrids).groupby(['x', 'y', 'z'], as_index=False)[["steric", "electrostatic"]].sum()
    wfold_grids = pd.concat(wfold_grids).groupby(['x', 'y', 'z'], as_index=False)[["steric", "electrostatic"]].sum()

    steric = pd.Series({f'steric_unfold {int(row.x)} {int(row.y)} {int(row.z)}': row.steric for _, row in wgrids.iterrows()})
    electrostatic = pd.Series({f'electrostatic_unfold {int(row.x)} {int(row.y)} {int(row.z)}': row.electrostatic for _, row in wgrids.iterrows()})
    fold_steric = pd.Series({f'steric_fold {int(row.x)} {int(row.y)} {int(row.z)}': row.steric for _, row in wfold_grids.iterrows()})
    fold_electrostatic = pd.Series({f'electrostatic_fold {int(row.x)} {int(row.y)} {int(row.z)}': row.electrostatic for _, row in wfold_grids.iterrows()})

    return pd.concat([steric, electrostatic, fold_steric, fold_electrostatic])


def calc_grid_(path):
    """
    Processes molecular grid data for a set of molecules and saves the results.
    """
    print(f'START PARCING {path}')
    df = pd.read_excel(path)
    results = []

    for inchikey, temperature in zip(df["InChIKey"], df["temperature"]):
        result = calc_grid_parallel(f'/Users/mac_poclab/CoMFA_calc/{inchikey}', temperature)
        results.append(result)

    data = pd.DataFrame(results)
    df = pd.concat([df, data], axis=1).fillna(0)
    path = path.replace(".xlsx", ".pkl")
    df.to_pickle(path)


if __name__ == '__main__':
    calc_grid_("/Users/mac_poclab/PycharmProjects/CoMFA_model/arranged_dataset/cbs.xlsx")
    calc_grid_("/Users/mac_poclab/PycharmProjects/CoMFA_model/arranged_dataset/DIP.xlsx")
    calc_grid_("/Users/mac_poclab/PycharmProjects/CoMFA_model/arranged_dataset/Ru.xlsx")
