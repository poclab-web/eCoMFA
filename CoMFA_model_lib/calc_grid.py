from itertools import product
import numpy as np
import pandas as pd
import glob
import cclib


def calc_grid__(log,T):
    """
    Extracts and processes grid data from molecular calculation output files.

    This function reads molecular calculation log and cube files to extract grid-based steric 
    and electrostatic data. It computes a data frame containing the grid points, steric contributions, 
    and electrostatic potential values, and calculates a thermodynamic weight based on enthalpy and 
    entropy at the given temperature.

    Args:
        log (str): Path to the log file containing molecular calculation results, readable by cclib.
                   The log file is expected to contain enthalpy and entropy information.
        T (float): Temperature in Kelvin, used to compute the thermodynamic weight.

    Returns:
        tuple:
            - df (pandas.DataFrame): A data frame with the following columns:
                - "x", "y", "z": Coordinates of grid points.
                - "steric": Steric contributions at each grid point from the `Dt` cube file.
                - "electrostatic": Electrostatic potential at each grid point from the `ESP` cube file.
            - weight (float): The computed thermodynamic weight as `enthalpy + entropy * T`.

    Notes:
        - The `Dt` and `ESP` cube files are inferred from the log file name by replacing `opt` with `Dt` 
          and `ESP`, and removing the `.log` extension.
        - The cube files are expected to contain grid data for steric and electrostatic contributions.
        - The `coord` array represents the coordinates of the grid points and is computed based on 
          orientation, axis, and grid size information from the cube files.

    Example:
        df, weight = calc_grid__("molecule_opt.log", T=298.15)
    """
    data = cclib.io.ccread(log)
    weight=data.enthalpy+data.entropy*T
    dt=log.replace('opt','Dt').replace('.log','.cube')
    esp=log.replace('opt','ESP').replace('.log','.cube')
    with open(dt, 'r', encoding='UTF-8') as f:
        f.readline()
        f.readline()
        n_atom,x,y,z=f.readline().split()
        n1,x1,y1,z1=f.readline().split()
        n2,x2,y2,z2=f.readline().split()
        n3,x3,y3,z3=f.readline().split()
        n_atom=int(n_atom)
        orient=np.array([x,y,z]).astype(float)
        size=np.array([n1,n2,n3]).astype(int)
        axis=np.array([[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]]).astype(float)
        coord = np.array(list(product(range(size[0]), range(size[1]), range(size[2])))) @ axis + orient

        for _ in range(n_atom):
            f.readline()
        dt=np.fromstring(f.read() ,dtype=float, sep=' ').reshape(-1,1)
    with open(esp, 'r', encoding='UTF-8') as f:
        for _ in range(6+n_atom):
            f.readline()
        esp=np.fromstring(f.read(), dtype=float, sep=' ').reshape(-1,1)
    df=pd.DataFrame(data=np.hstack((coord, dt,esp)), columns=["x", "y", "z", "steric", "electrostatic"])
    return df,weight

def calc_grid(path,T):
    """
    Aggregates and processes grid data for steric and electrostatic potentials from multiple log files.

    This function processes molecular grid data from log files in a given directory, applying
    steric and electrostatic adjustments, folding, and weighting based on thermodynamic properties.
    It generates steric and electrostatic potential data on a grid, both for unfolded and folded configurations.

    Args:
        path (str): The directory path containing the log files with names matching the pattern `opt*.log`.
        T (float): Temperature in Kelvin, used to calculate thermodynamic weights.

    Returns:
        pandas.Series: A combined series containing:
            - Steric and electrostatic potentials for unfolded grids.
            - Steric and electrostatic potentials for folded grids.
            Each entry is indexed by a string indicating the type and grid position:
                - `steric_unfold x y z`
                - `electrostatic_unfold x y z`
                - `steric_fold x y z`
                - `electrostatic_fold x y z`

    Workflow:
        1. Parse all `opt*.log` files in the specified directory.
        2. Extract steric and electrostatic grid data using `calc_grid__`.
        3. Filter and normalize steric values, then compute weighted electrostatic potentials.
        4. Align grids to integer positions and group data by grid points.
        5. Process folded grids (mirroring negative z-coordinates).
        6. Apply thermodynamic weights to the grid data based on enthalpy and entropy.
        7. Aggregate steric and electrostatic data into unfolded and folded forms.
        8. Return the combined series with labeled grid data.

    Notes:
        - The function skips files that fail to parse and logs the failure with an exception message.
        - Grid values are weighted using the formula: 
          `weights = exp(-Δweight / (3.1668114e-6 * T)) / sum(weights)` 
          where Δweight is relative to the minimum weight.

    Example:
        result = calc_grid("/path/to/logs", T=298.15)
    """
    grids=[]
    fold_grids=[]
    weights=[]
    for log in glob.glob(f'{path}/opt*.log'):
        try:
            df,weight=calc_grid__(log,T)
            print(f'PARCING SUCCESS {log}')

        except Exception as e:
            print(f'PARCING FAILURE {log}')
            print(e)
            continue
        df=df[df["steric"]>1e-6]
        df["steric"]=df["steric"].where(df["steric"]<1e-3,0)
        df["electrostatic"]=df["steric"]*df["electrostatic"]

        df[["x","y","z"]]/=1
        df[["x", "y", "z"]] = np.where(df[["x", "y", "z"]] > 0,np.ceil(df[["x", "y", "z"]]),np.floor(df[["x", "y", "z"]])).astype(int)
        
        df=df.groupby(['x', 'y', 'z'], as_index=False)[["steric", "electrostatic"]].sum()
        grids.append(df.copy())
        
        df.loc[df['z'] < 0, ['steric','electrostatic']] *= -1
        df[["y","z"]]=df[["y","z"]].abs()
        df=df.groupby(['x', 'y', 'z'], as_index=False)[["steric", "electrostatic"]].sum()

        fold_grids.append(df)
        weights.append(weight)
        
    weights=np.array(weights)-np.min(weights)
    weights=np.exp(-weights/3.1668114e-6/T)
    weights/=np.sum(weights)
    wgrids=[]
    wfold_grids=[]
    for weight,grid,foldgrid in zip(weights,grids,fold_grids):
        grid[["steric", "electrostatic"]] *= weight
        foldgrid[["steric", "electrostatic"]] *= weight
        wgrids.append(grid)
        wfold_grids.append(foldgrid)
    wgrids=pd.concat(wgrids).groupby(['x', 'y', 'z'], as_index=False)[["steric", "electrostatic"]].sum().astype({'x': int,'y': int,'z': int})    
    wfold_grids=pd.concat(wfold_grids).groupby(['x', 'y', 'z'], as_index=False)[["steric", "electrostatic"]].sum().astype({'x': int,'y': int,'z': int})
    steric=pd.Series({f'steric_unfold {int(row.x)} {int(row.y)} {int(row.z)}': row.steric for idx, row in wgrids.iterrows()})
    electrostatic=pd.Series({f'electrostatic_unfold {int(row.x)} {int(row.y)} {int(row.z)}': row.electrostatic for idx, row in wgrids.iterrows()})
    fold_steric=pd.Series({f'steric_fold {int(row.x)} {int(row.y)} {int(row.z)}': row.steric for idx, row in wfold_grids.iterrows()})
    fold_electrostatic=pd.Series({f'electrostatic_fold {int(row.x)} {int(row.y)} {int(row.z)}': row.electrostatic for idx, row in wfold_grids.iterrows()})
    return pd.concat([steric,electrostatic,fold_steric,fold_electrostatic])

def calc_grid_(path):
    """
    Processes molecular grid data for a set of molecules and saves the results.

    This function reads an Excel file containing molecular information, processes steric and electrostatic 
    grid data for each molecule using its InChIKey and temperature, and saves the resulting data to a 
    pickle file. Grid calculations are performed using the `calc_grid` function.

    Args:
        path (str): Path to the Excel file (.xlsx) containing molecular data.
                    The file must have the following columns:
                    - "InChIKey": A unique identifier for each molecule, used to locate calculation folders.
                    - "temperature": The temperature in Kelvin for each molecule.

    Returns:
        None: The function saves the resulting data as a pickle file (.pkl) in the same directory as the 
              input Excel file, with the same name.

    Workflow:
        1. Load molecular data from the specified Excel file.
        2. For each molecule, compute grid data using `calc_grid` with the specified path and temperature.
        3. Combine the computed grid data with the original data.
        4. Save the resulting data as a pickle file.

    Example:
        calc_grid_("/path/to/molecular_data.xlsx")
    """
    print(f'START PARCING {path}')
    df=pd.read_excel(path)
    l=[]
    for inchikey,temperature in zip(df["InChIKey"],df["temperature"]):
        _=calc_grid(f'/Volumes/SSD-PSM960U3-UW/CoMFA_calc/{inchikey}',temperature)
        l.append(_)
    data=pd.DataFrame(l)
    df=pd.concat([df,data],axis=1).fillna(0)
    path=path.replace(".xlsx",".pkl")
    df.to_pickle(path)

if __name__ == '__main__':
    #arranged_dataset読み込み
    calc_grid_("/Users/mac_poclab/PycharmProjects/CoMFA_model/arranged_dataset/cbs.xlsx")
    calc_grid_("/Users/mac_poclab/PycharmProjects/CoMFA_model/arranged_dataset/DIP.xlsx")
    calc_grid_("/Users/mac_poclab/PycharmProjects/CoMFA_model/arranged_dataset/Ru.xlsx")
