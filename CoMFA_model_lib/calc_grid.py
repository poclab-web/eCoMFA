from itertools import product
import numpy as np
import pandas as pd
import glob
import cclib
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem

def calc_grid__(log,T):
    print(log)
    data = cclib.io.ccread(log)
    weight=data.enthalpy+data.entropy*T
    dt=log.replace('opt','Dt').replace('.log','.cube')
    esp=log.replace('opt','ESP').replace('.log','.cube')
    with open(esp, 'r', encoding='UTF-8') as f:
        esp=f.read().splitlines()[2:]
    with open(dt, 'r', encoding='UTF-8') as f:
        dt=f.read().splitlines()[2:]
    n_atom=int(dt[0].split()[0])
    orient=np.array(dt[0].split()[1:]).astype(float)
    size=np.array([dt[1].split()[0],dt[2].split()[0],dt[3].split()[0]]).astype(int)
    axis=np.array([dt[1].split()[1:],dt[2].split()[1:],dt[3].split()[1:]]).astype(float)
    coord = np.array(list(product(range(size[0]), range(size[1]), range(size[2])))) @ axis + orient
    dt = np.fromiter(' '.join(dt[4 + n_atom:]).split(), dtype=float).reshape(-1,1)#np.concatenate([_.split() for _ in dt[4 + n_atom:]]).astype(float).reshape(-1, 1)
    esp = np.fromiter(' '.join(esp[4 + n_atom:]).split(), dtype=float).reshape(-1,1)#np.concatenate([_.split() for _ in esp[4 + n_atom:]]).astype(float).reshape(-1, 1)
    df=pd.DataFrame(data=np.hstack((coord, dt,esp)), columns=["x", "y", "z", "steric", "electrostatic"])
    return df,weight

def calc_grid(path,T):
    #折りたたみ前の値を加える
    grids=[]
    fold_grids=[]
    weights=[]
    for log in glob.glob(f'{path}/opt*.log'):
        df,weight=calc_grid__(log,T)
        df["steric"]=df["steric"].where(df["steric"]<1e-3,0)
        df["electrostatic"]=df["steric"]*df["electrostatic"]

        df[["x","y","z"]]/=1
        df[["x", "y", "z"]] = np.where(df[["x", "y", "z"]] < 0,np.ceil(df[["x", "y", "z"]]),np.floor(df[["x", "y", "z"]]))
        df=df.groupby(['x', 'y', 'z'], as_index=False)[["steric", "electrostatic"]].sum()
        grids.append(df)
        
        df.loc[df['z'] < 0, ['steric','electrostatic']] *= -1
        df["y"]=np.abs(df["y"].values)
        df["z"]=np.abs(df["z"].values)
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
    steric=pd.Series({f'steric {int(row.x)} {int(row.y)} {int(row.z)}': row.steric for idx, row in wgrids.iterrows()})
    electrostatic=pd.Series({f'electrostatic {int(row.x)} {int(row.y)} {int(row.z)}': row.electrostatic for idx, row in wgrids.iterrows()})
    fold_steric=pd.Series({f'steric fold {int(row.x)} {int(row.y)} {int(row.z)}': row.steric for idx, row in wfold_grids.iterrows()})
    fold_electrostatic=pd.Series({f'electrostatic fold {int(row.x)} {int(row.y)} {int(row.z)}': row.electrostatic for idx, row in wfold_grids.iterrows()})
    return pd.concat([steric,electrostatic,fold_steric,fold_electrostatic])

def calc_grid_(path):
    print(path)
    df=pd.read_excel(path)
    df["molwt"] = df["SMILES"].apply(lambda smiles: ExactMolWt(Chem.MolFromSmiles(smiles)))
    df=df.sort_values("molwt").reset_index().iloc[:35]
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