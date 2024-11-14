from itertools import product
import numpy as np
import pandas as pd
import glob
import cclib
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem
from rdkit.Chem import PandasTools

def calc_grid(path):
    dfs=[]
    weights=[]
    for log in glob.glob(f'{path}/opt*.log'):
        print(log)
        data = cclib.io.ccread(log)
        print(data.enthalpy)
        print(data.entropy)
        weights.append(1)
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
        coord=coord/1
        coord=np.where(coord > 0, np.ceil(coord), np.floor(coord))
        dt = np.concatenate([_.split() for _ in dt[4 + n_atom:]]).astype(float).reshape(-1, 1)
        esp = np.concatenate([_.split() for _ in esp[4 + n_atom:]]).astype(float).reshape(-1, 1)
        steric=np.where(dt >1e-3, 0, dt)
        electrostatic=steric*esp
        steric[coord[:, 2] < 0] *= -1
        coord[:,1:3]=np.abs(coord[:,1:3])

        df=pd.DataFrame(data=np.concatenate([coord, steric,electrostatic], axis=1), columns=["x", "y", "z", "steric", "electrostatic"])
        df=df.groupby(['x', 'y', 'z'], as_index=False)[["steric", "electrostatic"]].sum()
        dfs.append(df)
    
    dfs=pd.concat(dfs).groupby(['x', 'y', 'z'], as_index=False)[["steric", "electrostatic"]].sum().astype({'x': int,'y': int,'z': int})
    # print(df)
    steric=pd.Series({f'steric {row.x} {row.y} {row.z}': row.steric for idx, row in dfs.iterrows()})
    electrostatic=pd.Series({f'electrostatic {row.x} {row.y} {row.z}': row.electrostatic for idx, row in dfs.iterrows()})
    return pd.concat([steric,electrostatic])



if __name__ == '__main__':
    #arranged_dataset読み込み
    path="/Users/mac_poclab/PycharmProjects/CoMFA_model/arranged_dataset/cbs.xlsx"
    df=pd.read_excel(path)
    df["molwt"] = df["SMILES"].apply(lambda smiles: ExactMolWt(Chem.MolFromSmiles(smiles)))
    df=df.sort_values("molwt").reset_index().iloc[:3]
    l=[]
    for inchikey in df["InChIKey"]:
        _=calc_grid(f'/Volumes/SSD-PSM960U3-UW/CoMFA_calc/{inchikey}')
        l.append(_)
    data=pd.DataFrame(l)
    print(data)
    df=pd.concat([df,data],axis=1).fillna(0)
    path=path.replace(".xlsx","_grid.xlsx")
    PandasTools.AddMoleculeColumnToFrame(df, "SMILES")
    PandasTools.SaveXlsxFromFrame(df, path, size=(100, 100))