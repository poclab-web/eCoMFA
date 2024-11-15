import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from sklearn.model_selection import train_test_split

def common(from_file_path):
    df = pd.read_excel(from_file_path, engine="openpyxl")
    df["SMILES"]=df["SMILES"].apply(lambda smiles: Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))
    df["mol"] = df["SMILES"].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=['er.', "mol", "SMILES"])
    df["InChIKey"] = df["mol"].apply(lambda mol: Chem.inchi.MolToInchiKey(Chem.AddHs(mol)))
    df["er."]=df["er."].apply(lambda x:np.clip(x,0.25,99.75))
    df = df[df["mol"].map(lambda mol: not mol.HasSubstructMatch(Chem.MolFromSmarts("[I]")))]
    # df["RT"] = 1.99 * 10 ** -3 * df["temperature"].values
    df["ΔΔG.expt."] = 1.99 * 10 ** -3 * df["temperature"] * np.log(100 / df["er."].values - 1)
    return df

def output(df,to_file_path):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=4)
    train_df['test'] = 0
    test_df['test'] = 1
    df = pd.concat([train_df, test_df])

    PandasTools.AddMoleculeColumnToFrame(df, "SMILES")
    df = df[["entry","SMILES", "ROMol", "InChIKey", "temperature","er.", "ΔΔG.expt.","Reference url","test"]]
    PandasTools.SaveXlsxFromFrame(df, to_file_path, size=(100, 100))

    df["aliphatic_aliphatic"]=df["ROMol"].map(lambda mol: mol.HasSubstructMatch(Chem.MolFromSmarts("CC(=O)C")))
    df["aliphatic_aromatic"]=df["ROMol"].map(lambda mol: mol.HasSubstructMatch(Chem.MolFromSmarts("CC(=O)c")))
    df["aromatic_aromatic"]=df["ROMol"].map(lambda mol: mol.HasSubstructMatch(Chem.MolFromSmarts("cC(=O)c")))
    df["ring"]=df["ROMol"].map(lambda mol: mol.HasSubstructMatch(Chem.MolFromSmarts("[#6][C;R](=O)[#6]")))
    print(f'aliphatic_aliphatic aliphatic_aromatic aromatic_aromatic ring')
    print(len(df[df["aliphatic_aliphatic"]&~df["ring"]&~df["test"]]),len(df[df["aliphatic_aliphatic"]&~df["ring"]&df["test"]]),
          len(df[df["aliphatic_aromatic"]&~df["ring"]&~df["test"]]),len(df[df["aliphatic_aromatic"]&~df["ring"]&df["test"]]),
          len(df[df["aromatic_aromatic"]&~df["ring"]&~df["test"]]),len(df[df["aromatic_aromatic"]&~df["ring"]&df["test"]]),
          len(df[df["ring"]&~df["test"]]),len(df[df["ring"]&df["test"]]),
          len(df),len(df[df["test"]==0]),len(df[df["test"]==1]))

if __name__ == '__main__':
    
    df_cbs=common("/Users/mac_poclab/PycharmProjects/CoMFA_model/sampledata/CBS.xlsx")#.drop_duplicates(subset="InChIKey")
    df_dip=common("/Users/mac_poclab/PycharmProjects/CoMFA_model/sampledata/DIP.xlsx")#.drop_duplicates(subset="InChIKey")
    df_ru=common("/Users/mac_poclab/PycharmProjects/CoMFA_model/sampledata/Ru.xlsx")#.drop_duplicates(subset="InChIKey")

    df_cbs = df_cbs[df_cbs["mol"].map(lambda mol: not mol.HasSubstructMatch(Chem.MolFromSmarts("n")))]
    df_dip=df = df_dip[df_dip["mol"].map(lambda mol:
                              not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6][#7,OH1]"))
                              and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6]*[#7,OH1]"))
                              and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6]**[#7,OH1]")))]
    
    to_dir_path = "/Users/mac_poclab/PycharmProjects/CoMFA_model/arranged_dataset"

    os.makedirs(to_dir_path, exist_ok=True)
    output(df_cbs,f'{to_dir_path}/CBS.xlsx')
    output(df_dip,f'{to_dir_path}/DIP.xlsx')
    output(df_ru,f'{to_dir_path}/Ru.xlsx')
    df_all=pd.concat([df_cbs,df_dip,df_ru])[["InChIKey","SMILES"]]
    print(len(df_all))
    df_all=df_all.drop_duplicates(subset=["InChIKey"])
    print(len(df_all))
    df_all.to_csv(f'{to_dir_path}/mol_list.csv',index=False)