import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools


def make_dataset(from_file_path, out_file_name):  # in ["dr.expt.BH3"]:
    # df = pd.read_csv(from_file_path)  # .iloc[:5]
    df = pd.read_excel(from_file_path,engine="openpyxl")
    # print(len(df))
    print(df)
    print(df.columns)
    df = df.dropna(subset=["smiles"])
    # print(len(df))
    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    # df["q"] =df["mol"].apply(lambda x:Chem.AddHs(x))
    df["inchikey"] = df["mol"].apply(lambda mol: Chem.inchi.MolToInchiKey(Chem.AddHs(mol)))
    df = df.dropna(subset=['er.', "mol", "smiles"])  # .dropna(subset=['smiles'])#順番重要！
    df.loc[df['er.'] <= 0.25, 'er.'] = 0.25
    df.loc[df['er.'] >= 99.75, 'er.'] = 99.75
    # if "βOH" in df.columns:
    #
    #     df=df[df["βOH"]!=True]
    if True:  # 計算できないものりすと
        # df=df[df["smiles"]!="C(=O)(c1ccc(Br)cc1)CCC(=C(C)O2)N=C2c1ccccc1"]
        # df=df[df["smiles"]!="O=C(C3=CCCC3)CC1=C(OC)C(C)=C2C(C(OC2)=O)=C1[Si](C(C)(C)C)(C)C"]
        # df = df[df["smiles"] != "C(=O)(c1ccc(F)cc1)CCCN2CCN(C3=NCC(F)C=N3)CC2"]
        # df = df[df["smiles"] != "c1ccccc1C(C)(C)C(=O)C#CCCCCCCCC"]
        # df = df[df["smiles"] != "CCCCCCCCC#CC(=O)C(C)(C)CC=C"]
        # df = df[df["smiles"] != "CCCCCCCCC#CC(=O)C(C)(C)CC=C"]
        # df = df[df["smiles"] != "O=C(C#C[Si](C)(C)C)C=C"]
        # df=df[df["smiles"] not in ["O=C1C(c2ccc(C(=O)OCC)cc2)=CCC1"]]
        df = df[~df["smiles"].isin(["C(=O)(c1ccc(F)cc1)CCCN2CCN(C3=NCC(F)C=N3)CC2",
                                    "c1ccccc1C(C)(C)C(=O)C#CCCCCCCCC",
                                    "c1ccccc1C(C)(C)C(=O)C#CCCCCCCCC",
                                    "CCCCCCCCC#CC(=O)C(C)(C)CC=C",
                                    "O=C(C#C[Si](C)(C)C)C=C",
                                    "O=C1C(c2ccc(C(=O)OCC)cc2)=CCC1"])]
    if True:
        df = df[df["mol"].map(lambda mol:
                              not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6][F,Cl]"))
                              and not mol.HasSubstructMatch(Chem.MolFromSmarts("[I,#7]"))
                              and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6][OH1]"))
                              and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6]*[OH1]"))
                              )]

    # print(df.columns)
    # print(len(df))
    df["RT"] = 1.99 * 10 ** -3 * df["temperature"].values
    df["ΔΔG.expt."] = df["RT"].values * np.log(100 / df["er."].values - 1)
    # df["ΔΔminG.expt."] =df["RT"].values* np.log(100 / 99 - 1)
    # df["ΔΔmaxG.expt."] = df["RT"].values * np.log(100 / 1 - 1)
    PandasTools.AddMoleculeColumnToFrame(df, "smiles")
    df = df[["smiles", "ROMol", "inchikey", "er.", "RT", "ΔΔG.expt."]].drop_duplicates(
        subset="inchikey")  # ,"ΔΔminG.expt.","ΔΔmaxG.expt."
    # df = df[["smiles", "ROMol", "er.", "RT"]]
    # print(df.columns)
    PandasTools.SaveXlsxFromFrame(df, out_file_name, size=(100, 100))

    # print(len(df))
    print("finish")


if __name__ == '__main__':
    to_dir_path = "../arranged_dataset"
    os.makedirs(to_dir_path, exist_ok=True)
    make_dataset("../sampledata/cbs_hand_read_0312.xlsx", to_dir_path + "/" + "cbs.xlsx")
    make_dataset("../sampledata/DIP-chloride_origin_0312.xlsx", to_dir_path + "/" + "DIP-chloride.xlsx")
    make_dataset("../sampledata/Ru_cat_0312.xlsx", to_dir_path + "/" + "RuSS.xlsx")
