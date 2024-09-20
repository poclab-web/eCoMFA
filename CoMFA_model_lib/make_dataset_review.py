import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools


def make_dataset(from_file_path, out_file_name,flag):  # in ["dr.expt.BH3"]:
    df = pd.read_excel(from_file_path, engine="openpyxl")
    print(from_file_path,len(df))
    df = df.dropna(subset=["smiles"])
    print("smiles_dupl_drop",len(df))
    # print(len(df))
    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    # q=[]
    # for i in range(len(df)):
    #     print(df["smiles"][q])
    #     inchikey = Chem.inchi.MolToInchiKey(Chem.AddHs(df["mol"][q]))
    #
    # raise ValueError
    df["inchikey"] = df["mol"].apply(lambda mol: Chem.inchi.MolToInchiKey(Chem.AddHs(mol)))
    df = df.dropna(subset=["mol", "smiles"])
    # df.loc[df['er.'] <= 0.25, 'er.'] = 0.25
    # df.loc[df['er.'] >= 99.75, 'er.'] = 99.75
    df = df[~df["smiles"].isin([
        # "C(=O)(c1ccc(F)cc1)CCCN2CCN(C3=NCC(F)C=N3)CC2",
        # "c1ccccc1C(C)(C)C(=O)C#CCCCCCCCC",
        # "CCCCCCCCC#CC(=O)C(C)(C)CC=C",
        # "O=C(C#C[Si](C)(C)C)C=C",
        # "O=C1C(c2ccc(C(=O)OCC)cc2)=CCC1",
        # "COc1cccc(OC)c1C(=O)C",
        # "C1=C(Br)C(=O)CC1"
    ])]
    print(len(df))
    df = df[df["mol"].map(lambda mol:
                          # not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6][F]"))
                          not mol.HasSubstructMatch(Chem.MolFromSmarts("[I,Li,#50,+,-]"))
                          and (mol.HasSubstructMatch(Chem.MolFromSmarts("[#6](=[#8])([c,C])([c,C])"))
                               or mol.HasSubstructMatch(Chem.MolFromSmarts("[#6](=[#8])([c,C])([p,P])"))
                               or mol.HasSubstructMatch(Chem.MolFromSmarts("[#6](=[#8])([p,P])([c,C])"))
                               or mol.HasSubstructMatch(Chem.MolFromSmarts("[13C](=[#8])([N])([c,C])"))
                          # and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6][OH1]"))
                          # and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6]*[OH1]"))
                          ))]
    # if flag:
    #     df=df[df["mol"].map(lambda mol:
    #                       not mol.HasSubstructMatch(Chem.MolFromSmarts("[I,#7]"))
    #                       )]
    print(len(df))
    # df["RT"] = 1.99 * 10 ** -3 * df["Me_temperature"].values


    PandasTools.AddMoleculeColumnToFrame(df, "smiles")

    df = df[["smiles", "ROMol", "inchikey"]].drop_duplicates(
        subset="inchikey")
    # df["RT"].fillna(0)
    # df = df[["smiles", "ROMol", "inchikey", "er.", "RT", "ΔΔG.expt."]].drop_duplicates(
    #     subset="inchikey")
    df.to_csv("../test.csv")
    PandasTools.SaveXlsxFromFrame(df, out_file_name, size=(100, 100))
    print("finish", len(df))


if __name__ == '__main__':
    to_dir_path = "../arranged_dataset/review"
    os.makedirs(to_dir_path, exist_ok=True)
    make_dataset("../sampledata/sample_0425/cbs_review_0619.xlsx", to_dir_path + "/" + "cbs_review.xlsx", False)
    make_dataset("../sampledata/sample_0425/alpineborane_review.xlsx", to_dir_path + "/" + "alpine_review.xlsx", False)

