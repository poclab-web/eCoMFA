import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools


def make_dataset(from_file_path, out_file_name,flag):  # in ["dr.expt.BH3"]:
    df = pd.read_excel(from_file_path, engine="openpyxl")
    # print(from_file_path,len(df))
    df = df.dropna(subset=["smiles"])
    # print("smiles_dupl_drop",len(df))
    # print(len(df))
    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=['er.', "mol", "smiles"])
    df["inchikey"] = df["mol"].apply(lambda mol: Chem.inchi.MolToInchiKey(Chem.AddHs(mol)))

    df.loc[df['er.'] <= 0.25, 'er.'] = 0.25
    df.loc[df['er.'] >= 99.75, 'er.'] = 99.75
    df = df[~df["smiles"].isin([
        # "C(=O)(c1ccc(F)cc1)CCCN2CCN(C3=NCC(F)C=N3)CC2",
        # "c1ccccc1C(C)(C)C(=O)C#CCCCCCCCC",
        # "CCCCCCCCC#CC(=O)C(C)(C)CC=C",
        # "O=C(C#C[Si](C)(C)C)C=C",
        # "O=C1C(c2ccc(C(=O)OCC)cc2)=CCC1",
        # "COc1cccc(OC)c1C(=O)C",
        # "C1=C(Br)C(=O)CC1"
    ])]
    df = df[df["mol"].map(lambda mol:
                          # not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6][F,Cl,OH1,#7]"))
                          not mol.HasSubstructMatch(Chem.MolFromSmarts("[I]"))
                          # and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6][OH1]"))
                          # and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6]*[OH1,#7]"))
                          # and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6]*C(=O)[OH1]"))
                          )]
    if flag=="cbs":
        df = df[df["mol"].map(lambda mol:
                              not mol.HasSubstructMatch(Chem.MolFromSmarts("[#7]"))
                              )]
    elif flag=="dip":
        df = df[df["mol"].map(lambda mol:
                              not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6][F,Cl,OH1]"))
                              and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6]*[#7,#8]"))
                              and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6]*C(=O)[OH1]"))
                              )]
    elif flag=="ru":
        df = df[df["mol"].map(lambda mol:
                              not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6][#7,#8]"))
                              # and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6]*[#7]"))
                              )]
    # if flag:
    #     df=df[df["mol"].map(lambda mol:
    #                       not mol.HasSubstructMatch(Chem.MolFromSmarts("[I,#7]"))
    #                       )]
    df["RT"] = 1.99 * 10 ** -3 * df["temperature"].values
    df["ΔΔG.expt."] = df["RT"].values * np.log(100 / df["er."].values - 1)
    PandasTools.AddMoleculeColumnToFrame(df, "smiles")
    # print(df[df.duplicated(subset='inchikey')][["inchikey","smiles"]])
    df = df[["smiles", "ROMol", "inchikey", "er.", "RT", "ΔΔG.expt."]].drop_duplicates(
        subset="inchikey")
    PandasTools.SaveXlsxFromFrame(df, out_file_name, size=(100, 100))
    print("finish",out_file_name, len(df))


if __name__ == '__main__':
    to_dir_path = "../arranged_dataset"
    os.makedirs(to_dir_path, exist_ok=True)
    make_dataset("../sampledata/sample_0425/cbs_sample.xlsx", to_dir_path + "/" + "cbs.xlsx","cbs")
    make_dataset("../sampledata/sample_0425/(+)DIP-chloride_sample.xlsx", to_dir_path + "/" + "DIP-chloride.xlsx","dip")
    make_dataset("../sampledata/sample_0425/RuSS_sample.xlsx", to_dir_path + "/" + "RuSS.xlsx","ru")
