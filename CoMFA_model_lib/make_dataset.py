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

    df["er."]=df["er."].apply(lambda x:np.clip(x,0.5,99.5))
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
                              not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6][F,#7,OH1]"))#F,Cl,
                              and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6]*[#7,OH1]"))
                              and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6]**[#7,OH1]"))
                              )]
    elif False:#flag=="ru":
        df = df[df["mol"].map(lambda mol:
                              not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6][#7]"))
                            #   not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6][#7,#8]"))
                              # and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6]*[#7]"))
                              )]
    df["RT"] = 1.99 * 10 ** -3 * df["temperature"].values
    df["ΔΔG.expt."] = df["RT"].values * np.log(100 / df["er."].values - 1)
    if True:
        PandasTools.AddMoleculeColumnToFrame(df, "smiles")
        # print(df[df.duplicated(subset='inchikey')][["inchikey","smiles"]])
        df = df[["smiles", "ROMol", "inchikey", "er.", "RT", "ΔΔG.expt.","Reference url"]].drop_duplicates(
            subset="inchikey")
        PandasTools.SaveXlsxFromFrame(df, out_file_name, size=(100, 100))
    else:
        df["aliphatic_aliphatic"]=df["mol"].map(lambda mol: mol.HasSubstructMatch(Chem.MolFromSmarts("CC(=O)C")))
        df["aliphatic_aromatic"]=df["mol"].map(lambda mol: mol.HasSubstructMatch(Chem.MolFromSmarts("CC(=O)c")))
        df["aromatic_aromatic"]=df["mol"].map(lambda mol: mol.HasSubstructMatch(Chem.MolFromSmarts("cC(=O)c")))
        df["ring"]=df["mol"].map(lambda mol: mol.HasSubstructMatch(Chem.MolFromSmarts("[#6][C;R](=O)[#6]")))

        print("finish",out_file_name, len(df))
        print(len(df[df["aliphatic_aliphatic"]&~df["ring"]]),len(df[df["aliphatic_aromatic"]&~df["ring"]]),len(df[df["aromatic_aromatic"]&~df["ring"]]),len(df[df["ring"]]))


if __name__ == '__main__':
    to_dir_path = "../arranged_dataset"
    os.makedirs(to_dir_path, exist_ok=True)
    make_dataset("../sampledata/sample_0425/cbs_sample.xlsx", to_dir_path + "/" + "cbs.xlsx","cbs")
    make_dataset("../sampledata/fluoro_add/(+)DIP-chloride_sample.xlsx", to_dir_path + "/" + "DIP-chloride.xlsx","dip")
    make_dataset("../sampledata/sample_0425/RuSS_sample.xlsx", to_dir_path + "/" + "RuSS.xlsx","ru")
    to_dir_path = "../all_dataset"
    os.makedirs(to_dir_path, exist_ok=True)
    make_dataset("../sampledata/sample_0425/cbs_sample.xlsx", to_dir_path + "/" + "cbs.xlsx","cbs_all")
    make_dataset("../sampledata/fluoro_add/(+)DIP-chloride_sample.xlsx", to_dir_path + "/" + "DIP-chloride.xlsx","dip_all")
    make_dataset("../sampledata/sample_0425/RuSS_sample.xlsx", to_dir_path + "/" + "RuSS.xlsx","ru_all")