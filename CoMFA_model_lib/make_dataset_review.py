import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
import xlsxwriter
def convert_smiles_to_romol(smiles):
    global invalid_smiles
    invalid_smiles =[]
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception as e:
        invalid_smiles.append(smiles)
        return None
    

def make_dataset(from_file_path, out_file_name,name,flag):  # in ["dr.expt.BH3"]:
    df = pd.read_excel(from_file_path,sheet_name=name, engine="openpyxl")
    invalid_smiles =[]
    # print(from_file_path,len(df))
    df = df.dropna(subset=["smiles"])
    # print("smiles_dupl_drop",len(df))
    # print(len(df))
    # df['ROMol_check'] = df['smiles'].apply(lambda x: convert_smiles_to_romol(x))
    # print(invalid_smiles)
    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=['er.', "mol", "smiles"])
    df["InChIKey"] = df["mol"].apply(lambda mol: Chem.inchi.MolToInchiKey(Chem.AddHs(mol)))

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
                              not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6][F,Cl,#7,OH1]"))#
                              and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6]*[#7,OH1]"))
                              and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6]**[#7,OH1]"))
                            #   and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6]*C(=O)[OH1]"))
                              )]
    elif False:#flag=="ru":
        df = df[df["mol"].map(lambda mol:
                              not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6][#7]"))
                            #   not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6][#7,#8]"))
                              # and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6]*[#7]"))
                              )]
    df["temperature"] =df["temperature"]+273.15
    df["RT"] = 1.99 * 10 ** -3 * df["temperature"].values
    df["ΔΔG.expt."] = df["RT"].values * np.log(100 / df["er."].values - 1)
    if True:
        PandasTools.AddMoleculeColumnToFrame(df, "smiles")
        # print(df[df.duplicated(subset='InChIKey')][["InChIKey","smiles"]])
        df = df[["smiles", "ROMol", "InChIKey", "er.", "RT", "ΔΔG.expt."]].drop_duplicates(
            subset="InChIKey")
        df = df[["smiles", "ROMol", "InChIKey", "er.", "RT", "ΔΔG.expt."]].drop_duplicates(
            subset="ROMol")
        print(df[df.duplicated(subset='InChIKey')][["InChIKey","smiles"]])
        df.to_csv("../test.csv")
        
        PandasTools.SaveXlsxFromFrame(df, out_file_name, size=(100, 100))
    else:
        df["aliphatic_aliphatic"]=df["mol"].map(lambda mol: mol.HasSubstructMatch(Chem.MolFromSmarts("CC(=O)C")))
        df["aliphatic_aromatic"]=df["mol"].map(lambda mol: mol.HasSubstructMatch(Chem.MolFromSmarts("CC(=O)c")))
        df["aromatic_aromatic"]=df["mol"].map(lambda mol: mol.HasSubstructMatch(Chem.MolFromSmarts("cC(=O)c")))
        df["ring"]=df["mol"].map(lambda mol: mol.HasSubstructMatch(Chem.MolFromSmarts("[#6][C;R](=O)[#6]")))

        print("finish",out_file_name, len(df))
        print(len(df[df["aliphatic_aliphatic"]&~df["ring"]]),len(df[df["aliphatic_aromatic"]&~df["ring"]]),len(df[df["aromatic_aromatic"]&~df["ring"]]),len(df[df["ring"]]))


if __name__ == '__main__':
    to_dir_path = "../../../arranged_dataset/review"
    os.makedirs(to_dir_path, exist_ok=True)
    # make_dataset("../sampledata/sample_0425/alpineborane.xlsx", to_dir_path + "/" + "alpine_review.xlsx","alpineborane", False)
    
    make_dataset("../sampledata/sample_0425/cbs_review_0621.xlsx", to_dir_path + "/" + "cbs_me_review.xlsx","cbs_me", False)
    make_dataset("../sampledata/sample_0425/cbs_review_0621.xlsx", to_dir_path + "/" + "cbs_nbu_review.xlsx","cbs_nbu", False)
    make_dataset("../sampledata/sample_0425/cbs_review_0621.xlsx", to_dir_path + "/" + "cbs_h_review.xlsx","cbs_h", False)
    # make_dataset("../sampledata/sample_0425/cbs_review_0621.xlsx", to_dir_path + "/" + "cbs_ph_review.xlsx","cbs_ph", False)
    # make_dataset("../sampledata/sample_0425/cbs_review_0621.xlsx", to_dir_path + "/" + "cbs_sime3_review.xlsx","cbs_sime3", False)


