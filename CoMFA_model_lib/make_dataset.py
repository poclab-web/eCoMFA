
import pandas as pd
from rdkit.Chem import PandasTools
from rdkit import Chem
import numpy as np
import os
import math

#from_file_path = "../sampledata/cbs_hand_read_0621.csv"
to_dir_path = "../arranged_dataset"
os.makedirs(to_dir_path, exist_ok=True)


def make_dataset(from_file_path, out_file_name):  # in ["dr.expt.BH3"]:
    df = pd.read_csv(from_file_path)  # .iloc[:5]
    print(len(df))
    print(df.columns)
    df = df.dropna(subset=["smiles"])
    print(len(df))
    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    df["q"] =df["mol"].apply(lambda x:Chem.AddHs(x))
    df["inchikey"] = df["q"].apply(lambda x: Chem.inchi.MolToInchiKey(x))
    df = df.dropna(subset=['er.', "mol", "smiles"])  # .dropna(subset=['smiles'])#順番重要！
    df.loc[df['er.'] <= 1, 'er.'] = 1
    df.loc[df['er.'] >= 99, 'er.'] = 99
    if "B_OH" in df.columns:

        df=df[df["B_OH"]==False]
    df["RT"] = 1.99 * 10 ** -3 * df["temperature"].values
    df["ΔΔG.expt."] = df["RT"].values * np.log(100 / df["er."].values - 1)
    df["ΔΔminG.expt."] =df["RT"].values* np.log(100 / 99 - 1)
    df["ΔΔmaxG.expt."] = df["RT"].values * np.log(100 / 1 - 1)
    PandasTools.AddMoleculeColumnToFrame(df, "smiles")
    df = df[[ "smiles", "ROMol","inchikey","er.", "RT", "ΔΔG.expt.","ΔΔminG.expt.","ΔΔmaxG.expt."]].drop_duplicates(subset="inchikey")
    #df = df[["smiles", "ROMol", "er.", "RT"]]
    print(df.columns)
    PandasTools.SaveXlsxFromFrame(df, to_dir_path + "/" + out_file_name, size=(100, 100))

    print(len(df))
    print("finish")

if __name__ == '__main__':
    # make_dataset("train","temperature","all_train.xls")
    make_dataset("../sampledata/cbs_hand_read_1207.csv", "cbs.xlsx")
    # make_dataset("../sampledata/cbs_hand_read_1030.csv","cbs.xls")
    make_dataset("../sampledata/DIP-chloride_0104.csv","DIP-chloride.xlsx")
    make_dataset("../sampledata/Ru_cat_1127.csv", "RuSS.xlsx")
    make_dataset("../sampledata/cbs_hand_scifinder.csv", "cbstestdata.xlsx")

    #make_dataset("../errortest/df3.csv","error.xls")
    # from CoMFA_model_lib import calculate_conformation
    #
    # df_dip=pd.read_excel(to_dir_path+"/DIP-chloride.xls")
    # df_cbs=pd.read_excel(to_dir_path+"/cbs.xls")
    # df_dip["mol"]=df_dip["smiles"].apply(lambda smiles:calculate_conformation.get_mol(smiles))
    # df_dip["InchyKey"]=df_dip["mol"].apply(lambda mol: mol.GetProp("InchyKey"))
    # df_cbs["mol"]=df_cbs["smiles"].apply(lambda smiles:calculate_conformation.get_mol(smiles))
    # df_cbs["InchyKey"]=df_cbs["mol"].apply(lambda mol: mol.GetProp("InchyKey"))
    # print(df_dip.columns)
    # l=[]
    # for er, inchykey in zip(df_dip["er."],df_dip["InchyKey"]):
    #     if inchykey in df_cbs["InchyKey"].values:
    #         ans=100-er
    #     else:
    #         ans=er
    #     l.append(ans)
    # df_dip["er."]=l
    # df_dip["ΔΔG.expt."] = df_dip["RT"].values * np.log(100 / df_dip["er."].values - 1)
    # print(df_dip["er."])
    # print(df_dip.columns)
    # PandasTools.AddMoleculeColumnToFrame(df_dip, "smiles")
    # df_cbs = df_cbs[["smiles", "ROMol", "er.", "RT", "ΔΔG.expt."]]
    # PandasTools.SaveXlsxFromFrame(df_dip,to_dir_path+"/DIP-chloride_revised.xls", size=(100, 100))
