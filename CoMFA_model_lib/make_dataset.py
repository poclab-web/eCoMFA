
import pandas as pd
from rdkit.Chem import PandasTools
from rdkit import Chem
import numpy as np
import os

#from_file_path = "../sampledata/cbs_hand_read_0621.csv"
to_dir_path = "../arranged_dataset"
os.makedirs(to_dir_path, exist_ok=True)


def make_dataset(from_file_path, out_file_name):  # in ["dr.expt.BH3"]:
    df = pd.read_csv(from_file_path)  # .iloc[:5]
    df = df.dropna(subset=["smiles"])
    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=['er.', "mol", "smiles"])  # .dropna(subset=['smiles'])#順番重要！
    df["RT"] = 1.99 * 10 ** -3 * df["temperature"].values
    df["ΔΔG.expt."] = df["RT"].values * np.log(100 / df["er."].values - 1)
    print(len(df))
    # df = df[(df["entry"] != 222)]  # &(df.index!=136)
    PandasTools.AddMoleculeColumnToFrame(df, "smiles")
    df = df[[ "smiles", "ROMol", "er.", "RT", "ΔΔG.expt."]]
    PandasTools.SaveXlsxFromFrame(df, to_dir_path + "/" + out_file_name, size=(100, 100))

    print(len(df))
    print("finish")


# make_dataset("train","temperature","all_train.xls")
make_dataset("../sampledata/cbs_hand_read_0621.csv","cbs.xls")
make_dataset("../sampledata/DIP-chloride.csv","DIP-chloride.xls")