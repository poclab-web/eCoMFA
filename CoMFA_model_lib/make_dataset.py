
import pandas as pd
from rdkit.Chem import PandasTools
from rdkit import Chem
import numpy as np
import os




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
    df.loc[df['er.'] <= 0.25, 'er.'] = 0.25
    df.loc[df['er.'] >= 99.75, 'er.'] = 99.75
    # if "βOH" in df.columns:
    #
    #     df=df[df["βOH"]!=True]
    if True :
        df=df[df["smiles"]!="C(=O)(c1ccc(Br)cc1)CCC(=C(C)O2)N=C2c1ccccc1"]
        df=df[df["smiles"]!="O=C(C3=CCCC3)CC1=C(OC)C(C)=C2C(C(OC2)=O)=C1[Si](C(C)(C)C)(C)C"]
        df=df[df["smiles"]!="C(=O)(c1ccc(F)cc1)CCCN2CCN(C3=NCC(F)C=N3)CC2"]
        df=df[df["smiles"]!="c1ccccc1C(C)(C)C(=O)C#CCCCCCCCC"]
        df = df[df["smiles"] != "CCCCCCCCC#CC(=O)C(C)(C)CC=C"]
        df = df[df["smiles"] != "CCCCCCCCC#CC(=O)C(C)(C)CC=C"]
    if True:
        df = df[df["mol"].map(lambda mol:
                          not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6][F,Cl]"))
                         and not mol.HasSubstructMatch(Chem.MolFromSmarts("[I]"))
                        and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6][OH1]"))
                          and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6]*[OH1]"))
                          )]

    print(df.columns)
    print(len(df))
    df["RT"] = 1.99 * 10 ** -3 * df["temperature"].values
    df["ΔΔG.expt."] = df["RT"].values * np.log(100 / df["er."].values - 1)
    # df["ΔΔminG.expt."] =df["RT"].values* np.log(100 / 99 - 1)
    # df["ΔΔmaxG.expt."] = df["RT"].values * np.log(100 / 1 - 1)
    PandasTools.AddMoleculeColumnToFrame(df, "smiles")
    df = df[[ "smiles", "ROMol","inchikey","er.", "RT", "ΔΔG.expt."]].drop_duplicates(subset="inchikey")#,"ΔΔminG.expt.","ΔΔmaxG.expt."
    #df = df[["smiles", "ROMol", "er.", "RT"]]
    print(df.columns)
    PandasTools.SaveXlsxFromFrame(df, out_file_name, size=(100, 100))

    print(len(df))
    print("finish")

if __name__ == '__main__':
    to_dir_path = "../arranged_dataset"
    os.makedirs(to_dir_path, exist_ok=True)
    # make_dataset("train","temperature","all_train.xls")
    make_dataset("../sampledata/cbs_hand_read_0116.csv", to_dir_path + "/" +"cbs.xlsx")
    # make_dataset("../sampledata/cbs_hand_read_1030.csv","cbs.xls")
    make_dataset("../sampledata/DIP-chloride_origin_0206.csv",to_dir_path + "/" + "DIP-chloride.xlsx")
    make_dataset("../sampledata/Ru_cat_1127.csv", to_dir_path + "/" +"RuSS.xlsx")
    # make_dataset("../sampledata/cbs_hand_scifinder.csv", "cbstestdata.xlsx")

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
