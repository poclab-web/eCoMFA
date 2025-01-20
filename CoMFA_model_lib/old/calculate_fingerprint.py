from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from rdkit import Chem
import os
import CoMFA_model_lib.old.calculate_conformation as calculate_conformation
import csv
import json
from rdkit.Chem import PandasTools

def make_fp(smiles):
    mol=Chem.MolFromSmiles(smiles)
    calculate_conformation.GetCommonStructure(mol,"[#6](=[#8])([c,C])([c,C])")
    bitI_morgan = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, 0, bitInfo=bitI_morgan)
    common_atom = [[atom.GetIdx() for atom in mol.GetAtoms() if atom.GetProp("alignment") == str(i)][0] for i in
                   range(4)]
    print(common_atom)
    print(bitI_morgan.values())




def make_fpweight(df):
    df["ROMol"] = df["smiles"].apply(Chem.MolFromSmiles)

    keys2 = []
    keys3 = []

    for i in df.index:
        m = df["ROMol"][i]
        com = m.GetSubstructMatch(Chem.MolFromSmarts("[#6](=[#8])([c,C])([c,C])"))
        com = list(com[0:2]) + list(reversed(com[2:4])) if com[2] > com[3] else com
        bitI_morgan = {}
        AllChem.GetMorganFingerprintAsBitVect(m, 0, bitInfo=bitI_morgan)
        k2 = [k for k, v in bitI_morgan.items() if (com[2], 0) in v]
        k3 = [k for k, v in bitI_morgan.items() if (com[3], 0) in v]
        keys2.append(int(k2[0]))
        keys3.append(int(k3[0]))

    df["keys2"] = keys2
    df["keys3"] = keys3

    #         for bit, value in bitI_morgan.items():
    df = df.replace({'keys2': {80: 926, 1019: 1, 114: 1060,694:1380,1873:1380},
                     'keys3': {80: 926, 1019: 1, 114: 1060,694:1380,1873:1380}})  # 同じものを削除どれとどれが同じFPなのかは可視化プログラムで判断
    key2 = pd.DataFrame(df['keys2'].value_counts())
    key3 = pd.DataFrame(df['keys3'].value_counts())
    keysum = pd.concat([key2, key3], axis=1).fillna(0)
    keysum["fpsum"] = keysum["keys2"] + keysum["keys3"]
    return df,keysum

def make_eachweight(df,keysum):
    FP = keysum.index
    for i in FP:
        if int(keysum["fpsum"][i]) <=5:#とりあえず入れてパラメーターで変えれるようにする
            # print(i)
            df = df.replace({'keys2': {i: 0}, 'keys3': {i: 0}})
            keysum = keysum.drop(i)
        else:
            None
    if False:  # 1除外するため
        df = df.replace({'keys2': {1: 0}, 'keys3': {1: 0}})
        keysum = keysum.drop(1)
    # print(type(str((df4.index))))
    fplist = keysum.index.unique().tolist()
    fplist = list(map(str, fplist))

    fp_append = pd.DataFrame(index=df.index, columns=fplist)
    # print(df_append)
    df = pd.concat([df, fp_append], axis=1).fillna(0)
    for i in df.index:
        p = str(df["keys2"][i])
        q = str(df["keys3"][i])
        if p in df.columns:
            df.loc[i, p] = df.loc[i, p] + 1
            # print(True)
            # print(type(df.loc[i, p]))
        if q in df.columns:
            df.loc[i, q] = df.loc[i, q] - 1
        else:
            print(q+"notincolumns")
    return df ,fplist

if __name__ == '__main__':
    param_file_name = "../parameter/parameter_dip-chloride_gaussian.txt"
    param_file_name = "../parameter/parameter_cbs_gaussian.txt"
    param_file_name = "../parameter/parameter_RuSS_gaussian.txt"
    with open(param_file_name, "r") as f:
        param = json.loads(f.read())

    df=pd.read_excel(param["data_file_path"]).dropna(subset=['smiles']).reset_index()
    print(df)
    #df["smiles"].apply(make_fp)
    df,keysum=make_fpweight(df)
    print(keysum)
    df,fplist=make_eachweight(df, keysum)
    # to_dir_name = "../fingerprint/fparranged_dataset"
    # to_file_name = "Dip-chloride.csv"
    # to_dir_name2 = param["fplist"]


    # os.makedirs(to_dir_name, exist_ok=True)
    # os.makedirs(to_dir_name2, exist_ok=True)
    # print(df)

    df.to_csv(param["fpdata_file_path"])
    with open(param["fplist"],"w") as f:
        writer =csv.writer(f)
        writer.writerow(fplist)
    # PandasTools.SaveXlsxFromFrame(df, to_dir_name + "/" + to_file_name, size=(100, 100))



    # print(keysum)