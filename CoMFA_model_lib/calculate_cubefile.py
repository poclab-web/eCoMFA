import json
from itertools import product

import numpy as np
import pandas as pd
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem
import calculate_conformation
import psi4
import os
import glob
import time


def psi4calculation(input_dir_name, output_dir_name, level="hf/sto-3g"):
    psi4.set_num_threads(nthread=2)
    psi4.set_memory("2GB")
    # psi4.set_options({'geom_maxiter': 1000})

    psi4.set_options({'cubeprop_filepath': output_dir_name})
    i = 0
    while os.path.isfile("{}/optimized{}.xyz".format(input_dir_name, i)):
        os.makedirs(output_dir_name, exist_ok=True)
        print("{}/optimized{}.xyz".format(input_dir_name, i))
        with open("{}/optimized{}.xyz".format(input_dir_name, i), "r") as f:
            rl = f.read().split("\n")
            mol_input = "0 1"
            mol_input += "\n nocom\n noreorient\n "
            mol_input += "\n".join(rl[2:])
            molecule = psi4.geometry(mol_input)
            input_energy = rl[1]
        energy, wfn = psi4.energy(level, molecule=molecule, return_wfn=True)
        psi4.set_options({'cubeprop_tasks': ['frontier_orbitals'],
                          "cubic_grid_spacing": [0.2, 0.2, 0.2]
                          })
        psi4.cubeprop(wfn)
        os.rename(glob.glob(output_dir_name + "/Psi_a_*_LUMO.cube")[0], output_dir_name + "/LUMO02_{}.cube".format(i))
        os.remove(glob.glob(output_dir_name + "/Psi_a_*_HOMO.cube")[0])

        psi4.set_options({'cubeprop_tasks': ['esp'],
                          "cubic_grid_spacing": [0.2, 0.2, 0.2]
                          })
        psi4.cubeprop(wfn)
        os.rename(output_dir_name + "/Dt.cube", output_dir_name + "/Dt02_{}.cube".format(i))
        os.rename(output_dir_name + "/ESP.cube", output_dir_name + "/ESP02_{}.cube".format(i))
        psi4.set_options({'cubeprop_tasks': ['dual_descriptor'],
                          "cubic_grid_spacing": [0.2, 0.2, 0.2]
                          })
        psi4.cubeprop(wfn)
        os.rename(glob.glob(output_dir_name + "/DUAL_*.cube")[0], output_dir_name + "/DUAL02_{}.cube".format(i))
        # os.rename(output_dir_name + "/geom.xyz",output_dir_name + "/optimized{}.xyz".format(i))
        with open(output_dir_name + "/geom.xyz", "r") as f:
            rl = f.read().split("\n")
            mol_output = rl[0] + "\n" + input_energy + "\n" + "\n".join(rl[2:])
        os.remove(output_dir_name + "/geom.xyz")
        with open(output_dir_name + "/optimized{}.xyz".format(i), "w") as f:
            print(mol_output, file=f)
        i += 1


def cube_to_pkl(dirs_name):

    i = 0
    while os.path.isfile("{}/optimized{}.xyz".format(dirs_name , i)):#+ "calculating"
        if os.path.isfile(dirs_name + "/data{}.pkl".format(i)):
            i+=1
            continue
        with open("{}/Dt02_{}.cube".format(dirs_name , i), 'r', encoding='UTF-8') as f:#+ "calculating"
            Dt = f.read().splitlines()
        with open("{}/ESP02_{}.cube".format(dirs_name , i), 'r', encoding='UTF-8') as f:#+ "calculating"
            ESP = f.read().splitlines()
        with open("{}/LUMO02_{}.cube".format(dirs_name , i), 'r', encoding='UTF-8') as f:#+ "calculating"
            LUMO = f.read().splitlines()
        with open("{}/DUAL02_{}.cube".format(dirs_name , i), 'r', encoding='UTF-8') as f:#+ "calculating"
            DUAL = f.read().splitlines()

        l = np.array([_.split() for _ in Dt[2:6]])
        n_atom = int(l[0, 0])
        x0 = l[0, 1:].astype(float)
        size = l[1:, 0].astype(int)
        axis = l[1:, 1:].astype(float)
        Dt = np.concatenate([_.split() for _ in Dt[3 + 3 + n_atom:]]).astype(float).reshape(-1, 1)
        ESP = np.concatenate([_.split() for _ in ESP[3 + 3 + n_atom:]]).astype(float).reshape(-1, 1)
        LUMO = np.concatenate([_.split() for _ in LUMO[3 + 3 + n_atom:]]).astype(float).reshape(-1, 1)
        DUAL = np.concatenate([_.split() for _ in DUAL[3 + 3 + n_atom:]]).astype(float).reshape(-1, 1)
        l = np.array(list(product(range(size[0]), range(size[1]), range(size[2])))) @ axis + x0
        l = l * psi4.constants.bohr2angstroms
        arr = np.concatenate([l, Dt, ESP, LUMO, DUAL], axis=1)
        df = pd.DataFrame(arr, columns=["x", "y", "z", "Dt", "ESP", "LUMO", "DUAL"])
        df.to_pickle(dirs_name + "/data{}.pkl".format(i))
        i += 1
    # if i != 0:
    #     os.rename(dirs_name + "calculating", dirs_name)


if __name__ == '__main__':
    param_file_name = "../parameter/single-point-calculation/wB97X-D_def2-TZVP.txt"#"../parameter_0221/parameter_cbs_gaussian.txt"  # _MP2
    #output_dir_name = "/Users/macmini_m1_2022/PycharmProjects/CoMFA_model/cube_aligned_b3lyp_6-31g(d)"  # "./psi4_cube_aligned"#_MP2
    with open(param_file_name, "r") as f:
        param = json.loads(f.read())
    print(param)
    dfs = []
    for path in glob.glob("../arranged_dataset/*.xlsx"):
        df = pd.read_excel(path)
        dfs.append(df)
    df = pd.concat(dfs).dropna(subset=['smiles']).drop_duplicates(subset=["smiles"])
    # df1 = pd.read_excel(data_file_path)
    # df2 = pd.read_excel("../arranged_dataset/DIP-chloride.xlsx")
    # df3 = pd.read_excel("../arranged_dataset/Russ.xlsx")
    # df = pd.concat([df1, df2,df3]).dropna(subset=['smiles']).drop_duplicates(subset=["smiles"])
    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=['mol'])
    df["molwt"] = df["smiles"].apply(lambda smiles: ExactMolWt(Chem.MolFromSmiles(smiles)))
    df = df.sort_values("molwt")  # [:2]
    #df = df[[os.path.isdir(features_dir_name + mol.GetProp("InchyKey")) for mol in df["mol"]]]

    print(df)

    if True:
        for smiles in df["smiles"]:
            print(smiles)
            mol = calculate_conformation.get_mol(smiles)
            input_dirs_name = param["psi4_aligned_dir_name"]+"/"+ mol.GetProp("InchyKey")
            output_dirs_name = param["cube_dir_name"] + "/" + mol.GetProp("InchyKey")
            if not os.path.isdir(output_dirs_name):
                print(mol.GetProp("InchyKey"))
                psi4calculation(input_dirs_name, output_dirs_name +"calculating", param["one_point_level"])
                try:
                    os.rename(output_dirs_name +"calculating",output_dirs_name)
                except:
                    None
            cube_to_pkl(output_dirs_name)
        time.sleep(10)
