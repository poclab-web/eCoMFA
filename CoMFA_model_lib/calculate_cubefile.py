import glob
import json
import multiprocessing
import os
import time
from itertools import product

import numpy as np
import pandas as pd
import psi4
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

import calculate_conformation
def gaussiansinglepoint(input_dir_name, output_dir_name, level="hf/sto-3g"):
    calc_condition = "# sp " + level
    charge_and_mult = '0 1'
    i = 0
    while os.path.isfile("{}/optimized{}.xyz".format(input_dir_name, i)):
        os.makedirs(output_dir_name, exist_ok=True)
        with open("{}/optimized{}.xyz".format(input_dir_name, i), "r") as f:
            ans = "\n".join(f.read().split("\n")[2:])
        with open("{}/gaussianinput{}.gjf".format(output_dir_name, i), 'w') as f:
            print("%nprocshared=8", file=f)
            print("%mem=6GB", file=f)
            print('%chk= {}.chk'.format(i), file=f)  # smiles
            print(calc_condition, file=f)

            print('', file=f)
            print('good luck!', file=f)
            print('', file=f)
            # print(head,file=f)
            print(charge_and_mult, file=f)

            print(ans, file=f)
        i += 1

def psi4calculation(input_dir_name, output_dir_name, level="hf/sto-3g"):
    psi4.set_num_threads(nthread=72)
    psi4.set_memory("64GB")
    # psi4.set_options({'geom_maxiter': 1000})

    psi4.set_options({'cubeprop_filepath': output_dir_name})
    i = 0
    print("{}/optimized{}.xyz".format(input_dir_name, i))
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
        # HOMO = scf_wfn.epsilon_a_subset("AO", "ALL")[scf_wfn.nalpha()]
        # LUMO = scf_wfn.epsilon_a_subset("AO", "ALL")[scf_wfn.nalpha() + 1]
        with open(output_dir_name + "/epsilon.txt", "w") as f:
            epsilon = wfn.epsilon_a_subset("AO", "ALL").np[wfn.nalpha() - 1]
            print(epsilon, file=f)
        # psi4.set_options({'cubeprop_tasks': ['frontier_orbitals'],
        #                   "cubic_grid_spacing": [0.2, 0.2, 0.2]
        #                   })
        # psi4.cubeprop(wfn)
        # tasks=['esp', "lol", "elf", 'frontier_orbitals','dual_descriptor']
        tasks=["esp"]
        psi4.set_options({'cubeprop_tasks': tasks,
                          "cubic_grid_spacing": [0.2, 0.2, 0.2],
                          "cubic_grid_overage": [8, 8, 8]
                          })
        psi4.cubeprop(wfn)
        try:
            os.rename(glob.glob(output_dir_name + "/Psi_a_*_LUMO.cube")[0], output_dir_name + "/LUMO02_{}.cube".format(i))
            os.rename(glob.glob(output_dir_name + "/Psi_a_*_HOMO.cube")[0], output_dir_name + "/HOMO02_{}.cube".format(i))
        except:
            print("Could not rename HOMO or LUMO")
        try:
            os.rename(glob.glob(output_dir_name + "/ELFa.cube")[0], output_dir_name + "/ELF02_{}.cube".format(i))
            os.remove(glob.glob(output_dir_name + "/ELFb.cube")[0])
        except:
            print("Could not rename ELF")
        try:
            os.rename(glob.glob(output_dir_name + "/LOLa.cube")[0], output_dir_name + "/LOL02_{}.cube".format(i))
            os.remove(glob.glob(output_dir_name + "/LOLb.cube")[0])
        except:
            print("Could not rename　LOL")
        try:
            os.rename(glob.glob(output_dir_name + "/DUAL_*.cube")[0], output_dir_name + "/DUAL02_{}.cube".format(i))
        except:
            print("Could not rename DUAL")
        try:
            os.rename(output_dir_name + "/Dt.cube", output_dir_name + "/Dt02_{}.cube".format(i))
        except:
            print("Could not rename Dt")
        try:
            os.rename(output_dir_name + "/ESP.cube", output_dir_name + "/ESP02_{}.cube".format(i))
        except:
            print("Could not rename ESP")
        with open(output_dir_name + "/geom.xyz", "r") as f:
            rl = f.read().split("\n")
            mol_output = rl[0] + "\n" + input_energy + "\n" + "\n".join(rl[2:])
        os.remove(output_dir_name + "/geom.xyz")
        with open(output_dir_name + "/optimized{}.xyz".format(i), "w") as f:
            print(mol_output, file=f)
        i += 1


def cube_to_pkl(dirs_name):
    i = 0
    while os.path.isfile("{}/optimized{}.xyz".format(dirs_name, i)):  # + "calculating"
        if os.path.isfile(dirs_name + "/data{}.pkl".format(i)):
            i += 1
            continue
        datas=[]
        columns=[]
        try:
            with open("{}/Dt02_{}.cube".format(dirs_name, i), 'r', encoding='UTF-8') as f:  # + "calculating"
                rl = f.read().splitlines()
            data = np.concatenate([_.split() for _ in rl[3 + 3 + n_atom:]]).astype(float).reshape(-1)
            datas.append(data)
            columns.append("Dt")
        except:
            print("couldn't read {}".format("{}/Dt02_{}.cube".format(dirs_name, i)))
        try:
            with open("{}/ESP02_{}.cube".format(dirs_name, i), 'r', encoding='UTF-8') as f:  # + "calculating"
                rl = f.read().splitlines()
            data = np.concatenate([_.split() for _ in rl[3 + 3 + n_atom:]]).astype(float).reshape(-1)
            datas.append(data)
            columns.append("ESP")
        except:
            print("couldn't read {}".format("{}/ESP02_{}.cube".format(dirs_name, i)))
        try:
            with open("{}/LUMO02_{}.cube".format(dirs_name, i), 'r', encoding='UTF-8') as f:  # + "calculating"
                rl = f.read().splitlines()
            data = np.concatenate([_.split() for _ in rl[3 + 3 + n_atom:]]).astype(float).reshape(-1)
            datas.append(data)
            columns.append("LUMO")
        except:
            print("couldn't read {}".format("{}/LUMO02_{}.cube".format(dirs_name, i)))
        try:
            with open("{}/DUAL02_{}.cube".format(dirs_name, i), 'r', encoding='UTF-8') as f:  # + "calculating"
                rl = f.read().splitlines()
            data = np.concatenate([_.split() for _ in rl[3 + 3 + n_atom:]]).astype(float).reshape(-1)

            datas.append(data)
            columns.append("DUAL")
        except:
            print("couldn't read {}".format("{}/DUAL02_{}.cube".format(dirs_name, i)))
        try:
            with open("{}/ELF02_{}.cube".format(dirs_name, i), 'r', encoding='UTF-8') as f:  # + "calculating"
                rl = f.read().splitlines()
            data = np.concatenate([_.split() for _ in rl[3 + 3 + n_atom:]]).astype(float).reshape(-1)
            datas.append(data)
            columns.append("ELF")
        except:
            print("couldn't read {}".format("{}/ELF02_{}.cube".format(dirs_name,i)))
        try:
            with open("{}/LOL02_{}.cube".format(dirs_name, i), 'r', encoding='UTF-8') as f:  # + "calculating"
                rl = f.read().splitlines()
            data = np.concatenate([_.split() for _ in rl[3 + 3 + n_atom:]]).astype(float).reshape(-1)
            datas.append(data)
            columns.append("LOL")
        except:
            print("couldn't read {}".format("{}/LOL02_{}.cube".format(dirs_name,i)))

        l = np.array([_.split() for _ in rl[2:6]])
        n_atom = int(l[0, 0])
        x0 = l[0, 1:].astype(float)
        size = l[1:, 0].astype(int)
        axis = l[1:, 1:].astype(float)
        l = np.array(list(product(range(size[0]), range(size[1]), range(size[2])))) @ axis + x0
        l = l * psi4.constants.bohr2angstroms
        print(dirs_name)
        df1=pd.DataFrame(l, columns=["x", "y", "z"]).astype("float32")
        df2=pd.DataFrame(datas,columns=columns).astype("float32")
        df= pd.concat([df1, df2], axis=1)
        # arr = np.concatenate(l.tolist()+data, axis=1)
        # df = pd.DataFrame(arr, columns=["x", "y", "z"]+columns).astype("float32")
        df.to_pickle(dirs_name + "/data{}.pkl".format(i))
        i += 1
    # if i != 0:
    #     os.rename(dirs_name + "calculating", dirs_name)

def calc(input):
    smiles,input_dirs_name,output_dirs_name,one_point_level=input
    print(smiles)
    mol = calculate_conformation.get_mol(smiles)
    if not os.path.isdir(output_dirs_name):
        print(mol.GetProp("InchyKey"))
        psi4calculation(input_dirs_name, output_dirs_name + "calculating", one_point_level)
        try:
            os.rename(output_dirs_name + "calculating", output_dirs_name)
        except:
            print("Not exist",output_dirs_name + "calculating")
    try:
        cube_to_pkl(output_dirs_name)
    except:
        None

if __name__ == '__main__':
    param_file_name = "../parameter/single-point-calculation/wB97X-D_def2-TZVP.txt"  # "../parameter_0221/parameter_cbs_gaussian.txt"  # _MP2
    # output_dir_name = "/Users/macmini_m1_2022/PycharmProjects/CoMFA_model/cube_aligned_b3lyp_6-31g(d)"  # "./psi4_cube_aligned"#_MP2
    with open(param_file_name, "r") as f:
        param = json.loads(f.read())
    print(param)
    dfs = []
    # for path in glob.glob("../arranged_dataset/*.xlsx"):
    for path in glob.glob("../arranged_dataset/newrea/newrea.xlsx"):

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
    # df = df[[os.path.isdir(features_dir_name + mol.GetProp("InchyKey")) for mol in df["mol"]]]

    print(df)


    while True:
        # for smiles in df["smiles"]:
        #     if True:  # smiles=="c1ccccc1OCN(C)CC(=O)c1ccccc1":#"Cl" in smiles:
        #         print(smiles)
        #         mol = calculate_conformation.get_mol(smiles)
        #         input_dirs_name = param["psi4_aligned_dir_name"] + "/" + mol.GetProp("InchyKey")
        #         output_dirs_name = param["cube_dir_name"] + "/" + mol.GetProp("InchyKey")
        #         if not os.path.isdir(output_dirs_name):
        #             print(mol.GetProp("InchyKey"))
        #             psi4calculation(input_dirs_name, output_dirs_name + "calculating", param["one_point_level"])
        #             os.rename(output_dirs_name + "calculating", output_dirs_name)
        #         cube_to_pkl(output_dirs_name)
        inputs=[]
        for smiles in df["smiles"]:
            mol = calculate_conformation.get_mol(smiles)
            input_dirs_name = param["psi4_aligned_dir_name"] + "/" + mol.GetProp("InchyKey")
            output_dirs_name = param["cube_dir_name"] + "/" + mol.GetProp("InchyKey")
            input=smiles,input_dirs_name,output_dirs_name,param["one_point_level"]
            inputs.append(input)
        p = multiprocessing.Pool(processes=1)
        p.map(calc, inputs)
        time.sleep(10)
