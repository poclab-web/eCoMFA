import glob
import heapq
import json
import os
import subprocess

import numpy as np
import pandas as pd
import psi4
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Geometry import Point3D

def get_mol(smiles):
   mol = Chem.MolFromSmiles(smiles)
   mol = Chem.AddHs(mol)
   carbon_13 = Chem.MolFromSmarts("[13C]")

   # 各部分構造を順次チェック
   found = GetCommonStructure(mol, "[#6](=[#8])([c,C])([c,C])")
   if not found:
       found = GetCommonStructure(mol, "[#6](=[#8])([p,P])([c,C])")
   if not found:
       found = GetCommonStructure(mol, "[#6](=[#8])([c,C])([p,P])")
   if mol.HasSubstructMatch(carbon_13):
       found = GetCommonStructure(mol, "[C;!13C](=[#8])([N])([c,C])")

   if found:
       mol.SetProp("InChIKey", Chem.MolToInchiKey(mol))
   else:
       print("No matching substructure found.")

   return mol

def GetCommonStructure(mol, common_structure):
    matches = mol.GetSubstructMatches(Chem.MolFromSmarts(common_structure))
    if not matches:
        print("False")
        return False

    com = matches[0]
    for atom in mol.GetAtoms():
        if atom.GetIdx() == com[0]:
            atom.SetProp("alignment", "0")
        elif atom.GetIdx() == com[1]:
            atom.SetProp("alignment", "1")
        elif atom.GetIdx() in com[2:]:
            atom.SetProp("alignment", "2")
        else:
            atom.SetProp("alignment", "-1")

    l = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetProp("alignment") == "2"]
    l.sort()
    if len(l) > 1:
        mol.GetAtomWithIdx(l[1]).SetProp("alignment", "3")

    return True



# def GetCommonStructure(mol, common_structure):
#     com = mol.GetSubstructMatches(Chem.MolFromSmarts(common_structure))[0]
#     for atom in mol.GetAtoms():
#         if atom.GetIdx() == com[0]:
#             atom.SetProp("alignment", "0")
#         elif atom.GetIdx() == com[1]:
#             atom.SetProp("alignment", "1")
#         elif atom.GetIdx() in com[2:]:
#             atom.SetProp("alignment", "2")
#         else:
#             atom.SetProp("alignment", "-1")
#     l = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetProp("alignment") == "2"]
#     l.sort()
#     mol.GetAtomWithIdx(l[1]).SetProp("alignment", "3")


# def get_mol(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     mol = Chem.AddHs(mol)
#     GetCommonStructure(mol, "[#6](=[#8])([c,C])([c,C])")
#     mol.SetProp("InChIKey", Chem.MolToInchiKey(mol))
#     return mol


# def CalcConfsEnergies(mol):
#     AllChem.EmbedMultipleConfs(mol, numConfs=param["numConfs"], randomSeed=1, pruneRmsThresh=0.01,
#                                numThreads=0)
#     for conf in mol.GetConformers():
#         mmff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol),
#                                                  confId=conf.GetId())  # nomal force field
#         # mmff=AllChem.UFFGetMoleculeForceField(mol,confId=cid) # other force field
#         mmff.Minimize()
#         # energy=psi4_calc.calc_energy(mol, confid=cidId) # calc energy by psi4
#         energy = mmff.CalcEnergy()
#         conf.SetProp("energy", str(energy))
def CalcConfsEnergies(mol, force_field):
    AllChem.EmbedMultipleConfs(mol, numConfs=param["numConfs"], randomSeed=1, pruneRmsThresh=0.01,
                               numThreads=0)
    for conf in mol.GetConformers():
        if force_field == "MMFF":
            ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol),
                                                   confId=conf.GetId())  # nomal force field
        if force_field == "UFF":
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf.GetId())  # other force field
        ff.Minimize()
        energy = ff.CalcEnergy()
        conf.SetProp("energy", str(energy))


def highenergycut(mol, energy):
    l = []
    for conf in mol.GetConformers():
        if float(conf.GetProp("energy")) - min([float(conf.GetProp("energy")) for conf in mol.GetConformers()]) >= \
                energy:
            l.append(conf.GetId())
    for i in l:
        mol.RemoveConformer(i)


def rmsdcut(mol, rmsd_):
    l = sorted([[float(conf.GetProp("energy")), conf.GetId()] for conf in mol.GetConformers()], reverse=True)
    for i, (_, conf1) in enumerate(l):
        for _, conf2 in l[1 + i:]:
            rmsd = AllChem.GetBestRMS(Chem.rdmolops.RemoveHs(mol), Chem.rdmolops.RemoveHs(mol), conf1, conf2)
            if rmsd < rmsd_:
                mol.RemoveConformer(conf1)
                break


def delconformer(mol, n):
    l = [float(conf.GetProp("energy")) for conf in mol.GetConformers()]
    dellist = [conf.GetId() for conf in mol.GetConformers() if
               float(conf.GetProp("energy")) not in heapq.nsmallest(n, l)]
    for id in dellist:
        mol.RemoveConformer(id)


def Rodrigues_rotation(n, sin, cos):
    ans = np.array(
        [[n[0] ** 2 * (1 - cos) + cos, n[0] * n[1] * (1 - cos) - n[2] * sin, n[0] * n[2] * (1 - cos) + n[1] * sin],
         [n[0] * n[1] * (1 - cos) + n[2] * sin, n[1] ** 2 * (1 - cos) + cos, n[1] * n[2] * (1 - cos) - n[0] * sin],
         [n[0] * n[2] * (1 - cos) - n[1] * sin, n[1] * n[2] * (1 - cos) + n[0] * sin, n[2] ** 2 * (1 - cos) + cos]
         ])
    return ans


def transform(conf, carbonyl_atom):
    conf = conf.GetPositions()
    c, o, c1, c2 = carbonyl_atom
    conf = conf - conf[c]
    a = conf[o] - conf[c]
    a = a / np.linalg.norm(a)
    cos1 = np.dot(a, np.array([1, 0, 0]))
    cros1 = np.cross(a, np.array([1, 0, 0]))
    sin1 = np.linalg.norm(cros1)
    n1 = cros1 / sin1

    b = conf[c2] - conf[c1]
    b_ = np.dot(Rodrigues_rotation(n1, sin1, cos1), b)
    byz = b_ * np.array([0, 1, 1])
    byz = byz / np.linalg.norm(byz)
    if False:

        cos2 = np.dot(byz, np.array([0, 1, 0]))
        cros2 = np.cross(byz, np.array([0, 1, 0]))

    else:
        cos2 = np.dot(byz, np.array([0, 0, 1]))
        cros2 = np.cross(byz, np.array([0, 0, 1]))
    sin2 = np.linalg.norm(cros2)
    n2 = cros2 / sin2
    conf = np.dot(Rodrigues_rotation(n1, sin1, cos1), conf.T).T
    conf = np.dot(Rodrigues_rotation(n2, sin2, cos2), conf.T).T

    return conf


def ConfTransform(mol):
    common_atom = [[atom.GetIdx() for atom in mol.GetAtoms() if atom.GetProp("alignment") == str(i)][0] for i in
                   range(4)]
    for conf in mol.GetConformers():
        conf_transformed = transform(conf, common_atom)
        for i, conf_ in enumerate(conf_transformed):
            conf.SetAtomPosition(i, Point3D(conf_[0], conf_[1], conf_[2]))


def conf_to_xyz(mol, out_dir_name):
    os.makedirs(out_dir_name + "calculating", exist_ok=True)
    for i, conf in enumerate(mol.GetConformers()):
        file_name = "{}/optimized{}.xyz".format(out_dir_name + "calculating", i)
        Chem.MolToXYZFile(mol, file_name, conf.GetId())
        with open(file_name, "r") as f:
            rl = f.read().split("\n")
        with open(file_name, "w") as f:
            print(rl[0], file=f)
            print(conf.GetProp("energy"), file=f)
            print("\n".join(rl[2:]), file=f)
    try:
        os.rename(out_dir_name + "calculating", out_dir_name)
    except:
        None


def psi4optimization(input_dir_name, output_dir_name, level="hf/sto-3g"):
    psi4.set_num_threads(nthread=8)
    psi4.set_memory("8GB")
    psi4.set_options({'geom_maxiter': 10000})
    i = 0
    while os.path.isfile("{}/optimized{}.xyz".format(input_dir_name, i)):
        try:
            # psi4.set_output_file(dir + "/{}/calculation.log".format(i))
            with open("{}/optimized{}.xyz".format(input_dir_name, i), "r") as f:
                ans = f.read()
                molecule = psi4.geometry(ans)

            energy = psi4.optimize(level, molecule=molecule)
            os.makedirs(output_dir_name, exist_ok=True)
            file = "{}/optimized{}.xyz".format(output_dir_name, i)
            print(file)
            open(file, 'w')
            with open(file, 'w') as f:
                print(molecule.natom(), file=f)
                print(energy * psi4.constants.hartree2kcalmol, file=f)
                print("\n".join(molecule.save_string_xyz().split('\n')[1:]), file=f)
            i += 1
        except:
            i += 1
            None


def read_xyz(mol, input_dir_name):
    mol.RemoveAllConformers()
    # i = 0
    # while os.path.isfile("{}/optimized{}.xyz".format(input_dir_name, i)):
    for filename in sorted(glob.glob("{}/optimized?.xyz".format(input_dir_name))):
        # conf = Chem.MolFromXYZFile("{}/optimized{}.xyz".format(input_dir_name, i)).GetConformer(-1)
        conf = Chem.MolFromXYZFile(filename).GetConformer(-1)
        # with open("{}/optimized{}.xyz".format(input_dir_name, i), "r") as f:
        with open(filename, "r") as f:
            energy = f.read().split("\n")[1]
            conf.SetProp("energy", energy)
        mol.AddConformer(conf, assignId=True)
        # i += 1


def gaussianfrequency(input_dir_name, output_dir_name, level="hf/sto-3g"):
    calc_condition = "# freq " + level
    charge_and_mult = '0 1'
    i = 0
    while os.path.isfile("{}/optimized{}.xyz".format(input_dir_name, i)):
        os.makedirs(output_dir_name, exist_ok=True)
        with open("{}/optimized{}.xyz".format(input_dir_name, i), "r") as f:
            ans = "\n".join(f.read().split("\n")[2:])
        with open("{}/gaussianinput{}.gjf".format(output_dir_name, i), 'w') as f:
            print("%nprocshared=10", file=f)
            print("%mem=12GB", file=f)
            print('%chk= {}.chk'.format(i), file=f)  # smiles
            print(calc_condition, file=f)

            print('', file=f)
            print('good luck!', file=f)
            print('', file=f)
            # print(head,file=f)
            print(charge_and_mult, file=f)

            print(ans, file=f)
        i += 1


def run_gaussian(filenames):
    """
    Gaussianを使用して指定されたファイルを計算する関数
    """

    for filename in glob.glob(filenames + "/gaussianinput?.gjf"):
        filename=filename.replace("\\","/")
        try:
            cmd = "source ~/.bash_profile ; g16 {}".format(filename)
            print(cmd)
            subprocess.call(cmd, shell=True)
            print(f'{filename} の計算が完了しました。')

            # outファイルを開くコマンドを実行
            log_file = filename.replace('.gjf', '.log')
            with open(log_file, "r") as file:
                lines = file.readlines()
                last_line = lines[-1].strip()
                if last_line.startswith("Normal termination"):
                    print("正常終了")
                else:
                    print("エラー")

        except subprocess.CalledProcessError as e:
            print(f'{filename} の計算中にエラーが発生しました: {e}')


if __name__ == '__main__':
    print("!!")
    #param_file_name = "./parameter/structural optimization/structural optimization.txt"
    param_file_name = "../parameter/structural optimization/structural_optimization.json"# "../parameter/parameter_cbs.txt"
    with open(param_file_name, "r") as f:
        param = json.loads(f.read())
    print(param)
    dfs = []
    # for path in glob.glob("../arranged_dataset/*.xlsx"):
    for path in glob.glob("../arranged_dataset/newrea/newrea.xlsx"):
        df = pd.read_excel(path)
        dfs.append(df)
    print(dfs)

    df = pd.concat(dfs).dropna(subset=['smiles']).drop_duplicates(subset=["smiles"])

    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=['mol'])
    df["molwt"] = df["smiles"].apply(lambda smiles: ExactMolWt(Chem.MolFromSmiles(smiles)))
    df = df.sort_values("molwt")  # [:2]
    print(df)
    for smiles in df["smiles"]:
        print(smiles)
        mol = get_mol(smiles)
        # if True or smiles != "C1CCCCC1C#CC(=O)C(C)(C)C":
        MMFF_out_dirs_name = param["MMFF_out_dir_name"] + "/" + mol.GetProp("InChIKey")
        psi4_out_dirs_name = param["psi4_out_dir_name"] + "/" + mol.GetProp(
            "InChIKey")  # "/"+param["optimize_level"] +
        psi4_aligned_dirs_name = param["psi4_aligned_dir_name"] + "/" + mol.GetProp(
            "InChIKey")  # "/" +param["optimize_level"]+
        psi4_out_dirs_name_freq = param["psi4_aligned_dir_name"] + "_freq" + "/" + mol.GetProp("InChIKey")

        if not os.path.isdir(MMFF_out_dirs_name):
            CalcConfsEnergies(mol, "MMFF")
            highenergycut(mol, param["cut_MMFF_energy"])
            rmsdcut(mol, param["cut_MMFF_rmsd"])
            delconformer(mol, param["max_conformer"])
            ConfTransform(mol)
            conf_to_xyz(mol, MMFF_out_dirs_name)

        if not os.path.isdir(psi4_aligned_dirs_name):
            psi4optimization(MMFF_out_dirs_name, psi4_out_dirs_name, param["optimize_level"])
            read_xyz(mol, psi4_out_dirs_name)
            highenergycut(mol, param["cut_psi4_energy"])
            rmsdcut(mol, param["cut_psi4_rmsd"])
            ConfTransform(mol)
            conf_to_xyz(mol, psi4_aligned_dirs_name)

        if False and not os.path.isdir(psi4_out_dirs_name_freq):
            gaussianfrequency(psi4_aligned_dirs_name, psi4_out_dirs_name_freq + "_calculating",
                              param["optimize_level"])
            run_gaussian(psi4_out_dirs_name_freq + "_calculating")
            os.rename(psi4_out_dirs_name_freq + "_calculating", psi4_out_dirs_name_freq)

        if not os.path.isfile(psi4_aligned_dirs_name + "/optimized0.xyz"):
            # try:
            MMFF_out_dirs_name = param["MMFF_out_dir_name"] + "/" + mol.GetProp("InChIKey") + "UFF"
            psi4_out_dirs_name = param["psi4_out_dir_name"] + "/" + mol.GetProp(
                "InChIKey") + "UFF"  # "/"+param["optimize_level"] +
            psi4_aligned_dirs_name = param["psi4_aligned_dir_name"] + "/" + mol.GetProp(
                "InChIKey") + "UFF"  # "/" +param["optimize_level"]+
            if not os.path.isdir(MMFF_out_dirs_name):
                CalcConfsEnergies(mol, "UFF")
                highenergycut(mol, param["cut_MMFF_energy"])
                rmsdcut(mol, param["cut_MMFF_rmsd"])
                delconformer(mol, param["max_conformer"])
                ConfTransform(mol)
                conf_to_xyz(mol, MMFF_out_dirs_name)

            if not os.path.isdir(psi4_aligned_dirs_name):
                psi4optimization(MMFF_out_dirs_name, psi4_out_dirs_name, param["optimize_level"])
                read_xyz(mol, psi4_out_dirs_name)
                highenergycut(mol, param["cut_psi4_energy"])
                rmsdcut(mol, param["cut_psi4_rmsd"])
                ConfTransform(mol)
                conf_to_xyz(mol, psi4_aligned_dirs_name)
