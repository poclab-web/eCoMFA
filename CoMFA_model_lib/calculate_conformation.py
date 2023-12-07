import json
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem import AllChem
import heapq
from rdkit.Geometry import Point3D
import numpy as np
import psi4


def GetCommonStructure(mol, common_structure):
    com = mol.GetSubstructMatches(Chem.MolFromSmarts(common_structure))[0]
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
    mol.GetAtomWithIdx(l[1]).SetProp("alignment", "3")


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    GetCommonStructure(mol, "[#6](=[#8])([c,C])([c,C])")
    mol.SetProp("InchyKey", Chem.MolToInchiKey(mol))
    return mol


def CalcConfsEnergies(mol):
    AllChem.EmbedMultipleConfs(mol, numConfs=param["numConfs"], randomSeed=1, pruneRmsThresh=0.01,
                               numThreads=0)
    for conf in mol.GetConformers():
        mmff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol),
                                                 confId=conf.GetId())  # nomal force field
        # mmff=AllChem.UFFGetMoleculeForceField(mol,confId=cid) # other force field
        mmff.Minimize()
        # energy=psi4_calc.calc_energy(mol, confid=cidId) # calc energy by psi4
        energy = mmff.CalcEnergy()
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
    psi4.set_num_threads(nthread=4)
    psi4.set_memory("4GB")
    psi4.set_options({'geom_maxiter': 10000})
    i = 0
    while os.path.isfile("{}/optimized{}.xyz".format(input_dir_name, i)):
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


def read_xyz(mol, input_dir_name):
    mol.RemoveAllConformers()
    i = 0
    while os.path.isfile("{}/optimized{}.xyz".format(input_dir_name, i)):
        conf = Chem.MolFromXYZFile("{}/optimized{}.xyz".format(input_dir_name, i)).GetConformer(-1)
        with open("{}/optimized{}.xyz".format(input_dir_name, i), "r") as f:
            energy = f.read().split("\n")[1]
            conf.SetProp("energy", energy)
        mol.AddConformer(conf, assignId=True)
        i += 1


if __name__ == '__main__':
    param_file_name = "../parameter/parameter_cbs.txt"
    with open(param_file_name, "r") as f:
        param = json.loads(f.read())
    print(param)
    data_file_path = "../arranged_dataset/cbs.xls"

    df1 = pd.read_excel(data_file_path)
    df2=pd.read_excel("../arranged_dataset/DIP-chloride.xls")
    df3 =pd.read_excel("../arranged_dataset/Russ.xls")
    df = pd.concat([df1, df2,df3]).dropna(subset=['smiles']).drop_duplicates(subset=["smiles"])

    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=['mol'])
    df["molwt"] = df["smiles"].apply(lambda smiles: ExactMolWt(Chem.MolFromSmiles(smiles)))
    df = df.sort_values("molwt")  # [:2]
    print(df)

    MMFF_out_dir_name = "../MMFF_optimization"
    psi4_out_dir_name = "../psi4_optimization"
    psi4_aligned_dir_name = "../psi4_optimization_aligned"
    #df=df[df["smiles"] != "C(=O)(CN(C)c1ccccc1)C"]
    # df = df[df["smiles"] != "C(=O)(C1=CC=CO1)CCCCC"]
    # df = df[df["smiles"] != "C(=O)(c1ccc(F)cc1)CCCN2CCN(C3=NCC(F)C=N3)CC2"]
    #df = df[df["smiles"] != "c1cc(Br)ccc1c2cc(Cl)ccc2C(=O)CCC"]
    #df = df[df["smiles"] != "C(=O)(C(c1ccccc1)(c1ccccc1)c1ccccc1)C"]
    #df = df[df["smiles"] != "c1ccccc1C(C)(C)C(=O)C#CCCCCCCCC"]

    # for smiles in df["smiles"]:
    for smiles in df["smiles"][df["smiles"] == "C(=O)(c1ccc(F)cc1)CCCN2CCN(C3=NCC(F)C=N3)CC2"]:
        print("longsmiles")
        print(smiles)
        mol = get_mol(smiles)
        MMFF_out_dirs_name = MMFF_out_dir_name + "/" + mol.GetProp("InchyKey")
        psi4_out_dirs_name = psi4_out_dir_name + "/"+param["optimize_level"] + "/" + mol.GetProp("InchyKey")
        psi4_aligned_dirs_name = psi4_aligned_dir_name + "/" +param["optimize_level"]+"/" +mol.GetProp("InchyKey")
        if not os.path.isdir(MMFF_out_dirs_name):
            CalcConfsEnergies(mol)
            highenergycut(mol, param["cut_MMFF_energy"])
            rmsdcut(mol, param["cut_MMFF_rmsd"])
            delconformer(mol, param["max_conformer"])
            ConfTransform(mol)
            conf_to_xyz(mol, MMFF_out_dirs_name)
        if not os.path.isdir(psi4_aligned_dirs_name):
            try:
                psi4optimization(MMFF_out_dirs_name, psi4_out_dirs_name , param["optimize_level"])
                read_xyz(mol, psi4_out_dirs_name )
                highenergycut(mol, param["cut_psi4_energy"])
                rmsdcut(mol, param["cut_psi4_rmsd"])
                ConfTransform(mol)
                conf_to_xyz(mol, psi4_aligned_dirs_name)
                #ヨウ素に対する計算を考える。
            except:
                continue
        print("{}smilescomplete".format(smiles))