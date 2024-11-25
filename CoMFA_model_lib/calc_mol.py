import os
import subprocess
import numpy as np
import pandas as pd
import psi4
from rdkit import Chem
from rdkit.Chem import AllChem
import heapq
import cclib

psi4.set_num_threads(nthread=30)
psi4.set_memory("30GB")

def energy_cut(mol, energy):
    l = []
    for conf in mol.GetConformers():
        if float(conf.GetProp("energy")) - min([float(conf.GetProp("energy")) for conf in mol.GetConformers()]) >= \
                energy:
            l.append(conf.GetId())
    for i in l:
        mol.RemoveConformer(i)

def rmsd_cut(mol, rmsd_):
    l = sorted([[float(conf.GetProp("energy")), conf.GetId()] for conf in mol.GetConformers()], reverse=True)
    for i, (_, conf1) in enumerate(l):
        for _, conf2 in l[1 + i:]:
            rmsd = AllChem.GetBestRMS(Chem.rdmolops.RemoveHs(mol), Chem.rdmolops.RemoveHs(mol), conf1, conf2)
            if rmsd < rmsd_:
                mol.RemoveConformer(conf1)
                break
            
def max_n_cut(mol, max_n):
    l = [float(conf.GetProp("energy")) for conf in mol.GetConformers()]
    dellist = [conf.GetId() for conf in mol.GetConformers() if
               float(conf.GetProp("energy")) not in heapq.nsmallest(max_n, l)]
    for id in dellist:
        mol.RemoveConformer(id)

def log_to_xyz(log):
    table = {
        1: "H",    2: "He",   3: "Li",   4: "Be",   5: "B",    6: "C",    7: "N",    8: "O",    9: "F",   10: "Ne",
    11: "Na",  12: "Mg",  13: "Al",  14: "Si",  15: "P",   16: "S",   17: "Cl",  18: "Ar",  19: "K",   20: "Ca",
    21: "Sc",  22: "Ti",  23: "V",   24: "Cr",  25: "Mn",  26: "Fe",  27: "Co",  28: "Ni",  29: "Cu",  30: "Zn",
    31: "Ga",  32: "Ge",  33: "As",  34: "Se",  35: "Br",  36: "Kr",  37: "Rb",  38: "Sr",  39: "Y",   40: "Zr",
    41: "Nb",  42: "Mo",  43: "Tc",  44: "Ru",  45: "Rh",  46: "Pd",  47: "Ag",  48: "Cd",  49: "In",  50: "Sn",
    51: "Sb",  52: "Te",  53: "I",   54: "Xe",  55: "Cs",  56: "Ba",  57: "La",  58: "Ce",  59: "Pr",  60: "Nd",
    61: "Pm",  62: "Sm",  63: "Eu",  64: "Gd",  65: "Tb",  66: "Dy",  67: "Ho",  68: "Er",  69: "Tm",  70: "Yb",
    71: "Lu",  72: "Hf",  73: "Ta",  74: "W",   75: "Re",  76: "Os",  77: "Ir",  78: "Pt",  79: "Au",  80: "Hg",
    81: "Tl",  82: "Pb",  83: "Bi",  84: "Po",  85: "At",  86: "Rn",  87: "Fr",  88: "Ra",  89: "Ac",  90: "Th",
    91: "Pa",  92: "U",   93: "Np",  94: "Pu",  95: "Am",  96: "Cm",  97: "Bk",  98: "Cf",  99: "Es", 100: "Fm",
    101: "Md", 102: "No", 103: "Lr", 104: "Rf", 105: "Db", 106: "Sg", 107: "Bh", 108: "Hs", 109: "Mt", 110: "Ds",
    111: "Rg", 112: "Cn", 113: "Nh", 114: "Fl", 115: "Mc", 116: "Lv", 117: "Ts", 118: "Og"
    }
    # Gaussianのlogファイルを読み込む
    data = cclib.io.ccread(log)

    # 最適化された原子座標の抽出
    atomnos = data.atomnos      # 原子番号のリスト
    atomcoords = data.atomcoords[-1]  # 最終構造の座標

    # XYZフォーマットに変換して保存
    xyz=log.replace('.log','.xyz')
    with open(xyz, "w") as f:
        f.write(f"{len(atomnos)}\n\n")  # 原子数と空行
        for num, coord in zip(atomnos, atomcoords):
            element = table[num]   # 原子番号から元素記号へ変換
            f.write(f"{element} {coord[0]} {coord[1]} {coord[2]}\n")
def Rodrigues_rotation(n, sin, cos):
    ans = np.array(
        [[n[0] ** 2 * (1 - cos) + cos, n[0] * n[1] * (1 - cos) - n[2] * sin, n[0] * n[2] * (1 - cos) + n[1] * sin],
         [n[0] * n[1] * (1 - cos) + n[2] * sin, n[1] ** 2 * (1 - cos) + cos, n[1] * n[2] * (1 - cos) - n[0] * sin],
         [n[0] * n[2] * (1 - cos) - n[1] * sin, n[1] * n[2] * (1 - cos) + n[0] * sin, n[2] ** 2 * (1 - cos) + cos]
         ])
    return ans

def transform(conf, carbonyl_atom):
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
    cos2 = np.dot(byz, np.array([0, 0, 1]))
    cros2 = np.cross(byz, np.array([0, 0, 1]))
    sin2 = np.linalg.norm(cros2)
    n2 = cros2 / sin2
    conf = np.dot(Rodrigues_rotation(n1, sin1, cos1), conf.T).T
    conf = np.dot(Rodrigues_rotation(n2, sin2, cos2), conf.T).T
    return conf


def calc(out_path,smiles):
    try:
        os.makedirs(out_path)
    except Exception as e:
        print(e)
        return
    mol=Chem.MolFromSmiles(smiles)
    mol=Chem.AddHs(mol)
    Chem.AssignStereochemistry(mol,cleanIt=True,force=True,flagPossibleStereoCenters=True)

    substruct=Chem.MolFromSmarts("[#6](=[#8])([#6])([#6])")
    substruct=mol.GetSubstructMatch(substruct)
    mol.GetAtoms()[0].GetProp('_CIPRank')
    if int(mol.GetAtomWithIdx(substruct[2]).GetProp('_CIPRank'))<int(mol.GetAtomWithIdx(substruct[3]).GetProp('_CIPRank')):
        substruct=(substruct[0],substruct[1],substruct[3],substruct[2])

    AllChem.EmbedMultipleConfs(mol, numConfs=10000, randomSeed=1, pruneRmsThresh=0.1, numThreads=0)
    for conf in mol.GetConformers():
        ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol),confId=conf.GetId())
        ff.Minimize()
        energy = ff.CalcEnergy()
        conf.SetProp("energy", str(energy))

    energy_cut(mol,2)
    rmsd_cut(mol,0.5)
    max_n_cut(mol,5)

    for _,conf in enumerate(mol.GetConformers()):
        gjf=f'{out_path}/opt{_}.gjf'
        with open(gjf, 'w')as f:
            xyz="\n".join(Chem.rdmolfiles.MolToXYZBlock(mol,confId=conf.GetId()).split("\n")[2:])
            input=f'%nprocshared=30\n%mem=30GB\n%chk= {_}.chk\n# freq opt=tight b3lyp 6-31g(d)\n\ngood luck!\n\n0 1\n{xyz}'
            print(input,file=f)
        try:
            subprocess.call(f'source ~/.bash_profile ; g16 {gjf}', shell=True)
            print(f'FINISH CALCULATION {gjf}')
            log=gjf.replace('.gjf', '.log')
            data = cclib.io.ccread(log)
            coords = data.atomcoords[-1]  # 最終構造の座標
            coords=transform(coords, substruct)
            nos = data.atomnos

            input = "0 1\n nocom\n noreorient\n "
            for no,coord in zip(nos,coords):
                input+=f"{no} {coord[0]} {coord[1]} {coord[2]}\n"
            psi4.set_output_file(f'{out_path}/sp{_}.log')
            molecule = psi4.geometry(input)
            energy, wfn = psi4.energy("wB97X-D/def2-TZVP", molecule=molecule, return_wfn=True)
            psi4.set_options({'cubeprop_filepath': out_path})
            psi4.set_options({'cubeprop_tasks': ["esp"],
                          "cubic_grid_spacing": [0.2, 0.2, 0.2],
                          "cubic_grid_overage": [8, 8, 8]
                          })
            psi4.cubeprop(wfn)
            os.rename(f'{out_path}/geom.xyz', f'{out_path}/geom{_}.xyz')
            os.rename(f'{out_path}/Dt.cube', f'{out_path}/Dt{_}.cube')
            os.rename(f'{out_path}/ESP.cube', f'{out_path}/ESP{_}.cube')
        except Exception as e:
            print(e)

if __name__ == '__main__':
    out_path='/Volumes/SSD-PSM960U3-UW/CoMFA_calc'
    df=pd.read_excel("/Users/mac_poclab/PycharmProjects/CoMFA_model/arranged_dataset/mol_list.xlsx")
    df[["InChIKey","SMILES"]].apply(lambda _:calc(f'{out_path}/{_[0]}',_[1]),axis=1)