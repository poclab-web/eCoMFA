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
    """
    Removes conformers from a molecular object that exceed a specified energy threshold.

    This function iterates over the conformers in a molecular object, identifies those whose energy
    difference from the lowest-energy conformer exceeds a given threshold, and removes them from the molecule.

    Args:
        mol (rdkit.Chem.rdchem.Mol): A molecule object containing conformers. Each conformer is expected
                                     to have an "energy" property.
        energy (float): The energy threshold. Conformers with energy differences greater than this
                        value (compared to the lowest-energy conformer) will be removed.

    Returns:
        None: The function modifies the input molecule `mol` in place by removing conformers.
    """
    l = []
    for conf in mol.GetConformers():
        if float(conf.GetProp("energy")) - min([float(conf.GetProp("energy")) for conf in mol.GetConformers()]) >= \
                energy:
            l.append(conf.GetId())
    for i in l:
        mol.RemoveConformer(i)

def rmsd_cut(mol, rmsd_):
    """
    Removes conformers from a molecular object based on the Root Mean Square Deviation (RMSD) threshold.

    This function calculates the RMSD between pairs of conformers in the molecule. If the RMSD between
    two conformers is below a specified threshold, the conformer with the higher energy is removed.
    Conformers are processed in descending order of energy.

    Args:
        mol (rdkit.Chem.rdchem.Mol): A molecule object containing conformers. Each conformer is expected
                                     to have an "energy" property, and RMSD calculations are performed
                                     using 3D coordinates.
        rmsd_ (float): The RMSD threshold. Conformers with RMSD values below this threshold are considered
                       redundant, and the higher-energy conformer is removed.

    Returns:
        None: The function modifies the input molecule `mol` in place by removing conformers.
    """
    l = sorted([[float(conf.GetProp("energy")), conf.GetId()] for conf in mol.GetConformers()], reverse=True)
    for i, (_, conf1) in enumerate(l):
        for _, conf2 in l[1 + i:]:
            rmsd = AllChem.GetBestRMS(Chem.rdmolops.RemoveHs(mol), Chem.rdmolops.RemoveHs(mol), conf1, conf2)
            if rmsd < rmsd_:
                mol.RemoveConformer(conf1)
                break
            
def max_n_cut(mol, max_n):
    """
    Retains only the `max_n` lowest-energy conformers in a molecular object and removes the rest.

    This function evaluates the energies of all conformers in a molecule, retains the `max_n` conformers 
    with the lowest energies, and removes all other conformers. Conformers are selected based on their 
    energy properties.

    Args:
        mol (rdkit.Chem.rdchem.Mol): A molecule object containing conformers. Each conformer is expected 
                                     to have an "energy" property.
        max_n (int): The maximum number of conformers to retain. Conformers are ranked by energy, 
                     and only the `max_n` lowest-energy conformers are kept.

    Returns:
        None: The function modifies the input molecule `mol` in place by removing conformers.
    """
    l = [float(conf.GetProp("energy")) for conf in mol.GetConformers()]
    dellist = [conf.GetId() for conf in mol.GetConformers() if
               float(conf.GetProp("energy")) not in heapq.nsmallest(max_n, l)]
    for id in dellist:
        mol.RemoveConformer(id)

def log_to_xyz(log):
    """
    Converts a computational chemistry log file into an XYZ format file.

    This function reads a log file generated by quantum chemistry calculations (e.g., Gaussian),
    extracts atomic numbers and coordinates from the last geometry step, and writes them into
    an XYZ format file. The function uses the cclib library to parse the log file and a lookup
    table to map atomic numbers to element symbols.

    Args:
        log (str): Path to the log file to be converted. The file must be readable by the cclib library
                   and should contain atomic numbers (`atomnos`) and coordinates (`atomcoords`).

    Returns:
        None: The function writes an XYZ format file to disk, replacing the `.log` extension in the
              input filename with `.xyz`.
    """
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
    data = cclib.io.ccread(log)

    atomnos = data.atomnos
    atomcoords = data.atomcoords[-1]

    xyz=log.replace('.log','.xyz')
    with open(xyz, "w") as f:
        f.write(f"{len(atomnos)}\n\n")
        for num, coord in zip(atomnos, atomcoords):
            element = table[num]
            f.write(f"{element} {coord[0]} {coord[1]} {coord[2]}\n")

def Rodrigues_rotation(n, sin, cos):
    """
    Computes the Rodrigues' rotation matrix for a given axis and rotation angle.

    This function constructs a 3x3 rotation matrix based on the Rodrigues' rotation formula.
    The formula is used to perform a rotation around a given axis in 3D space, specified by a unit vector `n`.
    The sine (`sin`) and cosine (`cos`) of the rotation angle are used directly as inputs.

    Args:
        n (numpy.ndarray): A 3-element array representing the unit vector of the rotation axis.
        sin (float): The sine of the rotation angle.
        cos (float): The cosine of the rotation angle.

    Returns:
        numpy.ndarray: A 3x3 rotation matrix as a NumPy array.

    Formula:
        R = I * cos(θ) + (1 - cos(θ)) * (n ⊗ n) + [n]_x * sin(θ)

        Where:
        - `I` is the identity matrix.
        - `n ⊗ n` is the outer product of `n`.
        - `[n]_x` is the skew-symmetric cross-product matrix of `n`.
    """
    ans = np.array(
        [[n[0] ** 2 * (1 - cos) + cos, n[0] * n[1] * (1 - cos) - n[2] * sin, n[0] * n[2] * (1 - cos) + n[1] * sin],
         [n[0] * n[1] * (1 - cos) + n[2] * sin, n[1] ** 2 * (1 - cos) + cos, n[1] * n[2] * (1 - cos) - n[0] * sin],
         [n[0] * n[2] * (1 - cos) - n[1] * sin, n[1] * n[2] * (1 - cos) + n[0] * sin, n[2] ** 2 * (1 - cos) + cos]
         ])
    return ans

def transform(conf, carbonyl_atom):
    """
    Transforms the coordinates of a molecular conformer to align a carbonyl group with specific axes.

    This function takes a conformer's coordinates and reorients the molecule such that:
    - The carbonyl C=O bond is aligned along the x-axis.
    - The plane defined by the carbonyl group and its neighboring atoms is aligned with the xz-plane.

    The transformation involves translating the carbonyl carbon atom to the origin and applying a series
    of Rodrigues' rotations to align the specified geometric features with the desired axes.

    Args:
        conf (numpy.ndarray): A 2D NumPy array of shape (N, 3) where N is the number of atoms,
                              and each row represents the 3D coordinates (x, y, z) of an atom.
        carbonyl_atom (list or tuple): A list/tuple of four integers [c, o, c1, c2], representing the indices
                                       of the atoms in the carbonyl group:
                                       - `c`: Index of the carbon atom in the carbonyl group.
                                       - `o`: Index of the oxygen atom in the carbonyl group.
                                       - `c1`: Index of a neighboring atom bonded to `c`.
                                       - `c2`: Index of another neighboring atom bonded to `c1`.

    Returns:
        numpy.ndarray: The transformed coordinates of the conformer as a 2D NumPy array of shape (N, 3).

    Steps:
        1. Translate the molecule such that the carbonyl carbon atom (`c`) is at the origin.
        2. Align the C=O bond to the x-axis using Rodrigues' rotation.
        3. Align the neighboring atoms to the xz-plane by further rotating the molecule.

    Notes:
        - The function assumes that the input coordinates (`conf`) and the atom indices (`carbonyl_atom`)
          are consistent with the molecular structure.
        - Proper normalization is performed to avoid numerical instability during rotations.
    """
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
    """
    Performs computational chemistry calculations for a molecule generated from its SMILES representation.

    This function takes a SMILES string as input, generates a 3D molecular structure, optimizes the
    conformers, filters the conformers based on energy and RMSD criteria, and generates Gaussian input
    files for further quantum chemical calculations. It also calculates single-point energy and grid
    properties using Psi4 and processes the output data.

    Args:
        out_path (str): The directory path to store calculation outputs, including Gaussian input files,
                        log files, and Psi4 output files.
        smiles (str): The SMILES string representation of the molecule to be processed.

    Returns:
        None: Outputs are saved in the specified directory. No value is returned.

    Workflow:
        1. Create the output directory.
        2. Generate a molecule object from the SMILES string and add hydrogens.
        3. Assign stereochemistry and identify substructures.
        4. Generate conformers and minimize their energies using MMFF force fields.
        5. Filter conformers based on:
            - Energy difference threshold (`energy_cut`).
            - RMSD threshold (`rmsd_cut`).
            - Maximum number of conformers (`max_n_cut`).
        6. Generate Gaussian input files for the remaining conformers and execute calculations.
        7. Use cclib to extract optimized geometries and transform coordinates for Psi4 calculations.
        8. Run single-point energy calculations with Psi4 and generate ESP grid data.
        9. Rename and save output files for further analysis.

    Notes:
        - The Gaussian software (`g16`) and Psi4 must be installed and properly configured in the environment.
        - Requires RDKit for SMILES parsing, substructure matching, and conformer generation.
        - Assumes that the output directory does not already exist; if it does, the function returns early.

    Example:
        calc("/path/to/output", "CC(=O)C(C)C")

    Error Handling:
        - If the output directory cannot be created, an exception is logged, and the function exits.
        - Gaussian and Psi4 calculations are wrapped in try-except blocks to handle potential errors.
    """
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
            coords = data.atomcoords[-1]
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
    out_path='/Users/mac_poclab/CoMFA_calc'#'/Volumes/SSD-PSM960U3-UW/CoMFA_calc'
    df=pd.read_excel("/Users/mac_poclab/PycharmProjects/CoMFA_model/dataset/mol_list.xlsx")
    df[["InChIKey","SMILES"]].apply(lambda _:calc(f'{out_path}/{_[0]}',_[1]),axis=1)