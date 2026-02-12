"""Generate conformers, run quantum calculations, and save outputs for CoMFA."""

import glob
import os
import subprocess

import cclib
import numpy as np
import pandas as pd
import psi4
from rdkit import Chem
from rdkit.Chem import AllChem

def _default_memory_gb():
    """Return half of total system RAM in GB (integer), used as default Psi4/Gaussian memory."""
    page_size = os.sysconf("SC_PAGE_SIZE")
    phys_pages = os.sysconf("SC_PHYS_PAGES")
    total_gb = (page_size * phys_pages) / (1024 ** 3)
    return int(total_gb / 2)


AUTO_NUM_THREADS = os.cpu_count() or 1
AUTO_MEMORY_GB = _default_memory_gb()

# You can set arbitrary values here if needed.
NUM_THREADS = AUTO_NUM_THREADS
MEMORY_GB = AUTO_MEMORY_GB
MEMORY_STR = f"{MEMORY_GB}GB"
OUTPUT_ROOT = os.path.join(os.path.expanduser("~"), "CoMFA_calc")

psi4.set_num_threads(nthread=NUM_THREADS)
psi4.set_memory(MEMORY_STR)


def energy_cut(mol, res, max_energy):
    """Remove unconverged/high-energy conformers and store relative MMFF energies."""
    res = np.array(res)
    res[:, 1] -= res[:, 1].min()

    remove_ids = []
    for conf, res_ in zip(mol.GetConformers(), res):
        not_converged, energy = res_
        if not_converged or energy > max_energy:
            remove_ids.append(conf.GetId())
            continue
        conf.SetProp("energy", str(energy))

    for conf_id in remove_ids:
        mol.RemoveConformer(conf_id)


def conformer_cut(mol, min_rmse, max_num_conformer):
    """
    Keep up to `max_num_conformer` conformers in ascending energy order while
    removing geometrically redundant conformers based on an RMSD threshold.

    Args:
        mol (rdkit.Chem.Mol): Molecule containing conformers.
        min_rmse (float): Minimum RMSD required to keep two conformers distinct.
        max_num_conformer (int): Maximum number of conformers to retain.
    """
    new_mol = Chem.RemoveHs(mol)

    # Sort conformers by energy in ascending order.
    conformer_list = sorted(
        [(float(conf.GetProp("energy")), conf.GetId()) for conf in mol.GetConformers()],
        key=lambda x: x[0],
    )
    selected_ids = []

    for _, conf1_id in conformer_list:
        if len(selected_ids) >= max_num_conformer:
            break

        keep = True
        for conf2_id in selected_ids:
            rmsd = AllChem.GetBestRMS(new_mol, new_mol, conf1_id, conf2_id)
            if rmsd < min_rmse:
                keep = False
                break

        if keep:
            selected_ids.append(conf1_id)

    # Remove all conformers that were not selected.
    conf_ids = [conf.GetId() for conf in mol.GetConformers()]
    for conf_id in conf_ids:
        if conf_id not in selected_ids:
            mol.RemoveConformer(conf_id)


def Rodrigues_rotation(n, sin, cos):
    """
    Compute a 3D Rodrigues rotation matrix from axis direction and angle components.

    Args:
        n (numpy.ndarray): A 3-element array representing the unit vector of the rotation axis.
        sin (float): The sine of the rotation angle.
        cos (float): The cosine of the rotation angle.

    Returns:
        numpy.ndarray: A 3x3 rotation matrix as a NumPy array.

    Notes:
        This is used to align conformer coordinates to a common orientation.
    """
    ans = np.array(
        [
            [
                n[0] ** 2 * (1 - cos) + cos,
                n[0] * n[1] * (1 - cos) - n[2] * sin,
                n[0] * n[2] * (1 - cos) + n[1] * sin,
            ],
            [
                n[0] * n[1] * (1 - cos) + n[2] * sin,
                n[1] ** 2 * (1 - cos) + cos,
                n[1] * n[2] * (1 - cos) - n[0] * sin,
            ],
            [
                n[0] * n[2] * (1 - cos) - n[1] * sin,
                n[1] * n[2] * (1 - cos) + n[0] * sin,
                n[2] ** 2 * (1 - cos) + cos,
            ],
        ]
    )
    return ans


def transform(conf, carbonyl_atom):
    """
    Align a conformer to a fixed carbonyl-based coordinate system.

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
        numpy.ndarray: Rotated coordinates with consistent orientation across molecules.
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


def calc(out_path, smiles):
    """
    Run the full conformer-to-quantum workflow for one molecule.

    Args:
        out_path (str): The directory path to store calculation outputs, including Gaussian input files,
                        log files, and Psi4 output files.
        smiles (str): The SMILES string representation of the molecule to be processed.

    Returns:
        None. Files are written under `out_path`, and `done` is created on success.

    Summary:
        1. Build/optimize conformers from SMILES.
        2. Keep only low-energy, diverse conformers.
        3. Run Gaussian optimization/frequency calculations.
        4. Run Psi4 single-point and cube property calculations.
        5. Write renamed cube/geometry outputs for downstream feature generation.
    """
    done_path = os.path.join(out_path, "done")
    if os.path.isfile(done_path):
        print(f"SKIP CALCULATION (done exists): {out_path}")
        return

    try:
        os.makedirs(out_path, exist_ok=True)
    except Exception as e:
        print(e)
        return

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    Chem.AssignStereochemistry(
        mol,
        cleanIt=True,
        force=True,
        flagPossibleStereoCenters=True,
    )

    substruct = Chem.MolFromSmarts("[#6](=[#8])([#6])([#6,#1])")
    substruct = mol.GetSubstructMatch(substruct)
    print(substruct)
    mol.GetAtoms()[0].GetProp("_CIPRank")
    if int(mol.GetAtomWithIdx(substruct[2]).GetProp("_CIPRank")) < int(
        mol.GetAtomWithIdx(substruct[3]).GetProp("_CIPRank")
    ):
        substruct = (substruct[0], substruct[1], substruct[3], substruct[2])

    # pruneRmsThresh=0.1
    AllChem.EmbedMultipleConfs(mol, numConfs=10000, randomSeed=1, numThreads=0)
    res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=1000, numThreads=0)

    print(len(mol.GetConformers()))
    energy_cut(mol, res, 2)
    print(len(mol.GetConformers()))
    conformer_cut(mol, min_rmse=0.5, max_num_conformer=5)
    print(len(mol.GetConformers()))

    calc_failed = False
    for _, conf in enumerate(mol.GetConformers()):
        gjf = f"{out_path}/opt{_}.gjf"
        with open(gjf, "w") as f:
            xyz = "\n".join(Chem.rdmolfiles.MolToXYZBlock(mol, confId=conf.GetId()).split("\n")[2:])
            input_text = (
                f"%nprocshared={NUM_THREADS}\n"
                f"%mem={MEMORY_STR}\n"
                f"%chk= {_}.chk\n"
                f"# freq opt=tight b3lyp 6-31g(d)\n\n"
                f"good luck!\n\n"
                f"0 1\n"
                f"{xyz}"
            )
            print(input_text, file=f)

        try:
            subprocess.call(f"source ~/.bash_profile ; g16 {gjf}", shell=True)
            print(f"FINISH CALCULATION {gjf}")

            log = gjf.replace(".gjf", ".log")
            data = cclib.io.ccread(log)
            coords = data.atomcoords[-1]
            coords = transform(coords, substruct)
            nos = data.atomnos

            input_text = "0 1\n nocom\n noreorient\n "
            for no, coord in zip(nos, coords):
                input_text += f"{no} {coord[0]} {coord[1]} {coord[2]}\n"

            psi4.set_output_file(f"{out_path}/sp{_}.log")
            molecule = psi4.geometry(input_text)
            energy, wfn = psi4.energy("wB97X-D/def2-TZVP", molecule=molecule, return_wfn=True)
            psi4.set_options({"cubeprop_filepath": out_path})
            psi4.set_options(
                {
                    "cubeprop_tasks": ["esp", "orbitals"],
                    "cubeprop_orbitals": [wfn.nalpha() + 1, wfn.nalpha() + 2],
                    "cubic_grid_spacing": [0.2, 0.2, 0.2],
                    "cubic_grid_overage": [8, 8, 8],
                }
            )
            psi4.cubeprop(wfn)
            os.rename(f"{out_path}/geom.xyz", f"{out_path}/geom{_}.xyz")
            os.rename(f"{out_path}/Dt.cube", f"{out_path}/Dt{_}.cube")
            os.rename(f"{out_path}/ESP.cube", f"{out_path}/ESP{_}.cube")
            os.rename(
                glob.glob(f"{out_path}/Psi_a_{wfn.nalpha() + 1}_*.cube")[0],
                f"{out_path}/LUMO{_}.cube",
            )
            os.rename(
                glob.glob(f"{out_path}/Psi_a_{wfn.nalpha() + 2}_*.cube")[0],
                f"{out_path}/LUMO+1_{_}.cube",
            )
        except Exception as e:
            print(e)
            calc_failed = True

    if not calc_failed:
        with open(done_path, "w", encoding="utf-8"):
            pass


if __name__ == "__main__":
    df = pd.read_excel("dataset/mol_list.xlsx")
    df[["InChIKey", "SMILES"]].apply(lambda _: calc(f"{OUTPUT_ROOT}/{_[0]}", _[1]), axis=1)
