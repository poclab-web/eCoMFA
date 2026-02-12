"""Build curated molecular datasets from raw Excel files for CoMFA workflows."""

import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem.Descriptors import ExactMolWt
from sklearn.model_selection import train_test_split


def common(from_file_path):
    """
    Load one raw dataset file and standardize the core molecular fields.

    The function canonicalizes SMILES, creates RDKit molecules and InChIKeys,
    removes invalid records, applies simple chemistry filters, and computes
    experimental `ΔΔG.expt.` from `er.` and temperature.
    """
    df = pd.read_excel(from_file_path, engine="openpyxl").dropna(subset="SMILES")
    df["SMILES"] = df["SMILES"].apply(
        lambda smiles: Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    )
    df["mol"] = df["SMILES"].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=["er.", "mol", "SMILES"])
    df["InChIKey"] = df["mol"].apply(lambda mol: Chem.inchi.MolToInchiKey(Chem.AddHs(mol)))
    df["er."] = df["er."].apply(lambda x: np.clip(x, 0.25, 99.75))
    df = df[df["mol"].map(lambda mol: not mol.HasSubstructMatch(Chem.MolFromSmarts("[I]")))]
    df["ΔΔG.expt."] = 1.99 * 10 ** -3 * df["temperature"] * np.log(100 / df["er."].values - 1)
    return df


def output(df, to_file_path):
    """
    Build train/test labels, assign reaction-class labels, and export an Excel dataset.

    This function adds substructure-based class labels, applies the current
    train/test split policy, prints class counts, and writes an Excel file
    including rendered molecule images.
    """
    print(len(df))
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
    train_df["test"] = 0
    test_df["test"] = 1
    df = pd.concat([train_df, test_df])
    df = df.drop_duplicates(subset="InChIKey")
    PandasTools.AddMoleculeColumnToFrame(df, "SMILES")
    df["aliphatic_aliphatic"] = df["ROMol"].map(
        lambda mol: mol.HasSubstructMatch(Chem.MolFromSmarts("CC(=O)C"))
    )
    df["aliphatic_aromatic"] = df["ROMol"].map(
        lambda mol: mol.HasSubstructMatch(Chem.MolFromSmarts("CC(=O)c"))
    )
    df["aromatic_aromatic"] = df["ROMol"].map(
        lambda mol: mol.HasSubstructMatch(Chem.MolFromSmarts("cC(=O)c"))
    )
    df["aldehyde"] = df["ROMol"].map(
        lambda mol: mol.HasSubstructMatch(Chem.MolFromSmarts("[2H]C(=O)[#6]"))
    )
    df["ring"] = df["ROMol"].map(
        lambda mol: mol.HasSubstructMatch(Chem.MolFromSmarts("[#6][C;R](=O)[#6]"))
    )
    # Initialize all labels with NaN.
    df["label"] = np.nan

    # Assign labels in priority order.
    df.loc[df["aliphatic_aliphatic"] & ~df["ring"], "label"] = 1
    df.loc[df["aliphatic_aromatic"] & ~df["ring"], "label"] = 2
    df.loc[df["aromatic_aromatic"] & ~df["ring"], "label"] = 3
    df.loc[df["ring"], "label"] = 4
    df.loc[df["aldehyde"], "label"] = 5

    # train_df, test_df = df[df["label"] != 4], df[df["label"] == 4]
    train_df = df[~df["label"].isin([3, 4])]
    test_df = df[df["label"].isin([3, 4])]

    train_df["test"] = 0
    test_df["test"] = 1
    # df = pd.concat([train_df, test_df])

    print("aliphatic_aliphatic aliphatic_aromatic aromatic_aromatic ring")

    print(
        len(df[df["aliphatic_aliphatic"] & ~df["ring"] & ~df["test"]]),
        len(df[df["aliphatic_aliphatic"] & ~df["ring"] & df["test"]]),
        len(df[df["aliphatic_aromatic"] & ~df["ring"] & ~df["test"]]),
        len(df[df["aliphatic_aromatic"] & ~df["ring"] & df["test"]]),
        len(df[df["aromatic_aromatic"] & ~df["ring"] & ~df["test"]]),
        len(df[df["aromatic_aromatic"] & ~df["ring"] & df["test"]]),
        len(df[df["ring"] & ~df["test"]]),
        len(df[df["ring"] & df["test"]]),
        len(df[df["aldehyde"] & ~df["test"]]),
        len(df[df["aldehyde"] & df["test"]]),
        len(df),
        len(df[df["test"] == 0]),
        len(df[df["test"] == 1]),
    )
    PandasTools.AddMoleculeColumnToFrame(df, "SMILES")
    df = df[
        [
            "entry",
            "SMILES",
            "ROMol",
            "InChIKey",
            "temperature",
            "er.",
            "ΔΔG.expt.",
            "citation",
            "test",
            "label",
        ]
    ]
    PandasTools.SaveXlsxFromFrame(df, to_file_path, size=(100, 100))


if __name__ == "__main__":
    df_cbs = common("sampledata_local/CBS.xlsx")
    df_dip = common("sampledata_local/DIP.xlsx")
    df_ru = common("sampledata_local/alpine_borane.xlsx")

    df_cbs = df_cbs[
        df_cbs["mol"].map(lambda mol: not mol.HasSubstructMatch(Chem.MolFromSmarts("n")))
    ]
    df_dip = df_dip[
        df_dip["mol"].map(
            lambda mol: not mol.HasSubstructMatch(Chem.MolFromSmarts("[Li]"))
            and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6][N,OH1]"))
            and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6]*[N,OH1]"))
            and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6]**[N,OH1]"))
        )
    ]

    to_dir_path = "dataset"

    os.makedirs(to_dir_path, exist_ok=True)
    output(df_cbs, f"{to_dir_path}/CBS.xlsx")
    output(df_dip, f"{to_dir_path}/DIP.xlsx")
    output(df_ru, f"{to_dir_path}/alpine_borane.xlsx")
    df_all = pd.concat([df_cbs, df_dip, df_ru])[["InChIKey", "SMILES"]]
    print(len(df_all))
    df_all = df_all.drop_duplicates(subset=["InChIKey"])
    print(len(df_all))
    df_all["molwt"] = df_all["SMILES"].apply(lambda smiles: ExactMolWt(Chem.MolFromSmiles(smiles)))
    df_all = df_all.sort_values("molwt")
    PandasTools.AddMoleculeColumnToFrame(df_all, "SMILES")
    PandasTools.SaveXlsxFromFrame(df_all, f"{to_dir_path}/mol_list.xlsx", size=(100, 100))
