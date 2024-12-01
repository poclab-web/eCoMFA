import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from sklearn.model_selection import train_test_split
from rdkit.Chem.Descriptors import ExactMolWt

def common(from_file_path):
    """
    Prepares and processes molecular data from an Excel file for further analysis.

    This function reads molecular data from an Excel file, processes the SMILES strings,
    computes additional properties, and filters the dataset based on specified criteria.

    Args:
        from_file_path (str): Path to the input Excel file containing molecular data.
                              The file must have the following columns:
                              - "SMILES": The SMILES string of the molecule.
                              - "er.": Experimental enantiomeric ratio (er.).
                              - "temperature": Experimental temperature in Kelvin.

    Returns:
        pandas.DataFrame: A processed DataFrame with the following added/modified columns:
            - "SMILES": Canonicalized SMILES strings.
            - "mol": RDKit `Mol` objects created from the SMILES strings.
            - "InChIKey": Unique identifier generated from the 3D structure of the molecule.
            - "er.": Experimental enantiomeric ratio, clipped to the range [0.25, 99.75].
            - "ΔΔG.expt.": Experimental free energy difference (kcal/mol), calculated as:
                          `ΔΔG.expt. = RT * ln((100 / er.) - 1)`
                          where `R = 1.99e-3 kcal/(mol·K)`.

    Processing Workflow:
        1. Load the Excel file using `openpyxl` engine.
        2. Canonicalize SMILES strings and convert them to RDKit `Mol` objects.
        3. Drop rows with missing values in critical columns ("er.", "mol", "SMILES").
        4. Filter molecules containing iodine atoms using SMARTS patterns.
        5. Clip enantiomeric ratios (`er.`) to the range [0.25, 99.75].
        6. Calculate `ΔΔG.expt.` based on `er.` and `temperature`.

    Example:
        df = common("/path/to/molecular_data.xlsx")

    Notes:
        - Requires the RDKit library for SMILES and molecule manipulation.
        - Assumes that the input file is well-structured and contains the required columns.

    Raises:
        - `KeyError`: If any required column is missing from the input Excel file.
        - `ValueError`: If invalid SMILES strings or data inconsistencies are encountered.
    """
    df = pd.read_excel(from_file_path, engine="openpyxl")
    df["SMILES"]=df["SMILES"].apply(lambda smiles: Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))
    df["mol"] = df["SMILES"].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=['er.', "mol", "SMILES"])
    df["InChIKey"] = df["mol"].apply(lambda mol: Chem.inchi.MolToInchiKey(Chem.AddHs(mol)))
    df["er."]=df["er."].apply(lambda x:np.clip(x,0.25,99.75))
    df = df[df["mol"].map(lambda mol: not mol.HasSubstructMatch(Chem.MolFromSmarts("[I]")))]
    # df["RT"] = 1.99 * 10 ** -3 * df["temperature"].values
    df["ΔΔG.expt."] = 1.99 * 10 ** -3 * df["temperature"] * np.log(100 / df["er."].values - 1)
    return df

def output(df,to_file_path):
    """
    Processes molecular data, splits it into training and test sets, adds substructure information, 
    and saves the resulting data to an Excel file.

    This function identifies specific molecules by their InChIKeys, assigns them to the test set, 
    splits the remaining data into training and test sets, and appends molecular substructure 
    information. The processed data is saved to an Excel file with embedded molecule images.

    Args:
        df (pandas.DataFrame): Input DataFrame containing molecular data. 
                               It must include the following columns:
                               - "SMILES": Canonical SMILES string of the molecule.
                               - "InChIKey": Unique molecular identifier.
                               - "temperature": Experimental temperature.
                               - "er.": Experimental enantiomeric ratio.
                               - "ΔΔG.expt.": Experimental free energy difference.
                               - "Reference url": Reference link for data source.
        to_file_path (str): Path to the output Excel file (.xlsx).

    Returns:
        None: The processed data is saved to an Excel file with embedded molecule images.

    Workflow:
        1. Identify and separate specific molecules into the test set based on their InChIKeys.
        2. Split the remaining data into training (80%) and test (20%) sets.
        3. Append the test-specific molecules to the test set.
        4. Add substructure information:
           - "aliphatic_aliphatic": Presence of an aliphatic-aliphatic ketone group.
           - "aliphatic_aromatic": Presence of an aliphatic-aromatic ketone group.
           - "aromatic_aromatic": Presence of an aromatic-aromatic ketone group.
           - "ring": Presence of a ketone group within a ring structure.
        5. Save the processed data to an Excel file with molecule images.
        6. Print counts of molecules with different substructure types in training and test sets.

    Example:
        output(df, "/path/to/output.xlsx")
    """
    bool=df['InChIKey'].isin(["KWOLFJPFCHCOCG-UHFFFAOYSA-N","KZJRKRQSDZGHEC-UHFFFAOYSA-N",
                                "GCSHUYKULREZSJ-UHFFFAOYSA-N","RYMBAPVTUHZCNF-UHFFFAOYSA-N","SKFLCXNDKRUHTA-UHFFFAOYSA-N"])
    df_=df[bool]
    train_df, test_df = train_test_split(df[~bool], test_size=0.2, random_state=4)
    df_["test"]=1
    train_df['test'] = 0
    test_df['test'] = 1
    df = pd.concat([train_df, df_,test_df])

    PandasTools.AddMoleculeColumnToFrame(df, "SMILES")
    df = df[["entry","SMILES", "ROMol", "InChIKey", "temperature","er.", "ΔΔG.expt.","Reference url","test"]]
    PandasTools.SaveXlsxFromFrame(df, to_file_path, size=(100, 100))

    df["aliphatic_aliphatic"]=df["ROMol"].map(lambda mol: mol.HasSubstructMatch(Chem.MolFromSmarts("CC(=O)C")))
    df["aliphatic_aromatic"]=df["ROMol"].map(lambda mol: mol.HasSubstructMatch(Chem.MolFromSmarts("CC(=O)c")))
    df["aromatic_aromatic"]=df["ROMol"].map(lambda mol: mol.HasSubstructMatch(Chem.MolFromSmarts("cC(=O)c")))
    df["ring"]=df["ROMol"].map(lambda mol: mol.HasSubstructMatch(Chem.MolFromSmarts("[#6][C;R](=O)[#6]")))
    print(f'aliphatic_aliphatic aliphatic_aromatic aromatic_aromatic ring')
    print(len(df[df["aliphatic_aliphatic"]&~df["ring"]&~df["test"]]),len(df[df["aliphatic_aliphatic"]&~df["ring"]&df["test"]]),
          len(df[df["aliphatic_aromatic"]&~df["ring"]&~df["test"]]),len(df[df["aliphatic_aromatic"]&~df["ring"]&df["test"]]),
          len(df[df["aromatic_aromatic"]&~df["ring"]&~df["test"]]),len(df[df["aromatic_aromatic"]&~df["ring"]&df["test"]]),
          len(df[df["ring"]&~df["test"]]),len(df[df["ring"]&df["test"]]),
          len(df),len(df[df["test"]==0]),len(df[df["test"]==1]))

if __name__ == '__main__':
    
    df_cbs=common("/Users/mac_poclab/PycharmProjects/CoMFA_model/sampledata/CBS.xlsx")#.drop_duplicates(subset="InChIKey")
    df_dip=common("/Users/mac_poclab/PycharmProjects/CoMFA_model/sampledata/DIP.xlsx")#.drop_duplicates(subset="InChIKey")
    df_ru=common("/Users/mac_poclab/PycharmProjects/CoMFA_model/sampledata/Ru.xlsx")#.drop_duplicates(subset="InChIKey")

    df_cbs = df_cbs[df_cbs["mol"].map(lambda mol: not mol.HasSubstructMatch(Chem.MolFromSmarts("n")))]
    df_dip=df = df_dip[df_dip["mol"].map(lambda mol:
                              not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6][#7,OH1]"))
                              and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6]*[#7,OH1]"))
                              and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6]**[#7,OH1]")))]
    
    to_dir_path = "/Users/mac_poclab/PycharmProjects/CoMFA_model/arranged_dataset"

    os.makedirs(to_dir_path, exist_ok=True)
    output(df_cbs,f'{to_dir_path}/CBS.xlsx')
    output(df_dip,f'{to_dir_path}/DIP.xlsx')
    output(df_ru,f'{to_dir_path}/Ru.xlsx')
    df_all=pd.concat([df_cbs,df_dip,df_ru])[["InChIKey","SMILES"]]
    print(len(df_all))
    df_all=df_all.drop_duplicates(subset=["InChIKey"])
    print(len(df_all))
    df_all["molwt"] = df_all["SMILES"].apply(lambda smiles: ExactMolWt(Chem.MolFromSmiles(smiles)))
    df_all=df_all.sort_values("molwt")#.reset_index()#.to_csv(f'{to_dir_path}/mol_list.csv',index=False)
    PandasTools.AddMoleculeColumnToFrame(df_all, "SMILES")
    PandasTools.SaveXlsxFromFrame(df_all, f'{to_dir_path}/mol_list.xlsx', size=(100, 100))
