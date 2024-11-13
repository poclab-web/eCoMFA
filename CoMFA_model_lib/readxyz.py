from rdkit import Chem

mol=Chem.MolFromXYZFile("/Users/mac_poclab/Desktop/optfreqspcube/opt_freq.xyz")
print(Chem.MolToMolBlock(mol))