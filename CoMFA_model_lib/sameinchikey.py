from rdkit import Chem
import pandas as pd
from rdkit.Chem import PandasTools
import xlwt

mol= Chem.MolFromSmiles("C(=O)(C(c1ccccc1)(c1ccccc1)c1ccccc1)C")
inchi=Chem.inchi.MolToInchiKey(mol)
print(inchi)



df1 = pd.read_excel("../arranged_dataset/cbs.xls").dropna(subset=['smiles'])
df2=pd.read_excel("../arranged_dataset/DIP-chloride.xls").dropna(subset=['smiles'])
df3 =pd.read_excel("../arranged_dataset/Russ.xls").dropna(subset=['smiles'])



    # print(df[df.duplicated(subset="inchi")].drop_duplicates(subset="inchi"))
df = pd.concat([df1,df2,df3])
print(df.reset_index(drop=True))
print(df.columns)

print(df[df.duplicated(subset="inchikey")])
df=df[df.duplicated(subset="inchikey")]
df=df["inchikey"]
print(df)
print(df.values.tolist())

df1=df1[df1["inchikey"].isin(df.values.tolist())]
df2=df2[df2["inchikey"].isin(df.values.tolist())]
df3=df3[df3["inchikey"].isin(df.values.tolist())]
print(df1)

df1=df1.rename(columns={'ΔΔG.expt.': 'ΔΔG.expt.cbs'})
df2=df2.rename(columns={'ΔΔG.expt.': 'ΔΔG.expt.dip'})
df3=df3.rename(columns={'ΔΔG.expt.': 'ΔΔG.expt.Ru'})

df4 = pd.concat([df1,df2,df3])
print(df4.columns)
PandasTools.AddMoleculeColumnToFrame(df4, "smiles")
df4 = df4[[ "smiles","ROMol","inchikey","er.",'ΔΔG.expt.cbs','ΔΔG.expt.dip','ΔΔG.expt.Ru']]
print(df4)

df4=df4.fillna("None")
print(df4)
PandasTools.SaveXlsxFromFrame(df4, "../datalist/datalist.xls", size=(100, 100))
