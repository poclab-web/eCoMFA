import pandas as pd
from rdkit.Chem import PandasTools
import os
df1 = pd.read_excel("../arranged_dataset/cbs.xlsx").dropna(subset=['smiles']).drop_duplicates(subset="inchikey")
df2=pd.read_excel("../arranged_dataset/DIP-chloride.xlsx").dropna(subset=['smiles']).drop_duplicates(subset="inchikey")
df3 =pd.read_excel("../arranged_dataset/Russ.xlsx").dropna(subset=['smiles']).drop_duplicates(subset="inchikey")
df = pd.concat([df1,df2,df3])
print(len(set(df["inchikey"])))
df=df[df.duplicated(subset="inchikey")]

df1=df1.rename(columns={'ΔΔG.expt.': 'ΔΔG.expt.cbs'})
df2=df2.rename(columns={'ΔΔG.expt.': 'ΔΔG.expt.dip'})
df3=df3.rename(columns={'ΔΔG.expt.': 'ΔΔG.expt.Ru'})
PandasTools.AddMoleculeColumnToFrame(df, "smiles")
PandasTools.AddMoleculeColumnToFrame(df1, "smiles")
PandasTools.AddMoleculeColumnToFrame(df2, "smiles")
PandasTools.AddMoleculeColumnToFrame(df3, "smiles")
df=df.reset_index(drop=True)
df = df[[ "smiles","ROMol","inchikey"]].drop_duplicates(subset="inchikey")
t=[]
for p in df.index:
    l = []
    for i in df1.index:
        if df["inchikey"][p]==df1["inchikey"][i]:

            q=df1['ΔΔG.expt.cbs'][i]
            l.append(q)
    t.append(l)
df['ΔΔG.expt.cbs']=t

s=[]
for p in df.index:
    l = []
    for i in df2.index:
        if str(df["inchikey"][p])==str(df2["inchikey"][i]):
            q=df2['ΔΔG.expt.dip'][i]
            l.append(q)
    s.append(l)
df['ΔΔG.expt.dip']=s

r=[]
for p in df.index:
    l = []
    for i in df3.index:
        if str(df["inchikey"][p])==str(df3["inchikey"][i]):
            q=df3['ΔΔG.expt.Ru'][i]
            l.append(q)
    r.append(l)
df['ΔΔG.expt.Ru']=r


df_cbs= pd.read_excel("../result/0206/cbs_gaussian/result_loo.xlsx")
df_dip = pd.read_excel("../result/0206/dip-chloride_gaussian/result_loo.xlsx")
df_Russ = pd.read_excel("../result/0206/RuSS_gaussian/result_loo.xlsx")

t=[]
for p in df.index:
    l = []
    for i in df_cbs.index:
        if df["inchikey"][p]==df_cbs["inchikey"][i]:

            q=df_cbs["ΔΔG.loo"][i]
            l.append(q)
    t.append(l)
df['ΔΔG.pre.cbs']=t

s=[]
for p in df.index:
    l = []
    for i in df_dip.index:
        if str(df["inchikey"][p])==str(df_dip["inchikey"][i]):
            q=df_dip["ΔΔG.loo"][i]
            l.append(q)
    s.append(l)
df['ΔΔG.pre.dip']=s

r=[]
for p in df.index:
    l = []
    for i in df_Russ.index:
        if str(df["inchikey"][p])==str(df_Russ["inchikey"][i]):
            q=df_Russ["ΔΔG.loo"][i]
            l.append(q)
    r.append(l)
df["ΔΔG.pre.RuSS"]=r


os.makedirs("../datalist", exist_ok=True)
PandasTools.SaveXlsxFromFrame(df, "../datalist/datalist.xlsx", size=(100, 100))
