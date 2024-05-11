import pandas as pd
from rdkit.Chem import PandasTools
import os
file1="../arranged_dataset/cbs.xlsx"
file1="C:/Users/poclabws/result/-4.375 -4.875 -5.875 28 40 48 0.25 20240413/cbs/0/λ_result.xlsx"
file2="../arranged_dataset/DIP-chloride.xlsx"
file2="C:/Users/poclabws/result/-4.375 -4.875 -5.875 28 40 48 0.25 20240413/DIP-chloride/0/λ_result.xlsx"
file3="../arranged_dataset/Russ.xlsx"
file3="C:/Users/poclabws/result/-4.375 -4.875 -5.875 28 40 48 0.25 20240413/RuSS/0/λ_result.xlsx"
df1 = pd.read_excel(file1).dropna(subset=['smiles']).drop_duplicates(subset="inchikey")[["inchikey","smiles","Gaussian_error",'ΔΔG.expt.',"Gaussian_test"]]
df2=pd.read_excel(file2).dropna(subset=['smiles']).drop_duplicates(subset="inchikey")[["inchikey","smiles","Gaussian_error",'ΔΔG.expt.',"Gaussian_test"]]
df3 =pd.read_excel(file3).dropna(subset=['smiles']).drop_duplicates(subset="inchikey")[["inchikey","smiles","Gaussian_error",'ΔΔG.expt.',"Gaussian_test"]]
df1=df1.rename(columns={"Gaussian_error":"Gaussian_error_CBS",'ΔΔG.expt.': 'ΔΔG.expt.CBS',"Gaussian_test":"Gaussian_test_CBS"})
df2=df2.rename(columns={"Gaussian_error":"Gaussian_error_DIP",'ΔΔG.expt.': 'ΔΔG.expt.DIP',"Gaussian_test":"Gaussian_test_DIP"})
df3=df3.rename(columns={"Gaussian_error":"Gaussian_error_Ru",'ΔΔG.expt.': 'ΔΔG.expt.Ru',"Gaussian_test":"Gaussian_test_Ru"})

df12 = pd.merge(df1,df2, on=["inchikey"])
df12["smiles"]=df12["smiles_x"].where(df12["smiles_x"]==df12["smiles_x"],df12["smiles_y"])
df12=df12.drop(["smiles_x","smiles_y"],axis=1)
# df12=df12.rename(columns={"Gaussian_error_x":"Gaussian_error_CBS","Gaussian_error_y":"Gaussian_error_DIP"})
print(df12)
df23=pd.merge(df2,df3, on=["inchikey"])
df23["smiles"]=df23["smiles_x"].where(df23["smiles_x"]==df23["smiles_x"],df23["smiles_y"])
df23=df23.drop(["smiles_x","smiles_y"],axis=1)

df31=pd.merge(df3,df1, on=["inchikey"])
df31["smiles"]=df31["smiles_x"].where(df31["smiles_x"]==df31["smiles_x"],df31["smiles_y"])
df31=df31.drop(["smiles_x","smiles_y"],axis=1)

df123=pd.merge(df12,df23,on=["inchikey","Gaussian_error_DIP",'ΔΔG.expt.DIP','Gaussian_test_DIP'])
df123["smiles"]=df123["smiles_x"].where(df123["smiles_x"]==df123["smiles_x"],df123["smiles_y"])
df123=df123.drop(["smiles_x","smiles_y"],axis=1)

df=pd.concat([df123,df12,df23,df31])#.rename(columns={"smiles_x":"smiles","smiles_y":"smiles","smiles_x_x:":"smiles"})
df=df.reindex(columns=df.columns.sort_values())
print(df123)

print(df.columns)

# df=df[[ "smiles","inchikey","Gaussian_error_CBS","Gaussian_error_DIP","Gaussian_error_Ru"]]
df=df.fillna(0)
print(len(set(df["inchikey"])))
# df=df[df.duplicated(subset="inchikey")]
print(len(df))
df.to_excel("../datalist/datalist.xlsx")
PandasTools.AddMoleculeColumnToFrame(df, "smiles")
# PandasTools.AddMoleculeColumnToFrame(df1, "smiles")
# PandasTools.AddMoleculeColumnToFrame(df2, "smiles")
# PandasTools.AddMoleculeColumnToFrame(df3, "smiles")
# df=df.reset_index(drop=True)
df = df.drop_duplicates(subset="inchikey")
# t=[]
# for p in df.index:
#     l = []
#     for i in df1.index:
#         if df["inchikey"][p]==df1["inchikey"][i]:
#
#             q=df1['ΔΔG.expt.cbs'][i]
#             l.append(q)
#     t.append(l)
# df['ΔΔG.expt.cbs']=t
#
# s=[]
# for p in df.index:
#     l = []
#     for i in df2.index:
#         if str(df["inchikey"][p])==str(df2["inchikey"][i]):
#             q=df2['ΔΔG.expt.dip'][i]
#             l.append(q)
#     s.append(l)
# df['ΔΔG.expt.dip']=s
#
# r=[]
# for p in df.index:
#     l = []
#     for i in df3.index:
#         if str(df["inchikey"][p])==str(df3["inchikey"][i]):
#             q=df3['ΔΔG.expt.Ru'][i]
#             l.append(q)
#     r.append(l)
# df['ΔΔG.expt.Ru']=r


# df_cbs= pd.read_excel("../result/0206/cbs_gaussian/result_loo.xlsx")
# df_dip = pd.read_excel("../result/0206/dip-chloride_gaussian/result_loo.xlsx")
# df_Russ = pd.read_excel("../result/0206/RuSS_gaussian/result_loo.xlsx")
#
# t=[]
# for p in df.index:
#     l = []
#     for i in df_cbs.index:
#         if df["inchikey"][p]==df_cbs["inchikey"][i]:
#
#             q=df_cbs["ΔΔG.loo"][i]
#             l.append(q)
#     t.append(l)
# df['ΔΔG.pre.cbs']=t
#
# s=[]
# for p in df.index:
#     l = []
#     for i in df_dip.index:
#         if str(df["inchikey"][p])==str(df_dip["inchikey"][i]):
#             q=df_dip["ΔΔG.loo"][i]
#             l.append(q)
#     s.append(l)
# df['ΔΔG.pre.dip']=s
#
# r=[]
# for p in df.index:
#     l = []
#     for i in df_Russ.index:
#         if str(df["inchikey"][p])==str(df_Russ["inchikey"][i]):
#             q=df_Russ["ΔΔG.loo"][i]
#             l.append(q)
#     r.append(l)
# df["ΔΔG.pre.RuSS"]=r


os.makedirs("../datalist", exist_ok=True)
PandasTools.SaveXlsxFromFrame(df, "../datalist/datalist.xlsx", size=(100, 100))
