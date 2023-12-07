import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
df1 = pd.read_excel("../arranged_dataset/cbs.xls").dropna(subset=['smiles']).drop_duplicates(subset="inchikey")
df2=pd.read_excel("../arranged_dataset/DIP-chloride.xls").dropna(subset=['smiles']).drop_duplicates(subset="inchikey")
df3 =pd.read_excel("../arranged_dataset/Russ.xls").dropna(subset=['smiles']).drop_duplicates(subset="inchikey")
for (df,name) in zip([df1,df2,df3],["cbs","dip","Russ"]):


    plt.hist(df["er."],bins=20)

    os.makedirs("../datahist",exist_ok=True)
    plt.savefig("../datahist"+"/{}erplot.png".format(name))
    plt.clf()

for (df, name) in zip([df1, df2, df3], ["cbs", "dip", "Russ"]):
    plt.hist(df["ΔΔG.expt."],bins=20)


    os.makedirs("../datahist", exist_ok=True)
    plt.savefig("../datahist" + "/{}ΔΔGplot.png".format(name))
    plt.clf()

