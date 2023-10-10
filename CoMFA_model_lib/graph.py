import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score
import json
param_file_name = "../parameter/parameter_cbs.txt"
with open(param_file_name, "r") as f:
    param = json.loads(f.read())
input_dir_name=param["out_dir_name"]
save_dir=param["fig_file_dir"]
fig = plt.figure(figsize=(3,3))
df = pd.read_excel("{}/result_loo.xls".format(input_dir_name))
ax = fig.add_subplot(1, 1, 1)
ax.plot([-2.5,2.5], [-2.5,2.5],color="Gray")
ax.plot(df["ΔΔG.expt."], df["ΔΔG.loo"], "s", color="red", label="loo")
ax.plot(df["ΔΔG.expt."], df["ΔΔG.train"], "x",color="Black", label="train")
ax.legend(loc = 'upper left') #凡例
ax.set_xticks([-2, 0, 2])
ax.set_yticks([-2, 0, 2])
ax.set_xlabel("ΔΔG.expt. [kcal/mol]")
ax.set_ylabel("ΔΔG.predict [kcal/mol]")
ax.set_title("$r^2$ = {:.2f}, $q^2$ = {:.2f}"
             .format(
                     r2_score(df["ΔΔG.expt."], df["ΔΔG.train"]),
                     r2_score(df["ΔΔG.expt."], df["ΔΔG.loo"])))
fig.tight_layout()              #レイアウトの設定
#plt.show()
os.makedirs(save_dir,exist_ok=True)
plt.savefig(save_dir+"/plot.png", dpi=300)