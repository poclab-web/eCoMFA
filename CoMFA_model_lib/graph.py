import csv

import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score
import json
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

#fig = plt.figure(figsize=(3*2, 3*4))
fig =plt.figure(figsize=(3*5, 3*1))
# for i,param_file_name in enumerate([
# "../parameter/parameter_cbs_PLS.txt",
# "../parameter/parameter_cbs_gaussian.txt",
#     "../parameter/parameter_cbs_FP.txt",
#
#     "../parameter/parameter_cbs_gaussian_FP.txt",
#     "../parameter/parameter_dip-chloride_PLS.txt",
# "../parameter/parameter_dip-chloride_FP.txt",
#     "../parameter/parameter_dip-chloride_gaussian.txt",
#     "../parameter/parameter_dip-chloride_gaussian_FP.txt"
#     ]):
# for i,param_file_name in enumerate([
# "../parameter/parameter_RuSS_PLS.txt",
# "../parameter/parameter_RuSS_gaussian.txt",
#
# "../parameter/parameter_RuSS_lassocv.txt",
# "../parameter/parameter_RuSS_ridgecv.txt",
# "../parameter/parameter_RuSS_elasticnetcv.txt",
#
#
#     ]):
for i,param_file_name in enumerate([
# "../parameter/parameter_cbs_gaussian.txt",
# "../parameter/parameter_dip-chloride_gaussian.txt",
# "../parameter/parameter_RuSS_gaussian.txt",
#  "../parameter_nomax/parameter_cbs_PLS.txt",
# "../parameter_nomax/parameter_cbs_gaussian.txt",
#  "../parameter_nomax/parameter_cbs_ridgecv.txt",
#  "../parameter_nomax/parameter_cbs_elasticnetcv.txt",
# "../parameter_nomax/parameter_cbs_lassocv.txt",
"../parameter/parameter_dip-chloride_PLS.txt",
"../parameter/parameter_dip-chloride_gaussian.txt",
#
"../parameter/parameter_dip-chloride_ridgecv.txt",
    "../parameter/parameter_dip-chloride_elasticnetcv.txt",
"../parameter/parameter_dip-chloride_lassocv.txt",
# "../parameter_nomax/parameter_RuSS_PLS.txt",
# "../parameter_nomax/parameter_RuSS_gaussian.txt",
#
# "../parameter_nomax/parameter_RuSS_lassocv.txt",
# "../parameter_nomax/parameter_RuSS_ridgecv.txt",
# "../parameter_nomax/parameter_RuSS_elasticnetcv.txt",

    ]):
# for i,param_file_name in enumerate([

#
#     ]):

# # for i, param_file_name in enumerate([
# #                                      "../parameter/parameter_cbs_gaussian.txt",
#
#                                      ]):
    with open(param_file_name, "r") as f:
        param = json.loads(f.read())
    input_dir_name=param["out_dir_name"]
    save_dir=param["fig_file_dir"]
    df = pd.read_excel("{}/result_loo.xlsx".format(input_dir_name))
    try:
        df_test = pd.read_excel("{}/result_5crossvalid.xlsx".format(input_dir_name))
        print("true")
    except:
        None
    # ax = fig.add_subplot(4, 2, i+1)
    ax = fig.add_subplot(1, 5, i+1)
    # ax =  subplot(2, 2, i)
    ax.plot([-2.5,2.5], [-2.5,2.5],color="Gray")
    #ax.plot(df["ΔΔG.expt."], df["ΔΔG.loo"], "s", color="red",markersize=4 , alpha=0.5,label="loo $q^2$ = {:.2f}".format(r2_score(df["ΔΔG.expt."], df["ΔΔG.loo"])))
    # ax.plot(df["ΔΔG.expt."], df["ΔΔG.train"], "x",color="Black", markersize=4 ,alpha=0.5,label="train $r^2$ = {:.2f}".format(r2_score(df["ΔΔG.expt."], df["ΔΔG.train"])))
    try:
        ax.plot(df_test["ΔΔG.expt."], df_test["ΔΔG.crosstest"], "o",color="Blue",markersize=4 , alpha=0.5,label="test $r^2$ = {:.2f}".format(r2_score(df_test["ΔΔG.expt."], df_test["ΔΔG.crosstest"])))
    except:
        None
    ax.legend(loc = 'lower right',fontsize=6) #凡例

    # ax.text(-2.5, 2.0, "$\mathrm{N_{training}}$ = "+"{}\n".format(len(df))+"$\mathrm{N_{test}}$ = "+"{}".format(len(df_test)), fontsize=8)
    ax.text(0.8, 0.15, "$\mathrm{N_{}}$ = "+"{}".format(len(df_test)), fontsize=9,transform=ax.transAxes)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    ax.set_xlabel("ΔΔG.expt. [kcal/mol]")
    ax.set_ylabel("ΔΔG.predict [kcal/mol]")
    ax.set_title(param["title"])
    # ax.set_title(param["title"]+"(N={})".format(len(df)))
    RMSE = np.sqrt(mean_squared_error(df_test["ΔΔG.expt."], df_test["ΔΔG.crosstest"]))
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)

    #np.savetxt("{}/RMSE.csv".format(param["out_dir_name"]), RMSE)
    rmse=[]
    rmse.append(RMSE)
    p=pd.DataFrame(rmse)
    p.to_csv("{}/RMSE.csv".format(param["out_dir_name"]))




save_dir="../figs"
fig.tight_layout()
os.makedirs(save_dir,exist_ok=True)
plt.savefig(save_dir+"/plot.png", dpi=300)