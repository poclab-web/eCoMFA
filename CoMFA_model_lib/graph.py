import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from sklearn.metrics import mean_squared_error, r2_score

def draw_RMSE(dir):
    fig = plt.figure(figsize=(9,3.5))

    dfs=[]
    ridge = pd.read_csv(dir+  "/Ridge.csv")
    ridge["method"]="Ridge"
    ridge["dataset"]=ridge.index*3//len(ridge)
    dfs.append(ridge)

    gaussian = pd.read_csv(dir+ "/Gaussian.csv")
    gaussian["dataset"]=gaussian.index*3//len(gaussian)
    for sigma in gaussian["sigma"].drop_duplicates().sort_values(ascending=False):
        _=gaussian[gaussian["sigma"]==sigma]
        _["method"]="Gaussian σ = {} Å".format(sigma)
        dfs.append(_)

    df=pd.concat(dfs)

    lasso = pd.read_csv(dir+  "/Lasso.csv")
    lasso["method"]="Lasso"
    lasso["dataset"]=lasso.index*3//len(lasso)

    for _, label in zip(range(3), ["($S$)-Me-CBS", "(-)-DIP-Chloride", "$trans$-[RuC$\mathrm{l_2}$\n{($S$)-XylBINAP}{($S$)-DAIPEN}]"]):
        ax = fig.add_subplot(1, 3, _ + 1)
        plt.xscale("log")
        ax2 = ax.twiny()
        plt.xscale("log")
        ax.set_title(label)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.0,0.5,1.0])
        ax.set_xticks([1, 10,100])
        ax2.set_xticks([0.01,0.1,1])
        ax.set_xlabel("λ (Gaussian, Ridge)", fontsize=10)
        ax2.set_xlabel("λ (Lasso)", fontsize=10)
        ax.set_ylabel("ΔΔG")
        sns.lineplot(x="alpha", y="RMSE_validation", data=df[df["dataset"]==_],hue="method",errorbar=None,style="method",
                    legend="full" if _==2 else None, color="blue", ax=ax,alpha=1)
        sns.lineplot(x="alpha", y="RMSE_validation", data=lasso[lasso["dataset"]==_],hue="method",errorbar=None,style="method",
                    legend="full" if _==2 else None, color="black", ax=ax2,alpha=1)
        sns.lineplot(x="alpha", y="RMSE_regression", data=df[df["dataset"]==_],hue="method",errorbar=None,style="method",
                    legend=None, color="blue", ax=ax,alpha=0.5)
        sns.lineplot(x="alpha", y="RMSE_regression", data=lasso[lasso["dataset"]==_],hue="method",errorbar=None,style="method",
                    legend=None, color="black", ax=ax2,alpha=0.5)
    lines_labels=[ax.get_legend_handles_labels() for ax in [ax2,ax]]
    lines, labels=[sum(lol,[]) for lol in zip(*lines_labels)]
    ax.legend(lines,labels,bbox_to_anchor=(1, 1), loc='upper left', fontsize=6, ncols=1)
    ax2.get_legend().remove()
    
    fig.tight_layout()
    plt.savefig(dir+  "/RMSE.png", transparent=False, dpi=300)

def draw_yyplot(dir):
    fig = plt.figure(figsize=(3 * 4, 3 * 1))
    for _, name in zip(range(4), ["Gaussian", "Ridge", "Lasso", "PLS"]):
        ax = fig.add_subplot(1, 4, _ + 1)
        ax.set_ylim(-4, 4)
        ax.set_yticks([-4, 0, 4])
        ax.set_xlim(-4, 4)
        ax.set_xticks([-4, 0, 4])
        ax.set_aspect("equal")
        ax.set_title(name)
        ax.set_xlabel("ΔΔ${G_{expt}}$ [kcal/mol]", fontsize=10)
        ax.set_ylabel("ΔΔ${G_{predict}}$ [kcal/mol]", fontsize=10)
        df=pd.read_csv(dir+  "/{}.csv".format(name))
        df["dataset"]=df.index*3//len(df)
        dfs=[]
        for __, (file, data) in enumerate(zip(["../arranged_dataset/cbs.xlsx", "../arranged_dataset/DIP-chloride.xlsx",
                                          "../arranged_dataset/RuSS.xlsx"], ["($S$)-Me-CBS", "(-)-DIP-Chloride",
                                             "$trans$-[RuC$\mathrm{l_2}${($S$)-XylBINAP}{($S$)-DAIPEN}]"])):
            df_=df[df["dataset"]==__]
            df_ = pd.read_excel(df_["savefilename"][df_["RMSE_validation"]==df_["RMSE_validation"].min()].iloc[0]+"_prediction.xlsx").sort_values(["smiles"])
            dfs.append(df_)
        df=pd.concat(dfs)
        r2s=[]
        RMSEs=[]
        for column in df.columns:
            if "validation" in column and "validation_PCA" not in column:
                r2s.append(r2_score(df["ΔΔG.expt."], df[column]))
                RMSEs.append(mean_squared_error(df["ΔΔG.expt."], df[column],squared=False))
                ax.scatter(df["ΔΔG.expt."], df[column], s=10, c="red", edgecolor="none",  alpha=0.2)
        r2=r2_score(df["ΔΔG.expt."], df["regression"])
        RMSE=mean_squared_error(df["ΔΔG.expt."], df[column],squared=False)
        ax.scatter(df["ΔΔG.expt."], df["regression"], s=10, c="blue", edgecolor="none",  alpha=0.6)
        ax.scatter([],[],c="red",label="validation \nRMSE = {:.3f}".format(np.average(RMSEs))
                   +"\n$\mathrm{r^2}$ = " + "{:.3f}".format(np.average(r2s)))
        ax.scatter([],[],c="blue",label="regression \nRMSE = {:.3f}".format(np.average(RMSE))
                   +"\n$\mathrm{r^2}$ = " + "{:.3f}".format(r2))
        ax.legend(loc='lower right', fontsize=6, ncols=1)
    fig.tight_layout()
    plt.savefig(dir+  "/yy-plot.png", transparent=False, dpi=300)

# time.sleep(60*60*6)
for param_name in glob.glob("../parameter/cube_to_grid/cube_to_grid0.250510.txt"):
    with open(param_name, "r") as f:
        param = json.loads(f.read())
    os.makedirs(param["out_dir_name"], exist_ok=True)
    draw_RMSE(param["out_dir_name"])
    draw_yyplot(param["out_dir_name"])


# rang=10
# for param_name in sorted(glob.glob("../parameter/cube_to_grid/20240531/cube_to_grid0.250510.txt"),
#                          reverse=True):  # -4.5~2.5 -3~3 -5~5
#     print(param_name)
#     with open(param_name, "r") as f:
#         param = json.loads(f.read())
#     print(param)
#     os.makedirs(param["fig_dir"], exist_ok=True)
#     if False:
#         fig = plt.figure(figsize=(3 * 3, 1 * 3 + 0.5))
#         i = 0
#         for file, label in zip(
#                 ["../arranged_dataset/cbs.xlsx", "../arranged_dataset/DIP-chloride.xlsx", "../arranged_dataset/RuSS.xlsx"],
#                 ["($S$)-Me-CBS", "(-)-DIP-Chloride", "$trans$-[RuC$\mathrm{l_2}$\n{($S$)-XylBINAP}{($S$)-DAIPEN}]", ]):
#             print(file)
#             i += 1
#             file_name = os.path.splitext(os.path.basename(file))[0]
#             ax = fig.add_subplot(1, 3, i)
#             plt.xscale("log")

#             ax.set_title(label, fontsize=10)
#             ax.set_ylim(0.55, 0.85)
#             ax.set_yticks([ 0.6, 0.7, 0.8])
#             # ax.set_ylim(0.5, 5.0)
#             # ax.set_yticks([0.5, 1.0, 1.5, 2.0,2.5,3.0])
#             Gaussian = [[] for i in range(10)]
#             dfps = []
#             for _ in range(rang):
#                 save_path = param["out_dir_name"] + "/" + file_name + "/" + str(_)
#                 # print(save_path)
#                 dfp = pd.read_csv(save_path + "/σ_result.csv")
#                 for j, name in zip(range(10), sorted([name for name in dfp.columns if "Gaussian_test_RMSE" in name])):
#                     # dfp["Gaussian_test_r2"] = dfp["Gaussian_test_RMSE{}".format(j)]
#                     dfp["Gaussian_test_r2"] = dfp[name]
#                     # dfp["n"] = j + 1
#                     dfps.append(dfp.copy())
#                     # ax.plot(dfp["lambda"], dfp["Gaussian_test_r2{}".format(j)], color=cm.hsv(j / 10), linewidth=1, alpha=0.05)
#                     # Gaussian[j].append(dfp["Gaussian_test_RMSE{}".format(j)].values.tolist())
#                     Gaussian[j].append(dfp[name].values.tolist())
#             dfs = pd.concat(dfps)
#             for j, name in zip(range(10), sorted([name for name in dfp.columns if "Gaussian_test_RMSE" in name])):
#                 # print(name)
#                 sigma = re.findall("Gaussian_test_RMSE(.*)", name)[0]
#                 ax.plot(dfp["lambda"], np.average(Gaussian[j], axis=0), "-", label="σ = {:.2f} Å".format(float(sigma)),
#                         # + "\n{:.3f}".format(np.max(np.average(Gaussian[j ], axis=0))),
#                         color=cm.hsv(j / 10), alpha=1)
#             # sns.lineplot(x="lambda",y="Gaussian_test_r2",data=dfs.sort_values(["n"]),ci=None,hue="n",legend="full",palette="hls")
#             # ax.legend(bbox_to_anchor=(0, 1), loc='upper left', fontsize=6, ncols=3)
#             ax.set_xlabel("λ [-]", fontsize=10)
#             ax.set_ylabel("RMSE [kcal/mol]", fontsize=10)  # "$r^2_{test}$"
#             ax.set_xticks([0.01,1, 100])
#         # plt.colorbar(ax=ax)
#         plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
#         fig.tight_layout()

#         # fig.colorbar(ax)
#         plt.savefig(param["fig_dir"] + "/σ_result.png", transparent=False, dpi=300)
#     if True:
#         fig = plt.figure(figsize=(3 * 3, 1 * 3 + 0.5))
#         i = 0
#         for file, label in zip(
#                 ["../arranged_dataset/cbs.xlsx", "../arranged_dataset/DIP-chloride.xlsx", "../arranged_dataset/RuSS.xlsx"],
#                 ["($S$)-Me-CBS", "(-)-DIP-Chloride", "$trans$-[RuC$\mathrm{l_2}$\n{($S$)-XylBINAP}{($S$)-DAIPEN}]", ]):
#             i += 1
#             file_name = os.path.splitext(os.path.basename(file))[0]
#             ax = fig.add_subplot(1, 3, i)
#             plt.xscale("log")

#             ax2 = ax.twiny()
#             plt.xscale("log")

#             ax.set_title(label)
#             # ax.set_ylim(0.5, 3)
#             # ax.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
#             ax.set_ylim(0, 1.0)
#             # ax2.set_xlim(100,10000)
#             ax.set_yticks([0.0,0.5,1.0])
#             ax.set_xticks([1, 10,100])
#             ax2.set_xticks([0.01,0.1,1])

#             # Gaussian = []
#             # Lasso = []
#             # Ridge = []
#             PLS = []
#             dfps = []
#             for _ in range(rang):
#                 save_path = param["out_dir_name"] + "/" + file_name + "/" + str(_)
#                 dfp = pd.read_csv(save_path + "/λ_result.csv")

#                 # dfp["validation_r2"] = dfp["Gaussian_validation_RMSE"]
#                 # dfp["method"] = "Gaussian"
#                 # dfp["Gaussian"]=True
#                 # dfps.append(dfp.copy())
#                 for _,columns in enumerate(sorted(dfp.columns)):
#                     if re.match("Gaussian..*_validation_RMSE", columns):#"Gaussian" in columns and "_validation_RMSE" in columns:
#                         dfp[["regression_RMSE","validation_RMSE"]] = dfp[[columns[:-16]+"_regression_RMSE",columns]]
#                         dfp["sort"]=_
#                         dfp["method"] = columns[:-16]+""
#                         dfp["Gaussian"]=True
#                         dfp["size"]=2
#                         dfps.append(dfp.copy())
#                 # dfp[["regression_r2","validation_r2"]] = dfp[["PLS_regression_RMSE","PLS_validation_RMSE"]]
#                 # dfp["method"] = "4PLS"
#                 # dfp["Gaussian"]=False
#                 # dfps.append(dfp.copy())
                

#                 dfp[["regression_RMSE","validation_RMSE"]] = dfp[["Ridge_regression_RMSE","Ridge_validation_RMSE"]]
#                 dfp["method"] = "Ridge"
#                 dfp["sort"]=_+2
#                 dfp["size"]=2
#                 dfp["Gaussian"]=False
#                 dfps.append(dfp.copy())
#                 dfp[["regression_RMSE","validation_RMSE"]] = dfp[["Lasso_regression_RMSE","Lasso_validation_RMSE"]]
#                 dfp["lambda"]/=100
#                 dfp["method"] = "Lasso"
#                 dfp["sort"]=_+3
#                 dfp["size"]=2
#                 dfp["Gaussian"]=False
#                 dfps.append(dfp.copy())
#                 # dfp["validation_r2"] = dfp["PLS_validation_RMSE"]
#                 # dfp["method"] = "PLS"
#                 # dfp["n"] = range(1, len(dfp["lambda"]) + 1)
#                 # dfps.append(dfp.copy())

#                 # Gaussian.append(dfp["Gaussian_validation_r2"].values.tolist())
#                 # Lasso.append(dfp["Lasso_validation_r2"].values.tolist())
#                 # Ridge.append(dfp["Ridge_validation_r2"].values.tolist())
#                 PLS.append(dfp["PLS_validation_RMSE"].values.tolist())
#             dfs = pd.concat(dfps)
#             print(dfs)
#             val=np.inf
#             for columns in dfs.columns:
#                 if re.match("Gaussian..*_validation_RMSE", columns):
#                     for lam in dfs["lambda"].drop_duplicates():
#                         ans=dfs[dfs["lambda"]==lam][columns].mean()
#                         if ans<val:
#                             val=ans
#                             column=columns[:-16]+""
            
#             # sns.lineplot(x="lambda", y="validation_r2", data=dfs[(dfs["method"]==column)&dfs["Gaussian"]].sort_values(["method"]), hue="method",errorbar=None,
#             #              legend=None, palette="jet_r", ax=ax)
#             sns.lineplot(x="lambda", y="validation_RMSE", data=dfs[dfs["method"]=="Lasso"].sort_values(["sort"],ascending=False), hue="method",errorbar=None,style="method",
#                          legend="full" if i==3 else None, color="black", ax=ax2,alpha=1)
#             sns.lineplot(x="lambda", y="regression_RMSE", data=dfs[dfs["method"]=="Lasso"].sort_values(["sort"],ascending=False), hue="method",errorbar=None,style="method",
#                          legend= None, color="black", ax=ax2,alpha=1,size="size")
#             sns.lineplot(x="lambda", y="validation_RMSE", data=dfs[dfs["method"]!="Lasso"].sort_values(["sort"],ascending=False), hue="method",errorbar=None,style="method",
#                          legend="full" if i==3 else None, palette="jet_r", ax=ax,alpha=1)
#             sns.lineplot(x="lambda", y="regression_RMSE", data=dfs[dfs["method"]!="Lasso"].sort_values(["sort"],ascending=False), hue="method",errorbar=None,style="method",
#                          legend= None, palette="jet_r", ax=ax,alpha=1,size="size")
            
#             # ax.fill_between(dfs[dfs["method"]=="3Lasso"]["lambda"], dfs[dfs["method"]=="3Lasso"]["validation_r2"]+0.1, dfs[dfs["method"]=="3Lasso"]["validation_r2"]-0.1, alpha=0.1)
#             # sns.lineplot(x="lambda", y="validation_r2", data=dfs[~dfs["Gaussian"]].sort_values(["method"]), hue="method", palette="winter",errorbar="ci",
#             #              legend="full" if i==3 else None, ax=ax,style="method")
#             # sns.lineplot(x="n", y="validation_r2", data=dfs[dfs["method"] == "PLS"].sort_values(["method"]), legend="full",
#             #              ax=ax2)

#             # # t検定の実行
#             # t_stat, p_value = stats.ttest_ind(dfs[(dfs["method"] == 'Gaussian') & (dfs["lambda"] == 512.0)]["validation_r2"],
#             #                                   dfs[(dfs["method"] == 'Ridge') & (dfs["lambda"] == 512.0)]["validation_r2"])

#             # # 結果の表示
#             # print('t統計量 =', t_stat, 'p値 =', p_value)
#             # ax.plot([], [], "-", label="PLS\n(≧{:.2f})".format(np.min(np.average(PLS, axis=0))), color="orange", alpha=1)
#             ax.set_xlabel("λ (Gaussian, Ridge)", fontsize=10)
#             ax2.set_xlabel("λ (Lasso)", fontsize=10)

#             # ax2.set_xlabel("n_components", fontsize=10)
#             ax.set_ylabel("RMSE [kcal/mol]", fontsize=10)  # $r^2_{test}$
#             # ax2.set_xticks([1, 5, 10, 15])  # print("Gaussian", np.std(Gaussian, axis=0))
#         lines_labels=[ax.get_legend_handles_labels() for ax in [ax2,ax]]
#         lines, labels=[sum(lol,[]) for lol in zip(*lines_labels)]
#         ax.legend(lines,labels,bbox_to_anchor=(1, 1), loc='upper left', fontsize=6, ncols=1)
#         ax2.get_legend().remove()
#         # ax2.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=6, ncols=1)

#         fig.tight_layout()
#         plt.savefig(param["fig_dir"] + "/λ_result.png", transparent=False, dpi=300)
#         print("rmse.png_complete")
#     if False:
#         fig = plt.figure(figsize=(3 * 3, 1 * 3))
#         i = 0

#         for file, label in zip(["../arranged_dataset/cbs.xlsx", "../arranged_dataset/DIP-chloride.xlsx",
#                                 "../arranged_dataset/RuSS.xlsx"], ["($S$)-Me-CBS", "(-)-DIP-Chloride",
#                                    "$trans$-[RuC$\mathrm{l_2}$\n{($S$)-XylBINAP}{($S$)-DAIPEN}]", ]):
#             i += 1
#             file_name = os.path.splitext(os.path.basename(file))[0]
#             ax = fig.add_subplot(1, 3, i)
#             ax.set_ylim(-5, 5)
#             ax.set_xlim(-5, 5)
#             ax.set_title(label)
#             ax.set_xticks([-5, 0, 5])
#             ax.set_yticks([-5, 0, 5])
#             dfps = []
#             for _ in range(rang):
#                 save_path = param["out_dir_name"] + "/" + file_name + "/" + str(_)
#                 dfp = pd.read_excel(save_path + "/result_test.xlsx")
#                 dfps.append(
#                     dfp)  # ax.plot(dfp["lambda"],dfp["Gaussian_test_RMSE"],color="blue",label="Gaussian")  # ax.plot(dfp["lambda"], dfp["lasso_test_RMSE"],color="red",label="lasso")  # ax.plot(dfp["lambda"], dfp["ridge_test_RMSE"],color="green",label="ridge")  # ax.plot(dfp["ΔΔG.expt."],dfp["Gaussian_predict"],"o",color="blue",label="Gaussian",alpha=0.5)  # ax.plot(dfp["ΔΔG.expt."], dfp["Ridge_predict"],"o",color="red",label="Lasso",alpha=0.5)  # ax.plot(dfp["ΔΔG.expt."], dfp["Lasso_predict"],"o",color="green",label="Ridge",alpha=0.5)

#                 # ax.plot(dfp["ΔΔG.expt."], dfp["Gaussian_test"], "o", label="Gaussian" if _ == 1 else None, color='blue',  #         markeredgecolor="none", alpha=0.1)  # ax.plot(dfp["ΔΔG.expt."], dfp["Ridge_test"], "o", label="Ridge" if _ == 1 else None, color="red",  #         markeredgecolor="none", alpha=0.1)  # ax.plot(dfp["ΔΔG.expt."], dfp["Lasso_test"], "o", label="Lasso" if _ == 1 else None, color="green",  #         markeredgecolor="none", alpha=0.1)  # ax.plot(dfp["ΔΔG.expt."], dfp["PLS_test"], "o", label="PLS" if _ == 1 else None, color="orange",  #         markeredgecolor="none", alpha=0.1)  # dfplegend["Gaussian_predict","Ridge_predict","Lasso_predict"] =0  # ax.set_xticks([-2, 0, 2])
#             dfp = pd.concat(dfps)
#             ax.plot(dfp["ΔΔG.expt."], dfp["Gaussian_test"], "o", label="Gaussian", color='blue', markeredgecolor="none",
#                     alpha=0.1)
#             ax.plot(dfp["ΔΔG.expt."], dfp["Ridge_test"], "o", label="Ridge", color="red", markeredgecolor="none",
#                     alpha=0.1)
#             ax.plot(dfp["ΔΔG.expt."], dfp["Lasso_test"], "o", label="Lasso", color="green", markeredgecolor="none",
#                     alpha=0.1)
#             ax.plot(dfp["ΔΔG.expt."], dfp["PLS_test"], "o", label="PLS", color="orange", markeredgecolor="none",
#                     alpha=0.1)
#             # ax.scatter([], [], color="blue", label="Gaussian", alpha=0.5)
#             # ax.scatter([], [], color="red", label="Lasso", alpha=0.5)
#             # ax.scatter([], [], color="green", label="Ridge", alpha=0.5)
#             ax.legend(loc='lower right', fontsize=6)
#             ax.set_xlabel("ΔΔ${G_{expt}}$ [kcal/mol]", fontsize=10)
#             ax.set_ylabel("ΔΔ${G_{predict}}$ [kcal/mol]", fontsize=10)
#             print("dfp", len(dfp))
#         fig.tight_layout()
#         plt.savefig(param["fig_dir"] + "/test_predict.png", transparent=False, dpi=300)

#     fig = plt.figure(figsize=(3 * 4, 3 * 1))
#     ax = []
#     for _, name in zip(range(4), ["Gaussian", "Ridge", "Lasso", "PLS"]):
#         ax_ = fig.add_subplot(1, 4, _ + 1)
#         ax_.set_ylim(-4, 4)
#         ax_.set_yticks([-4, 0, 4])
#         ax_.set_xlim(-4, 4)
#         ax_.set_xticks([-4, 0, 4])
#         ax_.set_aspect("equal")
#         ax_.set_title(name)
#         ax_.set_xlabel("ΔΔ${G_{expt}}$ [kcal/mol]", fontsize=10)
#         ax_.set_ylabel("ΔΔ${G_{predict}}$ [kcal/mol]", fontsize=10)
#         ax.append(ax_)
#     fig.tight_layout()
#     # for h, (file, data) in enumerate(zip(glob.glob("../arranged_dataset/*.xlsx"),
#     #                                      [
#     #                                         "($S$)-CBS-Me",
#     #                                          "(-)-DIP-Chloride",
#     #                                       "$trans$-[RuC$\mathrm{l_2}${($S$)-XylBINAP}{($S$)-DAIPEN}]",
#     #                                      ])):
#     Gaussians = []
#     Ridges = []
#     Lassos = []
#     PLSs = []
#     Gaussian = [[] for _ in range(rang)]
#     Ridge = [[] for _ in range(rang)]
#     Lasso = [[] for _ in range(rang)]
#     PLS = [[] for _ in range(rang)]
#     Gaussian_regressions = [[] for _ in range(rang)]
#     Ridge_regressions = [[] for _ in range(rang)]
#     Lasso_regressions = [[] for _ in range(rang)]
#     PLS_regressions = [[] for _ in range(rang)]
#     exp = []
#     for h, (file, data) in enumerate(zip(["../arranged_dataset/cbs.xlsx", "../arranged_dataset/DIP-chloride.xlsx",
#                                           "../arranged_dataset/RuSS.xlsx"], ["($S$)-Me-CBS", "(-)-DIP-Chloride",
#                                              "$trans$-[RuC$\mathrm{l_2}${($S$)-XylBINAP}{($S$)-DAIPEN}]"])):
#         file_name = os.path.splitext(os.path.basename(file))[0]
#         Gaussian_ = []
#         Ridge_ = []
#         Lasso_ = []
#         PLS_ = []
#         Gaussian_regression = []
#         Ridge_regression = []
#         Lasso_regression = []
#         PLS_regression = []
#         for _ in range(rang):
#             save_path = param["out_dir_name"] + "/" + file_name + "/" + str(_)
#             dfp = pd.read_excel(save_path + "/λ_result.xlsx").sort_values(["smiles"])
#             rmse=np.inf
#             for columns in dfp.columns:
#                 if "Gaussian0.50" in columns and "_validation" in columns:
#                     print(columns)
#                     rmse_ = mean_squared_error(dfp["ΔΔG.expt."], dfp[columns],squared=False)
#                     # print(rmse_)
#                     if rmse_ < rmse:
#                         rmse=rmse_
#                         column=columns

#             Gaussian_.append(dfp[column].where(dfp[column].abs()<4,4*np.sign(dfp[column].values)).values.tolist())#"Gaussian1.00_validation"
#             Ridge_.append(dfp["Ridge_validation"].where(dfp["Ridge_validation"].abs()<4,4*np.sign(dfp["Ridge_validation"].values)).values.tolist())
#             Lasso_.append(dfp["Lasso_validation"].where(dfp["Lasso_validation"].abs()<4,4*np.sign(dfp["Lasso_validation"].values)).values.tolist())
#             PLS_.append(dfp["PLS_validation"].where(dfp["PLS_validation"].abs()<4,4*np.sign(dfp["PLS_validation"].values)).values.tolist())
            
#             Gaussian_regressions[_].extend(dfp[column[:-11]+"regression"].where(dfp[column].abs()<4,4*np.sign(dfp[column].values)).values.tolist())#"Gaussian1.00_validation"
#             Ridge_regressions[_].extend(dfp["Ridgeregression"].where(dfp["Ridgeregression"].abs()<4,4*np.sign(dfp["Ridgeregression"].values)).values.tolist())
#             Lasso_regressions[_].extend(dfp["Lassoregression"].where(dfp["Lassoregression"].abs()<4,4*np.sign(dfp["Lassoregression"].values)).values.tolist())
#             PLS_regressions[_].extend(dfp["PLSregression"].where(dfp["PLSregression"].abs()<4,4*np.sign(dfp["PLSregression"].values)).values.tolist())
#             Gaussian_regression.append(dfp[column[:-11]+"regression"].where(dfp[column].abs()<4,4*np.sign(dfp[column].values)).values.tolist())#"Gaussian1.00_validation"
#             Ridge_regression.append(dfp["Ridgeregression"].where(dfp["Ridgeregression"].abs()<4,4*np.sign(dfp["Ridgeregression"].values)).values.tolist())
#             Lasso_regression.append(dfp["Lassoregression"].where(dfp["Lassoregression"].abs()<4,4*np.sign(dfp["Lassoregression"].values)).values.tolist())
#             PLS_regression.append(dfp["PLSregression"].where(dfp["PLSregression"].abs()<4,4*np.sign(dfp["PLSregression"].values)).values.tolist())

#             print(column)
#             Gaussian[_].extend(dfp[column].where(dfp[column].abs()<4,4*np.sign(dfp[column].values)).values.tolist())
#             Ridge[_].extend(dfp["Ridge_validation"].where(dfp["Ridge_validation"].abs()<4,4*np.sign(dfp["Ridge_validation"].values)).values.tolist())
#             Lasso[_].extend(dfp["Lasso_validation"].where(dfp["Lasso_validation"].abs()<4,4*np.sign(dfp["Lasso_validation"].values)).values.tolist())
#             PLS[_].extend(dfp["PLS_validation"].where(dfp["PLS_validation"].abs()<4,4*np.sign(dfp["PLS_validation"].values)).values.tolist())
#         Gaussians.append(Gaussian_)
#         Ridges.append(Ridge_)
#         Lassos.append(Lasso_)
#         PLSs.append(PLS_)
#         # Gaussian_regressions.append(Gaussian_regression)
#         # Ridge_regressions.append(Ridge_regression)
#         # Lasso_regressions.append(Lasso_regression)
#         # PLS_regressions.append(PLS_regression)
#         exp.extend(dfp["ΔΔG.expt."].values.tolist())
        
        
#         alpha=0.2
#         markersize=10
#         ax[0].scatter(np.tile(dfp["ΔΔG.expt."].values, np.array(Gaussian_).shape[0]), np.array(Gaussian_).ravel(), s=markersize, c="red",
#                    edgecolor="none",  alpha=alpha)
#         ax[1].scatter(np.tile(dfp["ΔΔG.expt."].values, np.array(Gaussian_).shape[0]), np.array(Ridge_).ravel(), s=markersize,c="red",
#                    edgecolor="none",alpha=alpha )
#         ax[2].scatter(np.tile(dfp["ΔΔG.expt."].values, np.array(Gaussian_).shape[0]), np.array(Lasso_).ravel(),s=markersize,c="red",
#                    edgecolor="red",  alpha=alpha)
#         ax[3].scatter(np.tile(dfp["ΔΔG.expt."].values, np.array(Gaussian_).shape[0]), np.array(PLS_).ravel(), s=markersize,c="red",
#                    edgecolor="red", alpha=alpha)
        
#         # alpha=0.2
#         # markersize=5
        
#         ax[0].scatter(np.tile(dfp["ΔΔG.expt."].values, np.array(Gaussian_regression).shape[0]), np.array(Gaussian_regression).ravel(), marker="x",s=markersize,
#                    linewidth=1,color="blue", alpha=alpha)
#         ax[1].scatter(np.tile(dfp["ΔΔG.expt."].values, np.array(Gaussian_regression).shape[0]), np.array(Ridge_regression).ravel(), marker="x",s=markersize,
#                    linewidth=1,color="blue", alpha=alpha)
#         ax[2].scatter(np.tile(dfp["ΔΔG.expt."].values, np.array(Gaussian_regression).shape[0]), np.array(Lasso_regression).ravel(), marker="x",s=markersize,
#                    linewidth=1,color="blue",  alpha=alpha)
#         ax[3].scatter(np.tile(dfp["ΔΔG.expt."].values, np.array(Gaussian_regression).shape[0]), np.array(PLS_regression).ravel(), marker="x",s=markersize,
#                    linewidth=1,color="blue", alpha=alpha)
#         # ax[3].legend(loc='upper center', bbox_to_anchor=(0.5 - 2, -0.17), ncol=4, fontsize=10, markerscale=4)
    
        
        
#     print(Gaussian)
#     print("!!!")
#     print(Gaussians)

#     ga = [r2_score(exp, pred) for pred in Gaussian]

#     ri = [r2_score(exp, pred) for pred in Ridge]
#     la = [r2_score(exp, pred) for pred in Lasso]
#     pls = [r2_score(exp, pred) for pred in PLS]

#     ga_regression = [r2_score(exp, pred) for pred in Gaussian_regressions]

#     ri_regression = [r2_score(exp, pred) for pred in Ridge_regressions]
#     la_regression = [r2_score(exp, pred) for pred in Lasso_regressions]
#     pls_regression = [r2_score(exp, pred) for pred in PLS_regressions]
#     # pls = [mean_squared_error(exp, pred) for pred in PLS]
#     print(np.average(ga), np.std(ga))
#     print(np.average(ri), np.std(ri))
#     print(np.average(la), np.std(la))
#     print(np.average(pls), np.std(pls))
#     print("Gaussian", np.average(np.std(Gaussian, axis=0)), )
#     print("Ridge", np.average(np.std(Ridge, axis=0)))
#     print("Lasso", np.average(np.std(Lasso, axis=0)))
#     print("PLS", np.average(np.std(PLS, axis=0)))
#     ax[0].scatter([],[],alpha=1,c="red",label="validation \n"+"$\mathrm{r^2_{val}}$ = " + "{:.3f} ± {:.3f}".format(np.average(ga), np.std(ga)),s=markersize)
#     ax[0].scatter([],[],alpha=1,c="blue",linewidth=1,label="regression \n"+"$\mathrm{r^2_{reg}}$ = " + "{:.3f} ± {:.3f}".format(np.average(ga_regression), np.std(ga_regression)),s=markersize,marker="x")
#     ax[0].legend(loc='lower right', fontsize=6, ncols=1)
#     # ax[0].text(-3.5, 3.5,"N = {}×{}".format(len(dfp),rang))
#     ax[1].scatter([],[],alpha=1,c="red",label="validation \n"+"$\mathrm{r^2_{val}}$ = " + "{:.3f} ± {:.3f}".format(np.average(ri), np.std(ri)),s=markersize)
#     ax[1].scatter([],[],alpha=1,c="blue",linewidth=1,label="regression \n "+"$\mathrm{r^2_{reg}}$ = " + "{:.3f} ± {:.3f}".format(np.average(ri_regression), np.std(ri_regression)),s=markersize,marker="x")
#     ax[1].legend(loc='lower right', fontsize=6, ncols=1)
#     ax[2].scatter([],[],alpha=1,c="red",label="validation \n " + "$\mathrm{r^2_{val}}$ ="+" {:.3f} ± {:.3f}".format(np.average(la), np.std(la)),s=markersize)
#     ax[2].scatter([],[],alpha=1,c="blue",linewidth=1,label="regression \n " + "$\mathrm{r^2_{reg}}$ ="+" {:.3f} ± {:.3f}".format(np.average(la_regression), np.std(la_regression)),s=markersize,marker="x")
#     ax[2].legend(loc='lower right', fontsize=6, ncols=1)
#     ax[3].scatter([],[],alpha=1,c="red",label="validation \n " + "$\mathrm{r^2_{val}}$ ="+" {:.3f} ± {:.3f}".format(np.average(pls), np.std(pls)),s=markersize)
#     ax[3].scatter([],[],alpha=1,c="blue",linewidth=1,label="regression \n " + "$\mathrm{r^2_{reg}}$ ="+" {:.3f} ± {:.3f}".format(np.average(pls_regression), np.std(pls_regression)),s=markersize,marker="x")
#     ax[3].legend(loc='lower right', fontsize=6, ncols=1)



#     # props = dict(boxstyle='round', facecolor='gray', alpha=0.2)
#     # ax[0].text(-3.5, 3.5, "$\mathrm{r^2_{test}}$ = " + "{:.3f} ± {:.3f}".format(np.average(ga), np.std(ga)),
#     #            verticalalignment='top', bbox=props, fontsize=8)
#     # ax[1].text(-3.5, 3.5, "$\mathrm{r^2_{test}}$ = " + "{:.3f} ± {:.3f}".format(np.average(ri), np.std(ri)),
#     #            verticalalignment='top', bbox=props, fontsize=8)
#     # ax[2].text(-3.5, 3.5, "$\mathrm{r^2_{test}}$ = " + "{:.3f} ± {:.3f}".format(np.average(la), np.std(la)),
#     #            verticalalignment='top', bbox=props, fontsize=8)
#     # ax[3].text(-3.5, 3.5, "$\mathrm{r^2_{test}}$ = " + "{:.3f} ± {:.3f}".format(np.average(pls), np.std(pls)),
#     #            verticalalignment='top', bbox=props, fontsize=8)
#     fig.tight_layout()

#     plt.savefig(param["fig_dir"] + "/validation.png", transparent=False, dpi=300)
#     print("calculate success")