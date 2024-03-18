import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from sklearn.metrics import r2_score

with open("../parameter/cube_to_grid/cube_to_grid.txt", "r") as f:
    param = json.loads(f.read())
print(param)

fig = plt.figure(figsize=(3 * 3, 1 * 3 + 0.5))
i = 0
for file, label in zip(["../arranged_dataset/cbs.xlsx",
                        "../arranged_dataset/DIP-chloride.xlsx",
                        "../arranged_dataset/RuSS.xlsx"
                        ],
                       [
                           "($S$)-Me-CBS",
                           "(-)-DIP-Chloride",
                           "$trans$-[RuC$\mathrm{l_2}$\n{($S$)-XylBINAP}{($S$)-DAIPEN}]",
                       ]):
    print(file)
    i += 1
    file_name = os.path.splitext(os.path.basename(file))[0]
    print(file_name)
    ax = fig.add_subplot(1, 3, i)
    plt.xscale("log")
    ax.set_ylim(0.6, 0.9)
    ax.set_title(label)
    # ax.set_yticks([0.50, 0.75, 1.00])
    Gaussian = [[] for i in range(10)]
    print(Gaussian)
    dfps = []
    for _ in range(10):
        save_path = param["out_dir_name"] + "/" + file_name + "/comparison" + str(_)
        print(save_path)
        dfp = pd.read_csv(save_path + "/n_comparison.csv")
        for j in range(1,11):
            dfp["Gaussian_test_r2"] = dfp["Gaussian_test_r2{}".format(j)]
            # dfp["n"] = j + 1
            dfps.append(dfp.copy())
            # ax.plot(dfp["lambda"], dfp["Gaussian_test_r2{}".format(j)], color=cm.hsv(j / 10), linewidth=1, alpha=0.05)
            Gaussian[j-1].append(dfp["Gaussian_test_r2{}".format(j)].values.tolist())
    print(Gaussian)
    dfs = pd.concat(dfps)
    for j in range(10):
        ax.plot(dfp["lambda"], np.average(Gaussian[j ], axis=0), "-",
                label="n = " + str(j) + "\n{:.3f}".format(np.max(np.average(Gaussian[j ], axis=0))),
                color=cm.hsv(j / 10), alpha=1)
    print(dfs)
    # sns.lineplot(x="lambda",y="Gaussian_test_r2",data=dfs.sort_values(["n"]),ci=None,hue="n",legend="full",palette="hls")
    ax.legend(bbox_to_anchor=(0, 1), loc='upper left', fontsize=6, ncols=3)
    ax.set_xlabel("λ", fontsize=10)
    ax.set_ylabel("$r^2_{test}$", fontsize=10)
fig.tight_layout()
plt.savefig("../figs/n_comparison.png", transparent=False, dpi=300)

fig = plt.figure(figsize=(3 * 3, 1 * 3 + 0.5))
i = 0
for file, label in zip(["../arranged_dataset/cbs.xlsx",
                        "../arranged_dataset/DIP-chloride.xlsx",
                        "../arranged_dataset/RuSS.xlsx"
                        ],
                       [
                           "($S$)-Me-CBS",
                           "(-)-DIP-Chloride",
                           "$trans$-[RuC$\mathrm{l_2}$\n{($S$)-XylBINAP}{($S$)-DAIPEN}]",
                       ]):
    i += 1
    file_name = os.path.splitext(os.path.basename(file))[0]
    ax = fig.add_subplot(1, 3, i)
    plt.xscale("log")
    ax2 = ax.twiny()
    ax.set_ylim(0.6, 0.9)
    ax.set_title(label)
    ax.set_yticks([0.6, 0.75, 0.9])
    Gaussian = []
    Lasso = []
    Ridge = []
    PLS = []
    dfps = []
    for _ in range(50):
        save_path = param["out_dir_name"] + "/" + file_name + "/" + str(_)
        dfp = pd.read_csv(save_path + "/result.csv")

        dfp["test_r2"] = dfp["Gaussian_test_r2"]
        dfp["method"] = "Gaussian"
        dfps.append(dfp.copy())
        dfp["test_r2"] = dfp["lasso_test_r2"]
        dfp["method"] = "Lasso"
        dfps.append(dfp.copy())

        dfp["test_r2"] = dfp["ridge_test_r2"]
        dfp["method"] = "Ridge"
        dfps.append(dfp.copy())

        dfp["test_r2"] = dfp["pls_test_r2"]
        dfp["method"] = "PLS"
        dfp["n"] = range(1, len(dfp["lambda"]) + 1)
        dfps.append(dfp.copy())

        Gaussian.append(dfp["Gaussian_test_r2"].values.tolist())
        Lasso.append(dfp["lasso_test_r2"].values.tolist())
        Ridge.append(dfp["ridge_test_r2"].values.tolist())
        PLS.append(dfp["pls_test_r2"].values.tolist())
        # ax.plot(dfp["lambda"], dfp["Gaussian_test_r2"], color="blue", linewidth=1, alpha=0.05)
        # ax.plot(dfp["lambda"], dfp["lasso_test_r2"], color="red", linewidth=1, alpha=0.05)
        # ax.plot(dfp["lambda"], dfp["ridge_test_r2"], color="green", linewidth=1, alpha=0.05)
        ax2.plot(range(1, len(dfp["lambda"]) + 1), dfp["pls_test_r2"], color="orange", linewidth=1, alpha=0.05)
    dfs = pd.concat(dfps)
    sns.lineplot(x="lambda", y="test_r2", data=dfs[~(dfs["method"] == "PLS")].sort_values(["method"]), hue="method",
                 legend="full", palette="hls", ax=ax)
    sns.lineplot(x="n", y="test_r2", data=dfs[dfs["method"] == "PLS"].sort_values(["method"]), hue=None, legend="full",
                 palette="hls", ax=ax2)
    # sns.lineplot(x="n",y="test_r2",data=dfs[dfs["method"]=="PLS"].sort_values(["method"])[:1],hue="method",legend="full",palette="hls",ax=ax2)

    # ax.plot(dfp["lambda"], np.average(Gaussian, axis=0), "o",
    #         label="Gaussian\nMAX{:.2f}".format(np.max(np.average(Gaussian, axis=0))), color="blue", alpha=1)
    # ax.plot(dfp["lambda"], np.average(Lasso, axis=0), "o",
    #         label="Lasso\nMAX{:.2f}".format(np.max(np.average(Lasso, axis=0))), color="red", alpha=1)
    # ax.plot(dfp["lambda"], np.average(Ridge, axis=0), "o",
    #         label="Ridge\nMAX{:.2f}".format(np.max(np.average(Ridge, axis=0))), color="green", alpha=1)
    ax2.plot(range(1, len(dfp["lambda"]) + 1), np.average(PLS, axis=0), "o",
             label="PLS\nMAX{:.2f}".format(np.max(np.average(PLS, axis=0))), color="orange", alpha=1)
    ax.plot([], [], "-",
            label="PLS\nMAX{:.2f}".format(np.max(np.average(PLS, axis=0))), color="orange", alpha=1)
    ax.legend(bbox_to_anchor=(0, 1), loc='upper left', fontsize=6, ncols=2)
    ax.set_xlabel("λ", fontsize=10)
    ax2.set_xlabel("n_components", fontsize=10)
    ax.set_ylabel("$r^2_{test}$", fontsize=10)
    ax2.set_xticks([1, 5, 10, 15])
    print("Gaussian", np.std(Gaussian, axis=0))

fig.tight_layout()
plt.savefig("../figs/rmse.png", transparent=False, dpi=300)
print("rmse.png_complete")

fig = plt.figure(figsize=(3 * 3, 1 * 3))
i = 0

for file, label in zip(["../arranged_dataset/cbs.xlsx",
                        "../arranged_dataset/DIP-chloride.xlsx",
                        "../arranged_dataset/RuSS.xlsx"
                        ],
                       [
                           "($S$)-Me-CBS",
                           "(-)-DIP-Chloride",
                           "$trans$-[RuC$\mathrm{l_2}$\n{($S$)-XylBINAP}{($S$)-DAIPEN}]",
                       ]):
    i += 1
    file_name = os.path.splitext(os.path.basename(file))[0]
    ax = fig.add_subplot(1, 3, i)
    ax.set_ylim(-5, 5)
    ax.set_xlim(-5, 5)
    ax.set_title(label)
    ax.set_xticks([-5, 0, 5])
    ax.set_yticks([-5, 0, 5])
    for _ in range(50):
        save_path = param["out_dir_name"] + "/" + file_name + "/" + str(_)
        dfp = pd.read_excel(save_path + "/result_test.xlsx")

        # ax.plot(dfp["lambda"],dfp["Gaussian_test_RMSE"],color="blue",label="Gaussian")
        # ax.plot(dfp["lambda"], dfp["lasso_test_RMSE"],color="red",label="lasso")
        # ax.plot(dfp["lambda"], dfp["ridge_test_RMSE"],color="green",label="ridge")
        # ax.plot(dfp["ΔΔG.expt."],dfp["Gaussian_predict"],"o",color="blue",label="Gaussian",alpha=0.5)
        # ax.plot(dfp["ΔΔG.expt."], dfp["Ridge_predict"],"o",color="red",label="Lasso",alpha=0.5)
        # ax.plot(dfp["ΔΔG.expt."], dfp["Lasso_predict"],"o",color="green",label="Ridge",alpha=0.5)

        ax.plot(dfp["ΔΔG.expt."], dfp["Gaussian_test"], "o", label="Gaussian" if _ == 1 else None, color='blue',
                markeredgecolor="none", alpha=0.1)
        ax.plot(dfp["ΔΔG.expt."], dfp["Ridge_test"], "o", label="Ridge" if _ == 1 else None, color="red",
                markeredgecolor="none", alpha=0.1)
        ax.plot(dfp["ΔΔG.expt."], dfp["Lasso_test"], "o", label="Lasso" if _ == 1 else None, color="green",
                markeredgecolor="none", alpha=0.1)
        ax.plot(dfp["ΔΔG.expt."], dfp["PLS_test"], "o", label="PLS" if _ == 1 else None, color="orange",
                markeredgecolor="none", alpha=0.1)
        # dfplegend["Gaussian_predict","Ridge_predict","Lasso_predict"] =0
        # ax.set_xticks([-2, 0, 2])
    # ax.scatter([], [], color="blue", label="Gaussian", alpha=0.5)
    # ax.scatter([], [], color="red", label="Lasso", alpha=0.5)
    # ax.scatter([], [], color="green", label="Ridge", alpha=0.5)
    ax.legend(loc='lower right', fontsize=6)
    ax.set_xlabel("ΔΔ${G_{expt}}$ [kcal/mol]", fontsize=10)
    ax.set_ylabel("ΔΔ${G_{predict}}$ [kcal/mol]", fontsize=10)
fig.tight_layout()
plt.savefig("../figs/test_predict.png", transparent=False, dpi=300)

fig = plt.figure(figsize=(3 * 4, 3 * 1 + 1))
ax = []
for _, name in zip(range(4), ["Gaussian", "Ridge", "Lasso", "PLS"]):
    ax_ = fig.add_subplot(1, 4, _ + 1)
    ax_.set_ylim(-5, 5)
    ax_.set_yticks([-5, 0, 5])
    ax_.set_xlim(-5, 5)
    ax_.set_xticks([-5, 0, 5])
    ax_.set_aspect("equal")
    ax_.set_title(name)
    ax_.set_xlabel("ΔΔ${G_{expt}}$ [kcal/mol]", fontsize=10)
    ax_.set_ylabel("ΔΔ${G_{predict}}$ [kcal/mol]", fontsize=10)
    ax.append(ax_)
fig.tight_layout()
# for h, (file, data) in enumerate(zip(glob.glob("../arranged_dataset/*.xlsx"),
#                                      [
#                                         "($S$)-CBS-Me",
#                                          "(-)-DIP-Chloride",
#                                       "$trans$-[RuC$\mathrm{l_2}${($S$)-XylBINAP}{($S$)-DAIPEN}]",
#                                      ])):
# Gaussians = []
# Ridges = []
# Lassos = []
# PLSs = []
Gaussian = [[] for _ in range(50)]
Ridge = [[] for _ in range(50)]
Lasso = [[] for _ in range(50)]
PLS = [[] for _ in range(50)]
exp=[]
for h, (file, data) in enumerate(zip(["../arranged_dataset/cbs.xlsx",
                                      "../arranged_dataset/DIP-chloride.xlsx",
                                      "../arranged_dataset/RuSS.xlsx"
                                      ],
                                     [
                                         "($S$)-Me-CBS",
                                         "(-)-DIP-Chloride",
                                         "$trans$-[RuC$\mathrm{l_2}${($S$)-XylBINAP}{($S$)-DAIPEN}]",
                                     ])):
    file_name = os.path.splitext(os.path.basename(file))[0]
    Gaussian_ = []
    Ridge_ = []
    Lasso_ = []
    PLS_ = []
    for _ in range(50):
        save_path = param["out_dir_name"] + "/" + file_name + "/" + str(_)
        dfp = pd.read_excel(save_path + "/result_test.xlsx").sort_values(["smiles"])
        for i, name in enumerate(["Gaussian_test", "Ridge_test", "Lasso_test", "PLS_test"]):
            ax[i].plot(dfp["ΔΔG.expt."], dfp[name], "o", color=["blue", "red", "green", "orange"][h],
                       markeredgecolor="none", alpha=0.05, markersize=4)
            # if _ == 0:
            #     ax[i].plot(dfp["ΔΔG.expt."], dfp[name], "o", color=["blue", "red", "green", "orange"][h],
            #                markeredgecolor="none", label=data, alpha=1, markersize=4)
            # ax.set_xticks([-2, 0, 2])
        Gaussian_.append(dfp["Gaussian_test"].values.tolist())
        Ridge_.append(dfp["Ridge_test"].values.tolist())
        Lasso_.append(dfp["Lasso_test"].values.tolist())
        PLS_.append(dfp["PLS_test"].values.tolist())
        Gaussian[_].extend(dfp["Gaussian_test"].values.tolist())
        Ridge[_].extend(dfp["Ridge_test"].values.tolist())
        Lasso[_].extend(dfp["Lasso_test"].values.tolist())
        PLS[_].extend(dfp["PLS_test"].values.tolist())
    # Gaussians.append(Gaussian)
    # Ridges.append(Ridge)
    # Lassos.append(Lasso)
    # PLSs.append(PLS)
    # l[0].extend(Gaussian)
    exp.extend(dfp["ΔΔG.expt."].values.tolist())
    ax[0].plot(dfp["ΔΔG.expt."], np.average(Gaussian_, axis=0), "o", color=["blue", "red", "green", "orange"][h],
               markeredgecolor="none", label=data, alpha=1, markersize=4)
    ax[1].plot(dfp["ΔΔG.expt."], np.average(Ridge_, axis=0), "o", color=["blue", "red", "green", "orange"][h],
               markeredgecolor="none", label=data, alpha=1, markersize=4)
    ax[2].plot(dfp["ΔΔG.expt."], np.average(Lasso_, axis=0), "o", color=["blue", "red", "green", "orange"][h],
               markeredgecolor="none", label=data, alpha=1, markersize=4)
    ax[3].plot(dfp["ΔΔG.expt."], np.average(PLS_, axis=0), "o", color=["blue", "red", "green", "orange"][h],
               markeredgecolor="none", label=data, alpha=1, markersize=4)
    ax[i].legend(loc='upper center', bbox_to_anchor=(0.5 - 2, -0.17), ncol=4, fontsize=10)
fig.tight_layout()

plt.savefig("../figs/test_predict_.png", transparent=False, dpi=300)
ga=[r2_score(exp,pred) for pred in Gaussian]
ri=[r2_score(exp,pred) for pred in Ridge]
la=[r2_score(exp,pred) for pred in Lasso]
pls=[r2_score(exp,pred) for pred in PLS]
print(np.average(ga),np.std(ga))
print(np.average(ri),np.std(ri))
print(np.average(la),np.std(la))
print(np.average(pls),np.std(pls))
print("Gaussian", np.average(np.std(Gaussian, axis=0)),)
print("Ridge", np.average(np.std(Ridge, axis=0)))
print("Lasso", np.average(np.std(Lasso, axis=0)))
print("PLS", np.average(np.std(PLS, axis=0)))

# Gaussians = np.array(Gaussians)
# Ridges = np.array(Ridges)
# Lassos = np.array(Lassos)
# PLSs = np.array(PLS)
# print(Gaussians.shape)
# Gaussians

