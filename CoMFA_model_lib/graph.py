import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from scipy import stats
from sklearn.metrics import r2_score

# time.sleep(60*60*6)
rang=10
for param_name in sorted(glob.glob("../parameter/cube_to_grid/cube_to_grid0.500510.txt"),
                         reverse=True):  # -4.5~2.5 -3~3 -5~5
    print(param_name)
    with open(param_name, "r") as f:
        param = json.loads(f.read())
    print(param)
    os.makedirs(param["fig_dir"], exist_ok=True)
    if True:
        fig = plt.figure(figsize=(3 * 3, 1 * 3 + 0.5))
        i = 0
        for file, label in zip(
                ["../arranged_dataset/cbs.xlsx", "../arranged_dataset/DIP-chloride.xlsx", "../arranged_dataset/RuSS.xlsx"],
                ["($S$)-Me-CBS", "(-)-DIP-Chloride", "$trans$-[RuC$\mathrm{l_2}$\n{($S$)-XylBINAP}{($S$)-DAIPEN}]", ]):
            print(file)
            i += 1
            file_name = os.path.splitext(os.path.basename(file))[0]
            ax = fig.add_subplot(1, 3, i)
            plt.xscale("log")

            ax.set_title(label, fontsize=10)
            ax.set_ylim(0.55, 0.85)
            ax.set_yticks([ 0.6, 0.7, 0.8])
            # ax.set_ylim(0.5, 5.0)
            # ax.set_yticks([0.5, 1.0, 1.5, 2.0,2.5,3.0])
            Gaussian = [[] for i in range(10)]
            dfps = []
            for _ in range(rang):
                save_path = param["out_dir_name"] + "/" + file_name + "/" + str(_)
                # print(save_path)
                dfp = pd.read_csv(save_path + "/σ_result.csv")
                for j, name in zip(range(10), sorted([name for name in dfp.columns if "Gaussian_test_RMSE" in name])):
                    # dfp["Gaussian_test_r2"] = dfp["Gaussian_test_RMSE{}".format(j)]
                    dfp["Gaussian_test_r2"] = dfp[name]
                    # dfp["n"] = j + 1
                    dfps.append(dfp.copy())
                    # ax.plot(dfp["lambda"], dfp["Gaussian_test_r2{}".format(j)], color=cm.hsv(j / 10), linewidth=1, alpha=0.05)
                    # Gaussian[j].append(dfp["Gaussian_test_RMSE{}".format(j)].values.tolist())
                    Gaussian[j].append(dfp[name].values.tolist())
            dfs = pd.concat(dfps)
            for j, name in zip(range(10), sorted([name for name in dfp.columns if "Gaussian_test_RMSE" in name])):
                # print(name)
                sigma = re.findall("Gaussian_test_RMSE(.*)", name)[0]
                ax.plot(dfp["lambda"], np.average(Gaussian[j], axis=0), "-", label="σ = {:.2f} Å".format(float(sigma)),
                        # + "\n{:.3f}".format(np.max(np.average(Gaussian[j ], axis=0))),
                        color=cm.hsv(j / 10), alpha=1)
            # sns.lineplot(x="lambda",y="Gaussian_test_r2",data=dfs.sort_values(["n"]),ci=None,hue="n",legend="full",palette="hls")
            # ax.legend(bbox_to_anchor=(0, 1), loc='upper left', fontsize=6, ncols=3)
            ax.set_xlabel("λ [-]", fontsize=10)
            ax.set_ylabel("RMSE [kcal/mol]", fontsize=10)  # "$r^2_{test}$"
            ax.set_xticks([0.01,1, 100])
        # plt.colorbar(ax=ax)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        fig.tight_layout()

        # fig.colorbar(ax)
        plt.savefig(param["fig_dir"] + "/σ_result.png", transparent=False, dpi=300)
    if True:
        fig = plt.figure(figsize=(3 * 3, 1 * 3 + 0.5))
        i = 0
        for file, label in zip(
                ["../arranged_dataset/cbs.xlsx", "../arranged_dataset/DIP-chloride.xlsx", "../arranged_dataset/RuSS.xlsx"],
                ["($S$)-Me-CBS", "(-)-DIP-Chloride", "$trans$-[RuC$\mathrm{l_2}$\n{($S$)-XylBINAP}{($S$)-DAIPEN}]", ]):
            i += 1
            file_name = os.path.splitext(os.path.basename(file))[0]
            ax = fig.add_subplot(1, 3, i)
            plt.xscale("log")
            ax2 = ax.twiny()
            ax.set_title(label)
            # ax.set_ylim(0.5, 3)
            # ax.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
            ax.set_ylim(0.5, 1.0)
            ax.set_yticks([0.5,  0.6,0.7,0.8,0.9,1.0])
            ax.set_xticks([0.001, 1, 100])
            Gaussian = []
            Lasso = []
            Ridge = []
            PLS = []
            dfps = []
            for _ in range(rang):
                save_path = param["out_dir_name"] + "/" + file_name + "/" + str(_)
                dfp = pd.read_csv(save_path + "/λ_result.csv")

                dfp["test_r2"] = dfp["Gaussian_test_RMSE"]
                dfp["method"] = "Gaussian"
                dfps.append(dfp.copy())
                dfp["test_r2"] = dfp["lasso_test_RMSE"]
                dfp["method"] = "Lasso"
                dfps.append(dfp.copy())

                dfp["test_r2"] = dfp["ridge_test_RMSE"]
                dfp["method"] = "Ridge"
                dfps.append(dfp.copy())

                dfp["test_r2"] = dfp["pls_test_RMSE"]
                dfp["method"] = "PLS"
                dfp["n"] = range(1, len(dfp["lambda"]) + 1)
                dfps.append(dfp.copy())

                Gaussian.append(dfp["Gaussian_test_r2"].values.tolist())
                Lasso.append(dfp["lasso_test_r2"].values.tolist())
                Ridge.append(dfp["ridge_test_r2"].values.tolist())
                PLS.append(dfp["pls_test_RMSE"].values.tolist())
                # ax.plot(dfp["lambda"], dfp["Gaussian_test_r2"], color="blue", linewidth=1, alpha=0.05)
                # ax.plot(dfp["lambda"], dfp["lasso_test_r2"], color="red", linewidth=1, alpha=0.05)
                # ax.plot(dfp["lambda"], dfp["ridge_test_r2"], color="green", linewidth=1, alpha=0.05)
                # ax2.plot(range(1, len(dfp["lambda"]) + 1), dfp["pls_test_r2"], color="orange", linewidth=1, alpha=0.05)
            dfs = pd.concat(dfps)
            sns.lineplot(x="lambda", y="test_r2", data=dfs[~(dfs["method"] == "PLS")].sort_values(["method"]), hue="method",
                         legend="full", palette="hls", ax=ax)
            sns.lineplot(x="n", y="test_r2", data=dfs[dfs["method"] == "PLS"].sort_values(["method"]), legend="full",
                         ax=ax2)

            # t検定の実行
            t_stat, p_value = stats.ttest_ind(dfs[(dfs["method"] == 'Gaussian') & (dfs["lambda"] == 512.0)]["test_r2"],
                                              dfs[(dfs["method"] == 'Ridge') & (dfs["lambda"] == 512.0)]["test_r2"])

            # 結果の表示
            print('t統計量 =', t_stat, 'p値 =', p_value)
            # sns.lineplot(x="n",y="test_r2",data=dfs[dfs["method"]=="PLS"].sort_values(["method"])[:1],hue="method",legend="full",palette="hls",ax=ax2)

            # ax.plot(dfp["lambda"], np.average(Gaussian, axis=0), "o",
            #         label="Gaussian\nMAX{:.2f}".format(np.max(np.average(Gaussian, axis=0))), color="blue", alpha=1)
            # ax.plot(dfp["lambda"], np.average(Lasso, axis=0), "o",
            #         label="Lasso\nMAX{:.2f}".format(np.max(np.average(Lasso, axis=0))), color="red", alpha=1)
            # ax.plot(dfp["lambda"], np.average(Ridge, axis=0), "o",
            #         label="Ridge\nMAX{:.2f}".format(np.max(np.average(Ridge, axis=0))), color="green", alpha=1)
            # ax2.plot(range(1, len(dfp["lambda"]) + 1), np.average(PLS, axis=0), "o",
            #          label="PLS\n(min:{:.2f})".format(np.max(np.average(PLS, axis=0))), color="orange", alpha=1)
            ax.plot([], [], "-", label="PLS\n(≧{:.2f})".format(np.min(np.average(PLS, axis=0))), color="orange", alpha=1)
            ax.legend(bbox_to_anchor=(0, 1), loc='upper left', fontsize=6, ncols=2)
            ax.set_xlabel("λ", fontsize=10)
            ax2.set_xlabel("n_components", fontsize=10)
            ax.set_ylabel("RMSE [kcal/mol]", fontsize=10)  # $r^2_{test}$
            ax2.set_xticks([1, 5, 10, 15])  # print("Gaussian", np.std(Gaussian, axis=0))

        fig.tight_layout()
        plt.savefig(param["fig_dir"] + "/λ_result.png", transparent=False, dpi=300)
        print("rmse.png_complete")
    if False:
        fig = plt.figure(figsize=(3 * 3, 1 * 3))
        i = 0

        for file, label in zip(["../arranged_dataset/cbs.xlsx", "../arranged_dataset/DIP-chloride.xlsx",
                                "../arranged_dataset/RuSS.xlsx"], ["($S$)-Me-CBS", "(-)-DIP-Chloride",
                                   "$trans$-[RuC$\mathrm{l_2}$\n{($S$)-XylBINAP}{($S$)-DAIPEN}]", ]):
            i += 1
            file_name = os.path.splitext(os.path.basename(file))[0]
            ax = fig.add_subplot(1, 3, i)
            ax.set_ylim(-5, 5)
            ax.set_xlim(-5, 5)
            ax.set_title(label)
            ax.set_xticks([-5, 0, 5])
            ax.set_yticks([-5, 0, 5])
            dfps = []
            for _ in range(rang):
                save_path = param["out_dir_name"] + "/" + file_name + "/" + str(_)
                dfp = pd.read_excel(save_path + "/result_test.xlsx")
                dfps.append(
                    dfp)  # ax.plot(dfp["lambda"],dfp["Gaussian_test_RMSE"],color="blue",label="Gaussian")  # ax.plot(dfp["lambda"], dfp["lasso_test_RMSE"],color="red",label="lasso")  # ax.plot(dfp["lambda"], dfp["ridge_test_RMSE"],color="green",label="ridge")  # ax.plot(dfp["ΔΔG.expt."],dfp["Gaussian_predict"],"o",color="blue",label="Gaussian",alpha=0.5)  # ax.plot(dfp["ΔΔG.expt."], dfp["Ridge_predict"],"o",color="red",label="Lasso",alpha=0.5)  # ax.plot(dfp["ΔΔG.expt."], dfp["Lasso_predict"],"o",color="green",label="Ridge",alpha=0.5)

                # ax.plot(dfp["ΔΔG.expt."], dfp["Gaussian_test"], "o", label="Gaussian" if _ == 1 else None, color='blue',  #         markeredgecolor="none", alpha=0.1)  # ax.plot(dfp["ΔΔG.expt."], dfp["Ridge_test"], "o", label="Ridge" if _ == 1 else None, color="red",  #         markeredgecolor="none", alpha=0.1)  # ax.plot(dfp["ΔΔG.expt."], dfp["Lasso_test"], "o", label="Lasso" if _ == 1 else None, color="green",  #         markeredgecolor="none", alpha=0.1)  # ax.plot(dfp["ΔΔG.expt."], dfp["PLS_test"], "o", label="PLS" if _ == 1 else None, color="orange",  #         markeredgecolor="none", alpha=0.1)  # dfplegend["Gaussian_predict","Ridge_predict","Lasso_predict"] =0  # ax.set_xticks([-2, 0, 2])
            dfp = pd.concat(dfps)
            ax.plot(dfp["ΔΔG.expt."], dfp["Gaussian_test"], "o", label="Gaussian", color='blue', markeredgecolor="none",
                    alpha=0.1)
            ax.plot(dfp["ΔΔG.expt."], dfp["Ridge_test"], "o", label="Ridge", color="red", markeredgecolor="none",
                    alpha=0.1)
            ax.plot(dfp["ΔΔG.expt."], dfp["Lasso_test"], "o", label="Lasso", color="green", markeredgecolor="none",
                    alpha=0.1)
            ax.plot(dfp["ΔΔG.expt."], dfp["PLS_test"], "o", label="PLS", color="orange", markeredgecolor="none",
                    alpha=0.1)
            # ax.scatter([], [], color="blue", label="Gaussian", alpha=0.5)
            # ax.scatter([], [], color="red", label="Lasso", alpha=0.5)
            # ax.scatter([], [], color="green", label="Ridge", alpha=0.5)
            ax.legend(loc='lower right', fontsize=6)
            ax.set_xlabel("ΔΔ${G_{expt}}$ [kcal/mol]", fontsize=10)
            ax.set_ylabel("ΔΔ${G_{predict}}$ [kcal/mol]", fontsize=10)
            print("dfp", len(dfp))
        fig.tight_layout()
        plt.savefig(param["fig_dir"] + "/test_predict.png", transparent=False, dpi=300)

    fig = plt.figure(figsize=(3 * 4, 3 * 1 + 1))
    ax = []
    for _, name in zip(range(4), ["Gaussian", "Ridge", "Lasso", "PLS"]):
        ax_ = fig.add_subplot(1, 4, _ + 1)
        ax_.set_ylim(-4, 4)
        ax_.set_yticks([-4, 0, 4])
        ax_.set_xlim(-4, 4)
        ax_.set_xticks([-4, 0, 4])
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
    Gaussian = [[] for _ in range(rang)]
    Ridge = [[] for _ in range(rang)]
    Lasso = [[] for _ in range(rang)]
    PLS = [[] for _ in range(rang)]
    exp = []
    for h, (file, data) in enumerate(zip(["../arranged_dataset/cbs.xlsx", "../arranged_dataset/DIP-chloride.xlsx",
                                          "../arranged_dataset/RuSS.xlsx"], ["($S$)-Me-CBS", "(-)-DIP-Chloride",
                                             "$trans$-[RuC$\mathrm{l_2}${($S$)-XylBINAP}{($S$)-DAIPEN}]", ])):
        file_name = os.path.splitext(os.path.basename(file))[0]
        Gaussian_ = []
        Ridge_ = []
        Lasso_ = []
        PLS_ = []
        for _ in range(rang):
            save_path = param["out_dir_name"] + "/" + file_name + "/" + str(_)
            dfp = pd.read_excel(save_path + "/λ_result.xlsx").sort_values(["smiles"])
            # for i, name in enumerate(["Gaussian_test", "Ridge_test", "Lasso_test", "PLS_test"]):
            #     None
            #     # ax[i].plot(dfp["ΔΔG.expt."], dfp[name], "o", color=["blue", "red", "green", "orange"][h],
            #     #            markeredgecolor="none", alpha=0.05, markersize=4)
            #     # if _ == 0:
            #     #     ax[i].plot(dfp["ΔΔG.expt."], dfp[name], "o", color=["blue", "red", "green", "orange"][h],
            #     #                markeredgecolor="none", label=data, alpha=1, markersize=4)
            #     # ax.set_xticks([-2, 0, 2])
            Gaussian_.append(dfp["Gaussian_test"].where(dfp["Gaussian_test"] > -3.5, -3.5).values.tolist())
            Ridge_.append(dfp["Ridge_test"].values.tolist())
            Lasso_.append(dfp["Lasso_test"].values.tolist())
            PLS_.append(dfp["PLS_test"].values.tolist())

            # Gaussian_pred=np.where(np.abs(dfp["Gaussian_test"].values) < 3.5, dfp["Gaussian_test"].values, 3.5 * np.sign(dfp["Gaussian_test"].values)).tolist()
            Gaussian_pred = dfp["Gaussian_test"].values.tolist()
            Gaussian[_].extend(Gaussian_pred)
            Ridge[_].extend(dfp["Ridge_test"].values.tolist())
            Lasso[_].extend(dfp["Lasso_test"].values.tolist())
            PLS[_].extend(dfp["PLS_test"].values.tolist())
        # Gaussians.append(Gaussian)
        # Ridges.append(Ridge)
        # Lassos.append(Lasso)
        # PLSs.append(PLS)
        # l[0].extend(Gaussian)
        exp.extend(dfp["ΔΔG.expt."].values.tolist())

        # ax[0].plot(np.tile(dfp["ΔΔG.expt."].values,np.array(Gaussian_).shape[0]), np.array(Gaussian_).ravel(), "o", color=["blue", "red", "green", "orange"][h],
        #            markeredgecolor="none", alpha=0.05, markersize=4)
        # ax[1].plot(np.tile(dfp["ΔΔG.expt."].values,np.array(Gaussian_).shape[0]), np.array(Ridge_).ravel(), "o", color=["blue", "red", "green", "orange"][h],
        #            markeredgecolor="none", alpha=0.05, markersize=4)
        # ax[2].plot(np.tile(dfp["ΔΔG.expt."].values,np.array(Gaussian_).shape[0]), np.array(Lasso_).ravel(), "o", color=["blue", "red", "green", "orange"][h],
        #            markeredgecolor="none", alpha=0.05, markersize=4)
        # ax[3].plot(np.tile(dfp["ΔΔG.expt."].values,np.array(Gaussian_).shape[0]), np.array(PLS_).ravel(), "o", color=["blue", "red", "green", "orange"][h],
        #            markeredgecolor="none", alpha=0.05, markersize=4)

        ax[0].plot(np.tile(dfp["ΔΔG.expt."].values, np.array(Gaussian_).shape[0]), np.array(Gaussian_).ravel(), "o",
                   color=["blue", "red", "green", "orange"][h], label=data, alpha=1, markersize=1)
        ax[1].plot(np.tile(dfp["ΔΔG.expt."].values, np.array(Gaussian_).shape[0]), np.array(Ridge_).ravel(), "o",
                   color=["blue", "red", "green", "orange"][h], label=data, alpha=1, markersize=1)
        ax[2].plot(np.tile(dfp["ΔΔG.expt."].values, np.array(Gaussian_).shape[0]), np.array(Lasso_).ravel(), "o",
                   color=["blue", "red", "green", "orange"][h], label=data, alpha=1, markersize=1)
        ax[3].plot(np.tile(dfp["ΔΔG.expt."].values, np.array(Gaussian_).shape[0]), np.array(PLS_).ravel(), "o",
                   color=["blue", "red", "green", "orange"][h], label=data, alpha=1, markersize=1)

        # ax[0].plot(dfp["ΔΔG.expt."], np.average(Gaussian_, axis=0), "o", color=["blue", "red", "green", "orange"][h],
        #            markeredgecolor="none", label=data, alpha=1, markersize=4)
        # ax[1].plot(dfp["ΔΔG.expt."], np.average(Ridge_, axis=0), "o", color=["blue", "red", "green", "orange"][h],
        #            markeredgecolor="none", label=data, alpha=1, markersize=4)
        # ax[2].plot(dfp["ΔΔG.expt."], np.average(Lasso_, axis=0), "o", color=["blue", "red", "green", "orange"][h],
        #            markeredgecolor="none", label=data, alpha=1, markersize=4)
        # ax[3].plot(dfp["ΔΔG.expt."], np.average(PLS_, axis=0), "o", color=["blue", "red", "green", "orange"][h],
        #            markeredgecolor="none", label=data, alpha=1, markersize=4)
        ax[3].legend(loc='upper center', bbox_to_anchor=(0.5 - 2, -0.17), ncol=4, fontsize=10, markerscale=4)

    ga = [r2_score(exp, pred) for pred in Gaussian]
    ri = [r2_score(exp, pred) for pred in Ridge]
    la = [r2_score(exp, pred) for pred in Lasso]
    pls = [r2_score(exp, pred) for pred in PLS]
    print(np.average(ga), np.std(ga))
    print(np.average(ri), np.std(ri))
    print(np.average(la), np.std(la))
    print(np.average(pls), np.std(pls))
    print("Gaussian", np.average(np.std(Gaussian, axis=0)), )
    print("Ridge", np.average(np.std(Ridge, axis=0)))
    print("Lasso", np.average(np.std(Lasso, axis=0)))
    print("PLS", np.average(np.std(PLS, axis=0)))

    props = dict(boxstyle='round', facecolor='gray', alpha=0.2)
    ax[0].text(-3.5, 3.5, "$\mathrm{r^2_{test}}$ = " + "{:.3f} ± {:.3f}".format(np.average(ga), np.std(ga)),
               verticalalignment='top', bbox=props, fontsize=8)
    ax[1].text(-3.5, 3.5, "$\mathrm{r^2_{test}}$ = " + "{:.3f} ± {:.3f}".format(np.average(ri), np.std(ri)),
               verticalalignment='top', bbox=props, fontsize=8)
    ax[2].text(-3.5, 3.5, "$\mathrm{r^2_{test}}$ = " + "{:.3f} ± {:.3f}".format(np.average(la), np.std(la)),
               verticalalignment='top', bbox=props, fontsize=8)
    ax[3].text(-3.5, 3.5, "$\mathrm{r^2_{test}}$ = " + "{:.3f} ± {:.3f}".format(np.average(pls), np.std(pls)),
               verticalalignment='top', bbox=props, fontsize=8)
    fig.tight_layout()

    plt.savefig(param["fig_dir"] + "/validation.png", transparent=False, dpi=300)
