import glob
import json
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def set_ax(ax):
    ax.set_ylim(-4, 4)
    ax.set_yticks([-4, 0, 4])
    ax.set_xlim(-4, 4)
    ax.set_xticks([-4, 0, 4])
    ax.set_aspect("equal")
    ax.set_xlabel("ΔΔ$\mathit{G}_{\mathrm{expt}}$ [kcal/mol]", fontsize=10)
    ax.set_ylabel("ΔΔ$\mathit{G}_{\mathrm{predict}}$ [kcal/mol]", fontsize=10)
    return ax

def draw_yyplot(dir):
    fig = plt.figure(figsize=(3 * 3, 3 * 1))

    df=pd.read_csv(dir+  "/{}.csv".format("ElasticNet"),index_col = 'Unnamed: 0').sort_index()

    df_lasso1=df.iloc[:len(df) // 3][df["l1ratio"]==1]
    src_file_path=df_lasso1["savefilename"][df_lasso1["RMSE_validation"]==df_lasso1["RMSE_validation"].min()].iloc[0]+"_prediction.xlsx"
    df_lasso1 = pd.read_excel(src_file_path).sort_values(["smiles"])
    


    ax = fig.add_subplot(1, 3, 1)
    ax=set_ax(ax)
    


def draw_yyplot(dir):
    fig = plt.figure(figsize=(3 * 3, 3 * 1))

    for _, name in zip(range(3), ["Lasso","Ridge","ElasticNet"]):
        ax = fig.add_subplot(1, 3, _ + 1)
        ax=set_ax(ax)
        df=pd.read_csv(dir+  "/{}.csv".format("ElasticNet"),index_col = 'Unnamed: 0').sort_index()
        if name=="Lasso":
            df=df[df["l1ratio"]==1]
        elif name=="Ridge":
            df=df[df["l1ratio"]==0]
        else:
            name="Elastic Net"
        ax.set_title(name)
        
        df.reset_index(drop=True, inplace=True)
        df["dataset"]=df.index*3//len(df)
        

        dfs=[]
        for __ in range(3):
            df_=df[(df["dataset"]==__)]
            src_file_path=df_["savefilename"][df_["RMSE_validation"]==df_["RMSE_validation"].min()].iloc[0]+"_prediction.xlsx"
            new_file_path=df_["savefilename"][df_["RMSE_validation"]==df_["RMSE_validation"].min()].iloc[0]+"_prediction_best.xlsx"
            df_ = pd.read_excel(src_file_path).sort_values(["smiles"])
            
            dfs.append(df_)
            shutil.copy(src_file_path, new_file_path)

        df_=pd.concat(dfs)
        df_train=df_[df_["test"]==False]
        df_test=df_[df_["test"]==True]
        r2s=[]
        RMSEs=[]
        for column in df_.columns:
            if "validation_PCA" not in column and "validation" in column:
                print(column)
                r2s.append(r2_score(df_train["ΔΔG.expt."], df_train[column]))
                RMSEs.append(mean_squared_error(df_train["ΔΔG.expt."], df_train[column],squared=False))
                ax.scatter(df_train["ΔΔG.expt."], df_train[column], s=10, c="dodgerblue", edgecolor="none",  alpha=0.6)
        ax.scatter(df_train["ΔΔG.expt."], df_train["regression"], s=10, c="black", edgecolor="none",  alpha=0.6)
        
        r2=r2_score(df_train["ΔΔG.expt."], df_train["regression"])
        RMSE=mean_squared_error(df_train["ΔΔG.expt."], df_train["regression"],squared=False)
        # dfs=[]
        # for __ in range(3):
        #     df_=df[df["dataset"]==__]
        #     df_ = pd.read_excel(df_["savefilename"][df_["RMSE_PCA_validation"]==df_["RMSE_PCA_validation"].min()].iloc[0]+"_prediction.xlsx").sort_values(["smiles"])
        #     dfs.append(df_)
        # df_=pd.concat(dfs)
        r2s_PCA=[]
        RMSEs_PCA=[]
        for column in df_test.columns:
            if "prediction" in column:
                r2s_PCA.append(r2_score(df_test["ΔΔG.expt."], df_test[column]))
                RMSEs_PCA.append(mean_squared_error(df_test["ΔΔG.expt."], df_test[column],squared=False))
                ax.scatter(df_test["ΔΔG.expt."], df_test[column], s=10, c="red", edgecolor="none",  alpha=0.8)
        ax.scatter([],[],c="red",label="$\mathrm{RMSE_{test}}$"+" = {:.2f}".format(np.average(RMSEs_PCA))
                +"\n$r_{test}^2$ = " + "{:.2f}".format(np.average(r2s_PCA)),  alpha=0.8, s=30)
        ax.scatter([],[],c="dodgerblue",label="$\mathrm{RMSE_{cv}}$"+" = {:.2f}".format(np.average(RMSEs))
                   +"\n$r^2_{cv}$ = " + "{:.2f}".format(np.average(r2s)),  alpha=0.6, s=30)
        ax.scatter([],[],c="black",label="$\mathrm{RMSE_{regression}}$"+" = {:.2f}".format(np.average(RMSE))
                   +"\n$r^2_{regression}$ = " + "{:.2f}".format(r2),  alpha=0.6, s=30)
        # 'df' は pandas データフレームであることを想定しています
        num_rows = len(df_test)  # データフレームの行数を取得
        ax.text(0.05, 0.95, "$N_{test}$"+f' = {num_rows}\n'+"$N_{training}$"+f' = {len(df_train)}', transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="white"))

        ax.legend(loc='lower right', fontsize=6, ncols=1)
    fig.tight_layout()
    plt.savefig(dir+  "/yy-plot.png", transparent=False, dpi=300)
    # plt.show()
def draw_coef_plot(dir):
    fig = plt.figure(figsize=(3 * 4, 3 * 1))
    for _, name in zip(range(4), [ "ElasticNet","PLS"]):
        ax = fig.add_subplot(1, 4, _ + 1)
        # ax.set_ylim(-4, 4)
        # ax.set_yticks([-4, 0, 4])
        # ax.set_xlim(-4, 4)
        # ax.set_xticks([-4, 0, 4])
        ax.set_aspect("equal")
        ax.set_title(name)
        # ax.set_xlabel("ΔΔ${G_{expt}}$ [kcal/mol]", fontsize=10)
        # ax.set_ylabel("ΔΔ${G_{predict}}$ [kcal/mol]", fontsize=10)
        df=pd.read_csv(dir+  "/{}.csv".format(name),index_col = 'Unnamed: 0').sort_index()
        df["dataset"]=df.index*3//len(df)


        # dfs=[]
        for __ in range(3):
            df_=df[(df["dataset"]==__)]
            df_ = pd.read_excel(df_["savefilename"][df_["RMSE_validation"]==df_["RMSE_validation"].min()].iloc[0]+"_prediction.xlsx").sort_values(["smiles"])
            ax.scatter(df_["steric_cont"], df_["electrostatic_cont"], s=10, c="dodgerblue", edgecolor="none",  alpha=0.2)
            # dfs.append(df_)
        # df_=pd.concat(dfs)
        # df_train=df_[df_["test"]==False]
        # df_test=df_[df_["test"]==True]
        # r2s=[]
        # RMSEs=[]
        # for column in df_.columns:
        #     if "validation_PCA" not in column and "validation" in column:
        #         print(column)
        #         r2s.append(r2_score(df_train["steric_cont"], df_train[column]))
        #         RMSEs.append(mean_squared_error(df_train["ΔΔG.expt."], df_train[column],squared=False))
        #         ax.scatter(df_train["ΔΔG.expt."], df_train[column], s=10, c="dodgerblue", edgecolor="none",  alpha=0.2)
        # ax.scatter(df_train["ΔΔG.expt."], df_train["regression"], s=10, c="black", edgecolor="none",  alpha=0.8)
        
        # r2=r2_score(df_train["ΔΔG.expt."], df_train["regression"])
        # RMSE=mean_squared_error(df_train["ΔΔG.expt."], df_train["regression"],squared=False)
        # # dfs=[]
        # # for __ in range(3):
        # #     df_=df[df["dataset"]==__]
        # #     df_ = pd.read_excel(df_["savefilename"][df_["RMSE_PCA_validation"]==df_["RMSE_PCA_validation"].min()].iloc[0]+"_prediction.xlsx").sort_values(["smiles"])
        # #     dfs.append(df_)
        # # df_=pd.concat(dfs)
        # r2s_PCA=[]
        # RMSEs_PCA=[]
        # for column in df_test.columns:
        #     if "prediction" in column:
        #         r2s_PCA.append(r2_score(df_test["ΔΔG.expt."], df_test[column]))
        #         RMSEs_PCA.append(mean_squared_error(df_test["ΔΔG.expt."], df_test[column],squared=False))
        #         ax.scatter(df_test["ΔΔG.expt."], df_test[column], s=10, c="tomato", edgecolor="none",  alpha=0.8)
        # ax.scatter([],[],c="tomato",label="PCA split \nRMSE = {:.3f}".format(np.average(RMSEs_PCA))
        #         +"\n$\mathrm{r^2}$ = " + "{:.3f}".format(np.average(r2s_PCA)),  alpha=0.8)
        # ax.scatter([],[],c="dodgerblue",label="random split \nRMSE = {:.3f}".format(np.average(RMSEs))
        #            +"\n$\mathrm{r^2}$ = " + "{:.3f}".format(np.average(r2s)),  alpha=0.8)
        # ax.scatter([],[],c="black",label="regression \nRMSE = {:.3f}".format(np.average(RMSE))
        #            +"\n$\mathrm{r^2}$ = " + "{:.3f}".format(r2),  alpha=0.8)
        # # 'df' は pandas データフレームであることを想定しています
        # num_rows = len(df_test)  # データフレームの行数を取得
        # ax.text(0.05, 0.95, f'N={num_rows}', transform=ax.transAxes, 
        #         fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="white"))

        # ax.legend(loc='lower right', fontsize=5, ncols=1)
    fig.tight_layout()
    plt.savefig(dir+  "/coef_plot.png", transparent=False, dpi=300)
    # plt.show()
def importance(dir):
    for name in [ "ElasticNet","PLS"]:
        df=pd.read_csv(dir+  "/{}.csv".format(name),index_col = 'Unnamed: 0').sort_index()
        df["dataset"]=df.index*3//len(df)
        dfs=[]
        for __ in range(3):
            df_=df[(df["dataset"]==__)]
            df_=df_[df_["RMSE_validation"]==df_["RMSE_validation"].min()]
            dfs.append(df_)
        df=pd.concat(dfs)
        print(df)

# time.sleep(60*60*6)
for param_name in glob.glob("C:/Users/poclabws/PycharmProjects/CoMFA_model/parameter/cube_to_grid0.50.txt"):
    with open(param_name, "r") as f:
        param = json.loads(f.read())
    os.makedirs(param["out_dir_name"], exist_ok=True)
    importance(param["out_dir_name"])
    draw_yyplot(param["out_dir_name"])
    draw_coef_plot(param["out_dir_name"])
