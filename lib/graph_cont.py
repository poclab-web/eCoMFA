"""Utilities for contribution-space analysis and plotting in the CoMFA workflow."""

from itertools import product
import os
import re
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit.Chem import PandasTools
import time

def nan_rmse(x, y):
    """Compute RMSE while ignoring entries where prediction values are NaN."""
    """
    Calculates the Root Mean Square Error (RMSE) while ignoring NaN values.

    This function computes the RMSE between two arrays, where NaN values in the
    first array (`x`) are ignored in the calculation.

    Args:
        x (numpy.ndarray or pandas.Series): Predicted values, which may contain NaN values.
        y (numpy.ndarray or pandas.Series): Actual values, corresponding to `x`.

    Returns:
        float: The RMSE value, calculated as:
               \[
               \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - x_i)^2}
               \]
               where \( N \) is the number of non-NaN values in `x`.
    """
    return np.sqrt(np.nanmean((y-x)**2))

def nan_r2(x, y):
    """Compute R2 while ignoring entries where prediction values are NaN."""
    """
    Calculates the coefficient of determination (R²) while ignoring NaN values.

    This function computes the R² score between two arrays, where NaN values in
    the first array (`x`) are ignored. The R² score indicates the proportion of
    variance in `y` that is predictable from `x`.

    Args:
        x (numpy.ndarray or pandas.Series): Predicted values, which may contain NaN values.
        y (numpy.ndarray or pandas.Series): Actual values, corresponding to `x`.

    Returns:
        float: The R² value, calculated as:
               \[
               R^2 = 1 - \frac{\sum (y_i - x_i)^2}{\sum (y_i - \bar{y})^2}
               \]
               where:
               - \( \bar{y} \) is the mean of the non-NaN `y` values.
               - The summations ignore NaN values in `x`.
    """
    x,y=x[~np.isnan(x)],y[~np.isnan(x)]
    return 1-np.sum((y-x)**2)/np.sum((y-np.mean(y))**2)

def evaluate_result(path):
    """Evaluate regression results from a pickle file and write a metrics CSV."""
    start=time.time()
    df=pd.read_pickle(path)
    print(time.time()-start)
    df_results=pd.DataFrame(index=df.filter(like='cv').columns)
    df_results["cv_RMSE"]=df_results.index.map(lambda column: nan_rmse(df[column].values,df["ΔΔG.expt."].values))
    df_results["cv_r2"]=df_results.index.map(lambda column: nan_r2(df[column].values,df["ΔΔG.expt."].values))
    df_results["regression_RMSE"]=df.filter(like='regression').columns.map(lambda column: nan_rmse(df[column].values,df["ΔΔG.expt."].values))
    df_results["regression_r2"]=df.filter(like='regression').columns.map(lambda column: nan_r2(df[column].values,df["ΔΔG.expt."].values))
    df_results.to_csv(path.replace("_regression.pkl","_results.csv"))
    best_cv_column=df_results["cv_RMSE"].idxmin()
    print(best_cv_column,np.log2(float(best_cv_column.split()[1])))
    return df_results["cv_RMSE"].idxmin()



def best_parameter(path):
    """Select the best CV setting and reconstruct contribution-level columns."""
    best_cv_column=pd.read_csv(path,index_col=0)["cv_RMSE"].idxmin()
    # print(best_cv_column)
    coef=pd.read_csv(path.replace("_results.csv","_regression.csv"), index_col=0)
    coef = coef[[best_cv_column.replace("cv", "electronic_coef"), best_cv_column.replace("cv", "electrostatic_coef")]]
    coef.columns = ["electronic_coef", "electrostatic_coef"]

    df=pd.read_pickle(path.replace("_results.csv","_regression.pkl"))

    start=time.time()
    columns=df.filter(like='electronic_unfold').columns.tolist()+df.filter(like='electrostatic_unfold').columns.tolist()
    def calc_cont(column):
        x,y,z=map(int, re.findall(r'[+-]?\d+', column))
        coef_column=column.replace(f"_unfold {x} {y} {z}","_coef")
        return df[column]*coef.at[f'{x} {abs(y)} {abs(z)}',coef_column]*np.sign(z)
    data = {col.replace("unfold","cont"): calc_cont(col) for col in columns}
    # data={col.replace("unfold","cont"): calc_cont(col) for col in df.filter(like='electronic_unfold').columns}
    data=pd.DataFrame(data=data)
    data["electronic_cont"],data["electrostatic_cont"]=data.iloc[:,:len(data.columns)//2].sum(axis=1),data.iloc[:,len(data.columns)//2:].sum(axis=1)
    df=pd.concat([df,data],axis=1)
    print("time",time.time()-start)

    df["cv"]=df[best_cv_column]
    df["prediction"]=df[best_cv_column.replace("cv","prediction")]
    # df["scaf_cv"]=df[best_cv_column.replace("cv","scaf")]
    df["er.prediction"]=100/(1+np.exp(df["prediction"]/1.99/df["temperature"]/0.001))
    df["er.cv"]=100/(1+np.exp(df["cv"]/1.99/df["temperature"]/0.001))
    df["regression"]=df[best_cv_column.replace("cv","regression")]
    df["cv_error"]=df["cv"]-df["ΔΔG.expt."]
    df["prediction_error"]=df["prediction"]-df["ΔΔG.expt."]
    # df = df.reindex(df[["prediction_error","cv_error"]].abs().sort_values(ascending=False).index)

    df_=df[["SMILES","InChIKey","ΔΔG.expt.","electronic_cont","electrostatic_cont","regression","prediction","er.prediction","er.cv","cv","prediction_error","cv_error"]].fillna("NAN")#.sort_values(["cv_error","prediction_error"])
    PandasTools.AddMoleculeColumnToFrame(df_, "SMILES")
    path=path.replace(".pkl",".xlsx")
    PandasTools.SaveXlsxFromFrame(df_,path.replace("_results.csv","_regression.xlsx"), size=(100, 100))
    return df#[["ΔΔG.expt.","regression","prediction","cv"]]


def make_cube(df, path):
    """Export per-molecule contribution cube files from reconstructed contributions."""
    grid = np.array([re.findall(r'[+-]?\d+', col) for col in df.filter(like='electronic_cont ').columns]).astype(int)
    min=np.min(grid,axis=0).astype(int)
    max=np.max(grid,axis=0).astype(int)
    rang=max-min

    columns=["ΔΔG.expt.","temperature"]
    for x,y,z in product(range(min[0],max[0]+1),range(min[1],max[1]+1),range(min[2],max[2]+1)):
        if x!=0 and y!=0 and z!=0:
            columns.append(f'electronic_cont {x} {y} {z}')
    for x,y,z in product(range(min[0],max[0]+1),range(min[1],max[1]+1),range(min[2],max[2]+1)):
        if x!=0 and y!=0 and z!=0:
            columns.append(f'electrostatic_cont {x} {y} {z}')
    df=df.set_index("InChIKey").reindex(columns=columns, fill_value=0)
    n=2
    # print(df.columns)
    min=' '.join(map(str, (min+np.array([0.5,0.5,0.5]))*n))
    for inchikey,expt,temp,value in zip(df.index,df["ΔΔG.expt."],df["temperature"],df.iloc[:,2:].values):
        dt=f'/Users/mac_poclab/CoMFA_calc/{inchikey}/Dt0.cube'
        # dt=f'/Volumes/SSD-PSM960U3-UW/CoMFA_calc/{inchikey}/Dt0.cube'
        with open(dt, 'r', encoding='UTF-8') as f:
            f.readline()
            f.readline()
            n_atom,x,y,z=f.readline().split()
            n_atom=int(n_atom)
            f.readline()
            f.readline()
            f.readline()
            coord=[f.readline() for _ in range(n_atom)]
        coord=''.join(coord)
        electronic='\n'.join([' '.join(f"{x}" for x in value[i:i + 6])for i in range(0, len(value)//2, 6)])
        electrostatic='\n'.join([' '.join(f"{x}" for x in value[i:i + 6])for i in range(len(value)//2, len(value), 6)])
        contribution=np.sum(value[:len(value)//2]),np.sum(value[len(value)//2:])
        pred=100/(1+np.exp(sum(contribution)/1.99/temp/0.001))
        os.makedirs(f'{path}/{inchikey}',exist_ok=True)
        with open(f'{path}/{inchikey}/electronic.cube','w') as f:
            print(f'contribution Gaussian Cube File.\nProperty: Default # color electronic {contribution[0]:.2f} predict {sum(contribution):.2f} expt {expt:.2f} pred {pred:.0f}\n{n_atom} {min}\n{rang[0]} {n} 0 0\n{rang[1]} 0 {n} 0\n{rang[2]} 0 0 {n}\n{coord}\n{electronic}',file=f)
        with open(f'{path}/{inchikey}/electrostatic.cube','w') as f:
            print(f'contribution Gaussian Cube File.\nProperty: ALIE # color electrostatic {contribution[1]:.2f} predict {sum(contribution):.2f} expt {expt:.2f} pred {pred:.0f}\n{n_atom} {min}\n{rang[0]} {n} 0 0\n{rang[1]} 0 {n} 0\n{rang[2]} 0 0 {n}\n{coord}\n{electrostatic}',file=f)


def graph_(df, path):
    """Legacy contribution scatter plot helper (kept for compatibility)."""
    #直線表示
    plt.figure(figsize=(3, 3))
    plt.yticks([-4,0,4])
    plt.xticks([-4,0,4])
    plt.ylim(-4,4)
    plt.xlim(-4,4)

    plt.scatter(df["electronic_cont"],df["electrostatic_cont"],c="blue",linewidths=0,s=10,alpha=0.5)
    # rmse=nan_rmse(df["regression"].values,df["ΔΔG.expt."].values)
    # r2=nan_r2(df["regression"].values,df["ΔΔG.expt."].values)
    # plt.scatter([],[],label="regression $r^2$ = " + f"{r2:.2f}"+"\n$\mathrm{RMSE}$"+f" = {rmse:.2f} kcal/mol"
    #                ,c="black",linewidths=0,  alpha=0.5, s=10)

    # rmse=nan_rmse(df["cv"].values,df["ΔΔG.expt."].values)
    # r2=nan_r2(df["cv"].values,df["ΔΔG.expt."].values)
    # plt.scatter([],[],label="LOOCV $r^2$ = " + f"{r2:.2f}"+"\n$\mathrm{RMSE}$"+f" = {rmse:.2f} kcal/mol"
    #                ,c="dodgerblue",linewidths=0,  alpha=0.6, s=10)

    # rmse=nan_rmse(df["prediction"].values,df["ΔΔG.expt."].values)
    # r2=nan_r2(df["prediction"].values,df["ΔΔG.expt."].values)
    # plt.scatter([],[],label="test $r^2$ = " + f"{r2:.2f}"+"\n$\mathrm{RMSE}$"+f" = {rmse:.2f} kcal/mol"
    #                ,c="red",linewidths=0,  alpha=0.8, s=10)

    # plt.scatter(df["ΔΔG.expt."],df["cv"],c="dodgerblue",linewidths=0,s=10,alpha=0.6)
    # # plt.scatter(df["ΔΔG.expt."],df["scaf_cv"],c="green",linewidths=0,s=10,alpha=0.6)
    # plt.scatter(df["ΔΔG.expt."],df["prediction"],c="red",linewidths=0,s=10,alpha=0.8)
    plt.xlabel("electronic contribution [kcal/mol]")
    plt.ylabel("electrostatic contribution [kcal/mol]")
    # plt.legend(loc='lower right', fontsize=5, ncols=1)

    plt.text(-3.6, 3.6, "$\mathit{N}_{\mathrm{test}}$"+f' = {len(df[df["test"]==1])}\n'+"$\mathit{N}_{\mathrm{training}}$"+f' = {len(df[df["test"]==0])}',# transform=ax.transAxes,
                fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(path.replace(".pkl",".png"),dpi=500,transparent=True)
    # df = df.reindex(df["error"].abs().sort_values(ascending=False).index)
def graph_(df, path):
    """Legacy contribution plot with colormap by experimental energy."""
    plt.figure(figsize=(3.5, 3))
    plt.yticks([-4, 0, 4])
    plt.xticks([-4, 0, 4])
    plt.ylim(-4, 4)
    plt.xlim(-4, 4)

    # 色付け対象の値
    values = df["ΔΔG.expt."].values

    # カラースケールの正規化: -4 〜 4 に固定
    norm = mpl.colors.Normalize(vmin=-4, vmax=4)

    # 散布図
    scatter = plt.scatter(
        df["electronic_cont"],
        df["electrostatic_cont"],
        c=values,
        cmap="coolwarm",
        norm=norm,
        linewidths=0,
        s=10,
        alpha=0.7
    )

    # カラーバー
    cbar = plt.colorbar(scatter)
    cbar.set_label(r"$\Delta\Delta G_{\mathrm{expt.}}$ [kcal/mol]")
    cbar.set_ticks([-4, 0, 4])  # 目盛を -4, 0, 4 に設定

    plt.xlabel("electronic [kcal/mol]")
    plt.ylabel("electrostatic [kcal/mol]")

    plt.text(-3.6, 3.6,
             "$\mathit{N}_{\mathrm{test}}$" + f' = {len(df[df["test"]==1])}\n' +
             "$\mathit{N}_{\mathrm{training}}$" + f' = {len(df[df["test"]==0])}',
             fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(path.replace(".pkl", ".png"), dpi=500, transparent=True)
import matplotlib.pyplot as plt
import matplotlib as mpl

import matplotlib.pyplot as plt
import matplotlib as mpl

def graph_(df, path):
    """Final contribution-space plot used by the current script entry point."""
    plt.figure(figsize=(3.5, 3))
    plt.yticks([-4, 0, 4])
    plt.xticks([-4, 0, 4])
    plt.ylim(-4, 4)
    plt.xlim(-4, 4)

    # カラースケールの正規化: -4 〜 4 に固定
    norm = mpl.colors.Normalize(vmin=-4, vmax=4)
    cmap = plt.get_cmap("coolwarm")

    # マスク作成
    mask_fill = df["InChIKey"].isin(["KWOLFJPFCHCOCG-UHFFFAOYSA-N","KZJRKRQSDZGHEC-UHFFFAOYSA-N","QQZOPKMRPOGIEB-UHFFFAOYSA-N","SYBYTAAJFKOIEJ-UHFFFAOYSA-N", "ZWEHNKRNPOVVGH-UHFFFAOYSA-N","PJGSXYOJTGTZAV-UHFFFAOYSA-N","LTNUSYNQZJZUSY-UICOGKGYSA-N"])
    df_outline = df[~mask_fill]
    df_fill = df[mask_fill]
    print(len(df_fill))

    # くり抜き丸（全体）
    plt.scatter(
        df_outline["electronic_cont"],
        df_outline["electrostatic_cont"],
        facecolors=cmap(norm(df_outline["ΔΔG.expt."])),
        linewidths=0,
        s=10,
        alpha=0.2
    )

    # 塗りつぶし丸（特定のInChIKey）
    scatter = plt.scatter(
        df_fill["electronic_cont"],
        df_fill["electrostatic_cont"],
        c=df_fill["ΔΔG.expt."],
        cmap=cmap,
        norm=norm,
        s=10,
        # edgecolors='black',
        linewidths=0,
        alpha=1
    )

    # カラーバー
    cbar = plt.colorbar(scatter)
    cbar.set_label(r"$\Delta\Delta G_{\mathrm{expt.}}$ [kcal/mol]")
    cbar.set_ticks([-4, 0, 4])

    plt.xlabel("electronic [kcal/mol]")
    plt.ylabel("electrostatic [kcal/mol]")

    plt.text(-3.6, 3.6,
             "$\mathit{N}_{\mathrm{test}}$" + f' = {len(df[df["test"]==1])}\n' +
             "$\mathit{N}_{\mathrm{training}}$" + f' = {len(df[df["test"]==0])}',
             fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(path.replace(".pkl", ".png"), dpi=500, transparent=True)


def bar():
    """Legacy LOOCV bar plot helper (kept for compatibility)."""
    path="/Users/mac_poclab/PycharmProjects/CoMFA_model/dataset/"
    cbs=pd.read_csv(path+"cbs_electronic_electrostatic_results.csv", index_col=0)
    dip=pd.read_csv(path+"DIP_electronic_electrostatic_results.csv", index_col=0)
    alpine_borane=pd.read_csv(path+"alpine_borane_electronic_electrostatic_results.csv", index_col=0)

    left=np.arange(3.0)*4

    array=np.array([cbs.filter(regex=r'PLS [+-]?\d+ cv',axis=0).max()["cv_r2"],
                    dip.filter(regex=r'PLS [+-]?\d+ cv',axis=0).max()["cv_r2"],
                    alpine_borane.filter(regex=r'PLS [+-]?\d+ cv',axis=0).max()["cv_r2"]])
    plt.figure(figsize=(4.8, 3.2))
    plt.bar(left,array,color="red",label='PLS',alpha=0.25)
    for i, v in enumerate(array):
        plt.text(left[i], v + 0.05, f"{v:.2f}", ha='center', fontsize=8)
    left+=0.9
    print(array)
    array=np.array([cbs.filter(regex=r"^Ridge .{0,} cv",axis=0).max()["cv_r2"],
                    dip.filter(regex=r"^Ridge .{0,} cv",axis=0).max()["cv_r2"],
                    alpine_borane.filter(regex=r"^Ridge .{0,} cv",axis=0).max()["cv_r2"]])
    print(array)
    plt.bar(left,array,color="red",label='Ridge',alpha=0.5)
    for i, v in enumerate(array):
        plt.text(left[i], v + 0.05, f"{v:.2f}", ha='center', fontsize=8)
    left+=0.9

    array=np.array([cbs.filter(regex=r"^ElasticNet .{0,} cv",axis=0).max()["cv_r2"],
                    dip.filter(regex=r"^ElasticNet .{0,} cv",axis=0).max()["cv_r2"],
                    alpine_borane.filter(regex=r"^ElasticNet .{0,} cv",axis=0).max()["cv_r2"]])
    plt.bar(left,array,color="red",label='Elastic Net',alpha=0.75)
    for i, v in enumerate(array):
        plt.text(left[i], v + 0.05, f"{v:.2f}", ha='center', fontsize=8)
    left+=0.9

    array=np.array([cbs.filter(regex=r"^Lasso .{0,} cv",axis=0).max()["cv_r2"],
                    dip.filter(regex=r"^Lasso .{0,} cv",axis=0).max()["cv_r2"],
                    alpine_borane.filter(regex=r"^Lasso .{0,} cv",axis=0).max()["cv_r2"]])
    plt.bar(left,array,color="red",label='Lasso',alpha=1)
    for i, v in enumerate(array):
        plt.text(left[i], v + 0.05, f"{v:.2f}", ha='center', fontsize=8)

    label = [r"$\mathit{(S)}$-CBS", r"$\mathit{(+)}$-DIP-Cl", r"$\mathit{(S)}$-alpine borane"]
    plt.bar(left-1.35, 0, tick_label=label, align="center")

    plt.axhline(0, color='black', linewidth=1.0)  # y=0の枠線
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.gca().xaxis.set_ticks_position('none')  # 横軸の目盛り線を消す
    plt.gca().yaxis.set_ticks_position('none')  # 横軸の目盛り線を消す

    plt.legend(ncol=4, bbox_to_anchor=(0.5, 1.01), loc='lower center', frameon=True)
    # plt.xlabel("Dataset")
    plt.ylabel("$r^2_{\mathrm{LOOCV}}$")
    plt.yticks(np.arange(0, 1.1, 0.5))
    plt.tight_layout()
    plt.savefig(path+"results.png",dpi=500,transparent=False)
def bar():
    """Final combined LOOCV benchmark plot with R2 (scatter) and RMSE (bar)."""
    path = "/Users/mac_poclab/PycharmProjects/CoMFA_model/dataset/"
    cbs = pd.read_csv(path + "cbs_electronic_electrostatic_results.csv", index_col=0)
    dip = pd.read_csv(path + "DIP_electronic_electrostatic_results.csv", index_col=0)
    alpine_borane = pd.read_csv(path + "alpine_borane_electronic_electrostatic_results.csv", index_col=0)

    dataset_labels = [r"$\mathit{(S)}$-CBS", r"$\mathit{(+)}$-DIP-Cl", r"$\mathit{(S)}$-alpine borane"]
    base_x = np.arange(3.0) * 4  # 基準となるx位置

    models = [
        (r'PLS [+-]?\d+ cv', "tab:red", 'PLS'),
        (r'^Ridge .{0,} cv', "tab:orange", 'Ridge'),
        (r'^ElasticNet .{0,} cv', "tab:green", 'Elastic Net'),
        (r'^Lasso .{0,} cv', "tab:blue", 'Lasso'),
    ]

    fig, ax1 = plt.subplots(figsize=(4.8, 3.2))
    ax2 = ax1.twinx()

    color_r2 = "red"
    color_rmse = "blue"

    handles = []  # for custom legend
    labels = []
    r2_array_max = np.array([
        cbs.max()["cv_r2"],
        dip.max()["cv_r2"],
        alpine_borane.max()["cv_r2"]
    ])
    for model_idx, (regex, alpha, label) in enumerate(models):
        x_positions = base_x + model_idx * 0.9

        r2_array = np.array([
            cbs.filter(regex=regex, axis=0).max()["cv_r2"],
            dip.filter(regex=regex, axis=0).max()["cv_r2"],
            alpine_borane.filter(regex=regex, axis=0).max()["cv_r2"]
        ])

        rmse_array = np.array([
            cbs.filter(regex=regex, axis=0).min()["cv_RMSE"],
            dip.filter(regex=regex, axis=0).min()["cv_RMSE"],
            alpine_borane.filter(regex=regex, axis=0).min()["cv_RMSE"]
        ])


        face_colors = []
        for r2_val, r2_max in zip(r2_array, r2_array_max):
            if np.isclose(r2_val, r2_max):  # 浮動小数点比較には isclose を推奨
                face_colors.append(alpha)  # 塗りつぶし（例: 'tab:red'など）
            else:
                face_colors.append("white")  # 白抜き
        s = ax1.scatter(x_positions, r2_array, color=alpha, alpha=1,facecolor=face_colors)
        s = ax1.scatter(x_positions, r2_array, color=alpha, alpha=1,label=label+r" $r^2$",  facecolor="none")
        b = ax2.bar(x_positions, rmse_array, color=alpha, alpha=1, width=0.4,label=label+" RMSE")

        handles.append(s)
        labels.append(label )
        handles.append(b)
        labels.append(label )

    ax1.set_ylabel(r"$r^2_{\mathrm{LOOCV}}$")#, color=color_r2)
    ax1.set_yticks(np.arange(0, 1.1, 0.1))
    ax1.set_ylim(0.5, 0.9)
    ax1.tick_params(axis='y')#, colors=color_r2)

    ax2.set_ylabel("RMSE"+r"$_{\mathrm{LOOCV}}$"+ " [kcal/mol]")#, color=color_rmse)
    ax2.set_ylim(0.5, 1)
    ax2.tick_params(axis='y')#, colors=color_rmse)

    # Set dataset labels at center of grouped bars
    mid_x = base_x + 1.35  # shift to middle of all bars
    ax1.set_xticks(mid_x)
    ax1.set_xticklabels(dataset_labels)

    ax1.axhline(0, color='black', linewidth=1.0)
    # ax1.spines['top'].set_visible(False)
    # ax1.spines['right'].set_visible(False)
    # ax1.spines['left'].set_visible(False)

    ax1.xaxis.set_ticks_position('none')
    # ax1.yaxis.set_ticks_position('none')

    plt.legend(handles=handles, ncol=4, bbox_to_anchor=(0.5, 1.02), loc='lower center',frameon=True,fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path + "results_with_rmse.png", dpi=500, transparent=False)


if __name__ == '__main__':
    # start=time.time()
    # for cond in ["cbs","DIP","alpine_borane"]:
    #     evaluate_result(f"/Users/mac_poclab/PycharmProjects/CoMFA_model/dataset/{cond}_electronic_electrostatic_regression.pkl")

    # print(time.time()-start)

    df_cbs=best_parameter("/Users/mac_poclab/PycharmProjects/CoMFA_model/dataset/cbs_electronic_electrostatic_results.csv")
    df_dip=best_parameter("/Users/mac_poclab/PycharmProjects/CoMFA_model/dataset/DIP_electronic_electrostatic_results.csv")
    df_alp=best_parameter("/Users/mac_poclab/PycharmProjects/CoMFA_model/dataset/alpine_borane_electronic_electrostatic_results.csv")
    # bar()

    # make_cube(df_cbs,'/Users/mac_poclab/CoMFA_results/CBS')
    # make_cube(df_dip,'/Users/mac_poclab/CoMFA_results/DIP')
    # make_cube(df_alp,'/Users/mac_poclab/CoMFA_results/alp')
    graph_(df_cbs,"/Users/mac_poclab/PycharmProjects/CoMFA_model/dataset/cont_cbs.png")
    graph_(df_dip,"/Users/mac_poclab/PycharmProjects/CoMFA_model/dataset/cont_dip.png")
    graph_(df_alp,"/Users/mac_poclab/PycharmProjects/CoMFA_model/dataset/cont_alpine_borane.png")
    graph_(pd.concat([df_cbs,df_dip,df_alp]),"/Users/mac_poclab/PycharmProjects/CoMFA_model/dataset/cont_all.png")
