from itertools import product
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit.Chem import PandasTools
import time

def nan_rmse(x,y):
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

def nan_r2(x,y):
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

def best_parameter(path):
    df=pd.read_pickle(path)
    cv_columns=df.filter(like='cv').columns
    df_results=pd.DataFrame(index=cv_columns)
    df_results["cv_RMSE"]=df_results.index.map(lambda column: nan_rmse(df[column].values,df["ΔΔG.expt."].values))
    df_results["cv_r2"]=df_results.index.map(lambda column: nan_r2(df[column].values,df["ΔΔG.expt."].values))
    best_cv_column=df_results["cv_RMSE"].idxmin()
    df_results.to_csv(path.replace("_regression.pkl","_results.csv"))

    coef=pd.read_csv(path.replace(".pkl",".csv"), index_col=0)
    coef = coef[[best_cv_column.replace("cv", "steric_coef"), best_cv_column.replace("cv", "electrostatic_coef")]]
    coef.columns = ["steric_coef", "electrostatic_coef"]

    start=time.time()
    columns=df.filter(like='steric_unfold').columns.tolist()+df.filter(like='electrostatic_unfold').columns.tolist()
    
    def calc_cont(column):
        x,y,z=map(int, re.findall(r'[+-]?\d+', column))
        coef_column=column.replace(f"_unfold {x} {y} {z}","_coef")
        return df[column]*coef.at[f'{x} {abs(y)} {abs(z)}',coef_column]*np.sign(z)
    data = {col.replace("unfold","cont"): calc_cont(col) for col in columns}   
    data=pd.DataFrame(data=data)
    data["steric_cont"],data["electrostatic_cont"]=data.iloc[:,:len(data.columns)//2].sum(axis=1),data.iloc[:,len(data.columns)//2:].sum(axis=1)
    df=pd.concat([df,data],axis=1)
    print(time.time()-start)
    
    df["cv"]=df[best_cv_column]
    df["prediction"]=df[best_cv_column.replace("cv","prediction")]
    df["regression"]=df[best_cv_column.replace("cv","regression")]
    df["cv_error"]=df["cv"]-df["ΔΔG.expt."]
    df["prediction_error"]=df["prediction"]-df["ΔΔG.expt."]
    # df = df.reindex(df[["prediction_error","cv_error"]].abs().sort_values(ascending=False).index)
    
    df_=df[["SMILES","InChIKey","ΔΔG.expt.","steric_cont","electrostatic_cont","regression","prediction","cv","prediction_error","cv_error"]].sort_values(["cv_error","prediction_error"]).fillna("NAN")
    PandasTools.AddMoleculeColumnToFrame(df_, "SMILES")
    path=path.replace(".pkl",".xlsx")
    PandasTools.SaveXlsxFromFrame(df_,path, size=(100, 100))
    return df#[["ΔΔG.expt.","regression","prediction","cv"]]

def make_cube(df,path):
    grid = np.array([re.findall(r'[+-]?\d+', col) for col in df.filter(like='steric_cont ').columns]).astype(int)
    min=np.min(grid,axis=0).astype(int)
    max=np.max(grid,axis=0).astype(int)
    rang=max-min
    columns=["InChIKey"]
    for x,y,z in product(range(min[0],max[0]+1),range(min[1],max[1]+1),range(min[2],max[2]+1)):
        if x!=0 and y!=0 and z!=0:
            columns.append(f'steric_cont {x} {y} {z}')
    for x,y,z in product(range(min[0],max[0]+1),range(min[1],max[1]+1),range(min[2],max[2]+1)):
        if x!=0 and y!=0 and z!=0:
            columns.append(f'electrostatic_cont {x} {y} {z}')
    df=df.reindex(columns=columns, fill_value=0)
    min=' '.join(map(str, min+np.array([0.5,0.5,0.5])))
    for inchikey,value in zip(df["InChIKey"],df.iloc[:,1:].values):
        dt=f'/Volumes/SSD-PSM960U3-UW/CoMFA_calc/{inchikey}/Dt0.cube'
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
        steric='\n'.join([' '.join(f"{x}" for x in value[i:i + 6])for i in range(0, len(value)//2, 6)])
        electrostatic='\n'.join([' '.join(f"{x}" for x in value[i:i + 6])for i in range(len(value)//2, len(value), 6)])
        os.makedirs(f'{path}/{inchikey}',exist_ok=True)
        with open(f'{path}/{inchikey}/steric.cube','w') as f:
            print(f'contribution Gaussian Cube File.\nProperty: Default # color steric {np.sum(value[:len(value)//2]):.2f} kcal/mol\n{n_atom} {min}\n{rang[0]} 1 0 0\n{rang[1]} 0 1 0\n{rang[2]} 0 0 1\n{coord}\n{steric}',file=f)
        with open(f'{path}/{inchikey}/electrostatic.cube','w') as f:
            print(f'contribution Gaussian Cube File.\nProperty: ALIE # color electrostatic {np.sum(value[len(value)//2:]):.2f}\n{n_atom} {min}\n{rang[0]} 1 0 0\n{rang[1]} 0 1 0\n{rang[2]} 0 0 1\n{coord}\n{electrostatic}',file=f)


def graph_(df,path):
    #直線表示

    
    plt.figure(figsize=(3, 3))
    plt.yticks([-4,0,4])
    plt.xticks([-4,0,4])
    plt.ylim(-4,4)
    plt.xlim(-4,4)
    
    plt.scatter(df["ΔΔG.expt."],df["regression"],c="black",linewidths=0,s=10,alpha=0.5)
    rmse=nan_rmse(df["regression"].values,df["ΔΔG.expt."].values)
    r2=nan_r2(df["regression"].values,df["ΔΔG.expt."].values)
    plt.scatter([],[],label="$\mathrm{RMSE_{regression}}$"+f" = {rmse:.2f}"
                   +"\n$r^2_{regression}$ = " + f"{r2:.2f}",c="black",linewidths=0,  alpha=0.5, s=10)
    
    rmse=nan_rmse(df["cv"].values,df["ΔΔG.expt."].values)
    r2=nan_r2(df["cv"].values,df["ΔΔG.expt."].values)
    plt.scatter([],[],label="$\mathrm{RMSE_{cv}}$"+f" = {rmse:.2f}"
                   +"\n$r^2_{cv}$ = " + f"{r2:.2f}",c="dodgerblue",linewidths=0,  alpha=0.6, s=10)
    
    rmse=nan_rmse(df["prediction"].values,df["ΔΔG.expt."].values)
    r2=nan_r2(df["prediction"].values,df["ΔΔG.expt."].values)
    plt.scatter([],[],label="$\mathrm{RMSE_{prediction}}$"+f" = {rmse:.2f}"
                   +"\n$r^2_{cv}$ = " + f"{r2:.2f}",c="red",linewidths=0,  alpha=0.8, s=10)

    plt.scatter(df["ΔΔG.expt."],df["cv"],c="dodgerblue",linewidths=0,s=10,alpha=0.6)
    plt.scatter(df["ΔΔG.expt."],df["prediction"],c="red",linewidths=0,s=10,alpha=0.8)
    plt.xlabel("ΔΔ$\mathit{G}_{\mathrm{expt}}$ [kcal/mol]")
    plt.ylabel("ΔΔ$\mathit{G}_{\mathrm{predict}}$ [kcal/mol]")
    plt.legend(loc='lower right', fontsize=5, ncols=1)

    plt.text(-3.6, 3.6, "$\mathit{N}_{\mathrm{test}}$"+f' = {len(df[df["test"]==1])}\n'+"$\mathit{N}_{\mathrm{training}}$"+f' = {len(df[df["test"]==0])}',# transform=ax.transAxes, 
                fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(path.replace(".pkl",".png"),dpi=500,transparent=True)
    # df = df.reindex(df["error"].abs().sort_values(ascending=False).index)

def bar():
    path="/Users/mac_poclab/PycharmProjects/CoMFA_model/arranged_dataset/"
    cbs=pd.read_csv(path+"CBS_results.csv", index_col=0)
    dip=pd.read_csv(path+"DIP_results.csv", index_col=0)
    ru=pd.read_csv(path+"Ru_results.csv", index_col=0)

    left=np.arange(3.0)*4

    array=np.array([cbs.filter(regex=r'PLS [+-]?\d+ cv',axis=0).min()["cv_RMSE"],
                    dip.filter(regex=r'PLS [+-]?\d+ cv',axis=0).min()["cv_RMSE"],
                    ru.filter(regex=r'PLS [+-]?\d+ cv',axis=0).min()["cv_RMSE"]])
    plt.bar(left,array,color="red",label='PLS',alpha=0.2)
    left+=0.9
    print(array)
    array=np.array([cbs.filter(regex=r"^ElasticNet \d+\.\d+ 0.0 cv",axis=0).min()["cv_RMSE"],
                    dip.filter(regex=r"^ElasticNet \d+\.\d+ 0.0 cv",axis=0).min()["cv_RMSE"],
                    ru.filter(regex=r"^ElasticNet \d+\.\d+ 0.0 cv",axis=0).min()["cv_RMSE"]])
    print(array)
    plt.bar(left,array,color="red",label='Ridge',alpha=0.4)
    left+=0.9

    array=np.array([cbs.filter(regex=r"^ElasticNet \d+\.\d+ 1.0 cv",axis=0).min()["cv_RMSE"],
                    dip.filter(regex=r"^ElasticNet \d+\.\d+ 1.0 cv",axis=0).min()["cv_RMSE"],
                    ru.filter(regex=r"^ElasticNet \d+\.\d+ 1.0 cv",axis=0).min()["cv_RMSE"]])
    plt.bar(left,array,color="red",label='Lasso',alpha=0.6)
    left+=0.9

    array=np.array([cbs.filter(regex=r"^ElasticNet \d+\.\d+ \d+\.\d+ cv",axis=0).min()["cv_RMSE"],
                    dip.filter(regex=r"^ElasticNet \d+\.\d+ \d+\.\d+ cv",axis=0).min()["cv_RMSE"],
                    ru.filter(regex=r"^ElasticNet \d+\.\d+ \d+\.\d+ cv",axis=0).min()["cv_RMSE"]])
    plt.bar(left,array,color="red",label='ElasticNet',alpha=0.8)
    label=["CBS","DIP","Ru"]
    plt.bar(left-1.35, 0, tick_label=label, align="center")
    plt.legend(ncol=4)
    plt.xlabel("Dataset")
    plt.ylabel("RMSE [kcal/mol]")
    plt.savefig(path+"results.png",dpi=500,transparent=True)

if __name__ == '__main__':
    df_cbs=best_parameter("/Users/mac_poclab/PycharmProjects/CoMFA_model/arranged_dataset/cbs_regression.pkl")
    df_dip=best_parameter("/Users/mac_poclab/PycharmProjects/CoMFA_model/arranged_dataset/DIP_regression.pkl")
    df_ru=best_parameter("/Users/mac_poclab/PycharmProjects/CoMFA_model/arranged_dataset/Ru_regression.pkl")
    make_cube(df_cbs,'/Volumes/SSD-PSM960U3-UW/CoMFA_results/CBS')
    make_cube(df_dip,'/Volumes/SSD-PSM960U3-UW/CoMFA_results/DIP')
    make_cube(df_ru,'/Volumes/SSD-PSM960U3-UW/CoMFA_results/Ru')
    bar()
    df=pd.concat([df_cbs,df_dip,df_ru])
    graph_(df,"/Users/mac_poclab/PycharmProjects/CoMFA_model/arranged_dataset/regression.png")