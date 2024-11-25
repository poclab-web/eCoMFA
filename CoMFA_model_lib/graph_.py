from itertools import product
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit.Chem import PandasTools
import time

def nan_rmse(x,y):
    return np.sqrt(np.nanmean((y-x)**2))

def nan_r2(x,y):
    x,y=x[~np.isnan(x)],y[~np.isnan(x)]
    return 1-np.sum((y-x)**2)/np.sum((y-np.mean(y))**2)

def best_parameter(path):
    #値を丸める
    df=pd.read_pickle(path)
    cv_columns=df.filter(like='ElasticNet cv').columns
    rmses=[nan_rmse(df[column].values,df["ΔΔG.expt."].values) for column in cv_columns]
    best_cv_column=cv_columns[rmses.index(min(rmses))]
    coef=pd.read_csv(path.replace(".pkl",".csv"), index_col=0)
    coef = coef[[best_cv_column.replace("cv", "steric_coef"), best_cv_column.replace("cv", "electrostatic_coef")]]
    coef.columns = ["steric_coef", "electrostatic_coef"]

    start=time.time()
    columns=[]
    data=[]
    for column in df.filter(like='steric_unfold').columns:
        x,y,z=map(int, re.findall(r'[+-]?\d+', column))
        columns.append(column.replace("unfold","cont"))
        data.append(df[column]*coef.at[f'{x} {abs(y)} {abs(z)}',"steric_coef"]*np.sign(z))
    data=pd.DataFrame(data=np.array(data).T,columns=columns)
    data["steric_cont"]=data.sum(axis=1)
    df=pd.concat([df,data],axis=1)
    print(time.time()-start)
    
    # df["steric_cont"]=np.sum(df[[column.replace("unfold","cont")for column in df.filter(like='steric_unfold')]].values,axis=1)
    df["cv"]=df[best_cv_column]
    prediction_column=best_cv_column.replace("cv","prediction")
    df["prediction"]=df[prediction_column]
    regression_column=best_cv_column.replace("cv","regression")
    df["regression"]=df[regression_column]
    df["cv_error"]=df["cv"]-df["ΔΔG.expt."]
    df["prediction_error"]=df["prediction"]-df["ΔΔG.expt."]
    # df = df.reindex(df[["prediction_error","cv_error"]].abs().sort_values(ascending=False).index)
    
    df_=df[["SMILES","InChIKey","ΔΔG.expt.","steric_cont","regression","prediction","cv","prediction_error","cv_error"]].sort_values(["cv_error","prediction_error"]).fillna("NAN")
    PandasTools.AddMoleculeColumnToFrame(df_, "SMILES")
    path=path.replace(".pkl",".xlsx")
    PandasTools.SaveXlsxFromFrame(df_,path, size=(100, 100))
    return df#[["ΔΔG.expt.","regression","prediction","cv"]]

def make_cube(df,path):
    print(df.filter(like='steric_cont').columns)
    grid = np.array([re.findall(r'[+-]?\d+', col) for col in df.filter(like='steric_cont ').columns]).astype(int)
    min=np.min(grid,axis=0)
    max=np.max(grid,axis=0)
    rang=max-min+np.array([1,1,1])
    columns=["InChIKey"]
    for x,y,z in product(range(min[0],max[0]+1),range(min[1],max[1]+1),range(min[2],max[2]+1)):
        columns.append(f'steric_cont {x} {y} {z}')
        
    df=df.reindex(columns=columns, fill_value=0)
    min=' '.join(map(str, min))
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
            # dt=f.read().splitlines()[2:]
        # n_atom=int(dt[0].split()[0])
        # coord=dt[4:n_atom+4]
        coord=''.join(coord)
        value='\n'.join([' '.join(f"{x}" for x in value[i:i + 6])for i in range(0, len(value), 6)])
        os.makedirs(f'{path}/{inchikey}',exist_ok=True)
        with open(f'{path}/{inchikey}/steric.cube','w') as f:
            print(f'contribution Gaussian Cube File.\nsteric\n{n_atom} {min}\n{rang[0]} 1 0 0\n{rang[1]} 0 1 0\n{rang[2]} 0 0 1\n{coord}\n{value}',file=f)


def graph_(df,path):
    #r2,rmse表示
    #直線表示
    #軸ラベル表示
    # データ数表示
    #グリッド寄与度→cube
    
    plt.figure(figsize=(3, 3))
    plt.yticks([-4,0,4])
    plt.xticks([-4,0,4])
    plt.ylim(-4,4)
    plt.xlim(-4,4)
    
    plt.scatter(df["ΔΔG.expt."],df["regression"],c="black",linewidths=0,s=10,alpha=0.5)
    rmse=nan_rmse(df["prediction"].values,df["ΔΔG.expt."].values)
    r2=nan_r2(df["prediction"].values,df["ΔΔG.expt."].values)
    plt.scatter([],[],label="$\mathrm{RMSE_{regression}}$"+f" = {rmse:.2f}"
                   +"\n$r^2_{regression}$ = " + f"{r2:.2f}",c="black",linewidths=0,  alpha=0.6, s=10)
    plt.scatter(df["ΔΔG.expt."],df["cv"],c="blue",linewidths=0,s=10,alpha=0.5)
    plt.scatter(df["ΔΔG.expt."],df["prediction"],c="red",linewidths=0,s=10,alpha=0.5)
    plt.xlabel("ΔΔ$\mathit{G}_{\mathrm{expt}}$ [kcal/mol]")
    plt.ylabel("ΔΔ$\mathit{G}_{\mathrm{predict}}$ [kcal/mol]")
    plt.legend(loc='lower right', fontsize=5, ncols=1)
    plt.tight_layout()
    plt.savefig(path.replace(".pkl",".png"),dpi=500,transparent=True)
    # df = df.reindex(df["error"].abs().sort_values(ascending=False).index)

if __name__ == '__main__':
    df_cbs=best_parameter("/Users/mac_poclab/PycharmProjects/CoMFA_model/arranged_dataset/cbs_regression.pkl")
    df_dip=best_parameter("/Users/mac_poclab/PycharmProjects/CoMFA_model/arranged_dataset/DIP_regression.pkl")
    df_ru=best_parameter("/Users/mac_poclab/PycharmProjects/CoMFA_model/arranged_dataset/Ru_regression.pkl")
    make_cube(df_cbs,'/Volumes/SSD-PSM960U3-UW/CoMFA_results/CBS')
    make_cube(df_dip,'/Volumes/SSD-PSM960U3-UW/CoMFA_results/DIP')
    make_cube(df_ru,'/Volumes/SSD-PSM960U3-UW/CoMFA_results/Ru')
    df=pd.concat([df_cbs,df_dip,df_ru])
    graph_(df,"/Users/mac_poclab/PycharmProjects/CoMFA_model/arranged_dataset/regression.png")