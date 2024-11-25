from itertools import product
import re
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold

def regression_ElasticNet(X_train,X_test,y_train,y,alpha,l1ratio):
    model=ElasticNet(alpha=alpha,l1_ratio=l1ratio, fit_intercept=False)
    model.fit(X_train, y_train)
    coef=model.coef_
    predict=model.predict(X_test)
    predict=np.clip(predict, np.min(y), np.max(y))
    return coef,predict

def regression_PLS(X_train,X_test,y_train,y,n_components):
    model = PLSRegression(n_components=n_components)
    model.fit(X_train, y_train)
    coef=model.coef_[0]
    predict=model.predict(X_test)[:,0]
    predict=np.clip(predict, np.min(y), np.max(y))
    return coef,predict

def regression_(path):
    df=pd.read_pickle(path).sort_values(by="test")
    df_train=df[df["test"]==0]
    steric_train = df_train.filter(like='steric_fold').to_numpy()
    steric = df.filter(like='steric_fold').to_numpy()
    electrostatic_train = df_train.filter(like='electrostatic_fold').to_numpy()
    electrostatic = df.filter(like='electrostatic_fold').to_numpy()
    
    steric_std,electrostatic_std=np.linalg.norm(steric_train),np.linalg.norm(electrostatic_train)
    steric_train/=steric_std
    steric/=steric_std
    electrostatic_train/=electrostatic_std
    electrostatic/=electrostatic_std
    
    X_train,X=np.concatenate([steric_train,electrostatic_train],axis=1),np.concatenate([steric,electrostatic],axis=1)
    y_train,y=df_train["ΔΔG.expt."].values,df["ΔΔG.expt."].values
    grid=pd.DataFrame(index=[col.replace("steric_fold ","") for col in df.filter(like='steric_fold ').columns])
    for alpha,l1ratio in product(np.logspace(-6,-3,4,base=10),np.round(np.linspace(0, 1, 3),decimals=10)):
        coef,predict=regression_ElasticNet(X_train,X,y_train,y,alpha,l1ratio)
        grid[f"ElasticNet steric_coef {alpha} {l1ratio}"]=coef[:len(coef) // 2]/steric_std
        grid[f"ElasticNet electrostatic_coef {alpha} {l1ratio}"]=coef[len(coef) // 2:]/electrostatic_std
        df[f'ElasticNet regression {alpha} {l1ratio}'] = np.where(df["test"] == 0, predict, np.nan)
        df[f'ElasticNet prediction {alpha} {l1ratio}'] = np.where(df["test"] == 1, predict, np.nan)

        cvs=[]
        sort_index=[]
        kf = KFold(n_splits=5, shuffle=True, random_state=1)
        for train_index, test_index in kf.split(y_train):
            _,cv=regression_ElasticNet(X_train[train_index],X_train[test_index],y_train[train_index],y,alpha,l1ratio)
            cvs.extend(cv)
            sort_index.extend(test_index)
        
        original_array = np.empty_like(cvs)
        original_array[sort_index] = cvs
        
        df.loc[df["test"]==0,f'ElasticNet cv {alpha} {l1ratio}']=original_array

    #PLS
    for n_components in range(1,10):
        coef,predict=regression_PLS(X_train,X,y_train,y,n_components)
        grid[f"PLS steric_coef {n_components}"],grid[f"PLS electrostatic_coef {n_components}"]=coef[:len(coef) // 2],coef[len(coef) // 2:]
        df[f'PLS regression {n_components}'] = np.where(df["test"] == 0, predict, np.nan)
        df[f'PLS prediction {n_components}'] = np.where(df["test"] == 1, predict, np.nan)
        cvs=[]
        sort_index=[]
        kf = KFold(n_splits=5, shuffle=True, random_state=1)
        for train_index, test_index in kf.split(y_train):
            _,cv=regression_PLS(X_train[train_index],X_train[test_index],y_train[train_index],y,n_components)
            cvs.extend(cv)
            sort_index.extend(test_index)
        
        original_array = np.empty_like(cvs)
        original_array[sort_index] = cvs
        df.loc[df["test"]==0,f'PLS cv {n_components}']=original_array

    path=path.replace(".pkl","_regression.pkl")
    df.to_pickle(path)
    path=path.replace(".pkl",".csv")
    grid.to_csv(path)
    print(grid.index)
    
if __name__ == '__main__':
    regression_("/Users/mac_poclab/PycharmProjects/CoMFA_model/arranged_dataset/cbs.pkl")
    regression_("/Users/mac_poclab/PycharmProjects/CoMFA_model/arranged_dataset/DIP.pkl")
    regression_("/Users/mac_poclab/PycharmProjects/CoMFA_model/arranged_dataset/Ru.pkl")
