from itertools import product
import re
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet
from rdkit.Chem import PandasTools
from sklearn.model_selection import KFold

def regression_ElasticNet(X_train,X_test,y_train,alpha,l1ratio):
    model=ElasticNet(alpha=alpha,l1_ratio=l1ratio, fit_intercept=False)
    model.fit(X_train, y_train)
    coef=model.coef_
    predict=model.predict(X_test)
    return coef,predict

def regression_PLS(X_train,X_test,y_train,n_components):
    model = PLSRegression(n_components=n_components)
    model.fit(X_train, y_train)
    coef=model.coef_
    predict=model.predict(X_test)
    return coef,predict

def regression_(path):
    df=pd.read_pickle(path).sort_values(by="test")
    df_train,df_test=df[df["test"]==0],df[df["test"]==1]

    steric_train = df_train.filter(like='steric fold').to_numpy()
    steric_test = df_test.filter(like='steric fold').to_numpy()
    electrostatic_train = df_train.filter(like='electrostatic fold').to_numpy()
    electrostatic_test = df_test.filter(like='electrostatic fold').to_numpy()
    
    steric_std,electrostatic_std=np.linalg.norm(steric_train),np.linalg.norm(electrostatic_train)
    steric_train/=steric_std
    steric_test/=steric_std
    electrostatic_train/=electrostatic_std
    electrostatic_test/=electrostatic_std
    
    X_train,X_test=np.concatenate([steric_train,electrostatic_train],axis=1),np.concatenate([steric_test,electrostatic_test],axis=1)
    y_train,y_test=df_train["ΔΔG.expt."].values,df_test["ΔΔG.expt."].values
    grid = [re.findall(r'\d+', col) for col in df.columns[df.columns.str.contains('steric fold')]]
    grid=pd.DataFrame(data=grid,columns=["x", "y", "z"])

    for alpha,l1ratio in product(np.logspace(-6,0,10,base=10),np.linspace(0, 1, 11)):
        coef,predict=regression_ElasticNet(X_train,np.concatenate([X_train,X_test],axis=0),y_train,alpha,l1ratio)
        grid[f"ElasticNet steric {alpha} {l1ratio}"],grid[f"ElasitNet electrostatic {alpha} {l1ratio}"]=coef[:len(coef) // 2],coef[len(coef) // 2:]
        df[f'predict ElasticNet {alpha} {l1ratio}']=predict
        cvs=[]
        sort_index=[]
        kf = KFold(n_splits=5, shuffle=True, random_state=1)
        for train_index, test_index in kf.split(y_train):
            _,cv=regression_ElasticNet(X_train[train_index],X_train[test_index],y_train[train_index],alpha,l1ratio)
            cvs.extend(cv)
            sort_index.extend(test_index)
        
        original_array = np.empty_like(cvs)
        original_array[sort_index] = cvs
        df[f'cv ElasticNet {alpha} {l1ratio}']=np.concatenate([original_array,np.zeros_like(y_test)])

    #PLS
    for n_components in range(1,10):
        coef,predict=regression_PLS(X_train,X_test,y_train,n_components)
        grid[f"PLS steric {n_components}"],grid[f"PLS electrostatic {n_components}"]=coef[:len(coef) // 2],coef[len(coef) // 2:]
        df[f'predict PLS {alpha} {l1ratio}']=predict
        cvs=[]
        sort_index=[]
        kf = KFold(n_splits=5, shuffle=True, random_state=1)
        for train_index, test_index in kf.split(y_train):
            _,cv=regression_PLS(X_train[train_index],X_train[test_index],y_train[train_index],n_components)
            cvs.extend(cv)
            sort_index.extend(test_index)
        
        original_array = np.empty_like(cvs)
        original_array[sort_index] = cvs
        df[f'cv PLS {n_components}']=np.concatenate([original_array,np.zeros_like(y_test)])

    path=path.replace(".pkl","_regression.pkl")
    df.to_pickle(path)
    path=path.replace(".pkl",".csv")
    grid.to_csv(path)
    
    # PandasTools.AddMoleculeColumnToFrame(df, "SMILES")
    # PandasTools.SaveXlsxFromFrame(df, path, size=(100, 100))

if __name__ == '__main__':
    path="/Users/mac_poclab/PycharmProjects/CoMFA_model/arranged_dataset/cbs.pkl"
    regression_(path)
