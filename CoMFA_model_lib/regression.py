from itertools import product
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from rdkit.Chem import PandasTools

def regression_(path):
    df=pd.read_pickle(path).sort_values(by="test")
    #訓練データの値で標準化
    #回帰
    df_train,df_test=df[df["test"]==0],df[df["test"]==1]
    print(len(df_train))
    steric_train,steric_test=df_train.loc[:, df_train.columns.str.contains('steric')].values,df_test.loc[:, df_test.columns.str.contains('steric')].values
    electrostatic_train,electrostatic_test=df_train.loc[:, df_train.columns.str.contains('steric')].values,df_test.loc[:, df_test.columns.str.contains('steric')].values
    steric_std,electrostatic_std=np.average(steric_train),np.average(electrostatic_train)
    steric_train/=steric_std
    steric_test/=steric_std
    electrostatic_train/=electrostatic_std
    electrostatic_test/=electrostatic_std
    X_train,X_test=np.concatenate([steric_train,electrostatic_train],axis=1),np.concatenate([steric_test,electrostatic_test],axis=1)
    y_train,y_test=df_train["ΔΔG.expt."],df_train["ΔΔG.expt."]
    print(steric_train)
    for alpha,l1ratio in product([1],[1]):
        model=ElasticNet(alpha=alpha,l1_ratio=l1ratio, fit_intercept=False)
        model.fit(X_train, y_train)
        coef=model.coef_
        df[f'predict {alpha} {l1ratio}']=model.predict(np.concatenate([X_train,X_test],axis=0))
        print(coef)
    path=path.replace(".pkl","_regression.pkl")
    df.to_pickle(path)
    # PandasTools.AddMoleculeColumnToFrame(df, "SMILES")
    # PandasTools.SaveXlsxFromFrame(df, path, size=(100, 100))

if __name__ == '__main__':
    path="/Users/mac_poclab/PycharmProjects/CoMFA_model/arranged_dataset/cbs.pkl"
    regression_(path)
