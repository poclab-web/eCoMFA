import glob
import json
import multiprocessing
import os
import re
import time
import warnings
import numpy as np
import pandas as pd
import scipy.linalg
from rdkit.Chem import PandasTools
from sklearn import linear_model
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
import time
import calculate_conformation

warnings.simplefilter('ignore')

# シャッフル前の順序に戻す関数
def unshuffle_array(shuffled_array, shuffled_indices):
    original_array = np.empty_like(shuffled_array)
    original_array[shuffled_indices] = shuffled_array
    return original_array

def split_data_pca(X, n_splits=5):
    # PCAを使用してデータを主成分に投影
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X).flatten()
    
    # 主成分に基づいてデータを分割
    percentiles = np.percentile(X_pca, np.linspace(0, 100, n_splits + 1))
    indices=[]
    for _ in range(n_splits):
        if _==n_splits-1:
            test=np.where((X_pca >= percentiles[_]) & (X_pca <= percentiles[_+1]))[0]
            train=np.where((X_pca < percentiles[_]) | (X_pca > percentiles[_+1]))[0]
        else:
            test=np.where((X_pca >= percentiles[_]) & (X_pca < percentiles[_+1]))[0]
            train=np.where((X_pca < percentiles[_]) | (X_pca >= percentiles[_+1]))[0]
        indices.append([train,test])
    return indices

# def regularizing_ridge(X, y, alpha, Q_dir):
#     # Calculate X.T @ X and store the result directly in the same variable if possible
#     XT_X = X.T @ X
#     XT_X += alpha * len(y) * np.load(Q_dir)  # In-place modification of XT_X to save memory
#     # Perform Cholesky factorization in place
#     c, lower = scipy.linalg.cho_factor(XT_X, lower=True, overwrite_a=True, check_finite=False)
#     # Solve the system using the Cholesky factorization result
#     result = scipy.linalg.cho_solve((c, lower), X.T @ y, overwrite_b=True, check_finite=False)
#     return result
def generalized_ridge_regression_transformed(X, y, P,L):
    X= np.block([X[:,:X.shape[1]//2]@P,X[:,X.shape[1]//2:]@P])
    beta_hat=linear_model.Ridge(alpha=L,fit_intercept=False,solver='cholesky').fit(X,y).coef_
    beta_hat= np.block([P@beta_hat[:beta_hat.shape[0]//2], P@beta_hat[beta_hat.shape[0]//2:]])
    return beta_hat

def Gaussian(input):
    X,X1,X2,y,alpha,Q,Q_dir,n_splits,n_repeats,df,df_coord,save_name=input
    
    # start=time.time()
    # gaussian=scipy.linalg.solve(X.T @ X + alpha * len(y) * np.load(Q_dir),
    #                                        X.T @ y, assume_a="pos")
    # time2=time.time()-start
    # start=time.time()
    # gaussian_, info=scipy.sparse.linalg.cg((X.T @ X + alpha * len(y) * np.load(Q_dir)).astype("float32"),
    #                                        (X.T @ y).astype("float32"),x0=gaussian,atol=1e-03)
    # c, lower = scipy.linalg.cho_factor(X.T @ X + alpha * len(y) * np.load(Q_dir), lower=True, overwrite_a=True, check_finite=False)
    print(Q_dir,np.std(np.load(Q_dir)))
    # gaussian = scipy.linalg.cho_solve(scipy.linalg.cho_factor(X.T @ X + alpha * len(y) * np.load(Q_dir), lower=True, overwrite_a=True, check_finite=False),
    #                                    X.T @ y, overwrite_b=True, check_finite=False)
    # gaussian=np.load(Q_dir)@linear_model.Ridge(alpha=alpha * len(y),fit_intercept=False,solver='cholesky').fit(X@np.load(Q_dir),y).coef_
    gaussian=generalized_ridge_regression_transformed(X, y, np.load(Q_dir),alpha * len(y))
    # print(gaussian.shape)
    # gaussian=regularizing_ridge(X,y,alpha,Q_dir)
    # time1=time.time()-start
    # print("norm",time1,time2)
    df["regression"]=X@gaussian
    df["R1_contribution_steric"]=X1[:,:len(df_coord)]@gaussian[:len(df_coord)]
    df["R2_contribution_steric"]=X2[:,:len(df_coord)]@gaussian[:len(df_coord)]
    df["R1_contribution_electric"]=X1[:,len(df_coord):]@gaussian[len(df_coord):]
    df["R2_contribution_electric"]=X2[:,len(df_coord):]@gaussian[len(df_coord):]
    df_coord["coef_steric"]=gaussian[:len(df_coord)]
    df_coord["coef_electric"]=gaussian[len(df_coord):]
    df_coord.to_csv( save_name + "_coef.csv", index=False)
    result=[]
    for repeat in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=repeat)
        predict=[]
        sort_index=[]
        for train_index, test_index in kf.split(X):
            # X_train, X_test = X[train_index], X[test_index]
            # X_train, X_test = X[train_index]/np.sqrt(np.average(X[train_index]**2)), X[test_index]/np.sqrt(np.average(X[train_index]**2))

            # c, lower = scipy.linalg.cho_factor(X_train.T @ X_train + alpha * len(train_index) * np.load(Q_dir), lower=True, overwrite_a=True, check_finite=False)
            # predict_cv=X[test_index]@scipy.linalg.cho_solve(scipy.linalg.cho_factor(X[train_index].T @ X[train_index] + alpha * len(train_index) * np.load(Q_dir), lower=True, overwrite_a=True, check_finite=False),
            #                                           X[train_index].T @ y[train_index], overwrite_b=True, check_finite=False)
            # predict_cv=X[test_index]@np.load(Q_dir)@linear_model.Ridge(alpha=alpha * len(train_index),fit_intercept=False,solver='cholesky').fit(X[train_index]@np.load(Q_dir),y[train_index]).coef_
            predict_cv=X[test_index]@generalized_ridge_regression_transformed(X[train_index], y[train_index], np.load(Q_dir),alpha * len(train_index))
            # predict_cv=X_test@scipy.linalg.solve((X_train.T @ X_train + alpha * len(train_index) * np.load(Q_dir)).astype("float32"),
            #                                (X_train.T @ y[train_index]).astype("float32"), assume_a="pos")
            # gaussian__, info=scipy.sparse.linalg.cg((X_train.T @ X_train + alpha * len(train_index) * np.load(Q_dir)).astype("float32"),
            #                                 (X_train.T @ y[train_index]).astype("float32"),x0=gaussian,atol=1e-03)
            predict.extend(predict_cv.tolist())
            sort_index.extend(test_index.tolist())
        predict=unshuffle_array(np.clip(predict, np.min(y), np.max(y)),sort_index)
        RMSE=mean_squared_error(y,predict,squared=False)
        r2=r2_score(y,predict)
        result.append([RMSE,r2])
        df["validation{}".format(repeat)]=predict

    split = np.empty(y.shape[0], dtype=np.uint8)
    predict=[]
    sort_index=[]
    for _,(train_index, test_index) in enumerate(split_data_pca(X,n_splits)):
        split[test_index]=_
        # X_train, X_test = X[train_index], X[test_index]
        # X_train, X_test = X[train_index]/np.sqrt(np.average(X[train_index]**2)), X[test_index]/np.sqrt(np.average(X[train_index]**2))
        # predict_cv=X_test@scipy.linalg.solve((X_train.T @ X_train + alpha * len(train_index) * np.load(Q_dir)).astype("float32"),
        #                                 (X_train.T @ y[train_index]).astype("float32"), assume_a="pos")
        # c, lower = scipy.linalg.cho_factor(X[train_index].T @ X[train_index] + alpha * len(train_index) * np.load(Q_dir), lower=True, overwrite_a=True, check_finite=False)
        # predict_cv=X[test_index]@scipy.linalg.cho_solve(scipy.linalg.cho_factor(X[train_index].T @ X[train_index] + alpha * len(train_index) * np.load(Q_dir), lower=True, overwrite_a=True, check_finite=False),
        #                                                  X[train_index].T @ y[train_index], overwrite_b=True, check_finite=False)
        # predict_cv=X[test_index]@np.load(Q_dir)@linear_model.Ridge(alpha=alpha * len(train_index),fit_intercept=False,solver='cholesky').fit(X[train_index]@np.load(Q_dir),y[train_index]).coef_
        predict_cv=X[test_index]@generalized_ridge_regression_transformed(X[train_index], y[train_index], np.load(Q_dir),alpha * len(train_index))
        predict.extend(predict_cv.tolist())
        sort_index.extend(test_index.tolist())
    predict=unshuffle_array(np.clip(predict, np.min(y), np.max(y)),sort_index)
    RMSE_PCA=mean_squared_error(y,predict,squared=False)
    
    df["split_PCA"]=split
    df["validation_PCA"]=predict
    PandasTools.AddMoleculeColumnToFrame(df, "smiles")
    PandasTools.SaveXlsxFromFrame(df, save_name + "_prediction.xlsx", size=(100, 100))
    r2_PCA=r2_score(y,predict)
    RMSE=mean_squared_error(y,X@gaussian,squared=False)
    r2=r2_score(y,df["regression"].values)
    integ=np.sum(df_coord["coef_steric"].values)
    moment=np.sum(df_coord["coef_steric"].values.reshape(-1,1)*df_coord[["x","y","z"]].values,axis=0)/integ
    return [save_name,alpha,Q,RMSE,r2]+np.average(result,axis=0).tolist()+[RMSE_PCA,r2_PCA]+[integ]+moment.tolist()

# def Gaussian_(input):
#     X,X1,X2,y,alpha,Q,Q_dir,n_splits,n_repeats,df,df_coord,save_name=input
#     # gaussian = scipy.linalg.cho_solve(scipy.linalg.cho_factor(X.T @ X + alpha * len(y) * np.load(Q_dir), lower=True, overwrite_a=True, check_finite=False),
#     #                                    X.T @ y, overwrite_b=True, check_finite=False)
#     # df["regression"]=X@gaussian
#     # df["R1_contribution_steric"]=X1[:,:len(df_coord)]@gaussian[:len(df_coord)]
#     # df["R2_contribution_steric"]=X2[:,:len(df_coord)]@gaussian[:len(df_coord)]
#     # df["R1_contribution_electric"]=X1[:,len(df_coord):]@gaussian[len(df_coord):]
#     # df["R2_contribution_electric"]=X2[:,len(df_coord):]@gaussian[len(df_coord):]
#     # df_coord["coef_steric"]=gaussian[:len(df_coord)]
#     # df_coord["coef_electric"]=gaussian[len(df_coord):]
#     # df_coord.to_csv( save_name + "_coef.csv", index=False)
#     df_coord=pd.read_csv(save_name + "_coef.csv")
#     df=pd.read_excel(save_name + "_prediction.xlsx")
#     result=[]
#     for repeat in range(n_repeats):
#         # kf = KFold(n_splits=n_splits, shuffle=True, random_state=repeat)
#         # predict=[]
#         # sort_index=[]
#         # for train_index, test_index in kf.split(X):
#         #     # X_train, X_test = X[train_index], X[test_index]
#         #     # X_train, X_test = X[train_index]/np.sqrt(np.average(X[train_index]**2)), X[test_index]/np.sqrt(np.average(X[train_index]**2))

#         #     # c, lower = scipy.linalg.cho_factor(X_train.T @ X_train + alpha * len(train_index) * np.load(Q_dir), lower=True, overwrite_a=True, check_finite=False)
#         #     predict_cv=X[test_index]@scipy.linalg.cho_solve(scipy.linalg.cho_factor(X[train_index].T @ X[train_index] + alpha * len(train_index) * np.load(Q_dir), lower=True, overwrite_a=True, check_finite=False),
#         #                                               X[train_index].T @ y[train_index], overwrite_b=True, check_finite=False)

#         #     # predict_cv=X_test@scipy.linalg.solve((X_train.T @ X_train + alpha * len(train_index) * np.load(Q_dir)).astype("float32"),
#         #     #                                (X_train.T @ y[train_index]).astype("float32"), assume_a="pos")
#         #     # gaussian__, info=scipy.sparse.linalg.cg((X_train.T @ X_train + alpha * len(train_index) * np.load(Q_dir)).astype("float32"),
#         #     #                                 (X_train.T @ y[train_index]).astype("float32"),x0=gaussian,atol=1e-03)
#         #     predict.extend(predict_cv.tolist())
#         #     sort_index.extend(test_index.tolist())
#         predict=df["validation{}".format(repeat)].values
#         RMSE=mean_squared_error(y,predict,squared=False)
#         r2=r2_score(y,predict)
#         result.append([RMSE,r2])

#     # split = np.empty(y.shape[0], dtype=np.uint8)
#     # predict=[]
#     # sort_index=[]
#     # for _,(train_index, test_index) in enumerate(split_data_pca(X,n_splits)):
#     #     split[test_index]=_
#     #     # X_train, X_test = X[train_index], X[test_index]
#     #     # X_train, X_test = X[train_index]/np.sqrt(np.average(X[train_index]**2)), X[test_index]/np.sqrt(np.average(X[train_index]**2))
#     #     # predict_cv=X_test@scipy.linalg.solve((X_train.T @ X_train + alpha * len(train_index) * np.load(Q_dir)).astype("float32"),
#     #     #                                 (X_train.T @ y[train_index]).astype("float32"), assume_a="pos")
#     #     # c, lower = scipy.linalg.cho_factor(X[train_index].T @ X[train_index] + alpha * len(train_index) * np.load(Q_dir), lower=True, overwrite_a=True, check_finite=False)
#     #     predict_cv=X[test_index]@scipy.linalg.cho_solve(scipy.linalg.cho_factor(X[train_index].T @ X[train_index] + alpha * len(train_index) * np.load(Q_dir), lower=True, overwrite_a=True, check_finite=False),
#     #                                                      X[train_index].T @ y[train_index], overwrite_b=True, check_finite=False)
#     #     predict.extend(predict_cv.tolist())
#     #     sort_index.extend(test_index.tolist())
#     # predict=unshuffle_array(np.clip(predict, np.min(y), np.max(y)),sort_index)
#     RMSE_PCA=mean_squared_error(y,df["validation_PCA"],squared=False)
    
#     # df["split_PCA"]=split
#     # df["validation_PCA"]=predict
#     # PandasTools.AddMoleculeColumnToFrame(df, "smiles")
#     # PandasTools.SaveXlsxFromFrame(df, save_name + "_prediction.xlsx", size=(100, 100))
#     r2_PCA=r2_score(y,df["validation_PCA"].values)
#     RMSE=mean_squared_error(y,df["regression"],squared=False)
#     r2=r2_score(y,df["regression"].values)
#     integ=np.sum(df_coord["coef_steric"].values)
#     moment=np.sum(df_coord["coef_steric"].values.reshape(-1,1)*df_coord[["x","y","z"]].values,axis=0)/integ
#     return [save_name,alpha,Q,RMSE,r2]+np.average(result,axis=0).tolist()+[RMSE_PCA,r2_PCA]+[integ]+moment.tolist()

def Ridge(input):
    X,X1,X2,y,alpha,n_splits,n_repeats,df,df_coord,save_name=input
    ridge = linear_model.Ridge(alpha=alpha * len(y), fit_intercept=False).fit(X, y)
    df["regression"]=ridge.predict(X)
    # df["R1_contribution"]=ridge.predict(X1)
    # df["R2_contribution"]=ridge.predict(X2)
    df["R1_contribution_steric"]=X1[:,:len(df_coord)]@ridge.coef_[:len(df_coord)]
    df["R2_contribution_steric"]=X2[:,:len(df_coord)]@ridge.coef_[:len(df_coord)]
    df["R1_contribution_electric"]=X1[:,len(df_coord):]@ridge.coef_[len(df_coord):]
    df["R2_contribution_electric"]=X2[:,len(df_coord):]@ridge.coef_[len(df_coord):]
    df_coord["coef_steric"]=ridge.coef_[:len(df_coord)]
    df_coord["coef_electric"]=ridge.coef_[len(df_coord):]
    df_coord.to_csv( save_name + "_coef.csv", index=False)
    result=[]
    for repeat in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=repeat)
        predict=[]
        sort_index=[]
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            # X_train, X_test = X[train_index]/np.sqrt(np.average(X[train_index]**2)), X[test_index]/np.sqrt(np.average(X[train_index]**2))
            predict_cv = linear_model.Ridge(alpha=alpha * len(train_index), fit_intercept=False).fit(X_train, y[train_index]).predict(X_test)
            predict.extend(predict_cv.tolist())
            sort_index.extend(test_index.tolist())
        predict=unshuffle_array(np.clip(predict, np.min(y), np.max(y)),sort_index)
        RMSE=mean_squared_error(y,predict,squared=False)
        r2=r2_score(y,predict)
        result.append([RMSE,r2])
        df["validation{}".format(repeat)]=predict
    
    split = np.empty(y.shape[0], dtype=np.uint8)
    predict=[]
    sort_index=[]
    for _,(train_index, test_index) in enumerate(split_data_pca(X,n_splits)):
        split[test_index]=_
        X_train, X_test = X[train_index], X[test_index]
        # X_train, X_test = X[train_index]/np.sqrt(np.average(X[train_index]**2)), X[test_index]/np.sqrt(np.average(X[train_index]**2))
        predict_cv = linear_model.Ridge(alpha=alpha * len(train_index), fit_intercept=False).fit(X_train, y[train_index]).predict(X_test)
        predict.extend(predict_cv.tolist())
        sort_index.extend(test_index.tolist())
    predict=unshuffle_array(np.clip(predict, np.min(y), np.max(y)),sort_index)
    RMSE_PCA=mean_squared_error(y,predict,squared=False)
    r2_PCA=r2_score(y,predict)
    df["split_PCA"]=split
    df["validation_PCA"]=predict
    PandasTools.AddMoleculeColumnToFrame(df, "smiles")
    PandasTools.SaveXlsxFromFrame(df, save_name + "_prediction.xlsx", size=(100, 100))
    RMSE=mean_squared_error(y,ridge.predict(X),squared=False)
    r2=r2_score(y,ridge.predict(X))
    return [save_name,alpha,RMSE,r2]+np.average(result,axis=0).tolist()+[RMSE_PCA,r2_PCA]

def PLS(input):
    X,X1,X2,y,alpha,n_splits,n_repeats,df,df_coord,save_name=input
    pls = PLSRegression(n_components=alpha).fit(X, y)
    df["regression"]=pls.predict(X)
    # df["R1_contribution"]=pls.predict(X1)
    # df["R2_contribution"]=pls.predict(X2)
    df["R1_contribution_steric"]=X1[:,:len(df_coord)]@pls.coef_[0][:len(df_coord)]
    df["R2_contribution_steric"]=X2[:,:len(df_coord)]@pls.coef_[0][:len(df_coord)]
    df["R1_contribution_electric"]=X1[:,len(df_coord):]@pls.coef_[0][len(df_coord):]
    df["R2_contribution_electric"]=X2[:,len(df_coord):]@pls.coef_[0][len(df_coord):]
    # df_coord["coef"]=pls.coef_[0][:len(df_coord)]
    df_coord["coef_steric"]=pls.coef_[0][:len(df_coord)]
    df_coord["coef_electric"]=pls.coef_[0][len(df_coord):]
    df_coord.to_csv( save_name + "_coef.csv", index=False)
    result=[]
    for repeat in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=repeat)
        predict=[]
        sort_index=[]
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            # X_train, X_test = X[train_index]/np.sqrt(np.average(X[train_index]**2)), X[test_index]/np.sqrt(np.average(X[train_index]**2))
            predict_cv = PLSRegression(n_components=alpha).fit(X_train, y[train_index]).predict(X_test)
            predict.extend(predict_cv.tolist())
            sort_index.extend(test_index.tolist())
        predict=unshuffle_array(np.clip(predict, np.min(y), np.max(y)),sort_index)
        RMSE=mean_squared_error(y,predict,squared=False)
        r2=r2_score(y,predict)
        result.append([RMSE,r2])
        df["validation{}".format(repeat)]=predict
    
    split = np.empty(y.shape[0], dtype=np.uint8)
    predict=[]
    sort_index=[]
    for _,(train_index, test_index) in enumerate(split_data_pca(X,n_splits)):
        split[test_index]=_
        X_train, X_test = X[train_index], X[test_index]
        # X_train, X_test = X[train_index]/np.sqrt(np.average(X[train_index]**2)), X[test_index]/np.sqrt(np.average(X[train_index]**2))
        predict_cv = PLSRegression(n_components=alpha).fit(X_train, y[train_index]).predict(X_test)
        predict.extend(predict_cv.tolist())
        sort_index.extend(test_index.tolist())
    predict=unshuffle_array(np.clip(predict, np.min(y), np.max(y)),sort_index)
    RMSE_PCA=mean_squared_error(y,predict,squared=False)
    r2_PCA=r2_score(y,predict)
    df["split_PCA"]=split
    df["validation_PCA"]=predict

    PandasTools.AddMoleculeColumnToFrame(df, "smiles")
    PandasTools.SaveXlsxFromFrame(df, save_name + "_prediction.xlsx", size=(100, 100))
    RMSE=mean_squared_error(y,pls.predict(X),squared=False)
    r2=r2_score(y,pls.predict(X))
    return [save_name,alpha,RMSE,r2]+np.average(result,axis=0).tolist()+[RMSE_PCA,r2_PCA]

def Lasso(input):
    X,X1,X2,y,alpha,n_splits,n_repeats,df,df_coord,save_name=input
    lasso = linear_model.Lasso(alpha=alpha /2, fit_intercept=False).fit(X, y)
    df["regression"]=lasso.predict(X)
    # df["R1_contribution"]=lasso.predict(X1)
    # df["R2_contribution"]=lasso.predict(X2)
    df["R1_contribution_steric"]=X1[:,:len(df_coord)]@lasso.coef_[:len(df_coord)]
    df["R2_contribution_steric"]=X2[:,:len(df_coord)]@lasso.coef_[:len(df_coord)]
    df["R1_contribution_electric"]=X1[:,len(df_coord):]@lasso.coef_[len(df_coord):]
    df["R2_contribution_electric"]=X2[:,len(df_coord):]@lasso.coef_[len(df_coord):]
    # df_coord["coef"]=lasso.coef_[:len(df_coord)]
    df_coord["coef_steric"]=lasso.coef_[:len(df_coord)]
    df_coord["coef_electric"]=lasso.coef_[len(df_coord):]
    df_coord.to_csv( save_name + "_coef.csv", index=False)
    result=[]
    for repeat in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=repeat)
        predict=[]
        sort_index=[]
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            # X_train, X_test = X[train_index]/np.sqrt(np.average(X[train_index]**2)), X[test_index]/np.sqrt(np.average(X[train_index]**2))
            predict_cv = linear_model.Lasso(alpha=alpha/2, fit_intercept=False).fit(X_train, y[train_index]).predict(X_test)
            predict.extend(predict_cv.tolist())
            sort_index.extend(test_index.tolist())
        predict=unshuffle_array(np.clip(predict, np.min(y), np.max(y)),sort_index)
        RMSE=mean_squared_error(y,predict,squared=False)
        r2=r2_score(y,predict)
        result.append([RMSE,r2])
        df["validation{}".format(repeat)]=predict
    
    split = np.empty(y.shape[0], dtype=np.uint8)
    predict=[]
    sort_index=[]
    for _,(train_index, test_index) in enumerate(split_data_pca(X,n_splits)):
        split[test_index]=_
        X_train, X_test = X[train_index], X[test_index]
        # X_train, X_test = X[train_index]/np.sqrt(np.average(X[train_index]**2)), X[test_index]/np.sqrt(np.average(X[train_index]**2))
        predict_cv = linear_model.Lasso(alpha=alpha/2, fit_intercept=False).fit(X_train, y[train_index]).predict(X_test)
        predict.extend(predict_cv.tolist())
        sort_index.extend(test_index.tolist())
    predict=unshuffle_array(np.clip(predict, np.min(y), np.max(y)),sort_index)

    RMSE_PCA=mean_squared_error(y,predict,squared=False)
    r2_PCA=r2_score(y,predict)
    df["split_PCA"]=split
    df["validation_PCA"]=predict
    PandasTools.AddMoleculeColumnToFrame(df, "smiles")
    PandasTools.SaveXlsxFromFrame(df, save_name + "_prediction.xlsx", size=(100, 100))
    RMSE=mean_squared_error(y,lasso.predict(X),squared=False)
    r2=r2_score(y,lasso.predict(X))
    return [save_name,alpha,RMSE,r2]+np.average(result,axis=0).tolist()+[RMSE_PCA,r2_PCA]

def energy_to_Boltzmann_distribution(mol, RT=1.99e-3 * 273):
    energies = []
    for conf in mol.GetConformers():
        # print(conf.GetId())
        line = json.loads(conf.GetProp("freq"))
        energies.append(float(line[0] - line[1] * RT / 1.99e-3))
    energies = np.array(energies)
    energies = energies - np.min(energies)
    rates = np.exp(-energies / RT)
    rates = rates / np.sum(rates)
    for conf, rate in zip(mol.GetConformers(), rates):
        conf.SetProp("Boltzmann_distribution", str(rate))


def is_normal_frequencies(filename):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            frequencies_lines = [line for line in lines if 'Frequencies' in line]
            if len(frequencies_lines)==0:
                f.close()
                print(filename, False)
                return False
            for l in frequencies_lines:
                splited = l.split()
                values = splited[2:]
                values = [float(v)>0 for v in values]
                ans=all(values)
                if not ans:
                    f.close()
                    print(filename,ans)
                    return ans
            f.close()
        return True
    except:
        print(filename," is bad conformer")
        return False

def get_free_energy(mol,dir):
    # dirs_name_freq = param["freq_dir"] + "/" + mol.GetProp("InchyKey") + "/gaussianinput?.log"
    del_list=[]

    for conf in mol.GetConformers():
        path=dir + "/" + mol.GetProp("InchyKey") + "/gaussianinput{}.log".format(conf.GetId())
        if is_normal_frequencies(path):
            try:
                with open(path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if 'Sum of electronic and thermal Enthalpies=' in line:
                            ent=float(line.split()[6])* 627.5095
                        if "Total     " in line:
                            entr=float(line.split()[3])/1000
                conf.SetProp("freq", json.dumps([ent, entr]))

            except:
                print("read error,",path)
        else:
            del_list.append(conf.GetId())
    for _ in del_list:
        mol.RemoveConformer(_)

def get_grid_feat(mol,RT,dir):
    energy_to_Boltzmann_distribution(mol, RT)
    feat = []
    # ESP = []
    we = []
    for conf in mol.GetConformers():
        data = pd.read_pickle(
            "{}/{}/data{}.pkl".format(dir, mol.GetProp("InchyKey"), conf.GetId()))
        Bd = float(conf.GetProp("Boltzmann_distribution"))
        we.append(Bd)
        feat.append(data[["Dt","ESP"]].values.tolist())
        # ESP.append(data["ESP"].values.tolist())
    # Dt = np.array(Dt)
    # ESP = np.array(ESP)
    # w = np.exp(-Dt / np.sqrt(np.average(Dt ** 2, axis=0)).reshape(1, -1))
    # w = np.exp(-Dt / np.max(Dt, axis=0).reshape(1, -1))
    # w = np.exp(-Dt/np.sqrt(np.average(Dt**2)))
    # w = np.exp(-Dt)
    # print(np.max(w),np.min(w),np.average(w))
    data[["Dt","ESP"]] = np.nan_to_num(np.average(feat, weights=we, axis=0))
    # w = np.exp(ESP / np.sqrt(np.average(ESP ** 2, axis=0)).reshape(1, -1))
    # data["ESP"] = np.nan_to_num(
    #     np.average(ESP, weights=np.array(we).reshape(-1, 1) * np.ones(shape=ESP.shape), axis=0))
    data_y = data[data["y"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])

    data_y[["Dt","ESP"]] = data[data["y"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])[["Dt","ESP"]].values + \
        data[data["y"] < 0].sort_values(['x', 'y', "z"], ascending=[True, False, True])[["Dt","ESP"]].values
    data_yz = data_y[data_y["z"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])

    # data_yz["Dt"] = data_y[data_y["z"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])["Dt"].values - \
    #                 data_y[data_y["z"] < 0].sort_values(['x', 'y', "z"], ascending=[True, True, False])["Dt"].values

    data_yz[["DtR1","ESPR1"]] = data_y[data_y["z"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])[["Dt","ESP"]].values
    data_yz[["DtR2","ESPR2"]] = data_y[data_y["z"] < 0].sort_values(['x', 'y', "z"], ascending=[True, True, False])[["Dt","ESP"]].values
    data_yz[["Dt","ESP"]]=data_yz[["DtR1","ESPR1"]].values-data_yz[["DtR2","ESPR2"]].values
    dfp_yz = data_yz.copy()
    return dfp_yz[["Dt","DtR1","DtR2","ESP","ESPR1","ESPR2"]].values.T.tolist()


if __name__ == '__main__':
    # time.sleep(60*60*24*1)
    # inputs_=[]
    ridge_input=[]
    lasso_input=[]
    gaussian_input=[]
    pls_input=[]
    for param_name in sorted(glob.glob("../parameter/run/cube_to_grid0.250510.txt"),reverse=True):
        print(param_name)
        with open(param_name, "r") as f:
            param = json.loads(f.read())
        print(param)
        start = time.perf_counter()  # 計測開始
        for file in glob.glob("../arranged_dataset/*.xlsx"):
            df = pd.read_excel(file).dropna(subset=['smiles']).reset_index(drop=True)#[:10]
            file_name = os.path.splitext(os.path.basename(file))[0]
            features_dir_name = param["grid_coordinates"] + file_name
            print(len(df),features_dir_name)
            df["mol"] = df["smiles"].apply(calculate_conformation.get_mol)
            df = df[
                [len(glob.glob("{}/{}/*".format(param["grid_coordinates"], mol.GetProp("InchyKey"))))>0 for mol in
                 df["mol"]]]
            print(len(df),features_dir_name)
            df["mol"].apply(
                lambda mol: calculate_conformation.read_xyz(mol,
                                                            param["opt_structure"] + "/" + mol.GetProp("InchyKey")))
            df["mol"].apply(lambda mol : get_free_energy(mol,param["freq_dir"]))
            df=df[[mol.GetNumConformers()>0 for mol in df["mol"]]]
            print(len(df),features_dir_name)
            df[["Dt","DtR1","DtR2","ESP","ESPR1","ESPR2"]]=df[["mol", "RT"]].apply(lambda _: get_grid_feat(_[0], _[1],param["grid_coordinates"]), axis=1, result_type='expand')
            print("feature_calculated")

            X=np.array(df["Dt"].values.tolist())/np.sqrt(np.average(np.array(df["Dt"].values.tolist())**2))
            X1=np.array(df["DtR1"].values.tolist())/np.sqrt(np.average(np.array(df["Dt"].values.tolist())**2))
            X2=np.array(df["DtR2"].values.tolist())/np.sqrt(np.average(np.array(df["Dt"].values.tolist())**2))

            X_esp=np.array(df["ESP"].values.tolist())/np.sqrt(np.average(np.array(df["ESP"].values.tolist())**2))
            X1_esp=np.array(df["ESPR1"].values.tolist())/np.sqrt(np.average(np.array(df["ESP"].values.tolist())**2))
            X2_esp=np.array(df["ESPR2"].values.tolist())/np.sqrt(np.average(np.array(df["ESP"].values.tolist())**2))
            X=np.concatenate([X,X_esp],1).astype("float32")
            X1=np.concatenate([X1,X1_esp],1).astype("float32")
            X2=np.concatenate([X2,X2_esp],1).astype("float32")
            # X = np.concatenate(features_all / std, axis=1)
            print(X.dtype)
            y=df["ΔΔG.expt."].values.astype("float32")
            # 正則化パラメータの候補
            alphas = np.logspace(0,10,11,base=2)
            df_coord = pd.read_csv(param["grid_coordinates"] + "/coordinates_yz.csv").sort_values(['x', 'y', "z"],
                                                                                  ascending=[True, True, True])
            # 5分割交差検証を10回実施
            n_splits = int(param["n_splits"])
            n_repeats = int(param["n_repeats"])
            dir=param["out_dir_name"] + "/{}".format(file_name)
            os.makedirs(dir,exist_ok=True)

            for alpha in alphas:
                save_name=dir + "/Ridge_alpha_{}".format(alpha)
                ridge_input.append([X,X1,X2,y,alpha,n_splits,n_repeats,df,df_coord,save_name])
            Qs=param["sigmas"]
            for alpha in alphas:
                for Q in Qs:
                    save_name=dir + "/Gaussian_alpha_{}_sigma_{}".format(alpha,Q)
                    Q_dir=param["grid_coordinates"]+ "/1eig{}.npy".format(Q)
                    gaussian_input.append([X,X1,X2,y,alpha,Q,Q_dir,n_splits,n_repeats,df,df_coord,save_name])
            alphas = np.logspace(-5,5,11,base=2)
            for alpha in alphas:
                save_name=dir + "/Lasso_alpha_{}".format(alpha)
                lasso_input.append([X,X1,X2,y,alpha,n_splits,n_repeats,df,df_coord,save_name])
            alphas = np.arange(1,12)
            for alpha in alphas:
                save_name=dir + "/PLS_alpha_{}".format(alpha)
                pls_input.append([X,X1,X2,y,alpha,n_splits,n_repeats,df,df_coord,save_name])
    p = multiprocessing.Pool(processes=int(param["processes"]))
    columns=["savefilename","alpha","RMSE_regression", "r2_regression","RMSE_validation", "r2_validation"]
    decimals = {"RMSE_regression": 5,"r2_regression":5,"RMSE_validation":5, "r2_validation":5}
    pd.DataFrame(p.map(Gaussian,gaussian_input), columns=["savefilename","alpha","sigma","RMSE_regression", "r2_regression","RMSE_validation", "r2_validation", "RMSE_PCA_validation","r2_PCA_validation"]+["integral","x_moment","y_moment","z_moment"]).round(decimals).to_csv(param["out_dir_name"] + "/Gaussian.csv")
    pd.DataFrame(p.map(Ridge,ridge_input), columns=columns+["RMSE_PCA_validation","r2_PCA_validation"]).round(decimals).to_csv(param["out_dir_name"] + "/Ridge.csv")
    pd.DataFrame(p.map(Lasso,lasso_input), columns=columns+["RMSE_PCA_validation","r2_PCA_validation"]).round(decimals).to_csv(param["out_dir_name"] + "/Lasso.csv")
    pd.DataFrame(p.map(PLS,pls_input), columns=columns+["RMSE_PCA_validation","r2_PCA_validation"]).round(decimals).to_csv(param["out_dir_name"] + "/PLS.csv")
    end = time.perf_counter()  # 計測終了
    print('Finish{:.2f}'.format(end - start))