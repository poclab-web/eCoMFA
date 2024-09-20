import glob
import json
import multiprocessing
import os
import time
import warnings
import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
from sklearn import linear_model
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
import time
import calculate_conformation
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
warnings.simplefilter('ignore')
# シャッフル前の順序に戻す関数
def unshuffle_array(shuffled_array, shuffled_indices):
    original_array = np.empty_like(shuffled_array)
    original_array[shuffled_indices] = shuffled_array
    return original_array

# def split_data_pca(X, n_splits=5):
#     # PCAを使用してデータを主成分に投影
#     pca = PCA(n_components=1)
#     X_pca = pca.fit_transform(X).flatten()
    
#     # 主成分に基づいてデータを分割
#     percentiles = np.percentile(X_pca, np.linspace(0, 100, n_splits + 1))
#     indices=[]
#     for _ in range(n_splits):
#         if _==n_splits-1:
#             test=np.where((X_pca >= percentiles[_]) & (X_pca <= percentiles[_+1]))[0]
#             train=np.where((X_pca < percentiles[_]) | (X_pca > percentiles[_+1]))[0]
#         else:
#             test=np.where((X_pca >= percentiles[_]) & (X_pca < percentiles[_+1]))[0]
#             train=np.where((X_pca < percentiles[_]) | (X_pca >= percentiles[_+1]))[0]
#         indices.append([train,test])
#     return indices


# def split_data_kmeans(X, n_splits=5, n_clusters=5):
#     # Apply KMeans to cluster the data into different groups
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     clusters = kmeans.fit_predict(X)
    
#     # Initialize an empty list to hold the indices for each split
#     indices = [[] for _ in range(n_splits)]
    
#     # Distribute data points from each cluster into different splits
#     for cluster in range(n_clusters):
#         cluster_indices = np.where(clusters == cluster)[0]
#         np.random.shuffle(cluster_indices)  # Shuffle to ensure randomness
        
#         # Split the cluster indices into n_splits parts
#         split_size = len(cluster_indices) // n_splits
#         for i in range(n_splits):
#             start = i * split_size
#             end = (i + 1) * split_size if i != n_splits - 1 else len(cluster_indices)
#             indices[i].extend(cluster_indices[start:end])
    
#     # Convert the list of indices to arrays and create train/test splits
#     result_indices = []
#     for i in range(n_splits):
#         test_indices = np.array(indices[i])
#         train_indices = np.array([idx for j in range(n_splits) if j != i for idx in indices[j]])
#         train_indices = np.array([idx for j in range(n_splits) if (j+1)%n_splits == i for idx in indices[j]])[0:5]
#         result_indices.append([train_indices, test_indices])
#     return result_indices

# Example usage
# X = your_data_matrix
# splits = split_data_kmeans(X, n_splits=5, n_clusters=5)
# for i, (train, test) in enumerate(splits):
#     print(f"Split {i}:")
#     print(f"  Train indices: {train}")
#     print(f"  Test indices: {test}")


# Example usage
# X = your_data_matrix
# splits = split_data_kmeans(X, n_splits=5, n_clusters=5)
# for i, (train, test) in enumerate(splits):
#     print(f"Split {i}:")
#     print(f"  Train indices: {train}")
#     print(f"  Test indices: {test}")


# def regularizing_ridge(X, y, alpha, Q_dir):
#     # Calculate X.T @ X and store the result directly in the same variable if possible
#     XT_X = X.T @ X
#     XT_X += alpha * len(y) * np.load(Q_dir)  # In-place modification of XT_X to save memory
#     # Perform Cholesky factorization in place
#     c, lower = scipy.linalg.cho_factor(XT_X, lower=True, overwrite_a=True, check_finite=False)
#     # Solve the system using the Cholesky factorization result
#     result = scipy.linalg.cho_solve((c, lower), X.T @ y, overwrite_b=True, check_finite=False)
#     return result

# def generalized_ridge_regression_transformed(X, y, P,L):
#     X= np.block([X[:,:X.shape[1]//3]@P[0],X[:,X.shape[1]//3:X.shape[1]//3*2]@P[1],X[:,X.shape[1]//3*2:X.shape[1]//3*3]@P[2]])
#     beta_hat=linear_model.Ridge(alpha=L,fit_intercept=False,solver='cholesky').fit(X,y).coef_
#     beta_hat= np.block([P[0]@beta_hat[:beta_hat.shape[0]//3], P[1]@beta_hat[beta_hat.shape[0]//3:beta_hat.shape[0]//3*2], P[2]@beta_hat[beta_hat.shape[0]//3*2:beta_hat.shape[0]//3*3]])
#     return beta_hat

# def Gaussian(input):
#     X,X1,X2,y,alpha,Q,Q_dir,n_splits,n_repeats,df,df_coord,save_name=input
#     # start=time.time()
#     # gaussian=scipy.linalg.solve(X.T @ X + alpha * len(y) * np.load(Q_dir),
#     #                                        X.T @ y, assume_a="pos")
#     # time2=time.time()-start
#     # start=time.time()
#     # gaussian_, info=scipy.sparse.linalg.cg((X.T @ X + alpha * len(y) * np.load(Q_dir)).astype("float32"),
#     #                                        (X.T @ y).astype("float32"),x0=gaussian,atol=1e-03)
#     # c, lower = scipy.linalg.cho_factor(X.T @ X + alpha * len(y) * np.load(Q_dir), lower=True, overwrite_a=True, check_finite=False)
#     # print(Q_dir,np.std(np.load(Q_dir)))
#     # gaussian = scipy.linalg.cho_solve(scipy.linalg.cho_factor(X.T @ X + alpha * len(y) * np.load(Q_dir), lower=True, overwrite_a=True, check_finite=False),
#     #                                    X.T @ y, overwrite_b=True, check_finite=False)
#     # gaussian=np.load(Q_dir)@linear_model.Ridge(alpha=alpha * len(y),fit_intercept=False,solver='cholesky').fit(X@np.load(Q_dir),y).coef_
#     # P=np.load(Q_dir+"/eigenvectors.npy")
#     # P=P/np.sqrt(np.load(Q_dir+"/eigenvalues.npy")+Q)
#     # P=[np.load(f"{Q_dir}/1eig{Q:.2f}.npy"),np.eye(X.shape[1]//3),np.eye(X.shape[1]//3)]
#     P=[np.load(f"{Q_dir}/1eig{Q:.2f}.npy"),np.load(f"{Q_dir}/1eig{Q:.2f}.npy"),np.load(f"{Q_dir}/1eig{Q:.2f}.npy")]
#     # from scipy.linalg import eigh
#     # P=[]
#     # for i in range(3):
#     #     _=X.shape[1]//3
#     #     eigenvalues,eigenvectors=eigh(X[:,_*i:_*(i+1)].T@X[:,_*i:_*(i+1)]+ np.eye(_) * 1e-1, overwrite_a=True,check_finite=False)
#     #     P_=eigenvectors*np.sqrt(eigenvalues)
#     #     P.append(P_)

#     # P=np.identity(P.shape[0])
#     # print(np.isnan(P).any())
#     # print(P)
#     gaussian=generalized_ridge_regression_transformed(X, y, P,alpha * len(y))
#     # print(gaussian.shape)
#     # gaussian=regularizing_ridge(X,y,alpha,Q_dir)
#     # time1=time.time()-start
#     # print("norm",time1,time2)
#     df["regression"]=X@gaussian
#     df["R1_contribution_steric"]=X1[:,:len(df_coord)]@gaussian[:len(df_coord)]
#     df["R2_contribution_steric"]=X2[:,:len(df_coord)]@gaussian[:len(df_coord)]
#     df["R1_contribution_electric"]=X1[:,len(df_coord):2*len(df_coord)]@gaussian[len(df_coord):2*len(df_coord)]
#     df["R2_contribution_electric"]=X2[:,len(df_coord):2*len(df_coord)]@gaussian[len(df_coord):2*len(df_coord)]
#     df["R1_contribution_LUMO"]=X1[:,2*len(df_coord):3*len(df_coord)]@gaussian[2*len(df_coord):3*len(df_coord)]
#     df["R2_contribution_LUMO"]=X2[:,2*len(df_coord):3*len(df_coord)]@gaussian[2*len(df_coord):3*len(df_coord)]
#     df_coord["coef_steric"]=gaussian[:len(df_coord)]
#     df_coord["coef_electric"]=gaussian[len(df_coord):2*len(df_coord)]
#     df_coord["coef_LUMO"]=gaussian[len(df_coord):2*len(df_coord)]
    
#     df_coord.to_csv( save_name + "_coef.csv", index=False)
#     result=[]
#     for repeat in range(n_repeats):
#         kf = KFold(n_splits=n_splits, shuffle=True, random_state=repeat)
#         predict=[]
#         sort_index=[]
#         for train_index, test_index in kf.split(X):
#             # X_train, X_test = X[train_index], X[test_index]
#             # X_train, X_test = X[train_index]/np.sqrt(np.average(X[train_index]**2)), X[test_index]/np.sqrt(np.average(X[train_index]**2))
#             # c, lower = scipy.linalg.cho_factor(X_train.T @ X_train + alpha * len(train_index) * np.load(Q_dir), lower=True, overwrite_a=True, check_finite=False)
#             # predict_cv=X[test_index]@scipy.linalg.cho_solve(scipy.linalg.cho_factor(X[train_index].T @ X[train_index] + alpha * len(train_index) * np.load(Q_dir), lower=True, overwrite_a=True, check_finite=False),
#             #                                           X[train_index].T @ y[train_index], overwrite_b=True, check_finite=False)
#             # predict_cv=X[test_index]@np.load(Q_dir)@linear_model.Ridge(alpha=alpha * len(train_index),fit_intercept=False,solver='cholesky').fit(X[train_index]@np.load(Q_dir),y[train_index]).coef_
#             predict_cv=X[test_index]@generalized_ridge_regression_transformed(X[train_index], y[train_index], P,alpha * len(train_index))
#             # predict_cv=X_test@scipy.linalg.solve((X_train.T @ X_train + alpha * len(train_index) * np.load(Q_dir)).astype("float32"),
#             #                                (X_train.T @ y[train_index]).astype("float32"), assume_a="pos")
#             # gaussian__, info=scipy.sparse.linalg.cg((X_train.T @ X_train + alpha * len(train_index) * np.load(Q_dir)).astype("float32"),
#             #                                 (X_train.T @ y[train_index]).astype("float32"),x0=gaussian,atol=1e-03)
#             predict.extend(predict_cv.tolist())
#             sort_index.extend(test_index.tolist())
#         predict=unshuffle_array(np.clip(predict, np.min(y), np.max(y)),sort_index)
#         RMSE=mean_squared_error(y,predict,squared=False)
#         r2=r2_score(y,predict)
#         result.append([RMSE,r2])
#         df["validation{}".format(repeat)]=predict

#     split = np.empty(y.shape[0], dtype=np.uint8)
#     predict=[]
#     sort_index=[]
#     for _,(train_index, test_index) in enumerate(split_data_kmeans(X,n_splits,n_splits)):
#         split[test_index]=_
#         # X_train, X_test = X[train_index], X[test_index]
#         # X_train, X_test = X[train_index]/np.sqrt(np.average(X[train_index]**2)), X[test_index]/np.sqrt(np.average(X[train_index]**2))
#         # predict_cv=X_test@scipy.linalg.solve((X_train.T @ X_train + alpha * len(train_index) * np.load(Q_dir)).astype("float32"),
#         #                                 (X_train.T @ y[train_index]).astype("float32"), assume_a="pos")
#         # c, lower = scipy.linalg.cho_factor(X[train_index].T @ X[train_index] + alpha * len(train_index) * np.load(Q_dir), lower=True, overwrite_a=True, check_finite=False)
#         # predict_cv=X[test_index]@scipy.linalg.cho_solve(scipy.linalg.cho_factor(X[train_index].T @ X[train_index] + alpha * len(train_index) * np.load(Q_dir), lower=True, overwrite_a=True, check_finite=False),
#         #                                                  X[train_index].T @ y[train_index], overwrite_b=True, check_finite=False)
#         # predict_cv=X[test_index]@np.load(Q_dir)@linear_model.Ridge(alpha=alpha * len(train_index),fit_intercept=False,solver='cholesky').fit(X[train_index]@np.load(Q_dir),y[train_index]).coef_
#         predict_cv=X[test_index]@generalized_ridge_regression_transformed(X[train_index], y[train_index], P,alpha * len(train_index))
#         print("len train",len(train_index))
#         predict.extend(predict_cv.tolist())
#         sort_index.extend(test_index.tolist())
#     predict=unshuffle_array(np.clip(predict, np.min(y), np.max(y)),sort_index)
#     RMSE_PCA=mean_squared_error(y,predict,squared=False)
    
#     df["split_PCA"]=split
#     df["validation_PCA"]=predict
#     df.drop(columns=['Dt', 'DtR1',"DtR2",'ESP', 'ESPR1',"ESPR2",'DUAL', 'DUALR1',"DUALR2"], inplace=True)
#     df["error"]=df["ΔΔG.expt."]-df["validation0"]
#     df = df.reindex(df['error'].abs().sort_values(ascending=False).index)

#     PandasTools.AddMoleculeColumnToFrame(df, "smiles")
#     PandasTools.SaveXlsxFromFrame(df, save_name + "_prediction.xlsx", size=(100, 100))
#     r2_PCA=r2_score(y,predict)
#     RMSE=mean_squared_error(y,X@gaussian,squared=False)
#     r2=r2_score(y,df["regression"].values)
#     integ=np.sum(df_coord[["coef_steric","coef_electric","coef_LUMO"]].values,axis=0).tolist()
#     moment=np.sum(df_coord["coef_steric"].values.reshape(-1,1)*df_coord[["x","y","z"]].values,axis=0)/integ
#     return [save_name,alpha,Q,RMSE,r2]+np.average(result,axis=0).tolist()+[RMSE_PCA,r2_PCA]+integ+moment.tolist()

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

# def Ridge(input):
#     X,y,alpha,n_splits,n_repeats,df,df_coord,save_name=input
#     ridge = linear_model.Ridge(alpha=alpha, fit_intercept=False).fit(X, y)
#     df["regression"]=ridge.predict(X)
#     # df["R1_contribution"]=ridge.predict(X1)
#     # df["R2_contribution"]=ridge.predict(X2)
#     # df["R1_contribution_steric"]=X1[:,:len(df_coord)]@ridge.coef_[:len(df_coord)]
#     # df["R2_contribution_steric"]=X2[:,:len(df_coord)]@ridge.coef_[:len(df_coord)]
#     # df["R1_contribution_electric"]=X1[:,len(df_coord):2*len(df_coord)]@ridge.coef_[len(df_coord):2*len(df_coord)]
#     # df["R2_contribution_electric"]=X2[:,len(df_coord):2*len(df_coord)]@ridge.coef_[len(df_coord):2*len(df_coord)]
#     # df["R1_contribution_LUMO"]=X1[:,2*len(df_coord):3*len(df_coord)]@ridge.coef_[2*len(df_coord):3*len(df_coord)]
#     # df["R2_contribution_LUMO"]=X2[:,2*len(df_coord):3*len(df_coord)]@ridge.coef_[2*len(df_coord):3*len(df_coord)]
#     df["steric_cont"]=X[:,:len(df_coord)]@ridge.coef_[:len(df_coord)]
#     df["electrostatic_cont"]=X[:,len(df_coord):2*len(df_coord)]@ridge.coef_[len(df_coord):2*len(df_coord)]
#     df_coord["coef_steric"]=ridge.coef_[:len(df_coord)]
#     df_coord["coef_electric"]=ridge.coef_[len(df_coord):2*len(df_coord)]
#     # df_coord["coef_LUMO"]=ridge.coef_[2*len(df_coord):3*len(df_coord)]
#     df_coord[["coef_steric","coef_electric"]]=df_coord[["coef_steric","coef_electric"]].values*df_coord[["Dt_std","ESP_std"]].values

#     df_coord.to_csv( save_name + "_coef.csv", index=False)
#     result=[]
#     for repeat in range(n_repeats):
#         kf = KFold(n_splits=n_splits, shuffle=True, random_state=repeat)
#         predict=[]
#         sort_index=[]
#         for train_index, test_index in kf.split(X):
#             # X_train, X_test = X[train_index], X[test_index]
#             # X_train, X_test = X[train_index]/np.sqrt(np.average(X[train_index]**2)), X[test_index]/np.sqrt(np.average(X[train_index]**2))
#             predict_cv = linear_model.Ridge(alpha=alpha, fit_intercept=False).fit(X[train_index], y[train_index]).predict(X[test_index])
#             predict.extend(predict_cv.tolist())
#             sort_index.extend(test_index.tolist())
#         predict=unshuffle_array(np.clip(predict, np.min(y), np.max(y)),sort_index)
#         RMSE=mean_squared_error(y,predict,squared=False)
#         r2=r2_score(y,predict)
#         result.append([RMSE,r2])
#         df["validation{}".format(repeat)]=predict
    
#     split = np.empty(y.shape[0], dtype=np.uint8)
#     predict=[]
#     sort_index=[]
#     for _,(train_index, test_index) in enumerate(split_data_pca(X,n_splits)):
#         split[test_index]=_
#         X_train, X_test = X[train_index], X[test_index]
#         # X_train, X_test = X[train_index]/np.sqrt(np.average(X[train_index]**2)), X[test_index]/np.sqrt(np.average(X[train_index]**2))
#         predict_cv = linear_model.Ridge(alpha=alpha, fit_intercept=False).fit(X_train, y[train_index]).predict(X_test)
#         predict.extend(predict_cv.tolist())
#         sort_index.extend(test_index.tolist())
#     predict=unshuffle_array(np.clip(predict, np.min(y), np.max(y)),sort_index)
#     RMSE_PCA=mean_squared_error(y,predict,squared=False)
#     r2_PCA=r2_score(y,predict)
#     df["split_PCA"]=split
#     df["validation_PCA"]=predict
#     df.drop(columns=['Dt', 'DtR1',"DtR2",'ESP', 'ESPR1',"ESPR2",'DUAL', 'DUALR1',"DUALR2"], inplace=True, errors='ignore')
#     df["error"]=df["ΔΔG.expt."]-df["validation0"]
#     df = df.reindex(df['error'].abs().sort_values(ascending=False).index)
#     PandasTools.AddMoleculeColumnToFrame(df, "smiles")
#     PandasTools.SaveXlsxFromFrame(df, save_name + "_prediction.xlsx", size=(100, 100))
#     RMSE=mean_squared_error(y,ridge.predict(X),squared=False)
#     r2=r2_score(y,ridge.predict(X))
#     integ=np.sum(np.abs(df_coord[["coef_steric","coef_electric"]].values),axis=0).tolist()
#     return [save_name,alpha,RMSE,r2]+np.average(result,axis=0).tolist()+[RMSE_PCA,r2_PCA]+integ

# def ElasticNet(input):
#     X,y,alpha,l1ratio,n_splits,n_repeats,df,df_coord,save_name=input
#     df_train=df[df["test"]]
#     ridge = linear_model.ElasticNet(alpha=alpha,l1_ratio=l1ratio, fit_intercept=False).fit(X, y)
#     df["regression"]=ridge.predict(X)
#     df["steric_cont"]=X[:,:len(df_coord)]@ridge.coef_[:len(df_coord)]
#     df["electrostatic_cont"]=X[:,len(df_coord):2*len(df_coord)]@ridge.coef_[len(df_coord):2*len(df_coord)]
#     df_coord["coef_steric"]=ridge.coef_[:len(df_coord)]
#     df_coord["coef_electric"]=ridge.coef_[len(df_coord):2*len(df_coord)]
#     # df_coord["coef_LUMO"]=ridge.coef_[2*len(df_coord):3*len(df_coord)]
#     df_coord[["coef_steric","coef_electric"]]=df_coord[["coef_steric","coef_electric"]].values*df_coord[["Dt_std","ESP_std"]].values

#     df_coord.to_csv(save_name + "_coef.csv", index=False)
#     result=[]
#     for repeat in range(n_repeats):
#         kf = KFold(n_splits=n_splits, shuffle=True, random_state=repeat)
#         predict=[]
#         sort_index=[]
#         for train_index, test_index in kf.split(X):
#             # X_train, X_test = X[train_index], X[test_index]
#             # X_train, X_test = X[train_index]/np.sqrt(np.average(X[train_index]**2)), X[test_index]/np.sqrt(np.average(X[train_index]**2))
#             predict_cv = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1ratio, fit_intercept=False).fit(X[train_index], y[train_index]).predict(X[test_index])
#             predict.extend(predict_cv.tolist())
#             sort_index.extend(test_index.tolist())
#         predict=unshuffle_array(np.clip(predict, np.min(y), np.max(y)),sort_index)
#         RMSE=mean_squared_error(y,predict,squared=False)
#         r2=r2_score(y,predict)
#         result.append([RMSE,r2])
#         df["validation{}".format(repeat)]=predict
    
#     split = np.empty(y.shape[0], dtype=np.uint8)
#     predict=[]
#     sort_index=[]
#     for _,(train_index, test_index) in enumerate(split_data_pca(X,n_splits)):
#         split[test_index]=_
#         X_train, X_test = X[train_index], X[test_index]
#         # X_train, X_test = X[train_index]/np.sqrt(np.average(X[train_index]**2)), X[test_index]/np.sqrt(np.average(X[train_index]**2))
#         predict_cv = linear_model.ElasticNet(alpha=alpha,l1_ratio=l1ratio, fit_intercept=False).fit(X_train, y[train_index]).predict(X_test)
#         predict.extend(predict_cv.tolist())
#         sort_index.extend(test_index.tolist())
#     predict=unshuffle_array(np.clip(predict, np.min(y), np.max(y)),sort_index)
#     RMSE_PCA=mean_squared_error(y,predict,squared=False)
#     r2_PCA=r2_score(y,predict)


#     df["split_PCA"]=split
#     df["validation_PCA"]=predict
    
#     df.drop(columns=['Dt', 'DtR1',"DtR2",'ESP', 'ESPR1',"ESPR2",'DUAL', 'DUALR1',"DUALR2"], inplace=True, errors='ignore')
#     df["error"]=df["ΔΔG.expt."]-df["validation0"]
#     df = df.reindex(df['error'].abs().sort_values(ascending=False).index)
#     PandasTools.AddMoleculeColumnToFrame(df, "smiles")
#     PandasTools.SaveXlsxFromFrame(df, save_name + "_prediction.xlsx", size=(100, 100))
#     RMSE=mean_squared_error(y,ridge.predict(X),squared=False)
#     r2=r2_score(y,ridge.predict(X))
#     integ=np.sum(np.abs(df_coord[["coef_steric","coef_electric"]].values),axis=0).tolist()
#     return [save_name,alpha,RMSE,r2]+np.average(result,axis=0).tolist()+[RMSE_PCA,r2_PCA]+integ

def ElasticNet(input):
    alpha,l1ratio,n_splits,n_repeats,df,df_coord,save_name=input
    df_train=df[~df["test"]]
    df_test=df[df["test"]]
    ste_train=np.array(df_train["Dt"].values.tolist())/np.sqrt(np.sum(np.array(df_train["Dt"].values.tolist())**2))
    esp_train=np.array(df_train["ESP"].values.tolist())/np.sqrt(np.sum(np.array(df_train["ESP"].values.tolist())**2))
    df_coord["Dt_std"]=np.average(np.sum(ste_train**2,axis=0))
    df_coord["ESP_std"]=np.average(np.sum(esp_train**2,axis=0))
    X_train=np.concatenate([ste_train,esp_train],1).astype("float32")
    y_train=df_train["ΔΔG.expt."].values
    model = linear_model.ElasticNet(alpha=alpha,l1_ratio=l1ratio, fit_intercept=False).fit(X_train, y_train)
    df_train["regression"]=model.predict(X_train)
    # df_train["steric_cont"]=X_train[:,:len(df_coord)]@model.coef_[:len(df_coord)]
    # df_train["electrostatic_cont"]=X_train[:,len(df_coord):2*len(df_coord)]@model.coef_[len(df_coord):2*len(df_coord)]
    df_coord["coef_steric"]=model.coef_[:len(df_coord)]
    df_coord["coef_electric"]=model.coef_[len(df_coord):2*len(df_coord)]
    df_train["steric_cont"]=ste_train@df_coord["coef_steric"].values
    df_train["electrostatic_cont"]=esp_train@df_coord["coef_electric"].values
    df_coord[["coef_steric","coef_electric"]]=df_coord[["coef_steric","coef_electric"]].values*df_coord[["Dt_std","ESP_std"]].values
    df_coord.to_csv(save_name + "_coef.csv", index=False)

    result=[]
    for repeat in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=repeat)
        predict=[]
        sort_index=[]
        for train_index, test_index in kf.split(X_train):
            predict_cv = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1ratio, fit_intercept=False).fit(X_train[train_index], y_train[train_index]).predict(X_train[test_index])
            predict.extend(predict_cv.tolist())
            sort_index.extend(test_index.tolist())
        predict=unshuffle_array(np.clip(predict, np.min(y_train), np.max(y_train)),sort_index)
        RMSE=mean_squared_error(y_train,predict,squared=False)
        r2=r2_score(y_train,predict)
        result.append([RMSE,r2])
        df_train["validation{}".format(repeat)]=predict
    df_train["error"]=df_train["validation{}".format(repeat)]-df_train["ΔΔG.expt."]
    df_train=df_train.sort_values(by='error', ascending=False)
    ste_test=np.array(df_test["Dt"].values.tolist())/np.sqrt(np.sum(np.array(df_train["Dt"].values.tolist())**2))
    esp_test=np.array(df_test["ESP"].values.tolist())/np.sqrt(np.sum(np.array(df_train["ESP"].values.tolist())**2))
    X_test=np.concatenate([ste_test,esp_test],1).astype("float32")
    df_test["prediction"]=model.predict(X_test)
    df_test["steric_cont"]=X_test[:,:len(df_coord)]@model.coef_[:len(df_coord)]
    df_test["electrostatic_cont"]=X_test[:,len(df_coord):2*len(df_coord)]@model.coef_[len(df_coord):2*len(df_coord)]
    df_test["error"]=df_test["prediction"]-df_test["ΔΔG.expt."]
    df_test=df_test.sort_values(by='error', ascending=False)
    
    df_concat = pd.concat([df_train, df_test])
    df_concat["test"]=df_concat["test"].astype(str)
    df_concat.drop(columns=['Dt', 'DtR1',"DtR2",'ESP', 'ESPR1',"ESPR2"], inplace=True, errors='ignore')
    PandasTools.AddMoleculeColumnToFrame(df_concat, "smiles")
    df_concat.to_excel(save_name + "_prediction.xlsx")
    PandasTools.SaveXlsxFromFrame(df_concat.fillna("NaN"), save_name + "_prediction.xlsx", size=(100, 100))
    RMSE_train=mean_squared_error(df_train["ΔΔG.expt."],df_train["regression"],squared=False)
    r2_train=r2_score(df_train["ΔΔG.expt."],df_train["regression"])
    RMSE_test=mean_squared_error(df_test["ΔΔG.expt."],df_test["prediction"],squared=False)
    r2_test=r2_score(df_test["ΔΔG.expt."],df_test["prediction"])
    integ=np.sum(np.abs(df_coord[["coef_steric","coef_electric"]].values),axis=0).tolist()
    return [save_name,alpha,l1ratio,RMSE_train,r2_train]+np.average(result,axis=0).tolist()+[RMSE_test,r2_test]+integ

def PLS(input):
    alpha,n_splits,n_repeats,df,df_coord,save_name=input
    df_train=df[~df["test"]]
    df_test=df[df["test"]]
    ste_train=np.array(df_train["Dt"].values.tolist())/np.sqrt(np.average(np.array(df_train["Dt"].values.tolist())**2))
    esp_train=np.array(df_train["ESP"].values.tolist())/np.sqrt(np.average(np.array(df_train["ESP"].values.tolist())**2))
    df_coord["Dt_std"]=np.sqrt(np.average(ste_train**2,axis=0))
    df_coord["ESP_std"]=np.sqrt(np.average(esp_train**2,axis=0))
    X_train=np.concatenate([ste_train,esp_train],1).astype("float32")
    y_train=df_train["ΔΔG.expt."].values
    model = PLSRegression(n_components=alpha).fit(X_train, y_train)
    df_train["regression"]=model.predict(X_train)
    df_train["steric_cont"]=X_train[:,:len(df_coord)]@model.coef_[0,:len(df_coord)]
    df_train["electrostatic_cont"]=X_train[:,len(df_coord):2*len(df_coord)]@model.coef_[0,len(df_coord):2*len(df_coord)]
    df_coord["coef_steric"]=model.coef_[0,:len(df_coord)]
    df_coord["coef_electric"]=model.coef_[0,len(df_coord):2*len(df_coord)]
    df_coord[["coef_steric","coef_electric"]]=df_coord[["coef_steric","coef_electric"]].values*df_coord[["Dt_std","ESP_std"]].values
    df_coord.to_csv(save_name + "_coef.csv", index=False)

    result=[]
    for repeat in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=repeat)
        predict=[]
        sort_index=[]
        for train_index, test_index in kf.split(X_train):
            predict_cv = PLSRegression(n_components=alpha).fit(X_train[train_index], y_train[train_index]).predict(X_train[test_index])
            predict.extend(predict_cv.tolist())
            sort_index.extend(test_index.tolist())
        predict=unshuffle_array(np.clip(predict, np.min(y_train), np.max(y_train)),sort_index)
        RMSE=mean_squared_error(y_train,predict,squared=False)
        r2=r2_score(y_train,predict)
        result.append([RMSE,r2])
        df_train["validation{}".format(repeat)]=predict
    
    ste_test=np.array(df_test["Dt"].values.tolist())/np.sqrt(np.average(np.array(df_train["Dt"].values.tolist())**2))
    esp_test=np.array(df_test["ESP"].values.tolist())/np.sqrt(np.average(np.array(df_train["ESP"].values.tolist())**2))
    X_test=np.concatenate([ste_test,esp_test],1).astype("float32")
    df_test["prediction"]=model.predict(X_test)
    df_test["steric_cont"]=X_test[:,:len(df_coord)]@model.coef_[0,:len(df_coord)]
    df_test["electrostatic_cont"]=X_test[:,len(df_coord):2*len(df_coord)]@model.coef_[0,len(df_coord):2*len(df_coord)]
    df_concat = pd.concat([df_train, df_test])
    df_concat["test"]=df_concat["test"].astype(str)
    df_concat.drop(columns=['Dt', 'DtR1',"DtR2",'ESP', 'ESPR1',"ESPR2"], inplace=True, errors='ignore')
    PandasTools.AddMoleculeColumnToFrame(df_concat, "smiles")
    df_concat.to_excel(save_name + "_prediction.xlsx")
    PandasTools.SaveXlsxFromFrame(df_concat.fillna("NaN"), save_name + "_prediction.xlsx", size=(100, 100))
    RMSE_train=mean_squared_error(y_train,df_train["regression"],squared=False)
    r2_train=r2_score(y_train,df_train["regression"])
    RMSE_test=mean_squared_error(df_test["ΔΔG.expt."],df_test["prediction"],squared=False)
    r2_test=r2_score(df_test["ΔΔG.expt."],df_test["prediction"])
    integ=np.sum(np.abs(df_coord[["coef_steric","coef_electric"]].values),axis=0).tolist()
    return [save_name,alpha,RMSE_train,r2_train]+np.average(result,axis=0).tolist()+[RMSE_test,r2_test]+integ

def PCR(input):
    alpha,n_splits,n_repeats,df,df_coord,save_name=input
    df_train=df[~df["test"]]
    df_test=df[df["test"]]
    ste_train=np.array(df_train["Dt"].values.tolist())/np.sqrt(np.average(np.array(df_train["Dt"].values.tolist())**2))
    esp_train=np.array(df_train["ESP"].values.tolist())/np.sqrt(np.average(np.array(df_train["ESP"].values.tolist())**2))
    X_train=np.concatenate([ste_train,esp_train],1).astype("float32")
    y_train=df_train["ΔΔG.expt."].values
    # PCR回帰のパイプラインを構築（切片をゼロに設定）
    pcr = Pipeline([
        ('pca', PCA(n_components=10)),  # 主成分数を指定
        ('regression', linear_model.LinearRegression(fit_intercept=False))
    ])
    model = pcr.fit(X_train, y_train)
    df_train["regression"]=model.predict(X_train)
    df_train["steric_cont"]=X_train[:,:len(df_coord)]@model.coef_[0,:len(df_coord)]
    df_train["electrostatic_cont"]=X_train[:,len(df_coord):2*len(df_coord)]@model.coef_[0,len(df_coord):2*len(df_coord)]
    df_coord["coef_steric"]=model.coef_[0,:len(df_coord)]
    df_coord["coef_electric"]=model.coef_[0,len(df_coord):2*len(df_coord)]
    df_coord[["coef_steric","coef_electric"]]=df_coord[["coef_steric","coef_electric"]].values*df_coord[["Dt_std","ESP_std"]].values
    df_coord.to_csv(save_name + "_coef.csv", index=False)

    result=[]
    for repeat in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=repeat)
        predict=[]
        sort_index=[]
        for train_index, test_index in kf.split(X_train):
            predict_cv = pcr.fit(X_train[train_index], y_train[train_index]).predict(X_train[test_index])
            predict.extend(predict_cv.tolist())
            sort_index.extend(test_index.tolist())
        predict=unshuffle_array(np.clip(predict, np.min(y_train), np.max(y_train)),sort_index)
        RMSE=mean_squared_error(y_train,predict,squared=False)
        r2=r2_score(y_train,predict)
        result.append([RMSE,r2])
        df_train["validation{}".format(repeat)]=predict
    
    ste_test=np.array(df_test["Dt"].values.tolist())/np.sqrt(np.average(np.array(df_train["Dt"].values.tolist())**2))
    esp_test=np.array(df_test["ESP"].values.tolist())/np.sqrt(np.average(np.array(df_train["ESP"].values.tolist())**2))
    X_test=np.concatenate([ste_test,esp_test],1).astype("float32")
    df_test["prediction"]=model.predict(X_test)
    df_test["steric_cont"]=X_test[:,:len(df_coord)]@model.coef_[0,:len(df_coord)]
    df_test["electrostatic_cont"]=X_test[:,len(df_coord):2*len(df_coord)]@model.coef_[0,len(df_coord):2*len(df_coord)]
    df_concat = pd.concat([df_train, df_test])
    df_concat["test"]=df_concat["test"].astype(str)
    df_concat.drop(columns=['Dt', 'DtR1',"DtR2",'ESP', 'ESPR1',"ESPR2"], inplace=True, errors='ignore')
    PandasTools.AddMoleculeColumnToFrame(df_concat, "smiles")
    df_concat.to_excel(save_name + "_prediction.xlsx")
    PandasTools.SaveXlsxFromFrame(df_concat.fillna("NaN"), save_name + "_prediction.xlsx", size=(100, 100))
    RMSE_train=mean_squared_error(y_train,df_train["regression"],squared=False)
    r2_train=r2_score(y_train,df_train["regression"])
    RMSE_test=mean_squared_error(df_test["ΔΔG.expt."],df_test["prediction"],squared=False)
    r2_test=r2_score(df_test["ΔΔG.expt."],df_test["prediction"])
    integ=np.sum(np.abs(df_coord[["coef_steric","coef_electric"]].values),axis=0).tolist()
    return [save_name,alpha,RMSE_train,r2_train]+np.average(result,axis=0).tolist()+[RMSE_test,r2_test]+integ

# def PLS(input):
#     X,y,alpha,n_splits,n_repeats,df,df_coord,save_name=input
#     pls = PLSRegression(n_components=alpha).fit(X, y)
#     df["regression"]=pls.predict(X)
#     # df["R1_contribution"]=pls.predict(X1)
#     # df["R2_contribution"]=pls.predict(X2)
#     # df["R1_contribution_steric"]=X1[:,:len(df_coord)]@pls.coef_[0][:len(df_coord)]
#     # df["R2_contribution_steric"]=X2[:,:len(df_coord)]@pls.coef_[0][:len(df_coord)]
#     # df["R1_contribution_electric"]=X1[:,len(df_coord):2*len(df_coord)]@pls.coef_[0][len(df_coord):2*len(df_coord)]
#     # df["R2_contribution_electric"]=X2[:,len(df_coord):2*len(df_coord)]@pls.coef_[0][len(df_coord):2*len(df_coord)]
#     # df["R1_contribution_LUMO"]=X1[:,2*len(df_coord):3*len(df_coord)]@pls.coef_[0][2*len(df_coord):3*len(df_coord)]
#     # df["R2_contribution_LUMO"]=X2[:,2*len(df_coord):3*len(df_coord)]@pls.coef_[0][2*len(df_coord):3*len(df_coord)]
#     df["steric_cont"]=X[:,:len(df_coord)]@pls.coef_[0][:len(df_coord)]
#     df["electrostatic_cont"]=X[:,len(df_coord):2*len(df_coord)]@pls.coef_[0][len(df_coord):2*len(df_coord)]
#     # df_coord["coef"]=pls.coef_[0][:len(df_coord)]
#     df_coord["coef_steric"]=pls.coef_[0][:len(df_coord)]
#     df_coord["coef_electric"]=pls.coef_[0][len(df_coord):2*len(df_coord)]
#     # df_coord["coef_LUMO"]=pls.coef_[0][2*len(df_coord):3*len(df_coord)]
    
#     df_coord.to_csv( save_name + "_coef.csv", index=False)
#     result=[]
#     for repeat in range(n_repeats):
#         kf = KFold(n_splits=n_splits, shuffle=True, random_state=repeat)
#         predict=[]
#         sort_index=[]
#         for train_index, test_index in kf.split(X):
#             X_train, X_test = X[train_index], X[test_index]
#             # X_train, X_test = X[train_index]/np.sqrt(np.average(X[train_index]**2)), X[test_index]/np.sqrt(np.average(X[train_index]**2))
#             predict_cv = PLSRegression(n_components=alpha).fit(X_train, y[train_index]).predict(X_test)
#             predict.extend(predict_cv.tolist())
#             sort_index.extend(test_index.tolist())
#         predict=unshuffle_array(np.clip(predict, np.min(y), np.max(y)),sort_index)
#         RMSE=mean_squared_error(y,predict,squared=False)
#         r2=r2_score(y,predict)
#         result.append([RMSE,r2])
#         df["validation{}".format(repeat)]=predict
    
#     split = np.empty(y.shape[0], dtype=np.uint8)
#     predict=[]
#     sort_index=[]
#     for _,(train_index, test_index) in enumerate(split_data_pca(X,n_splits)):
#         split[test_index]=_
#         X_train, X_test = X[train_index], X[test_index]
#         # X_train, X_test = X[train_index]/np.sqrt(np.average(X[train_index]**2)), X[test_index]/np.sqrt(np.average(X[train_index]**2))
#         predict_cv = PLSRegression(n_components=alpha).fit(X_train, y[train_index]).predict(X_test)
#         predict.extend(predict_cv.tolist())
#         sort_index.extend(test_index.tolist())
#     predict=unshuffle_array(np.clip(predict, np.min(y), np.max(y)),sort_index)
#     RMSE_PCA=mean_squared_error(y,predict,squared=False)
#     r2_PCA=r2_score(y,predict)
#     df["split_PCA"]=split
#     df["validation_PCA"]=predict
#     df.drop(columns=['Dt', 'DtR1',"DtR2",'ESP', 'ESPR1',"ESPR2",'DUAL', 'DUALR1',"DUALR2"], inplace=True, errors='ignore')
#     df["error"]=df["ΔΔG.expt."]-df["validation0"]
#     df = df.reindex(df['error'].abs().sort_values(ascending=False).index)
#     PandasTools.AddMoleculeColumnToFrame(df, "smiles")
#     PandasTools.SaveXlsxFromFrame(df, save_name + "_prediction.xlsx", size=(100, 100))
#     RMSE=mean_squared_error(y,pls.predict(X),squared=False)
#     integ=np.sum(df_coord[["coef_steric","coef_electric"]].values,axis=0).tolist()
#     r2=r2_score(y,pls.predict(X))
#     return [save_name,alpha,RMSE,r2]+np.average(result,axis=0).tolist()+[RMSE_PCA,r2_PCA]+integ

# def Lasso(input):
#     X,y,alpha,n_splits,n_repeats,df,df_coord,save_name=input
#     lasso = linear_model.Lasso(alpha=alpha , fit_intercept=False).fit(X, y)
#     df["regression"]=lasso.predict(X)
#     # df["R1_contribution"]=lasso.predict(X1)
#     # df["R2_contribution"]=lasso.predict(X2)
#     # df["R1_contribution_steric"]=X1[:,:len(df_coord)]@lasso.coef_[:len(df_coord)]
#     # df["R2_contribution_steric"]=X2[:,:len(df_coord)]@lasso.coef_[:len(df_coord)]
#     # df["R1_contribution_electric"]=X1[:,len(df_coord):2*len(df_coord)]@lasso.coef_[len(df_coord):2*len(df_coord)]
#     # df["R2_contribution_electric"]=X2[:,len(df_coord):2*len(df_coord)]@lasso.coef_[len(df_coord):2*len(df_coord)]
#     # df["R1_contribution_LUMO"]=X1[:,2*len(df_coord):3*len(df_coord)]@lasso.coef_[2*len(df_coord):3*len(df_coord)]
#     # df["R2_contribution_LUMO"]=X2[:,2*len(df_coord):3*len(df_coord)]@lasso.coef_[2*len(df_coord):3*len(df_coord)]
#     df["steric_cont"]=X[:,:len(df_coord)]@lasso.coef_[:len(df_coord)]
#     df["electrostatic_cont"]=X[:,len(df_coord):2*len(df_coord)]@lasso.coef_[len(df_coord):2*len(df_coord)]
#     # df_coord["coef_LUMO"]=lasso.coef_[2*len(df_coord):3*len(df_coord)]
#     # df_coord["coef"]=lasso.coef_[:len(df_coord)]
#     df_coord["coef_steric"]=lasso.coef_[:len(df_coord)]
#     df_coord["coef_electric"]=lasso.coef_[len(df_coord):2*len(df_coord)]
#     df_coord[["coef_steric","coef_electric"]]=df_coord[["coef_steric","coef_electric"]].values*df_coord[["Dt_std","ESP_std"]].values

#     df_coord.to_csv( save_name + "_coef.csv", index=False)
#     result=[]
#     for repeat in range(n_repeats):
#         kf = KFold(n_splits=n_splits, shuffle=True, random_state=repeat)
#         predict=[]
#         sort_index=[]
#         for train_index, test_index in kf.split(X):
#             X_train, X_test = X[train_index], X[test_index]
#             # X_train, X_test = X[train_index]/np.sqrt(np.average(X[train_index]**2)), X[test_index]/np.sqrt(np.average(X[train_index]**2))
#             predict_cv = linear_model.Lasso(alpha=alpha, fit_intercept=False).fit(X_train, y[train_index]).predict(X_test)
#             predict.extend(predict_cv.tolist())
#             sort_index.extend(test_index.tolist())
#         predict=unshuffle_array(np.clip(predict, np.min(y), np.max(y)),sort_index)
#         RMSE=mean_squared_error(y,predict,squared=False)
#         r2=r2_score(y,predict)
#         result.append([RMSE,r2])
#         df["validation{}".format(repeat)]=predict
    
#     split = np.empty(y.shape[0], dtype=np.uint8)
#     predict=[]
#     sort_index=[]
#     for _,(train_index, test_index) in enumerate(split_data_pca(X,n_splits)):
#         split[test_index]=_
#         X_train, X_test = X[train_index], X[test_index]
#         # X_train, X_test = X[train_index]/np.sqrt(np.average(X[train_index]**2)), X[test_index]/np.sqrt(np.average(X[train_index]**2))
#         predict_cv = linear_model.Lasso(alpha=alpha, fit_intercept=False).fit(X_train, y[train_index]).predict(X_test)
#         predict.extend(predict_cv.tolist())
#         sort_index.extend(test_index.tolist())
#     predict=unshuffle_array(np.clip(predict, np.min(y), np.max(y)),sort_index)

#     RMSE_PCA=mean_squared_error(y,predict,squared=False)
#     r2_PCA=r2_score(y,predict)
#     df["split_PCA"]=split
#     df["validation_PCA"]=predict
#     df.drop(columns=['Dt', 'DtR1',"DtR2",'ESP', 'ESPR1',"ESPR2",'DUAL', 'DUALR1',"DUALR2"], inplace=True, errors='ignore')
#     df["error"]=df["ΔΔG.expt."]-df["validation0"]
#     df = df.reindex(df['error'].abs().sort_values(ascending=False).index)
#     PandasTools.AddMoleculeColumnToFrame(df, "smiles")
#     PandasTools.SaveXlsxFromFrame(df, save_name + "_prediction.xlsx", size=(100, 100))
#     RMSE=mean_squared_error(y,lasso.predict(X),squared=False)
#     # integ=np.sum(df_coord[["coef_steric","coef_electric","coef_LUMO"]].values,axis=0).tolist()
#     integ=np.sum(np.abs(df_coord[["coef_steric","coef_electric"]].values),axis=0).tolist()
#     r2=r2_score(y,lasso.predict(X))
#     return [save_name,alpha,RMSE,r2]+np.average(result,axis=0).tolist()+[RMSE_PCA,r2_PCA]+integ

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
    features=["Dt","ESP"]
    we = []
    for conf in mol.GetConformers():
        data = pd.read_pickle(
            "{}/{}/data{}.pkl".format(dir, mol.GetProp("InchyKey"), conf.GetId()))
        Bd = float(conf.GetProp("Boltzmann_distribution"))
        we.append(Bd)
        feat.append(data[features].values.tolist())
        # ESP.append(data["ESP"].values.tolist())
    # Dt = np.array(Dt)
    # ESP = np.array(ESP)
    # w = np.exp(-Dt / np.sqrt(np.average(Dt ** 2, axis=0)).reshape(1, -1))
    # w = np.exp(-Dt / np.max(Dt, axis=0).reshape(1, -1))
    # w = np.exp(-Dt/np.sqrt(np.average(Dt**2)))
    # w = np.exp(-Dt)
    # print(np.max(w),np.min(w),np.average(w))
    data[features] = np.nan_to_num(np.average(feat, weights=we, axis=0))
    # w = np.exp(ESP / np.sqrt(np.average(ESP ** 2, axis=0)).reshape(1, -1))
    # data["ESP"] = np.nan_to_num(
    #     np.average(ESP, weights=np.array(we).reshape(-1, 1) * np.ones(shape=ESP.shape), axis=0))
    ans1=data[(data["y"] > 0)&(data["z"]>0)].sort_values(['x', 'y', "z"], ascending=[True, True, True])[features].values
    ans2=data[(data["y"] < 0)&(data["z"]>0)].sort_values(['x', 'y', "z"], ascending=[True, False, True])[features].values
    ans3=data[(data["y"] > 0)&(data["z"]<0)].sort_values(['x', 'y', "z"], ascending=[True, True, False])[features].values
    ans4=data[(data["y"] < 0)&(data["z"]<0)].sort_values(['x', 'y', "z"], ascending=[True, False, False])[features].values
    ans=ans1+ans2-ans3-ans4
    
    # data_y[features] = data[data["y"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])[features].values + \
        # data[data["y"] < 0].sort_values(['x', 'y', "z"], ascending=[True, False, True])[features].values
    
    # data_y["Dt"]=np.minimum(data[data["y"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])["Dt"].values,
    #                         data[data["y"] < 0].sort_values(['x', 'y', "z"], ascending=[True, False, True])["Dt"].values)

    # data_yz = data_y[data_y["z"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])

    # data_yz["Dt"] = data_y[data_y["z"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])["Dt"].values - \
    #                 data_y[data_y["z"] < 0].sort_values(['x', 'y', "z"], ascending=[True, True, False])["Dt"].values

    # data_yz[[_+"R1" for _ in features]] = data_y[data_y["z"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])[features].values
    # data_yz[[_+"R2" for _ in features]] = data_y[data_y["z"] < 0].sort_values(['x', 'y', "z"], ascending=[True, True, False])[features].values
    # data_yz[features]=data_yz[[_+"R1" for _ in features]].values-data_yz[[_+"R2" for _ in features]].values
    # dfp_yz = data_yz.copy()
    # return dfp_yz[["Dt","ESP"]].values.T.tolist()
    return ans.T.tolist()



# def pca_kmeans_clustering(X):
#     # PCAの実行
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X)

#     # データのプロット
#     plt.figure(figsize=(14, 6))
    
#     plt.subplot(1, 3, 1)
#     plt.scatter(X_pca[:, 0], X_pca[:, 1])
#     plt.title('PCA result')
#     plt.xlabel('Principal Component 1')
#     plt.ylabel('Principal Component 2')

#     # エルボー法の実行
#     inertia = []
#     K = range(1, 11)
#     for k in K:
#         kmeans = KMeans(n_clusters=k, random_state=0)
#         kmeans.fit(X_pca)
#         inertia.append(kmeans.inertia_)

#     # エルボー法の結果プロット
#     plt.subplot(1, 3, 2)
#     plt.plot(K, inertia, 'bx-')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('Inertia')
#     plt.title('Elbow Method For Optimal k')

#     # 最適なクラスタ数を選択 (例えばエルボー法からk=4と仮定)
#     optimal_k = 4  # 実際のプロットを見て最適なkを決定してください
#     kmeans = KMeans(n_clusters=optimal_k, random_state=0)
#     y_kmeans = kmeans.fit_predict(X_pca)

#     # クラスタリング結果のプロット
#     plt.subplot(1, 3, 3)
#     plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, s=50, cmap='viridis')
#     centers = kmeans.cluster_centers_
#     plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
#     plt.title(f'k-means Clustering (k={optimal_k})')
#     plt.xlabel('Principal Component 1')
#     plt.ylabel('Principal Component 2')

#     plt.tight_layout()
#     plt.savefig("clustering")


# def plot_elbow_method(X, filename,max_clusters=10):
#     distortions = []
#     for i in range(1, max_clusters + 1):
#         kmeans = KMeans(n_clusters=i, random_state=42)
#         kmeans.fit(X)
#         distortions.append(kmeans.inertia_)  # Inertiaは各点からの距離の総和

#     plt.figure(figsize=(8, 5))
#     plt.plot(range(1, max_clusters + 1), distortions, marker='o')
#     plt.title('Elbow Method For Optimal k')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('Distortion (Inertia)')
#     plt.savefig(filename)
#     # plt.show()

# def select_data_by_clustering(X, n_clusters):
#     # KMeansクラスタリングを実行
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     kmeans.fit(X)
    
#     # 各クラスタの中心に最も近いデータポイントを選ぶ
#     closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
#     closest_mask = np.isin(np.arange(len(X)), closest)
#     return closest_mask



if __name__ == '__main__':
    # time.sleep(60*60*24*1)
    # inputs_=[]
    ridge_input=[]
    lasso_input=[]
    elasticnet_input=[]
    pls_input=[]
    pcr_input=[]
    for param_name in sorted(glob.glob("../parameter/run/cube_to_grid0.50.txt"),reverse=True):
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
            df[["Dt","ESP"]]=df[["mol", "RT"]].apply(lambda _: get_grid_feat(_[0], _[1],param["grid_coordinates"]), axis=1, result_type='expand')
            print("feature_calculated")
            # X=np.array(df["Dt"].values.tolist())/np.sqrt(np.sum(np.array(df["Dt"].values.tolist())**2))
            # X_esp=np.array(df["ESP"].values.tolist())/np.sqrt(np.sum(np.array(df["ESP"].values.tolist())**2))
            df_coord = pd.read_csv(param["grid_coordinates"] + "/coordinates_yz.csv").sort_values(['x', 'y', "z"],
                                                                                  ascending=[True, True, True])
            # X=np.concatenate([X,X_esp],1).astype("float32")
            # n_clusters = 10  # 例として5クラスターに設定
            # df["test"] = select_data_by_clustering(X, n_clusters)
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=4)

            # ラベルを追加する
            train_df['test'] = False
            test_df['test'] = True

            # 訓練データとテストデータを結合する
            df = pd.concat([train_df, test_df])
            # X1=np.concatenate([X1,X1_esp],1).astype("float32")
            # X2=np.concatenate([X2,X2_esp],1).astype("float32")
            # サンプル行列を作成

            # 下位10%の閾値を計算
            # threshold = np.percentile(np.average(X**2,axis=0), 90)

            # 行列の下位10%の要素を0に置き換え
            # X[:, np.count_nonzero(X, axis=0) <= X.shape[0]/2] = 0
            # X = np.concatenate(features_all / std, axis=1)
            # pca_kmeans_clustering(X)
            #testのフラグを立てる。
            # y=df["ΔΔG.expt."].values.astype("float32")
            # 正則化パラメータの候補
            # alphas = np.logspace(0,15,16,base=2)

            # 5分割交差検証を10回実施
            n_splits = int(param["n_splits"])
            n_repeats = int(param["n_repeats"])
            dir=param["out_dir_name"] + "/{}".format(file_name)
            os.makedirs(dir,exist_ok=True)
            # plot_elbow_method(X, dir+"/elbow.png",max_clusters=10)
            # for alpha in alphas:
            #     save_name=dir + "/Ridge_alpha_{}".format(alpha)
            #     ridge_input.append([X,y,alpha,n_splits,n_repeats,df,df_coord,save_name])
            # Qs=np.array(param["sigmas"], dtype=np.float32)
            # for alpha in alphas*1e-2:
            #     for Q in Qs:
            #         save_name=dir + "/Gaussian_alpha_{}_sigma_{}".format(alpha,Q)
            #         Q_dir=param["grid_coordinates"]#+ "/1eig{}.npy".format(Q)
            #         gaussian_input.append([X,X1,X2,y,alpha,Q,Q_dir,n_splits,n_repeats,df,df_coord,save_name])
            alphas = np.logspace(-20,-10,11,base=2)
            l1ratios = np.linspace(0, 1, 11)

            for alpha in alphas:
                for l1ratio in l1ratios:
                    save_name=dir + "/ElasticNet_alpha_{}_l1ratio{}".format(alpha,l1ratio)
                    elasticnet_input.append([alpha,l1ratio,n_splits,n_repeats,df,df_coord,save_name])
            # alphas = np.logspace(-10,5,11,base=2)
            # for alpha in alphas:
            #     save_name=dir + "/Lasso_alpha_{}".format(alpha)
            #     lasso_input.append([X,y,alpha,n_splits,n_repeats,df,df_coord,save_name])
            alphas = np.arange(1,10)
            for alpha in alphas:
                save_name=dir + "/PLS_alpha_{}".format(alpha)
                pls_input.append([alpha,n_splits,n_repeats,df,df_coord,save_name])
            alphas = np.arange(1,10)
            for alpha in alphas:
                save_name=dir + "/PCR_alpha_{}".format(alpha)
                pcr_input.append([alpha,n_splits,n_repeats,df,df_coord,save_name])
    p = multiprocessing.Pool(processes=int(param["processes"]))
    columns=["savefilename","alpha","RMSE_regression", "r2_regression","RMSE_validation", "r2_validation","RMSE_PCA_validation","r2_PCA_validation","integral_steric","integral_electric"]
    elasticnet_columns=["savefilename","alpha","l1ratio","RMSE_regression", "r2_regression","RMSE_validation", "r2_validation","RMSE_PCA_validation","r2_PCA_validation","integral_steric","integral_electric"]
    # decimals = {"RMSE_regression": 5,"r2_regression":5,"RMSE_validation":5, "r2_validation":5}
    # pd.DataFrame(p.map(PCR,pcr_input), columns=columns).apply(lambda x: x.round(4) if np.issubdtype(x.dtype, np.floating) else x).to_csv(param["out_dir_name"] + "/PCR.csv")
    # pd.DataFrame(p.map(Ridge,ridge_input), columns=columns).apply(lambda x: x.round(4) if np.issubdtype(x.dtype, np.floating) else x).to_csv(param["out_dir_name"] + "/Ridge.csv")
    # pd.DataFrame(p.map(Lasso,lasso_input), columns=columns).apply(lambda x: x.round(4) if np.issubdtype(x.dtype, np.floating) else x).to_csv(param["out_dir_name"] + "/Lasso.csv")
    pd.DataFrame(p.map(PLS,pls_input), columns=columns).apply(lambda x: x.round(4) if np.issubdtype(x.dtype, np.floating) else x).to_csv(param["out_dir_name"] + "/PLS.csv")
    pd.DataFrame(p.map(ElasticNet,elasticnet_input), columns=elasticnet_columns).apply(lambda x: x.round(4) if np.issubdtype(x.dtype, np.floating) else x).to_csv(param["out_dir_name"] + "/ElasticNet.csv")
    # pd.DataFrame(p.map(Gaussian,gaussian_input), columns=["savefilename","alpha","sigma","RMSE_regression", "r2_regression","RMSE_validation", "r2_validation", "RMSE_PCA_validation","r2_PCA_validation"]+["integral_steric","integral_electric","integral_LUMO","x_moment","y_moment","z_moment"]).apply(lambda x: x.round(4) if np.issubdtype(x.dtype, np.floating) else x).to_csv(param["out_dir_name"] + "/Gaussian.csv")
    end = time.perf_counter()  # 計測終了
    print('Finish{:.2f}'.format(end - start))
    