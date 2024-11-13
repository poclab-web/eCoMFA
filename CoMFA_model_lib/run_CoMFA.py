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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import time
import calculate_conformation
from sklearn.pipeline import Pipeline
warnings.simplefilter('ignore')

# シャッフル前の順序に戻す関数
def unshuffle_array(shuffled_array, shuffled_indices):
    original_array = np.empty_like(shuffled_array)
    original_array[shuffled_indices] = shuffled_array
    return original_array

def calc_x(df):
    df_train=df[df["test"]==0]
    ste_train=np.array(df_train["Dt"].values.tolist())/np.sqrt(np.sum(np.array(df_train["Dt"].values.tolist())**2))
    esp_train=np.array(df_train["ESP"].values.tolist())/np.sqrt(np.sum(np.array(df_train["ESP"].values.tolist())**2))
    df_coord["Dt_std"]=np.average(np.sum(ste_train**2,axis=0))
    df_coord["ESP_std"]=np.average(np.sum(esp_train**2,axis=0))
    X_train=np.concatenate([ste_train,esp_train],1).astype("float32")
    return X_train

def ElasticNet(input):
    alpha,l1ratio,n_splits,n_repeats,df,df_coord,save_name=input
    df_train=df[df["test"]==0]
    df_test=df[df["test"]==1]
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
    df_train=df[df["test"]==0]
    df_test=df[df["test"]==1]
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


<<<<<<< HEAD
# #regressionだけにする。
# def regression_comparison_(df, dfp, gaussian_penalize, save_name, n_splits,features):
#     os.makedirs(save_name + "/molecular_field_csv",exist_ok=True)
#     df_coord = pd.read_csv(gaussian_penalize + "/coordinates_yz.csv").sort_values(['x', 'y', "z"],
#                                                                                   ascending=[True, True, True])
#     # ptp_name=gaussian_penalize + "/{}ptp{}.npy".format(len(features),n)
#     features_all = np.array(df[features].values.tolist()).reshape(len(df), -1, len(features)).transpose(2, 0, 1)
#     std = np.sqrt(np.average(features_all**2, axis=(1, 2))).reshape(features_all.shape[0], 1, 1)
#     X = np.concatenate(features_all / std, axis=1)

#     predicts = []
#     coefs_=[]
#     for L, n_num in zip(dfp["lambda"], dfp["n_components"]):
#         ridge = linear_model.Ridge(alpha=L * len(df), fit_intercept=False).fit(X, df["ΔΔG.expt."])
#         lasso = linear_model.Lasso(alpha=L/2/100, fit_intercept=False).fit(X, df["ΔΔG.expt."])
#         pls = PLSRegression(n_components=n_num).fit(X, df["ΔΔG.expt."])
#         # gaussian_coef = scipy.linalg.solve((X.T @ X + L * len(df) * np.load(ptp_name)).astype("float32"),
#         #                                    (X.T @ df["ΔΔG.expt."].values).astype("float32"), assume_a="gen").T
#         l=[ridge.coef_,lasso.coef_,pls.coef_[0]]
#         columns=["Ridge","Lasso","PLS"]
#         for ptpname in sorted(glob.glob(gaussian_penalize + "/{}ptp*.npy".format(len(features)))):
#             gaussian_coef_all = scipy.linalg.solve((X.T @ X + L * len(df) * np.load(ptpname)).astype("float32"),
#                                            (X.T @ df["ΔΔG.expt."].values).astype("float32"), assume_a="gen").T
#             sigma = re.findall(gaussian_penalize +"/"+str(len(features))+"ptp"+ "(.*).npy", ptpname.replace(os.sep,'/'))
#             n = sigma[0]
#             columns.append("Gaussian"+n)
#             l.append(gaussian_coef_all)

#         coefs_.append(l)
#         n = int(ridge.coef_.shape[0] / features_all.shape[0])
#         df_coord[[_+"_Dt" for _ in columns]]=np.stack((l),axis=1)[:n]* std[0].reshape([1])#,lasso.coef_[:n].tolist(),pls.coef_[0][:n].tolist()]
#         # kf = KFold(n_splits=int(n_splits), shuffle=False)
#         # predicts_=[]
#         # for i, (train_index, test_index) in enumerate(kf.split(df)):
#         #     features_training = features_all[:, train_index]
#         #     std_ = np.sqrt(np.average(features_training**2, axis=(1, 2))).reshape(features_all.shape[0], 1, 1)
#         #     features_training = features_training / std_
#         #     features_training = np.concatenate(features_training, axis=1)
#         #     X_ = features_training
#         #     Y = df.iloc[train_index]["ΔΔG.expt."].values
#         #     # gaussian_coef_ = scipy.linalg.solve((X_.T @ X_ + L * len(train_index)  * np.load(ptp_name)).astype("float32"), (X_.T @ Y).astype("float32"),
#         #     #                                     assume_a="gen").T
#         #     ridge = linear_model.Ridge(alpha=L * len(train_index) , fit_intercept=False).fit(
#         #         features_training,
#         #         df.iloc[train_index][
#         #             "ΔΔG.expt."])
#         #     lasso = linear_model.Lasso(alpha=L/2/100, fit_intercept=False).fit(features_training,
#         #                                                                  df.iloc[train_index]["ΔΔG.expt."])
#         #     pls = PLSRegression(n_components=n_num).fit(features_training,
#         #                                                 df.iloc[train_index]["ΔΔG.expt."])
#         #     features_test = features_all[:, test_index]  # np.array(features_all)[test_index].transpose(2, 0, 1)
#         #     features_test = np.concatenate(features_test / std_, axis=1)
#         #     # gaussian_predict = np.sum(gaussian_coef_ * features_test, axis=1).tolist()  # model.predict(features_test)
#         #     ridge_predict = ridge.predict(features_test)
#         #     lasso_predict = lasso.predict(features_test)
#         #     pls_predict = pls.predict(features_test)[:,0]
#         #     l=[ridge_predict,lasso_predict,pls_predict]
#         #     for ptpname in sorted(glob.glob(gaussian_penalize + "/{}ptp*.npy".format(len(features)))):
#         #         gaussian_coef_ = scipy.linalg.solve((X_.T @ X_ + L * len(train_index)  * np.load(ptpname)).astype("float32"), (X_.T @ Y).astype("float32"),
#         #                                         assume_a="gen").T
#         #         sigma = re.findall(gaussian_penalize +"/"+str(len(features))+"ptp"+ "(.*).npy", ptpname.replace(os.sep,'/'))
#         #         n = sigma[0]
#         #         gaussian_predict = np.sum(gaussian_coef_ * features_test, axis=1).tolist()
#         #         l.append(gaussian_predict)
#         #     predicts_.append(l)
#         # predicts.append(np.concatenate(predicts_,axis=1).tolist())
#         df_coord.to_csv(save_name + "/molecular_field_csv" + "/molecular_field{}.csv".format(L))
    
#     dfp[[_+"_coef" for _ in columns]]=coefs_
#     dfp[[_+"_regression_predict" for _ in columns]]=dfp[[_+"_coef" for _ in columns]].applymap(lambda coef: np.sum(coef.reshape([1,-1])*X,axis=1))
#     dfp[[_+"_regression_r2" for _ in columns]] = dfp[[_+"_regression_predict" for _ in columns]].applymap(
#         lambda predict: r2_score(df["ΔΔG.expt."], predict))
#     dfp[[_+"_regression_RMSE" for _ in columns]] = dfp[[_+"_regression_predict" for _ in columns]].applymap(
#         lambda predict: np.sqrt(mean_squared_error(df["ΔΔG.expt."], predict)))
#     # dfp[[_+"_validation_predict" for _ in columns]] = predicts
#     # dfp[[_+"_validation_r2" for _ in columns]] = dfp[[_+"_validation_predict" for _ in columns]].applymap(
#     #     lambda predict: r2_score(df["ΔΔG.expt."], predict))
#     # dfp[[_+"_validation_r" for _ in columns]] = dfp[[_+"_validation_predict" for _ in columns]].applymap(
#     #     lambda predict: np.corrcoef(df["ΔΔG.expt."], predict)[1, 0])
#     # dfp[[_+"_validation_RMSE" for _ in columns]] = dfp[[_+"_validation_predict" for _ in columns]].applymap(
#     #     lambda predict: mean_squared_error(df["ΔΔG.expt."], predict,squared=False))
    
#     features_R1 = np.array(df["DtR1"].tolist()).reshape(len(df), -1, 1).transpose(2, 0, 1)
#     features_R1 = np.concatenate(features_R1 / std, axis=1)
#     dfp[[_+"_regression_predictR1" for _ in columns]]=dfp[[_+"_coef" for _ in columns]].applymap(lambda coef: np.sum(coef.reshape([1,-1])*features_R1,axis=1))
#     features_R2 = np.array(df["DtR2"].tolist()).reshape(len(df), -1, 1).transpose(2, 0, 1)
#     features_R2 = np.concatenate(features_R2 / std, axis=1)
#     dfp[[_+"_regression_predictR2" for _ in columns]]=dfp[[_+"_coef" for _ in columns]].applymap(lambda coef: np.sum(coef.reshape([1,-1])*features_R2,axis=1))
#     for _ in columns:
#         for L in dfp["lambda"]:
<<<<<<< HEAD
#             gaussian_coef = scipy.linalg.solve(((X.T @ X).astype("float32") + L * len(df)  * np.load(ptpname)).astype("float32"), XTY, assume_a="pos").T
#             gaussians.append(gaussian_coef)  # * a.tolist()
#             n_ = int(gaussian_coef.shape[0] / features_all.shape[0])
#             df_coord["Gaussian_Dt"] = gaussian_coef[:n_]
#             df_coord.to_csv(save_name  + "/molecular_field_csv"+"/molecular_filed{}{}.csv".format(n, L))

<<<<<<< HEAD
<<<<<<< HEAD
            kf = KFold(n_splits=2, shuffle=False)
            # kf = KFold(n_splits=len(df), shuffle=False)
=======
            kf = KFold(n_splits=int(n_splits), shuffle=False)
>>>>>>> cb7fd67 (from windows)
            gaussian_predicts = []
            start = time.time()
            for (train_index, test_index) in kf.split(df):
                features_training = features_all[:, train_index]
                std = np.std(features_training, axis=(1, 2)).reshape(features_all.shape[0], 1, 1)
                X_ = np.concatenate(features_training / std, axis=1)
                gaussian_coef_ = scipy.linalg.solve(
                    (X_.T @ X_ + L * len(train_index)  * np.load(ptpname)).astype("float32"),
                    (X_.T @ df.iloc[train_index]["ΔΔG.expt."]).astype("float32"),
                    assume_a="pos", check_finite=False, overwrite_a=True, overwrite_b=True).T
                features_test = features_all[:, test_index]
                features_test = np.concatenate(features_test / std, axis=1)
                predict = np.sum(gaussian_coef_ * features_test, axis=1).tolist()
                gaussian_predicts.extend(predict)
            predicts.append([gaussian_predicts])
        gaussians = np.array(gaussians)
        gaussians = gaussians.reshape(gaussians.shape[0], 1, -1)
        dfp["Gaussian_regression_predict{}".format(n)] = np.sum(gaussians * X.reshape(1, X.shape[0], -1),
                                                                axis=2).tolist()
        dfp[["Gaussian_regression_r2{}".format(n)]] = dfp[
            ["Gaussian_regression_predict{}".format(n)]].applymap(
            lambda predict: r2_score(df["ΔΔG.expt."], predict))
        dfp[["Gaussian_regression_RMSE{}".format(n)]] = \
            dfp[[
                "Gaussian_regression_predict{}".format(n)]].applymap(
                lambda predict: np.sqrt(mean_squared_error(df["ΔΔG.expt."], predict)))
=======
#             kf = KFold(n_splits=int(n_splits), shuffle=False)
#             gaussian_predicts = []
#             start = time.time()
#             for (train_index, test_index) in kf.split(df):
#                 features_training = features_all[:, train_index]
#                 std = np.std(features_training, axis=(1, 2)).reshape(features_all.shape[0], 1, 1)
#                 X_ = np.concatenate(features_training / std, axis=1)
#                 gaussian_coef_ = scipy.linalg.solve(
#                     (X_.T @ X_ + L * len(train_index)  * np.load(ptpname)).astype("float32"),
#                     (X_.T @ df.iloc[train_index]["ΔΔG.expt."]).astype("float32"),
#                     assume_a="pos", check_finite=False, overwrite_a=True, overwrite_b=True).T
#                 features_test = features_all[:, test_index]
#                 features_test = np.concatenate(features_test / std, axis=1)
#                 predict = np.sum(gaussian_coef_ * features_test, axis=1).tolist()
#                 gaussian_predicts.extend(predict)
#             predicts.append([gaussian_predicts])
#         gaussians = np.array(gaussians)
#         gaussians = gaussians.reshape(gaussians.shape[0], 1, -1)
#         dfp["Gaussian_regression_predict{}".format(n)] = np.sum(gaussians * X.reshape(1, X.shape[0], -1),
#                                                                 axis=2).tolist()
#         dfp[["Gaussian_regression_r2{}".format(n)]] = dfp[
#             ["Gaussian_regression_predict{}".format(n)]].applymap(
#             lambda predict: r2_score(df["ΔΔG.expt."], predict))
#         dfp[["Gaussian_regression_RMSE{}".format(n)]] = \
#             dfp[[
#                 "Gaussian_regression_predict{}".format(n)]].applymap(
#                 lambda predict: np.sqrt(mean_squared_error(df["ΔΔG.expt."], predict)))
>>>>>>> bd2c239 (from win)

#         dfp[["Gaussian_test_predict{}".format(n)]] = predicts
#         dfp[["Gaussian_test_r2{}".format(n)]] = dfp[
#             ["Gaussian_test_predict{}".format(n)]].applymap(
#             lambda predict: r2_score(df["ΔΔG.expt."], predict))
#         dfp[["Gaussian_test_r{}".format(n)]] = dfp[
#             ["Gaussian_test_predict{}".format(n)]].applymap(
#             lambda predict: np.corrcoef(df["ΔΔG.expt."], predict)[1, 0])
#         dfp[["Gaussian_test_RMSE{}".format(n)]] = dfp[
#             ["Gaussian_test_predict{}".format(n)]].applymap(
#             lambda predict: np.sqrt(mean_squared_error(df["ΔΔG.expt."], predict)))
#         df["Gaussian_predict{}".format(n)] = \
#             dfp[dfp["Gaussian_test_r{}".format(n)] == dfp["Gaussian_test_r{}".format(n)].max()].iloc[0][
#                 "Gaussian_test_predict{}".format(n)]
#         df["Gaussian_error{}".format(n)] = df["Gaussian_predict{}".format(n)] - df["ΔΔG.expt."]
#         print(dfp[["Gaussian_test_RMSE{}".format(n)]].min())

#     df = df.sort_values(by='Gaussian_error{}'.format(n), key=abs, ascending=[False])
=======
#             df[[_+"regression",_+"_R1",_+"_R2"]]=dfp.loc[dfp[_+"_validation_RMSE"].idxmin()][[_+"_regression_predict",_+"_regression_predictR1",_+"_regression_predictR2"]]
    
#     df["Ridge_error"] = df["Ridge_regression"] - df["ΔΔG.expt."]
#     df = df.sort_values(by='Ridge_error', key=abs, ascending=[False])
>>>>>>> f93d900 (code creanup)
#     PandasTools.AddMoleculeColumnToFrame(df, "smiles")
#     print(dfp[[_+"_regression_r2" for _ in columns]].max())
#     PandasTools.SaveXlsxFromFrame(df, save_name + "/λ_result.xlsx", size=(100, 100))
#     dfp.to_csv(save_name + "/λ_result.csv", index=False)
#     print(save_name)

# #CV
# def regression_comparison__(df, dfp, gaussian_penalize, save_name, n_splits,features):
#     os.makedirs(save_name + "/molecular_field_csv",exist_ok=True)
#     df_coord = pd.read_csv(gaussian_penalize + "/coordinates_yz.csv").sort_values(['x', 'y', "z"],
#                                                                                   ascending=[True, True, True])
#     # ptp_name=gaussian_penalize + "/{}ptp{}.npy".format(len(features),n)
#     features_all = np.array(df[features].values.tolist()).reshape(len(df), -1, len(features)).transpose(2, 0, 1)
#     std = np.sqrt(np.average(features_all**2, axis=(1, 2))).reshape(features_all.shape[0], 1, 1)
#     X = np.concatenate(features_all / std, axis=1)

#     predicts = []
#     coefs_=[]
#     for L, n_num in zip(dfp["lambda"], dfp["n_components"]):
#         ridge = linear_model.Ridge(alpha=L * len(df), fit_intercept=False).fit(X, df["ΔΔG.expt."])
#         lasso = linear_model.Lasso(alpha=L/2/100, fit_intercept=False).fit(X, df["ΔΔG.expt."])
#         pls = PLSRegression(n_components=n_num).fit(X, df["ΔΔG.expt."])
#         # gaussian_coef = scipy.linalg.solve((X.T @ X + L * len(df) * np.load(ptp_name)).astype("float32"),
#         #                                    (X.T @ df["ΔΔG.expt."].values).astype("float32"), assume_a="gen").T
#         l=[ridge.coef_,lasso.coef_,pls.coef_[0]]
#         columns=["Ridge","Lasso","PLS"]
#         for ptpname in sorted(glob.glob(gaussian_penalize + "/{}ptp*.npy".format(len(features)))):
#             gaussian_coef_all = scipy.linalg.solve((X.T @ X + L * len(df) * np.load(ptpname)).astype("float32"),
#                                            (X.T @ df["ΔΔG.expt."].values).astype("float32"), assume_a="gen").T
#             sigma = re.findall(gaussian_penalize +"/"+str(len(features))+"ptp"+ "(.*).npy", ptpname.replace(os.sep,'/'))
#             n = sigma[0]
#             columns.append("Gaussian"+n)
#             l.append(gaussian_coef_all)

#         coefs_.append(l)
#         n = int(ridge.coef_.shape[0] / features_all.shape[0])
#         df_coord[[_+"_Dt" for _ in columns]]=np.stack((l),axis=1)[:n]* std[0].reshape([1])#,lasso.coef_[:n].tolist(),pls.coef_[0][:n].tolist()]
#         kf = KFold(n_splits=int(n_splits), shuffle=False)
#         predicts_=[]
#         for i, (train_index, test_index) in enumerate(kf.split(df)):
#             features_training = features_all[:, train_index]
#             std_ = np.sqrt(np.average(features_training**2, axis=(1, 2))).reshape(features_all.shape[0], 1, 1)
#             features_training = features_training / std_
#             features_training = np.concatenate(features_training, axis=1)
#             X_ = features_training
#             Y = df.iloc[train_index]["ΔΔG.expt."].values
#             # gaussian_coef_ = scipy.linalg.solve((X_.T @ X_ + L * len(train_index)  * np.load(ptp_name)).astype("float32"), (X_.T @ Y).astype("float32"),
#             #                                     assume_a="gen").T
#             ridge = linear_model.Ridge(alpha=L * len(train_index) , fit_intercept=False).fit(
#                 features_training,
#                 df.iloc[train_index][
#                     "ΔΔG.expt."])
#             lasso = linear_model.Lasso(alpha=L/2/100, fit_intercept=False).fit(features_training,
#                                                                          df.iloc[train_index]["ΔΔG.expt."])
#             pls = PLSRegression(n_components=n_num).fit(features_training,
#                                                         df.iloc[train_index]["ΔΔG.expt."])
#             features_test = features_all[:, test_index]  # np.array(features_all)[test_index].transpose(2, 0, 1)
#             features_test = np.concatenate(features_test / std_, axis=1)
#             # gaussian_predict = np.sum(gaussian_coef_ * features_test, axis=1).tolist()  # model.predict(features_test)
#             ridge_predict = ridge.predict(features_test)
#             lasso_predict = lasso.predict(features_test)
#             pls_predict = pls.predict(features_test)[:,0]
#             l=[ridge_predict,lasso_predict,pls_predict]
#             for ptpname in sorted(glob.glob(gaussian_penalize + "/{}ptp*.npy".format(len(features)))):
#                 gaussian_coef_ = scipy.linalg.solve((X_.T @ X_ + L * len(train_index)  * np.load(ptpname)).astype("float32"), (X_.T @ Y).astype("float32"),
#                                                 assume_a="gen").T
#                 sigma = re.findall(gaussian_penalize +"/"+str(len(features))+"ptp"+ "(.*).npy", ptpname.replace(os.sep,'/'))
#                 n = sigma[0]
#                 gaussian_predict = np.sum(gaussian_coef_ * features_test, axis=1).tolist()
#                 l.append(gaussian_predict)
#             predicts_.append(l)
#         predicts.append(np.concatenate(predicts_,axis=1).tolist())
#         df_coord.to_csv(save_name + "/molecular_field_csv" + "/molecular_field{}.csv".format(L))
    
#     dfp[[_+"_coef" for _ in columns]]=coefs_
#     dfp[[_+"_regression_predict" for _ in columns]]=dfp[[_+"_coef" for _ in columns]].applymap(lambda coef: np.sum(coef.reshape([1,-1])*X,axis=1))
#     dfp[[_+"_regression_r2" for _ in columns]] = dfp[[_+"_regression_predict" for _ in columns]].applymap(
#         lambda predict: r2_score(df["ΔΔG.expt."], predict))
#     dfp[[_+"_regression_RMSE" for _ in columns]] = dfp[[_+"_regression_predict" for _ in columns]].applymap(
#         lambda predict: np.sqrt(mean_squared_error(df["ΔΔG.expt."], predict)))
#     dfp[[_+"_validation_predict" for _ in columns]] = predicts
#     dfp[[_+"_validation_r2" for _ in columns]] = dfp[[_+"_validation_predict" for _ in columns]].applymap(
#         lambda predict: r2_score(df["ΔΔG.expt."], predict))
#     dfp[[_+"_validation_r" for _ in columns]] = dfp[[_+"_validation_predict" for _ in columns]].applymap(
#         lambda predict: np.corrcoef(df["ΔΔG.expt."], predict)[1, 0])
#     dfp[[_+"_validation_RMSE" for _ in columns]] = dfp[[_+"_validation_predict" for _ in columns]].applymap(
#         lambda predict: mean_squared_error(df["ΔΔG.expt."], predict,squared=False))
    
#     features_R1 = np.array(df["DtR1"].tolist()).reshape(len(df), -1, 1).transpose(2, 0, 1)
#     features_R1 = np.concatenate(features_R1 / std, axis=1)
#     dfp[[_+"_regression_predictR1" for _ in columns]]=dfp[[_+"_coef" for _ in columns]].applymap(lambda coef: np.sum(coef.reshape([1,-1])*features_R1,axis=1))
#     features_R2 = np.array(df["DtR2"].tolist()).reshape(len(df), -1, 1).transpose(2, 0, 1)
#     features_R2 = np.concatenate(features_R2 / std, axis=1)
#     dfp[[_+"_regression_predictR2" for _ in columns]]=dfp[[_+"_coef" for _ in columns]].applymap(lambda coef: np.sum(coef.reshape([1,-1])*features_R2,axis=1))
#     for _ in columns:
#         df[[_+"regression",_+"_validation",_+"_R1",_+"_R2"]]=dfp.loc[dfp[_+"_validation_RMSE"].idxmin()][[_+"_regression_predict",_+"_validation_predict",_+"_regression_predictR1",_+"_regression_predictR2"]]
#     df["Ridge_error"] = df["Ridge_validation"] - df["ΔΔG.expt."]
#     df = df.sort_values(by='Ridge_error', key=abs, ascending=[False])
#     PandasTools.AddMoleculeColumnToFrame(df, "smiles")
#     print(dfp[[_+"_validation_r2" for _ in columns]].max())
#     PandasTools.SaveXlsxFromFrame(df, save_name + "/λ_result.xlsx", size=(100, 100))
#     dfp.to_csv(save_name + "/λ_result.csv", index=False)
#     print(save_name)
if False:
    None
    # def regression_comparison(df, dfp, gaussian_penalize, save_name, n,n_splits,features):
    #     os.makedirs(save_name + "/molecular_field_csv",exist_ok=True)
    #     df_coord = pd.read_csv(gaussian_penalize + "/coordinates_yz.csv").sort_values(['x', 'y', "z"],
    #                                                                                   ascending=[True, True, True])
    #     ptp_name=gaussian_penalize + "/{}ptp{}.npy".format(len(features),n)
    #     features_all = np.array(df[features].values.tolist()).reshape(len(df), -1, len(features)).transpose(2, 0, 1)
    #     std = np.sqrt(np.average(features_all**2, axis=(1, 2))).reshape(features_all.shape[0], 1, 1)
    #     X = np.concatenate(features_all / std, axis=1)

<<<<<<< HEAD
    # models = []
    # gaussians = []
    predicts = []
    coefs_=[]
    for L, n_num in zip(dfp["lambda"], dfp["n_components"]):
        ridge = linear_model.Ridge(alpha=L * len(df), fit_intercept=False).fit(X, df["ΔΔG.expt."])
        lasso = linear_model.Lasso(alpha=L/2/100, fit_intercept=False).fit(X, df["ΔΔG.expt."])
        pls = PLSRegression(n_components=n_num).fit(X, df["ΔΔG.expt."])
        #coefにして、いれる
        # models.append([ridge, lasso, pls])
        # X = features
        # start=time.time()
        gaussian_coef = scipy.linalg.solve((X.T @ X + L * len(df) * np.load(ptp_name)).astype("float32"),
                                           (X.T @ df["ΔΔG.expt."].values).astype("float32"), assume_a="gen").T
        l=[gaussian_coef,ridge.coef_,lasso.coef_,pls.coef_[0]]
        columns=["Gaussian","Ridge","Lasso","PLS"]
        for ptpname in sorted(glob.glob(gaussian_penalize + "/{}ptp*.npy".format(len(features)))):
            gaussian_coef_all = scipy.linalg.solve((X.T @ X + L * len(df) * np.load(ptpname)).astype("float32"),
                                           (X.T @ df["ΔΔG.expt."].values).astype("float32"), assume_a="gen").T
            # print(ptpname)
            # print(gaussian_penalize +"/"+str(len(features))+"ptp"+ "(.*).npy")
            sigma = re.findall(gaussian_penalize +"/"+str(len(features))+"ptp"+ "(.*).npy", ptpname.replace(os.sep,'/'))
            n = sigma[0]
            columns.append("Gaussian"+n)
            l.append(gaussian_coef_all)

        coefs_.append(l)
        # print("scipy time",time.time()-start)
        # gaussians.append(gaussian_coef.tolist())
        n = int(gaussian_coef.shape[0] / features_all.shape[0])
<<<<<<< HEAD
        # print(gaussian_coef[:n].shape)
        df_coord[["Gaussian_Dt","Ridge_Dt","Lasso_Dt","PLS_Dt"]]=np.stack((gaussian_coef,ridge.coef_,lasso.coef_,pls.coef_[0]),axis=1)[:n]* std[0].reshape([1])#,lasso.coef_[:n].tolist(),pls.coef_[0][:n].tolist()]
        # df_coord["Gaussian_Dt"] = gaussian_coef[:n] * std[0].reshape([1])
        # # df_coord["Gaussian_ESP"]=gaussian_coef[n:]
        # df_coord["Ridge_Dt"] = ridge.coef_[:n] * std[0].reshape([1])
        # # df_coord["Ridge_ESP"]=ridge.coef_[n:]
        # df_coord["Lasso_Dt"] = lasso.coef_[:n] * std[0].reshape([1])
        # # df_coord["Lasso_ESP"]=lasso.coef_[n:]
        # df_coord["PLS_Dt"] = pls.coef_[0][:n] * std[0].reshape([1])
        # # df_coord["PLS_Dt"] = pls.coef_[0][:n] * np.std(features, axis=0) * std[0].reshape([1])

<<<<<<< HEAD
<<<<<<< HEAD
        kf = KFold(n_splits=2, shuffle=False)
        # kf = KFold(n_splits=len(df), shuffle=False)
=======
=======
        #df_coord[["Gaussian_Dt","Ridge_Dt","Lasso_Dt","PLS_Dt"]]=(np.stack((gaussian_coef[:n],ridge.coef_[:n],lasso.coef_[:n],pls.coef_[0][:n])) * std[0]).tolist()#.reshape([1,1])
>>>>>>> 847d536 (crean up run comfa)
        kf = KFold(n_splits=int(n_splits), shuffle=False)
>>>>>>> cb7fd67 (from windows)
        gaussian_predicts = []
        ridge_predicts = []
        lasso_predicts = []
        pls_predicts = []

=======
        df_coord[[_+"_Dt" for _ in columns]]=np.stack((l),axis=1)[:n]* std[0].reshape([1])#,lasso.coef_[:n].tolist(),pls.coef_[0][:n].tolist()]
        kf = KFold(n_splits=int(n_splits), shuffle=False)
        # gaussian_predicts = []
        # ridge_predicts = []
        # lasso_predicts = []
        # pls_predicts = []
        predicts_=[]
>>>>>>> bd2c239 (from win)
        for i, (train_index, test_index) in enumerate(kf.split(df)):
            features_training = features_all[:, train_index]
            # std_ = np.sqrt(np.sum(features_training**2/features_training.shape[1], axis=(1, 2))).reshape(features_all.shape[0], 1, 1)

            std_ = np.sqrt(np.average(features_training**2, axis=(1, 2))).reshape(features_all.shape[0], 1, 1)
            # std_ = np.std(features_training, axis=(1, 2)).reshape(features_all.shape[0], 1, 1)
            features_training = features_training / std_
            features_training = np.concatenate(features_training, axis=1)
            # start = time.time()
            X_ = features_training
            Y = df.iloc[train_index]["ΔΔG.expt."].values
            gaussian_coef_ = scipy.linalg.solve((X_.T @ X_ + L * len(train_index)  * np.load(ptp_name)).astype("float32"), (X_.T @ Y).astype("float32"),
                                                assume_a="gen").T
            # print("after__",gaussian_coef_,time.time()-start)
            # print(time.time()-start)
            ridge = linear_model.Ridge(alpha=L * len(train_index) , fit_intercept=False).fit(
                features_training,
                df.iloc[train_index][
                    "ΔΔG.expt."])
            lasso = linear_model.Lasso(alpha=L/2/100, fit_intercept=False).fit(features_training,
                                                                         df.iloc[train_index]["ΔΔG.expt."])
            pls = PLSRegression(n_components=n_num).fit(features_training,
                                                        df.iloc[train_index]["ΔΔG.expt."])
            features_test = features_all[:, test_index]  # np.array(features_all)[test_index].transpose(2, 0, 1)
            features_test = np.concatenate(features_test / std_, axis=1)
            gaussian_predict = np.sum(gaussian_coef_ * features_test, axis=1).tolist()  # model.predict(features_test)
            # gaussian_predicts.extend(gaussian_predict)
            # predicts[0].extend(predict)
            ridge_predict = ridge.predict(features_test)
            # predicts[1].extend(ridge_predict)
            # ridge_predicts.extend(ridge_predict)
            lasso_predict = lasso.predict(features_test)
            # predicts[2].extend(lasso_predict)
            # lasso_predicts.extend(lasso_predict)
            # features_test_norm = features_test / np.std(features_training, axis=0)
            # pls_predict = pls.predict(features_test_norm)
            pls_predict = pls.predict(features_test)[:,0]
            # predicts[3].extend([_[0] for _ in pls_predict])
            # pls_predicts.extend(pls_predict)#[_[0] for _ in pls_predict])
            l=[gaussian_predict,ridge_predict,lasso_predict,pls_predict]
            for ptpname in sorted(glob.glob(gaussian_penalize + "/{}ptp*.npy".format(len(features)))):
                gaussian_coef_ = scipy.linalg.solve((X_.T @ X_ + L * len(train_index)  * np.load(ptpname)).astype("float32"), (X_.T @ Y).astype("float32"),
                                                assume_a="gen").T
                # print(ptpname)
                # print(gaussian_penalize +"/"+str(len(features))+"ptp"+ "(.*).npy")
                sigma = re.findall(gaussian_penalize +"/"+str(len(features))+"ptp"+ "(.*).npy", ptpname.replace(os.sep,'/'))
                n = sigma[0]
                # columns.append("Gaussian"+n)
                gaussian_predict = np.sum(gaussian_coef_ * features_test, axis=1).tolist()
                l.append(gaussian_predict)
            predicts_.append(l)
            # print(ridge_predict.shape,pls_predict.shape)
            # n = int(gaussian_coef_.shape[0] / features_all.shape[0])
            # df_coord[["Gaussian_Dt{}".format(i),"Ridge_Dt{}".format(i),"Lasso_Dt{}".format(i),"PLS_Dt{}".format(i)]]=np.stack((gaussian_coef_,ridge.coef_,lasso.coef_,pls.coef_[0]),axis=1)[:n] * std_[0].reshape([1])

        # predicts.append([gaussian_predicts, ridge_predicts, lasso_predicts, pls_predicts])
        predicts.append(np.concatenate(predicts_,axis=1).tolist())
        df_coord.to_csv(save_name + "/molecular_field_csv" + "/molecular_field{}.csv".format(L))
    
    dfp[[_+"_coef" for _ in columns]]=coefs_
    dfp[[_+"_regression_predict" for _ in columns]]=dfp[[_+"_coef" for _ in columns]].applymap(lambda coef: np.sum(coef.reshape([1,-1])*X,axis=1))
    # dfp[["Ridge_model", "Lasso_model", "PLS_model"]] = models
    # dfp[["Ridge_regression_predict", "Lasso_regression_predict", "PLS_regression_predict"]] \
    #     = dfp[["Ridge_model", "Lasso_model", "PLS_model"]].applymap(lambda model: model.predict(features))
    # gaussians = np.array(gaussians)
    # gaussians = gaussians.reshape(gaussians.shape[0], 1, -1)
    # features = features.reshape(1, features.shape[0], -1)
    # dfp["Gaussian_regression_predict"] = np.sum(gaussians * features, axis=2).tolist()

    dfp[[_+"_regression_r2" for _ in columns]] = dfp[[_+"_regression_predict" for _ in columns]].applymap(
        lambda predict: r2_score(df["ΔΔG.expt."], predict))
    dfp[[_+"_regression_RMSE" for _ in columns]] = dfp[[_+"_regression_predict" for _ in columns]].applymap(
        lambda predict: np.sqrt(mean_squared_error(df["ΔΔG.expt."], predict)))

    # dfp[["Gaussian_test_predict", "ridge_test_predict", "lasso_test_predict", "pls_test_predict"]] = predicts
    # dfp[["Gaussian_test_r2", "ridge_test_r2", "lasso_test_r2", "pls_test_r2"]] = dfp[
    #     ["Gaussian_test_predict", "ridge_test_predict", "lasso_test_predict", "pls_test_predict"]].applymap(
    #     lambda predict: r2_score(df["ΔΔG.expt."], predict))
    # dfp[["Gaussian_test_r", "ridge_test_r", "lasso_test_r", "pls_test_r"]] = dfp[
    #     ["Gaussian_test_predict", "ridge_test_predict", "lasso_test_predict", "pls_test_predict"]].applymap(
    #     lambda predict: np.corrcoef(df["ΔΔG.expt."], predict)[1, 0])
    # dfp[["Gaussian_test_RMSE", "ridge_test_RMSE", "lasso_test_RMSE", "pls_test_RMSE"]] = dfp[
    #     ["Gaussian_test_predict", "ridge_test_predict", "lasso_test_predict", "pls_test_predict"]].applymap(
    #     lambda predict: mean_squared_error(df["ΔΔG.expt."], predict,squared=False))
    dfp[[_+"_validation_predict" for _ in columns]] = predicts
    dfp[[_+"_validation_r2" for _ in columns]] = dfp[[_+"_validation_predict" for _ in columns]].applymap(
        lambda predict: r2_score(df["ΔΔG.expt."], predict))
    dfp[[_+"_validation_r" for _ in columns]] = dfp[[_+"_validation_predict" for _ in columns]].applymap(
        lambda predict: np.corrcoef(df["ΔΔG.expt."], predict)[1, 0])
    dfp[[_+"_validation_RMSE" for _ in columns]] = dfp[[_+"_validation_predict" for _ in columns]].applymap(
        lambda predict: mean_squared_error(df["ΔΔG.expt."], predict,squared=False))
    
    features_R1 = np.array(df["DtR1"].tolist()).reshape(len(df), -1, 1).transpose(2, 0, 1)
    features_R1 = np.concatenate(features_R1 / std, axis=1)
    # dfp["Gaussian_regression_predictR1"] = np.sum(gaussians * features_R1, axis=2).tolist()
    # dfp[["Ridge_regression_predictR1", "Lasso_regression_predictR1", "PLS_regression_predictR1"]] \
    #     = dfp[["Ridge_model", "Lasso_model", "PLS_model"]].applymap(lambda model: model.predict(features_R1))
    dfp[[_+"_regression_predictR1" for _ in columns]]=dfp[[_+"_coef" for _ in columns]].applymap(lambda coef: np.sum(coef.reshape([1,-1])*features_R1,axis=1))
    features_R2 = np.array(df["DtR2"].tolist()).reshape(len(df), -1, 1).transpose(2, 0, 1)
    features_R2 = np.concatenate(features_R2 / std, axis=1)
    # dfp["Gaussian_regression_predictR2"] = np.sum(gaussians * features_R2, axis=2).tolist()
    # dfp[["Ridge_regression_predictR2", "Lasso_regression_predictR2", "PLS_regression_predictR2"]] \
    #     = dfp[["Ridge_model", "Lasso_model", "PLS_model"]].applymap(lambda model: model.predict(features_R2))
    dfp[[_+"_regression_predictR2" for _ in columns]]=dfp[[_+"_coef" for _ in columns]].applymap(lambda coef: np.sum(coef.reshape([1,-1])*features_R2,axis=1))
    # df[["Gaussian_validation", "Gaussian_regression", "Gaussian_R1", "Gaussian_R2"]] = \
    #     dfp.loc[dfp["Gaussian_validation_r"].idxmax()][
    #         ["Gaussian_validation_predict", "Gaussian_regression_predict", "Gaussian_regression_predictR1",
    #          "Gaussian_regression_predictR2"]]
    # df["Ridge_validation"] = dfp.loc[dfp["Ridge_validation_r"].idxmax()]["Ridge_validation_predict"]
    # df["Lasso_validation"] = dfp.loc[dfp["Lasso_validation_r"].idxmax()]["Lasso_validation_predict"]
    # df["PLS_validation"] = dfp.loc[dfp["PLS_validation_r"].idxmax()]["PLS_validation_predict"]
    for _ in columns:
        df[[_+"regression",_+"_validation",_+"_R1",_+"_R2"]]=dfp.loc[dfp[_+"_validation_RMSE"].idxmin()][[_+"_regression_predict",_+"_validation_predict",_+"_regression_predictR1",_+"_regression_predictR2"]]
    df["Gaussian_error"] = df["Gaussian_validation"] - df["ΔΔG.expt."]
    # df[["Gaussian_error","Ridge_error","Lasso_error"]] = df[["Gaussian_test","Ridge_test","Lasso_test",]].applymap(lambda test:test - df["ΔΔG.expt."].values)
    df = df.sort_values(by='Gaussian_error', key=abs, ascending=[False])
    PandasTools.AddMoleculeColumnToFrame(df, "smiles")
    print(dfp[["Gaussian_validation_r2", "Ridge_validation_r2", "Lasso_validation_r2", "PLS_validation_r2"]].max())
    PandasTools.SaveXlsxFromFrame(df, save_name + "/λ_result.xlsx", size=(100, 100))
    dfp.to_csv(save_name + "/λ_result.csv", index=False)
    print(save_name)

#CV


# def energy_to_Boltzmann_distribution(mol, RT=1.99e-3 * 273):
#     energies = np.array([float(conf.GetProp("energy")) for conf in mol.GetConformers()])
#     energies = energies - np.min(energies)
#     rates = np.exp(-energies / RT)
#     rates = rates / sum(rates)
#     for conf, rate in zip(mol.GetConformers(), rates):
#         conf.SetProp("Boltzmann_distribution", str(rate))
=======
    #     predicts = []
    #     coefs_=[]
    #     for L, n_num in zip(dfp["lambda"], dfp["n_components"]):
    #         ridge = linear_model.Ridge(alpha=L * len(df), fit_intercept=False).fit(X, df["ΔΔG.expt."])
    #         lasso = linear_model.Lasso(alpha=L/2/100, fit_intercept=False).fit(X, df["ΔΔG.expt."])
    #         pls = PLSRegression(n_components=n_num).fit(X, df["ΔΔG.expt."])
    #         gaussian_coef = scipy.linalg.solve((X.T @ X + L * len(df) * np.load(ptp_name)).astype("float32"),
    #                                            (X.T @ df["ΔΔG.expt."].values).astype("float32"), assume_a="gen").T
    #         l=[gaussian_coef,ridge.coef_,lasso.coef_,pls.coef_[0]]
    #         columns=["Gaussian","Ridge","Lasso","PLS"]
    #         for ptpname in sorted(glob.glob(gaussian_penalize + "/{}ptp*.npy".format(len(features)))):
    #             gaussian_coef_all = scipy.linalg.solve((X.T @ X + L * len(df) * np.load(ptpname)).astype("float32"),
    #                                            (X.T @ df["ΔΔG.expt."].values).astype("float32"), assume_a="gen").T
    #             sigma = re.findall(gaussian_penalize +"/"+str(len(features))+"ptp"+ "(.*).npy", ptpname.replace(os.sep,'/'))
    #             n = sigma[0]
    #             columns.append("Gaussian"+n)
    #             l.append(gaussian_coef_all)
>>>>>>> f93d900 (code creanup)

    #         coefs_.append(l)
    #         n = int(gaussian_coef.shape[0] / features_all.shape[0])
    #         df_coord[[_+"_Dt" for _ in columns]]=np.stack((l),axis=1)[:n]* std[0].reshape([1])#,lasso.coef_[:n].tolist(),pls.coef_[0][:n].tolist()]
    #         kf = KFold(n_splits=int(n_splits), shuffle=False)
    #         predicts_=[]
    #         for i, (train_index, test_index) in enumerate(kf.split(df)):
    #             features_training = features_all[:, train_index]
    #             std_ = np.sqrt(np.average(features_training**2, axis=(1, 2))).reshape(features_all.shape[0], 1, 1)
    #             features_training = features_training / std_
    #             features_training = np.concatenate(features_training, axis=1)
    #             X_ = features_training
    #             Y = df.iloc[train_index]["ΔΔG.expt."].values
    #             gaussian_coef_ = scipy.linalg.solve((X_.T @ X_ + L * len(train_index)  * np.load(ptp_name)).astype("float32"), (X_.T @ Y).astype("float32"),
    #                                                 assume_a="gen").T
    #             ridge = linear_model.Ridge(alpha=L * len(train_index) , fit_intercept=False).fit(
    #                 features_training,
    #                 df.iloc[train_index][
    #                     "ΔΔG.expt."])
    #             lasso = linear_model.Lasso(alpha=L/2/100, fit_intercept=False).fit(features_training,
    #                                                                          df.iloc[train_index]["ΔΔG.expt."])
    #             pls = PLSRegression(n_components=n_num).fit(features_training,
    #                                                         df.iloc[train_index]["ΔΔG.expt."])
    #             features_test = features_all[:, test_index]  # np.array(features_all)[test_index].transpose(2, 0, 1)
    #             features_test = np.concatenate(features_test / std_, axis=1)
    #             gaussian_predict = np.sum(gaussian_coef_ * features_test, axis=1).tolist()  # model.predict(features_test)
    #             ridge_predict = ridge.predict(features_test)
    #             lasso_predict = lasso.predict(features_test)
    #             pls_predict = pls.predict(features_test)[:,0]
    #             l=[gaussian_predict,ridge_predict,lasso_predict,pls_predict]
    #             for ptpname in sorted(glob.glob(gaussian_penalize + "/{}ptp*.npy".format(len(features)))):
    #                 gaussian_coef_ = scipy.linalg.solve((X_.T @ X_ + L * len(train_index)  * np.load(ptpname)).astype("float32"), (X_.T @ Y).astype("float32"),
    #                                                 assume_a="gen").T
    #                 sigma = re.findall(gaussian_penalize +"/"+str(len(features))+"ptp"+ "(.*).npy", ptpname.replace(os.sep,'/'))
    #                 n = sigma[0]
    #                 gaussian_predict = np.sum(gaussian_coef_ * features_test, axis=1).tolist()
    #                 l.append(gaussian_predict)
    #             predicts_.append(l)
    #         predicts.append(np.concatenate(predicts_,axis=1).tolist())
    #         df_coord.to_csv(save_name + "/molecular_field_csv" + "/molecular_field{}.csv".format(L))
        
    #     dfp[[_+"_coef" for _ in columns]]=coefs_
    #     dfp[[_+"_regression_predict" for _ in columns]]=dfp[[_+"_coef" for _ in columns]].applymap(lambda coef: np.sum(coef.reshape([1,-1])*X,axis=1))
    #     dfp[[_+"_regression_r2" for _ in columns]] = dfp[[_+"_regression_predict" for _ in columns]].applymap(
    #         lambda predict: r2_score(df["ΔΔG.expt."], predict))
    #     dfp[[_+"_regression_RMSE" for _ in columns]] = dfp[[_+"_regression_predict" for _ in columns]].applymap(
    #         lambda predict: np.sqrt(mean_squared_error(df["ΔΔG.expt."], predict)))
    #     dfp[[_+"_validation_predict" for _ in columns]] = predicts
    #     dfp[[_+"_validation_r2" for _ in columns]] = dfp[[_+"_validation_predict" for _ in columns]].applymap(
    #         lambda predict: r2_score(df["ΔΔG.expt."], predict))
    #     dfp[[_+"_validation_r" for _ in columns]] = dfp[[_+"_validation_predict" for _ in columns]].applymap(
    #         lambda predict: np.corrcoef(df["ΔΔG.expt."], predict)[1, 0])
    #     dfp[[_+"_validation_RMSE" for _ in columns]] = dfp[[_+"_validation_predict" for _ in columns]].applymap(
    #         lambda predict: mean_squared_error(df["ΔΔG.expt."], predict,squared=False))
        
    #     features_R1 = np.array(df["DtR1"].tolist()).reshape(len(df), -1, 1).transpose(2, 0, 1)
    #     features_R1 = np.concatenate(features_R1 / std, axis=1)
    #     dfp[[_+"_regression_predictR1" for _ in columns]]=dfp[[_+"_coef" for _ in columns]].applymap(lambda coef: np.sum(coef.reshape([1,-1])*features_R1,axis=1))
    #     features_R2 = np.array(df["DtR2"].tolist()).reshape(len(df), -1, 1).transpose(2, 0, 1)
    #     features_R2 = np.concatenate(features_R2 / std, axis=1)
    #     dfp[[_+"_regression_predictR2" for _ in columns]]=dfp[[_+"_coef" for _ in columns]].applymap(lambda coef: np.sum(coef.reshape([1,-1])*features_R2,axis=1))
    #     for _ in columns:
    #         df[[_+"regression",_+"_validation",_+"_R1",_+"_R2"]]=dfp.loc[dfp[_+"_validation_RMSE"].idxmin()][[_+"_regression_predict",_+"_validation_predict",_+"_regression_predictR1",_+"_regression_predictR2"]]
    #     df["Gaussian_error"] = df["Gaussian_validation"] - df["ΔΔG.expt."]
    #     df = df.sort_values(by='Gaussian_error', key=abs, ascending=[False])
    #     PandasTools.AddMoleculeColumnToFrame(df, "smiles")
    #     print(dfp[["Gaussian_validation_r2", "Ridge_validation_r2", "Lasso_validation_r2", "PLS_validation_r2"]].max())
    #     PandasTools.SaveXlsxFromFrame(df, save_name + "/λ_result.xlsx", size=(100, 100))
    #     dfp.to_csv(save_name + "/λ_result.csv", index=False)
    #     print(save_name)
=======
>>>>>>> 36b620d (crean up)
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
                print(filename, "fail_freq_calculation")
                return False
            for l in frequencies_lines:
                splited = l.split()
                values = splited[2:]
                values = [float(v)>0 for v in values]
                ans=all(values)
                if not ans:
                    f.close()
                    print(filename,"imaginary freq")
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
    data[features] = np.nan_to_num(np.average(feat, weights=we, axis=0))
    # data.to_pickle( "{}/{}/dataBoltzmann.pkl".format(dir, mol.GetProp("InchyKey")))
    ans1=data[(data["y"] > 0)&(data["z"]>0)].sort_values(['x', 'y', "z"], ascending=[True, True, True])[features].values
    ans2=data[(data["y"] < 0)&(data["z"]>0)].sort_values(['x', 'y', "z"], ascending=[True, False, True])[features].values
    ans3=data[(data["y"] > 0)&(data["z"]<0)].sort_values(['x', 'y', "z"], ascending=[True, True, False])[features].values
    ans4=data[(data["y"] < 0)&(data["z"]<0)].sort_values(['x', 'y', "z"], ascending=[True, False, False])[features].values
    ans=ans1+ans2-ans3-ans4
    # ans.to_pickle( "{}/{}/dataBoltzmannfold.pkl".format(dir, mol.GetProp("InchyKey")))
    return ans.T.tolist()


if __name__ == '__main__':
    ridge_input=[]
    lasso_input=[]
    elasticnet_input=[]
    pls_input=[]
    pcr_input=[]
    for param_name in sorted(glob.glob("C:/Users/poclabws/PycharmProjects/CoMFA_model/parameter/cube_to_grid0.50.txt"),reverse=True):
        print(param_name)
        with open(param_name, "r") as f:
            param = json.loads(f.read())
        print(param)
        os.makedirs(param["out_dir_name"],exist_ok=True)
        start = time.perf_counter()  # 計測開始
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        # for file in glob.glob("../arranged_dataset/newrea/*"):
        for file in glob.glob("../arranged_dataset/*.xlsx"):
<<<<<<< HEAD
=======
        for file in glob.glob("../arranged_dataset/*.xlsx"):
        # for file in glob.glob("../arranged_dataset/*.xlsx"):
>>>>>>> cb7fd67 (from windows)

=======
        for file in glob.glob("../arranged_dataset/c*.xlsx"):
>>>>>>> 847d536 (crean up run comfa)
=======
        for file in glob.glob("../arranged_dataset/*.xlsx"):
>>>>>>> bd2c239 (from win)
=======
        for file in glob.glob("../arranged_dataset/RuSS.xlsx"):
>>>>>>> 8b8e38e (from win)
=======
        for file in glob.glob("../arranged_dataset/*.xlsx"):
>>>>>>> bed7344 (from win)
            df = pd.read_excel(file).dropna(subset=['smiles']).reset_index(drop=True)  # [:50]
=======
=======
        for file in glob.glob("C:/Users/poclabws/PycharmProjects/CoMFA_model/arranged_dataset/*.xlsx"):
>>>>>>> 66e5b02 (cleaning)
            df = pd.read_excel(file).dropna(subset=['smiles']).reset_index(drop=True)#[:10]
>>>>>>> e529ae1 (from win)
            file_name = os.path.splitext(os.path.basename(file))[0]
            features_dir_name = param["grid_coordinates"] + file_name
            print(len(df),features_dir_name)
            df["mol"] = df["smiles"].apply(calculate_conformation.get_mol)
            for mol in df["mol"]:
                if len(glob.glob(f'{param["grid_coordinates"]}/{mol.GetProp("InchyKey")}/*'))==0:
                    print(mol.GetProp("InchyKey"))
            # df = df[
            #     [len(glob.glob(f'{param["grid_coordinates"]}/{mol.GetProp("InchyKey")}/*'))>0 for mol in
            #      df["mol"]]]
            print(len(df),features_dir_name)
            df["mol"].apply(
                lambda mol: calculate_conformation.read_xyz(mol,
                                                            param["opt_structure"] + "/" + mol.GetProp("InchyKey")))
            df["mol"].apply(lambda mol : get_free_energy(mol,param["freq_dir"]))
            df=df[[mol.GetNumConformers()>0 for mol in df["mol"]]]
            print(len(df),features_dir_name)
            df[["Dt","ESP"]]=df[["mol", "RT"]].apply(lambda _: get_grid_feat(_[0], _[1],param["grid_coordinates"]), axis=1, result_type='expand')
            print("feature_calculated")
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
            inputs = []
            for _ in range(3):
                df_ = df.sample(frac=1, random_state=_)
                save_path = param["out_dir_name"] + "/" + file_name + "/σ_comparison"+"/validation" + str(_)
                field_csv_path = save_path +"/molecular_field_csv"
                # save_path = p
                os.makedirs(save_path, exist_ok=True)
                os.makedirs(field_csv_path, exist_ok=True)
                input = df_, dfp, param["grid_coordinates"], save_path
                inputs.append(input)
                # GP(input)
                # Gaussian_penalized(df_, dfp, param["grid_coordinates"], save_path)
            num_processes = multiprocessing.cpu_count()
            print(num_processes)
            p = multiprocessing.Pool(processes=4)
            p.map(GP, inputs)
            # from joblib import Parallel, delayed
            #
            # Parallel(n_jobs=-1)(delayed(GP)(input) for input in inputs)
            # n = select_σ_n(file_name)
            n = param["sigma"]
            inputs = []
            for _ in range(3):
=======
            # inputs = []
            # for _ in range(10):
            #     df_ = df.sample(frac=1, random_state=_)
            #     save_path = param["out_dir_name"] + "/" + file_name + "/comparison" + str(_)
            #     os.makedirs(save_path, exist_ok=True)
            #     input = df_, dfp, param["grid_coordinates"], save_path,param["n_splits"]
            #     inputs.append(input)
            # num_processes = multiprocessing.cpu_count()
            # print(num_processes)
            # p = multiprocessing.Pool(processes=20)
            # p.map(GP, inputs)
            n = param["sigma"]
            # inputs = []
            # for _ in range(10):
            #     df_ = df.sample(frac=1, random_state=_)
            #     save_path = param["out_dir_name"] + "/" + file_name + "/" + str(_)
            #     os.makedirs(save_path, exist_ok=True)
            #     input = df_, dfp, param["grid_coordinates"], save_path, n,param["n_splits"]
            #     inputs.append(input)
            # p.map(RC, inputs)
<<<<<<< HEAD
            for _ in range(10):
>>>>>>> cb7fd67 (from windows)
=======
            for _ in range(5):
>>>>>>> b2b0e28 (from windows)
=======
            n = param["sigma"]
=======
            n = "1.00"#param["sigma"]
>>>>>>> bd2c239 (from win)

            for _ in range(10):
>>>>>>> 847d536 (crean up run comfa)
                df_ = df.sample(frac=1, random_state=_)
                save_path = param["out_dir_name"] + "/" + file_name + "/λ_comparison"+"/validation" + str(_)
                field_csv_path = save_path +"/molecular_field_csv"
                os.makedirs(save_path, exist_ok=True)
<<<<<<< HEAD
<<<<<<< HEAD
                os.makedirs(field_csv_path, exist_ok=True)
                input = df_, dfp, param["grid_coordinates"], save_path, n
=======
                input = df_, dfp, param["grid_coordinates"], save_path, n, param["n_splits"],True
>>>>>>> cb7fd67 (from windows)
                inputs.append(input)
<<<<<<< HEAD
                input = df_, dfp, param["grid_coordinates"], save_path, n, param["n_splits"], False
=======
                input = df_, dfp, param["grid_coordinates"], save_path, n, param["n_splits"],True,param["features"]
                inputs.append(input)
                input = df_, dfp, param["grid_coordinates"], save_path, n, param["n_splits"], False,param["features"]
>>>>>>> b2b0e28 (from windows)
                inputs_.append(input)
            # p.map(run,inputs)
        # end = time.perf_counter()  # 計測終了
        # print('Finish{:.2f}'.format(end - start))
    p = multiprocessing.Pool(processes=30)
    # p.map(run, inputs)
    p = multiprocessing.Pool(processes=15)
    p.map(run, inputs_)
=======

            X=np.array(df["Dt"].values.tolist())/np.sqrt(np.average(np.array(df["Dt"].values.tolist())**2))
            X1=np.array(df["DtR1"].values.tolist())/np.sqrt(np.average(np.array(df["Dt"].values.tolist())**2))
            X2=np.array(df["DtR2"].values.tolist())/np.sqrt(np.average(np.array(df["Dt"].values.tolist())**2))

            X_esp=np.array(df["ESP"].values.tolist())/np.sqrt(np.average(np.array(df["ESP"].values.tolist())**2))
            X1_esp=np.array(df["ESPR1"].values.tolist())/np.sqrt(np.average(np.array(df["ESP"].values.tolist())**2))
            X2_esp=np.array(df["ESPR2"].values.tolist())/np.sqrt(np.average(np.array(df["ESP"].values.tolist())**2))
            X_dual=np.array(df["DUAL"].values.tolist())/np.sqrt(np.average(np.array(df["DUAL"].values.tolist())**2))
            X1_dual=np.array(df["DUALR1"].values.tolist())/np.sqrt(np.average(np.array(df["DUAL"].values.tolist())**2))
            X2_dual=np.array(df["DUALR2"].values.tolist())/np.sqrt(np.average(np.array(df["DUAL"].values.tolist())**2))
            print(X.shape)
=======
            # X=np.array(df["Dt"].values.tolist())/np.sqrt(np.sum(np.array(df["Dt"].values.tolist())**2))
            # X_esp=np.array(df["ESP"].values.tolist())/np.sqrt(np.sum(np.array(df["ESP"].values.tolist())**2))
>>>>>>> 7393452 (kisoyuuki_after)
            df_coord = pd.read_csv(param["grid_coordinates"] + "/coordinates_yz.csv").sort_values(['x', 'y', "z"],
                                                                                  ascending=[True, True, True])
            n_splits = int(param["n_splits"])
            n_repeats = int(param["n_repeats"])
            dir=param["out_dir_name"] + "/{}".format(file_name)
            os.makedirs(dir,exist_ok=True)

            alphas = np.logspace(-20,-10,11,base=2)
            l1ratios = np.linspace(0, 1, 11)

            for alpha in alphas:
                for l1ratio in l1ratios:
                    save_name=dir + "/ElasticNet_alpha_{}_l1ratio{}".format(alpha,l1ratio)
                    elasticnet_input.append([alpha,l1ratio,n_splits,n_repeats,df,df_coord,save_name])

            alphas = np.arange(1,10)
            for alpha in alphas:
                save_name=dir + "/PLS_alpha_{}".format(alpha)
                pls_input.append([alpha,n_splits,n_repeats,df,df_coord,save_name])
            # alphas = np.arange(1,10)
            # for alpha in alphas:
            #     save_name=dir + "/PCR_alpha_{}".format(alpha)
            #     pcr_input.append([alpha,n_splits,n_repeats,df,df_coord,save_name])
    p = multiprocessing.Pool(processes=int(param["processes"]))
<<<<<<< HEAD
    columns=["savefilename","alpha","RMSE_regression", "r2_regression","RMSE_validation", "r2_validation"]
    decimals = {"RMSE_regression": 5,"r2_regression":5,"RMSE_validation":5, "r2_validation":5}
    pd.DataFrame(p.map(Gaussian,gaussian_input), columns=["savefilename","alpha","sigma","RMSE_regression", "r2_regression","RMSE_validation", "r2_validation", "RMSE_PCA_validation","r2_PCA_validation"]+["integral","x_moment","y_moment","z_moment"]).round(decimals).to_csv(param["out_dir_name"] + "/Gaussian.csv")
    pd.DataFrame(p.map(Ridge,ridge_input), columns=columns+["RMSE_PCA_validation","r2_PCA_validation"]).round(decimals).to_csv(param["out_dir_name"] + "/Ridge.csv")
    pd.DataFrame(p.map(Lasso,lasso_input), columns=columns+["RMSE_PCA_validation","r2_PCA_validation"]).round(decimals).to_csv(param["out_dir_name"] + "/Lasso.csv")
    pd.DataFrame(p.map(PLS,pls_input), columns=columns+["RMSE_PCA_validation","r2_PCA_validation"]).round(decimals).to_csv(param["out_dir_name"] + "/PLS.csv")
<<<<<<< HEAD
<<<<<<< HEAD
    # p.map(run, inputs_)
>>>>>>> f93d900 (code creanup)
=======
    pd.DataFrame(p.map(Gaussian,gaussian_input), columns=["savefilename","alpha","sigma","RMSE_regression", "r2_regression","RMSE_validation", "r2_validation", "RMSE_PCA_validation","r2_PCA_validation"]+["integral","x_moment","y_moment","z_moment"]).round(decimals).to_csv(param["out_dir_name"] + "/Gaussian.csv")
>>>>>>> 3a83914 (code creanup)
=======
>>>>>>> e529ae1 (from win)
=======
    columns=["savefilename","alpha","RMSE_regression", "r2_regression","RMSE_validation", "r2_validation","RMSE_PCA_validation","r2_PCA_validation","integral_steric","integral_electric"]
    elasticnet_columns=["savefilename","alpha","l1ratio","RMSE_regression", "r2_regression","RMSE_validation", "r2_validation","RMSE_PCA_validation","r2_PCA_validation","integral_steric","integral_electric"]
    os.makedirs(param["out_dir_name"],exist_ok=True)
    pd.DataFrame(p.map(PLS,pls_input), columns=columns).apply(lambda x: x.round(4) if np.issubdtype(x.dtype, np.floating) else x).to_csv(param["out_dir_name"] + "/PLS.csv")
    pd.DataFrame(p.map(ElasticNet,elasticnet_input), columns=elasticnet_columns).apply(lambda x: x.round(4) if np.issubdtype(x.dtype, np.floating) else x).to_csv(param["out_dir_name"] + "/ElasticNet.csv")
<<<<<<< HEAD
    # pd.DataFrame(p.map(Gaussian,gaussian_input), columns=["savefilename","alpha","sigma","RMSE_regression", "r2_regression","RMSE_validation", "r2_validation", "RMSE_PCA_validation","r2_PCA_validation"]+["integral_steric","integral_electric","integral_LUMO","x_moment","y_moment","z_moment"]).apply(lambda x: x.round(4) if np.issubdtype(x.dtype, np.floating) else x).to_csv(param["out_dir_name"] + "/Gaussian.csv")
>>>>>>> ecb8078 (20240808commit)
=======
>>>>>>> 66e5b02 (cleaning)
    end = time.perf_counter()  # 計測終了
    print('Finish{:.2f}'.format(end - start))
<<<<<<< HEAD
=======
                # regression_comparison(df_, dfp, param["grid_coordinates"], save_path, n)
            # p = multiprocessing.Pool(5)
            p.map(RC, inputs)
        end = time.perf_counter()  # 計測終了
        print('Finish{:.2f}'.format(end - start))



>>>>>>> bbdeec7 (runcomfa 保存先変更)
=======
    
>>>>>>> 7393452 (kisoyuuki_after)
