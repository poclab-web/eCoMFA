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
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold

import calculate_conformation

warnings.simplefilter('ignore')


# def Gaussian_penalized(df, dfp, gaussian_penalize, save_name,n_splits,features):
#     df_coord = pd.read_csv(gaussian_penalize + "/coordinates_yz.csv").sort_values(['x', 'y', "z"],
#                                                                                   ascending=[True, True, True])
#     features_all = np.array(df[features].values.tolist()).reshape(len(df), -1, len(features)).transpose(2, 0, 1)
#     # std = np.std(features_all, axis=(1, 2)).reshape(features_all.shape[0], 1, 1)
#     std = np.average(features_all, axis=(1, 2)).reshape(features_all.shape[0], 1, 1)
#     X = np.concatenate(features_all / std, axis=1).astype("float32")
#     XTY = (X.T @ df["ΔΔG.expt."].values).astype("float32")
#     gaussian_penalize=gaussian_penalize.replace(os.sep,'/')
#     os.makedirs(save_name + "/molecular_field_csv", exist_ok=True)
#     if len(features)==1:
#         ptpname_="/ptp"
#     else:
#         ptpname_="/2ptp"
#     # ptpname_="/{}ptp".format(len(features))
#     for ptpname in sorted(glob.glob(gaussian_penalize + ptpname_+"*.npy")):
#         ptpname=ptpname.replace(os.sep,'/')
#         sigma = re.findall(gaussian_penalize + ptpname_+"(.*).npy", ptpname)
#         n = sigma[0]
#         start = time.time()
#         # print("before", time.time() - start)
#         gaussians = []
#         predicts = []
#         for L in dfp["lambda"]:
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
#     PandasTools.AddMoleculeColumnToFrame(df, "smiles")
#     PandasTools.SaveXlsxFromFrame(df, save_name  + "/σ_result.xlsx", size=(100, 100))
#     dfp.to_csv(save_name  + "/σ_result.csv")
#     print(save_name)


def regression_comparison(df, dfp, gaussian_penalize, save_name, n,n_splits,features):
    os.makedirs(save_name + "/molecular_field_csv",exist_ok=True)
    df_coord = pd.read_csv(gaussian_penalize + "/coordinates_yz.csv").sort_values(['x', 'y', "z"],
                                                                                  ascending=[True, True, True])
    # if len(features)==1:
    #     ptp_name=gaussian_penalize + "/{}ptp{}.npy".format(len(features),n)
    # else:
    #     ptp_name=gaussian_penalize+"/2ptp{}.npy".format(str(n))
    ptp_name=gaussian_penalize + "/{}ptp{}.npy".format(len(features),n)
    features_all = np.array(df[features].values.tolist()).reshape(len(df), -1, len(features)).transpose(2, 0, 1)
    # std = np.std(features_all, axis=(1, 2)).reshape(features_all.shape[0], 1, 1)
    std = np.sqrt(np.average(features_all**2, axis=(1, 2))).reshape(features_all.shape[0], 1, 1)
    # std = np.sqrt(np.sum(features_all**2/features_all.shape[1], axis=(1, 2))).reshape(features_all.shape[0], 1, 1)
    X = np.concatenate(features_all / std, axis=1)

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


# def energy_to_Boltzmann_distribution(mol, RT=1.99e-3 * 273):
#     energies = np.array([float(conf.GetProp("energy")) for conf in mol.GetConformers()])
#     energies = energies - np.min(energies)
#     rates = np.exp(-energies / RT)
#     rates = rates / sum(rates)
#     for conf, rate in zip(mol.GetConformers(), rates):
#         conf.SetProp("Boltzmann_distribution", str(rate))

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

# from cclib.io import ccread

def is_normal_frequencies(filename):
    try:
        start = time.perf_counter()
        # data = ccread(filename)
        # print(11,time.perf_counter() - start)

        # data.enthalpy * 627.5095  # hartree
        # data.entropy * 627.5095  # hartree

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
                # for v in values:
                #     if v < 0:
                #         f.close()
                #         print(filename, " is bad conformer")
                #         return False


            f.close()
        # print(time.perf_counter() - start)
        return True
    except:
        print(filename," is bad conformer")
        return False



# def select_σ_n(file_name):
#     Gaussian = [[] for i in range(10)]
#     for _ in range(5):
#         save_path = param["out_dir_name"] + "/" + file_name + "/comparison" + str(_)
#         print(save_path)
#         dfp_n = pd.read_csv(save_path + "/n_comparison.csv")
#         for j in range(1, 11):
#             Gaussian[j - 1].append(dfp_n["Gaussian_test_r2{}".format(j)].values.tolist())
#             # ax.plot(dfp["lambda"], dfp["lasso_test_r2"], color="red", linewidth=1, alpha=0.05)
#             # ax.plot(dfp["lambda"], dfp["ridge_test_r2"], color="green", linewidth=1, alpha=0.05)
#             # ax2.plot(range(1, len(dfp["lambda"]) + 1), dfp["pls_test_r2"], color="orange", linewidth=1, alpha=0.05)
#     gave = [[] for i in range(10)]
#     for j in range(1, 11):
#         gave[j - 1].append(np.average(Gaussian[j - 1]))
#     # print(Gaussian)
#     print("gave")
#     print(gave)
#
#     print(np.argmax(gave))
#     q = np.argmax(gave)
#     n = q + 1
#     print(n)
#     return n
# print(Gaussian[j-1])
# ax.plot(dfp["lambda"], np.average(Gaussian[j - 1], axis=0), "o",
#         label="n = " + str(j) + "\n{:.3f}".format(),
#         color=cm.hsv(j / 10), alpha=1)


# def GP(input):
#     df_, dfp, grid_coordinates, save_path,n_splits = input
#     Gaussian_penalized(df_, dfp, grid_coordinates, save_path,n_splits)


# def RC(input):
#     df_, dfp, grid_coordinates, save_path, n,n_splits = input
#     regression_comparison(df_, dfp, grid_coordinates, save_path, n,n_splits)
def run(input):
    df_, dfp, grid_coordinates, save_path, n, n_splits,flag,features = input
    if flag:
        Gaussian_penalized(df_, dfp, grid_coordinates, save_path, n_splits,features)
    else:
        regression_comparison(df_, dfp, grid_coordinates, save_path, n,n_splits,features)

if __name__ == '__main__':
    #time.sleep(60*60*24*2.5)
    inputs=[]
    inputs_=[]
    for param_name in sorted(glob.glob("../parameter/cube_to_grid/cube_to_grid0.250510.txt"),reverse=True):
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
        # for file in glob.glob("../arranged_dataset/newrea/*"):
        for file in glob.glob("../arranged_dataset/*.xlsx"):
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
            df = pd.read_excel(file).dropna(subset=['smiles']).reset_index(drop=True)  # [:50]
            file_name = os.path.splitext(os.path.basename(file))[0]
            features_dir_name = param["grid_coordinates"] + file_name
            print(len(df),features_dir_name)
            df["mol"] = df["smiles"].apply(calculate_conformation.get_mol)
            # df = df[[os.path.isdir(features_dir_name + "/" + mol.GetProp("InchyKey")) for mol in df["mol"]]]
            print(len(df))
            df = df[
                [len(glob.glob("{}/{}/*".format(param["grid_coordinates"], mol.GetProp("InchyKey"))))>0 for mol in
                 df["mol"]]]
            print(len(df))
            # df = df[[os.path.isdir("{}/{}".format(param["freq_dir"], mol.GetProp("InchyKey"))) for mol in
            #          df["mol"]]]
            # freq = []
            # for mol in df["mol"]:
            #     freq_ = any([is_normal_frequencies(path) for path in
            #                  sorted(
            #                      glob.glob(
            #                          param["freq_dir"] + "/" + mol.GetProp("InchyKey") + "/gaussianinput?.log"))])

            #     # print(mol.GetProp("InchyKey"), freq_)
            #     freq.append(freq_)
            # df=df[freq]
            df["mol"].apply(
                lambda mol: calculate_conformation.read_xyz(mol,
                                                            param["opt_structure"] + "/" + mol.GetProp("InchyKey")))
            # grid = []
            # for mol in df["mol"]:
            #     freq_ = all([os.path.isfile(
            #         "{}/{}/data{}.pkl".format(param["grid_coordinates"], mol.GetProp("InchyKey"), conf.GetId())) for
            #         conf in mol.GetConformers()])

            #     # print(mol.GetProp("InchyKey"), freq_)
            #     grid.append(freq_)
            # df = df[grid]
            print("dflen", len(df))

            for mol in df["mol"]:
                dirs_name_freq = param["freq_dir"] + "/" + mol.GetProp("InchyKey") + "/gaussianinput?.log"
                # print(dirs_name_freq)
                del_list=[]
                # for path, conf in zip(
                #         sorted(glob.glob(dirs_name_freq)),
                #         mol.GetConformers()):
                for conf in mol.GetConformers():
                    path=param["freq_dir"] + "/" + mol.GetProp("InchyKey") + "/gaussianinput{}.log".format(conf.GetId())
                    # print(path)
                    
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
            df=df[[mol.GetNumConformers()>0 for mol in df["mol"]]]
            print("dflen", len(df))
            dfp = pd.read_csv(param["grid_coordinates"] + "/penalty_param.csv")
            Dts = []
            DtR1 = []
            DtR2 = []
            ESPs = []
            ESPR1 = []
            ESPR2 = []
            for mol, RT in df[["mol", "RT"]].values:
                print(mol.GetProp("InchyKey"))
                energy_to_Boltzmann_distribution(mol, RT)
                Dt = []
                ESP = []
                we = []
                for conf in mol.GetConformers():
                    data = pd.read_pickle(
                        "{}/{}/data{}.pkl".format(param["grid_coordinates"], mol.GetProp("InchyKey"), conf.GetId()))
                    Bd = float(conf.GetProp("Boltzmann_distribution"))
                    we.append(Bd)
                    Dt.append(data["Dt"].values.tolist())
                    ESP.append(data["ESP"].values.tolist())
                Dt = np.array(Dt)
                ESP = np.array(ESP)
                w = np.exp(-Dt / np.sqrt(np.average(Dt ** 2, axis=0)).reshape(1, -1))
                w = np.exp(-Dt / np.max(Dt, axis=0).reshape(1, -1))
                # w = np.exp(-Dt/np.sqrt(np.average(Dt**2)))
                # w = np.exp(-Dt)
                print(np.max(w),np.min(w),np.average(w))
                data["Dt"] = np.nan_to_num(
                    np.average(Dt, weights=np.array(we).reshape(-1, 1) * np.ones(shape=Dt.shape), axis=0))
                w = np.exp(ESP / np.sqrt(np.average(ESP ** 2, axis=0)).reshape(1, -1))
                data["ESP"] = np.nan_to_num(
                    np.average(ESP, weights=np.array(we).reshape(-1, 1) * np.ones(shape=ESP.shape), axis=0))
                data_y = data[data["y"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])
                # data_y["Dt"] = data[data["y"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])[
                #                    "Dt"].values + \
                #                data[data["y"] < 0].sort_values(['x', 'y', "z"], ascending=[True, False, True])[
                #                    "Dt"].values
                # data_y["ESP"] = data[data["y"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])[
                #                     "ESP"].values + \
                #                 data[data["y"] < 0].sort_values(['x', 'y', "z"], ascending=[True, False, True])[
                #                     "ESP"].values
                data_y[["Dt","ESP"]] = data[data["y"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])[[
                    "Dt","ESP"]].values + \
                        data[data["y"] < 0].sort_values(['x', 'y', "z"], ascending=[True, False, True])[[
                    "Dt","ESP"]].values
                data_yz = data_y[data_y["z"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])
                # data_yz["Dt"] = data_y[data_y["z"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])[
                #                     "Dt"].values - \
                #                 data_y[data_y["z"] < 0].sort_values(['x', 'y', "z"], ascending=[True, True, False])[
                #                     "Dt"].values
                # data_yz["ESP"] = data_y[data_y["z"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])[
                #                      "ESP"].values - \
                #                  data_y[data_y["z"] < 0].sort_values(['x', 'y', "z"], ascending=[True, True, False])[
                #                      "ESP"].values
                data_yz[["Dt","ESP"]] = data_y[data_y["z"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])[["Dt","ESP"]].values - \
                                data_y[data_y["z"] < 0].sort_values(['x', 'y', "z"], ascending=[True, True, False])[["Dt","ESP"]].values
                # data_yz["DtR1"] = \
                #     data_y[data_y["z"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])["Dt"].values
                # data_yz["DtR2"] = \
                #     data_y[data_y["z"] < 0].sort_values(['x', 'y', "z"], ascending=[True, True, False])["Dt"].values
                # data_yz["ESPR1"] = \
                #     data_y[data_y["z"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])["ESP"].values
                # data_yz["ESPR2"] = \
                #     data_y[data_y["z"] < 0].sort_values(['x', 'y', "z"], ascending=[True, True, False])["ESP"].values
                data_yz[["DtR1","ESPR1"]] = \
                    data_y[data_y["z"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])[["Dt","ESP"]].values
                data_yz[["DtR2","ESPR2"]] = \
                    data_y[data_y["z"] < 0].sort_values(['x', 'y', "z"], ascending=[True, True, False])[["Dt","ESP"]].values
                dfp_yz = data_yz.copy()

                Dts.append(dfp_yz["Dt"].values.tolist())
                DtR1.append(dfp_yz["DtR1"].values.tolist())
                DtR2.append(dfp_yz["DtR2"].values.tolist())
                ESPs.append(dfp_yz["ESP"].values.tolist())
                ESPR1.append(dfp_yz["ESPR1"].values.tolist())
                ESPR2.append(dfp_yz["ESPR2"].values.tolist())

            df["Dt"] = Dts
            df["DtR1"] = DtR1
            df["DtR2"] = DtR2
            df["ESP"] = ESPs
            df["ESPR1"] = ESPR1
            df["ESPR2"] = ESPR2

            print("feature_calculated")
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
    end = time.perf_counter()  # 計測終了
    print('Finish{:.2f}'.format(end - start))
=======
                # regression_comparison(df_, dfp, param["grid_coordinates"], save_path, n)
            # p = multiprocessing.Pool(5)
            p.map(RC, inputs)
        end = time.perf_counter()  # 計測終了
        print('Finish{:.2f}'.format(end - start))



>>>>>>> bbdeec7 (runcomfa 保存先変更)
