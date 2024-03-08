import glob
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
from sklearn import linear_model
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold

import calculate_conformation

warnings.simplefilter('ignore')


def Gaussian_penalized(grid_dir, df, dfp, gaussian_penalize, save_name):
    # Gaussian Lasso Ridgeを行う。#PLSなどは別？
    df_coord = pd.read_csv(gaussian_penalize + "/coordinates_yz.csv").sort_values(['x', 'y', "z"],
                                                                                  ascending=[True, True, True])
    penalty = np.load(gaussian_penalize + "/penalty.npy")

    # features_all = []
    # for mol in df["mol"]:
    #     l=[]
    #     for conf in mol.GetConformers():
    #         data=pd.read_csv("{}/{}/data_yz_{}.pkl".format(grid_dir, mol.GetProp("InchyKey"),conf.GetId()))
    #         l.append(data["Dt"].values)
    #     #df_grid = pd.read_csv("{}/{}/feature_yz.csv".format(grid_dir, mol.GetProp("InchyKey")))
    #     feat = df_grid[["Dt"]].values
    #     features_all.append(feat)
    # features_all=df["Dt"].values

    # print(np.array(df["Dt"].tolist()).reshape(len(df),-1,1).shape)
    features_all = np.array(df["Dt"].tolist()).reshape(len(df), -1, 1).transpose(2, 0, 1)
    # print(features_all.shape)
    # print(features_all)
    std = np.std(features_all, axis=(1, 2)).reshape(features_all.shape[0], 1, 1)

    features = features_all / std
    features = np.concatenate(features, axis=1)
    zeros = np.zeros(penalty.shape[0] * features_all.shape[0])

    models = []
    # gaussian_predicts_list = []
    # ridge_predicts_list = []
    # lasso_predicts_list = []
    predicts = []
    for L ,n_num in zip(dfp["lambda"],range(1,len(dfp)+1)):
        penalty_L = []
        for _ in range(features_all.shape[0]):
            penalty_L_ = []
            for __ in range(features_all.shape[0]):
                if _ == __:
                    penalty_L_.append(L * penalty)
                else:
                    penalty_L_.append(np.zeros(penalty.shape))
            penalty_L.append(penalty_L_)

        # penalty_L=[[L * penalty, np.zeros(penalty.shape)],
        #               [np.zeros(penalty.shape), L * penalty]]
        X = np.block([[features]] + penalty_L)
        Y = np.concatenate([df["ΔΔG.expt."], zeros], axis=0)
        model = linear_model.Ridge(alpha=0, fit_intercept=False).fit(X, Y)
        # Gaussian_penalized_models.append(model)
        # Ridge_models.append(ridge)
        # Lasso_models.append(lasso)
        ridge = linear_model.Ridge(alpha=L, fit_intercept=False).fit(features, df["ΔΔG.expt."])
        lasso = linear_model.Lasso(alpha=L, fit_intercept=False).fit(features, df["ΔΔG.expt."])
        features_norm=features/np.std(features,axis=0)
        pls = PLSRegression(n_components=n_num).fit(features_norm, df["ΔΔG.expt."])
        models.append([model, ridge, lasso,pls])

        gaussian_coef = model.coef_
        n = int(gaussian_coef.shape[0] / features_all.shape[0])
        df_coord["Gaussian_Dt"] = gaussian_coef[:n]
        # df_coord["Gaussian_ESP"]=gaussian_coef[n:]
        df_coord["Ridge_Dt"] = ridge.coef_[:n]
        # df_coord["Ridge_ESP"]=ridge.coef_[n:]
        df_coord["Lasso_Dt"] = lasso.coef_[:n]
        # df_coord["Lasso_ESP"]=lasso.coef_[n:]
        df_coord["PLS_Dt"] = pls.coef_[0][:n]*np.std(features,axis=0)
        df_coord.to_csv(save_name + "/molecular_filed{}.csv".format(L))

        kf = KFold(n_splits=5, shuffle=False)
        gaussian_predicts = []
        ridge_predicts = []
        lasso_predicts = []
        pls_predicts = []
        for (train_index, test_index) in kf.split(df):
            features_training = features_all[:, train_index]  # np.array(features_all)[train_index].transpose(2, 0, 1)
            std = np.std(features_training, axis=(1, 2)).reshape(features_all.shape[0], 1, 1)
            features_training = features_training / std
            features_training = np.concatenate(features_training, axis=1)
            X = np.block([[features_training]] + penalty_L)
            Y = np.concatenate([df.iloc[train_index]["ΔΔG.expt."], zeros], axis=0)
            model = linear_model.Ridge(alpha=0, fit_intercept=False).fit(X, Y)
            # model=linear_model.LinearRegression(fit_intercept=False).fit(X,Y)
            ridge = linear_model.Ridge(alpha=L, fit_intercept=False).fit(features_training,
                                                                         df.iloc[train_index]["ΔΔG.expt."])
            lasso = linear_model.Lasso(alpha=L, fit_intercept=False).fit(features_training,
                                                                         df.iloc[train_index]["ΔΔG.expt."])
            features_training_norm = features_training / np.std(features_training, axis=0)
            pls = PLSRegression(n_components=n_num).fit(features_training_norm,
                                                                     df.iloc[train_index]["ΔΔG.expt."])

            features_test = features_all[:, test_index]  # np.array(features_all)[test_index].transpose(2, 0, 1)
            features_test = features_test / std
            features_test = np.concatenate(features_test, axis=1)

            predict = model.predict(features_test)
            gaussian_predicts.extend(predict)
            ridge_predict = ridge.predict(features_test)
            ridge_predicts.extend(ridge_predict)
            lasso_predict = lasso.predict(features_test)
            lasso_predicts.extend(lasso_predict)
            features_test_norm = features_test / np.std(features_test, axis=0)
            pls_predict = pls.predict(features_test_norm)
            pls_predicts.extend([_[0] for _ in pls_predict])

        # gaussian_predicts_list.append(gaussian_predicts)
        # ridge_predicts_list.append(ridge_predicts)
        # lasso_predicts_list.append(lasso_predicts)
        predicts.append([gaussian_predicts, ridge_predicts, lasso_predicts,pls_predicts])
    dfp[["Gaussian_model", "Ridge_model", "Lasso_model","PLS_model"]] = models
    dfp[["Gaussian_regression_predict", "Ridge_regression_predict", "Lasso_regression_predict", "PLS_regression_predict"]] \
        = dfp[["Gaussian_model", "Ridge_model", "Lasso_model" ,"PLS_model"]].applymap(lambda model: model.predict(features))
    dfp[["Gaussian_regression_r2", "Ridge_regression_r2", "Lasso_regression_r2" ,"PLS_regression_r2"]] = dfp[
        ["Gaussian_regression_predict", "Ridge_regression_predict", "Lasso_regression_predict" ,"PLS_regression_predict"]].applymap(
        lambda predict: r2_score(df["ΔΔG.expt."], predict))
    dfp[["Gaussian_regression_RMSE", "Ridge_regression_RMSE", "Lasso_regression_RMSE" , "PLS_regression_RMSE"]] = dfp[[
        "Gaussian_regression_predict", "Ridge_regression_predict", "Lasso_regression_predict" ,"PLS_regression_predict"]].applymap(
        lambda predict: np.sqrt(mean_squared_error(df["ΔΔG.expt."], predict)))


    # dfp["Gaussian_regression_predict"] = dfp["Gaussian_model"].apply(lambda model: model.predict(features))
    # dfp["Gaussian_regression_r2"] = dfp["Gaussian_regression_predict"].apply(
    #     lambda predict: r2_score(df["ΔΔG.expt."], predict))
    # dfp["Gaussian_regression_RMSE"] = dfp["Gaussian_regression_predict"].apply(
    #     lambda predict: np.sqrt(mean_squared_error(df["ΔΔG.expt."], predict)))

    # dfp["Ridge_model"] = Ridge_models
    # dfp["Ridge_regression_predict"] = dfp["Ridge_model"].apply(lambda model: model.predict(features))
    # dfp["Ridge_regression_r2"] = dfp["Ridge_regression_predict"].apply(
    #     lambda predict: r2_score(df["ΔΔG.expt."], predict))
    # dfp["Ridge_regression_RMSE"] = dfp["Ridge_regression_predict"].apply(
    #     lambda predict: np.sqrt(mean_squared_error(df["ΔΔG.expt."], predict)))

    # dfp["Lasso_model"] = Lasso_models
    # dfp["Lasso_regression_predict"] = dfp["Lasso_model"].apply(lambda model: model.predict(features))
    # dfp["Lasso_regression_r2"] = dfp["Lasso_regression_predict"].apply(
    #     lambda predict: r2_score(df["ΔΔG.expt."], predict))
    # dfp["Lasso_regression_RMSE"] = dfp["Lasso_regression_predict"].apply(
    #     lambda predict: np.sqrt(mean_squared_error(df["ΔΔG.expt."], predict)))
    dfp[["Gaussian_test_predict", "ridge_test_predict", "lasso_test_predict","pls_test_predict"]] = predicts
    dfp[["Gaussian_test_r2", "ridge_test_r2", "lasso_test_r2", "pls_test_r2"]] = dfp[
        ["Gaussian_test_predict", "ridge_test_predict", "lasso_test_predict" ,"pls_test_predict"]].applymap(
        lambda predict: r2_score(df["ΔΔG.expt."], predict))
    dfp[["Gaussian_test_r", "ridge_test_r", "lasso_test_r", "pls_test_r"]] = dfp[
        ["Gaussian_test_predict", "ridge_test_predict", "lasso_test_predict", "pls_test_predict"]].applymap(
        lambda predict: np.corrcoef(df["ΔΔG.expt."], predict)[1,0])
    dfp[["Gaussian_test_RMSE", "ridge_test_RMSE", "lasso_test_RMSE", "pls_test_RMSE"]] = dfp[
        ["Gaussian_test_predict", "ridge_test_predict", "lasso_test_predict", "pls_test_predict"]].applymap(
        lambda predict: np.sqrt(mean_squared_error(df["ΔΔG.expt."], predict)))

    # dfp["Gaussian_test_predict"] = gaussian_predicts_list
    # dfp["Gaussian_test_r2"] = dfp["Gaussian_test_predict"].apply(lambda predict: r2_score(df["ΔΔG.expt."], predict))
    # dfp["Gaussian_test_RMSE"] = dfp["Gaussian_test_predict"].apply(
    #     lambda predict: np.sqrt(mean_squared_error(df["ΔΔG.expt."], predict)))

    # dfp["ridge_test_predict"] = ridge_predicts_list
    # dfp["ridge_test_r2"] = dfp["ridge_test_predict"].apply(lambda predict: r2_score(df["ΔΔG.expt."], predict))
    # dfp["ridge_test_RMSE"] = dfp["ridge_test_predict"].apply(
    #     lambda predict: np.sqrt(mean_squared_error(df["ΔΔG.expt."], predict)))

    # dfp["lasso_test_predict"] = lasso_predicts_list
    # dfp["lasso_test_r2"] = dfp["lasso_test_predict"].apply(lambda predict: r2_score(df["ΔΔG.expt."], predict))
    # dfp["lasso_test_RMSE"] = dfp["lasso_test_predict"].apply(
    #     lambda predict: np.sqrt(mean_squared_error(df["ΔΔG.expt."], predict)))

    df["Gaussian_predict"]=dfp[dfp["Gaussian_test_r"] == dfp["Gaussian_test_r"].max()].iloc[0][
        "Gaussian_test_predict"]
    df["Ridge_predict"]=dfp[dfp["ridge_test_r"] == dfp["ridge_test_r"].max()].iloc[0][
        "ridge_test_predict"]
    df["Lasso_predict"] = dfp[dfp["lasso_test_r"] == dfp["lasso_test_r"].max()].iloc[0]["lasso_test_predict"]
    df["PLS_predict"] = dfp[dfp["pls_test_r"] == dfp["pls_test_r"].max()].iloc[0]["pls_test_predict"]
    df["Gaussian_error"]=df["Gaussian_predict"]-df["ΔΔG.expt."]
    # df[["Gaussian_error","Ridge_error","Lasso_error"]] = df[["Gaussian_predict","Ridge_predict","Lasso_predict",]].applymap(lambda test:test - df["ΔΔG.expt."].values)
    #print(df)
    df = df.sort_values(by='Gaussian_error', key=abs, ascending=[False])
    PandasTools.AddMoleculeColumnToFrame(df, "smiles")
    # df = df[[ "smiles", "ROMol","inchikey","er.", "RT", "ΔΔG.expt."]].drop_duplicates(subset="inchikey")#,"ΔΔminG.expt.","ΔΔmaxG.expt."
    # df = df[["smiles", "ROMol", "er.", "RT"]]
    print("gaussian r2")
    print(dfp["Gaussian_test_r2"].max())
    print("pls r2")
    print(dfp["pls_test_r2"].max())
    PandasTools.SaveXlsxFromFrame(df, save_name + "/result_test.xlsx", size=(100, 100))
    # df.to_excel(save_name + "/test.xlsx")
    # plt.plot(df["ΔΔG.expt."],df["test_predict"],"o")
    # plt.savefig("../figs/plot.png")
    dfp.to_csv(save_name + "/result.csv")
    print(save_name)
    # RMSE = np.sqrt(mean_squared_error(df["ΔΔG.expt."], l))
    # #罰則項を変えて訓練・k=5分割交差検証


# ダブルクロスバリデーション
def grid_search(fold, features_dir_name, regression_features, feature_number, df, dfp, out_file_name,
                regression_type, maxmin):
    os.makedirs("../errortest/", exist_ok=True)
    #
    # if True:  # fold:
    #     xyz = pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))[
    #         ["x", "y", "z"]].values
    #     print("yzfold")
    # # else:
    # #     xyz = pd.read_csv("{}/{}/feature_y.csv".format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))[
    # #         ["x", "y", "z"]].values
    # #     print("zfold")
    #
    # d = np.array([np.linalg.norm(xyz - _, axis=1) for _ in xyz])
    # d_y = np.array([np.linalg.norm(xyz - _ * np.array([1, -1, 1]), axis=1) for _ in xyz])
    # d_z = np.array([np.linalg.norm(xyz - _ * np.array([1, 1, -1]), axis=1) for _ in xyz])
    # d_yz = np.array([np.linalg.norm(xyz - _ * np.array([1, -1, -1]), axis=1) for _ in xyz])
    #
    # def gauss_func(d):
    #     sigma = 0.5
    #     leng = 1
    #     ans = 1 / (2 * np.pi * np.sqrt(2 * np.pi) * sigma ** 3) * leng ** 3 \
    #           * np.exp(-d ** 2 / (2 * sigma ** 2))
    #     return ans
    #
    # penalty = np.where(d < 0.5 * 3, gauss_func(d), 0)
    # penalty_y = np.where(d_y < 0.5 * 3, gauss_func(d_y), 0)
    # penalty_z = np.where(d_z < 0.5 * 3, gauss_func(d_z), 0)
    # penalty_yz = np.where(d_yz < 0.5 * 3, gauss_func(d_yz), 0)
    # if True:  # fold:
    #     penalty = penalty + penalty_y + penalty_z + penalty_yz
    #     grid_features_name = "{}/{}/feature_yz.csv"
    # else:
    #     penalty = penalty + penalty_y
    #     grid_features_name = "{}/{}/feature_y.csv"
    #
    # penalty = penalty / np.max(np.sum(penalty, axis=0))
    # # np.fill_diagonal(penalty, -1)
    # penalty = penalty - np.identity(penalty.shape[0]) / np.sum(penalty, axis=0)
    # os.makedirs("../penalty", exist_ok=True)
    # np.save('../penalty/penalty.npy', penalty)
    r2_list = []
    RMSE_list = []
    penalty = np.load("../grid_coordinates/[[-4.75, -2.75, -4.75],[14, 12, 20],0.5].npy")
    if True:  # feature_number == "2":
        # feature1param = list(dict.fromkeys(dfp["{}param".format(regression_features.split()[0])]))
        # feature2param = list(dict.fromkeys(dfp["{}param".format(regression_features.split()[1])]))
        # q = []
        # for L1 in feature1param:
        #     for L2 in feature2param:
        for L1, L2 in dfp[["Dtparam", "ESP_cutoffparam"]].values:  # zip(dfp["Dtparam"], dfp["ESP_cutoffparam"]):
            print([L1, L2])

            l = []
            kf = KFold(n_splits=5, shuffle=False)
            for (train_index, test_index) in kf.split(df):
                if False:
                    features1 = []
                    features2 = []
                    for mol in df.iloc[train_index]["mol"]:
                        df_grid = pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))
                        feat1 = df_grid[regression_features.split()[0]].values
                        feat2 = df_grid[regression_features.split()[1]].values
                        features1.append(feat1)
                        features2.append(feat2)

                    # features1 = [
                    #     pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                    #         "{}".format(regression_features.split()[0])].values
                    #     for
                    #     mol in df.iloc[train_index]["mol"]]
                    # features2 = [
                    #     pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                    #         "{}".format(regression_features.split()[1])].values
                    #     for
                    #     mol in df.iloc[train_index]["mol"]]
                    features1_ = np.array(features1)
                    features2_ = np.array(features2)
                    features1 = features1 / np.std(features1_)
                    features2 = features2 / np.std(features2_)
                    features = np.concatenate([features1, features2], axis=1)  # / np.std(features_train, axis=0)
                else:
                    features = []
                    for mol in df.iloc[train_index]["mol"]:
                        df_grid = pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))
                        feat = df_grid[["Dt", "ESP"]].values
                        features.append(feat)
                    # features1 = [
                    #     pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                    #         "{}".format(regression_features.split()[0])].values
                    #     for
                    #     mol in df.iloc[train_index]["mol"]]
                    # features2 = [
                    #     pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                    #         "{}".format(regression_features.split()[1])].values
                    #     for
                    #     mol in df.iloc[train_index]["mol"]]

                    features = np.array(features).transpose(2, 0, 1)
                    std = np.std(features, axis=(1, 2)).reshape(2, 1, 1)
                    features = features / std
                    features = np.concatenate(features, axis=1)
                # if True:#regression_type in ["gaussian"]:
                # penalty1 = np.concatenate([L1 * penalty, np.zeros(penalty.shape)], axis=1)
                # penalty2 = np.concatenate([np.zeros(penalty.shape), L2 * penalty], axis=1)
                # X = np.concatenate([features, penalty1, penalty2], axis=0)
                X = np.block([[features],
                              [L1 * penalty, np.zeros(penalty.shape)],
                              [np.zeros(penalty.shape), L2 * penalty]])
                Y = np.concatenate([df.iloc[train_index]["ΔΔG.expt."], np.zeros(penalty.shape[0] * 2)], axis=0)
                model = linear_model.Ridge(alpha=0, fit_intercept=False).fit(X, Y)
                # α=0だとただのlinear regression ただRidgeのほうが計算時間早い

                # ここから、テストセットの特徴量計算
                # features1 = [
                #     pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))[
                #         "{}".format(regression_features.split()[0])].values
                #     for
                #     mol in df.iloc[test_index]["mol"]]
                # features2 = [
                #     pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))[
                #         "{}".format(regression_features.split()[1])].values
                #     for
                #     mol in df.iloc[test_index]["mol"]]
                if False:
                    features1 = []
                    features2 = []
                    for mol in df.iloc[test_index]["mol"]:
                        df_grid = pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))
                        feat1 = df_grid[regression_features.split()[0]].values
                        feat2 = df_grid[regression_features.split()[1]].values
                        features1.append(feat1)
                        features2.append(feat2)
                    features1 = np.array(features1)
                    features2 = np.array(features2)
                    features1 = features1 / np.std(features1_)
                    features2 = features2 / np.std(features2_)
                    features = np.concatenate([features1, features2], axis=1)  # / np.std(features_train, axis=0)
                else:
                    features = []
                    for mol in df.iloc[test_index]["mol"]:
                        df_grid = pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))
                        feat = df_grid[["Dt", "ESP"]].values
                        features.append(feat)
                    features = np.array(features).transpose(2, 0, 1)
                    # std = np.std(features, axis=(1, 2)).reshape(2, 1, 1)
                    features = features / std
                    features = np.concatenate(features, axis=1)
                predict = model.predict(features)
                if False:  # maxmin == "True":
                    for i in range(len(predict)):
                        if predict[i] >= df.iloc[test_index]["ΔΔmaxG.expt."].values[i]:
                            predict[i] = df.iloc[test_index]["ΔΔmaxG.expt."].values[i]
                        if predict[i] <= df.iloc[test_index]["ΔΔminG.expt."].values[i]:
                            predict[i] = df.iloc[test_index]["ΔΔminG.expt."].values[i]
                l.extend(predict)
            print(l)
            r2 = r2_score(df["ΔΔG.expt."], l)
            RMSE = np.sqrt(mean_squared_error(df["ΔΔG.expt."], l))
            print("r2", r2)
            r2_list.append(r2)
            RMSE_list.append(RMSE)
        # print(q)
        # paramlist = pd.DataFrame(q)
        paramlist = dfp
        print(regression_features.split()[0])
        print(regression_features.split()[1])
        paramlist.rename(columns={0: "{}param".format(regression_features.split()[0]),
                                  1: "{}param".format(regression_features.split()[1])}, inplace=True)

    paramlist["r2"] = r2_list
    paramlist["RMSE"] = RMSE_list

    paramlist.to_csv(out_file_name)

    min_index = paramlist['RMSE'].idxmin()
    min_row = paramlist.loc[min_index, :]
    p = pd.DataFrame([min_row], index=[min_index])
    print(p)
    p.to_csv(param["out_dir_name"] + "/hyperparam.csv")


# def df1unfolding(df):
#     df_y = copy.deepcopy(df)
#     df_y["y"] = -df_y["y"]
#
#     df_z = copy.deepcopy(df)
#     df_z["z"] = -df_z["z"]
#     df_z["Dt"] = -df_z["Dt"]
#     df_z["ESP_cutoff"] = -df_z["ESP_cutoff"]
#     df_yz = copy.deepcopy(df)
#     df_yz["y"] = -df_yz["y"]
#     df_yz["z"] = -df_yz["z"]
#     df_yz["Dt"] = -df_yz["Dt"]
#     df_yz["ESP_cutoff"] = -df_yz["ESP_cutoff"]
#
#     df = pd.concat([df, df_y, df_z, df_yz]).sort_values(by=["x", "y", "z"])
#
#     return df


def zunfolding(df, regression_features):
    df.to_csv("../errortest/dfbefore.csv")
    df_z = copy.deepcopy(df)
    df_z["z"] = -df_z["z"]
    df_z["MF_Dt"] = -df_z["MF_Dt"]

    df_z["MF_{}".format(regression_features.split()[1])] = -df_z["MF_{}".format(regression_features.split()[1])]

    df = pd.concat([df, df_z]).sort_values(['x', 'y', "z"],
                                           ascending=[True, True, True])  # .sort_values(by=["x", "y", "z"])
    df.to_csv("../errortest/dfafter.csv")
    return df


def leave_one_out(fold, features_dir_name, regression_features, df, out_file_name, moleculer_field_dir,
                  regression_type, dfp, p=None):
    # penalty = np.load("../penalty/penalty.npy")
    penalty = np.load("../grid_coordinates/[[-4.75, -2.75, -4.75],[14, 12, 20],0.5].npy")
    # if True:  # fold:
    grid_features_name = "{}/{}/feature_yz.csv"
    # else:grid_features_name = "{}/{}/feature_y.csv"
    # if regression_features in ["LUMO"]:
    features1 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                     "{}".format(regression_features.split()[0])].values for mol
                 in df["mol"]]
    features2 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                     "{}".format(regression_features.split()[1])].values for mol
                 in df["mol"]]
    # features1unfold = [pd.read_csv("{}/{}/feature_y.csv".format(features_dir_name, mol.GetProp("InchyKey")))[
    #                        "{}".format(regression_features.split()[0])].values for mol
    #                    in df["mol"]]
    # features2unfold = [pd.read_csv("{}/{}/feature_y.csv".format(features_dir_name, mol.GetProp("InchyKey")))[
    #                        "{}".format(regression_features.split()[1])].values for mol
    #                    in df["mol"]]

    features1_ = np.array(features1)

    features1 = features1 / np.std(features1_)
    features2_ = np.array(features2)
    # features1_unfold = np.array(features1unfold)
    # features2_unfold = np.array(features2unfold)
    features2 = features2 / np.std(features2_)
    features_train = np.concatenate([features1, features2], axis=1)
    # np.save(param["moleculer_field_dir"]+"/trainstd", np.std(features1, axis=0))
    features = features_train  # / np.std(features_train, axis=0)
    features = np.nan_to_num(features)

    df_mf = pd.read_csv(grid_features_name.format(features_dir_name, "KWOLFJPFCHCOCG-UHFFFAOYSA-N"))
    df_mf.to_csv("../errortest/dfmf.csv")
    os.makedirs(moleculer_field_dir, exist_ok=True)
    if regression_type == "lassocv" or regression_type == "PLS" or regression_type == "ridgecv" or regression_type == "elasticnetcv":
        Y = df["ΔΔG.expt."].values

        if regression_type == "lassocv":
            model = linear_model.LassoCV(n_alphas=10, fit_intercept=False, cv=5).fit(features, Y)
            print("alphas_")
            print(model.alphas_)
            print("alpha")
            print(model.alpha_)
            print("model.mse_path_")
            print(model.mse_path_)
            # raise ValueError


        elif regression_type == "PLS":
            model = PLSRegression(n_components=3).fit(features, Y)
        elif regression_type == "ridgecv":

            model = RidgeCV(alphas=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024), fit_intercept=False, cv=5).fit(
                features, Y)
            print("alpha")
            print(model.alpha_)
            print("best_score_")
            print(model.best_score_)

            # raise ValueError
        elif regression_type == "elasticnetcv":
            model = ElasticNetCV(n_alphas=10, fit_intercept=False, cv=5).fit(features, Y)
            print("alphas_")
            print(model.alphas_)
            print("alpha")
            print(model.alpha_)
            # raise ValueError

        df["ΔΔG.train"] = model.predict(features)

        if regression_type == "PLS":
            df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[0][
                                                                    :int(model.coef_[0].shape[0] / 2)]
            df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[0][
                                                                    int(model.coef_[0].shape[0] / 2):int(
                                                                        model.coef_[0].shape[0])]
        else:  # regression_type == "lassocv":
            df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[:int(model.coef_.shape[0] / 2)]
            df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[int(model.coef_.shape[0] / 2):int(
                model.coef_.shape[0])]
        print("writemoleculerfield")
        df_mf.to_csv((moleculer_field_dir + "/" + "moleculer_field.csv"))

    if regression_type in ["gaussian"]:
        print("loocv printp")
        print(p)
        hparam1 = p['{}param'.format(regression_features.split()[0])].values
        hparam2 = p['{}param'.format(regression_features.split()[1])].values
        penalty1 = np.concatenate([hparam1 * penalty, np.zeros(penalty.shape)], axis=1)
        penalty2 = np.concatenate([np.zeros(penalty.shape), hparam2 * penalty], axis=1)
        X = np.concatenate([features, penalty1, penalty2], axis=0)
        Y = np.concatenate([df["ΔΔG.expt."], np.zeros(penalty.shape[0] * 2)], axis=0)
        model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)
        df["ΔΔG.train"] = model.predict(features)
        # os.makedirs(param["moleculer_field_dir"], exist_ok=True)
        df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[:penalty.shape[0]]
        df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[penalty.shape[0]:penalty.shape[0] * 2]
        all_hparam = True
        if all_hparam:
            for L1, L2 in zip(dfp["Dtparam"], dfp["ESP_cutoffparam"]):
                print([L1, L2])
                penalty1 = np.concatenate([L1 * penalty, np.zeros(penalty.shape)], axis=1)
                penalty2 = np.concatenate([np.zeros(penalty.shape), L2 * penalty], axis=1)
                X = np.concatenate([features, penalty1, penalty2], axis=0)
                Y = np.concatenate([df["ΔΔG.expt."], np.zeros(penalty.shape[0] * 2)], axis=0)
                model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)
                os.makedirs(moleculer_field_dir + "/{}/".format(L1), exist_ok=True)
                df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[:penalty.shape[0]]
                df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[
                                                                        penalty.shape[0]:penalty.shape[0] * 2]
                df_mf.to_csv((moleculer_field_dir + "/{}/".format(L1) + "moleculer_field.csv"))
                # df_mf.to_excel((param["moleculer_field_dir"] + "/{}/".format(L1) + "moleculer_field.csv"))

    df["{}_contribution".format(regression_features.split()[0])] = np.sum(
        features1 * df_mf["MF_{}".format(regression_features.split()[0])].values, axis=1)  # / np.std(features1, axis=0)

    df["{}_contribution".format(regression_features.split()[1])] = np.sum(
        features2 * df_mf["MF_{}".format(regression_features.split()[1])].values,
        axis=1)  # / (np.std(features2, axis=0)) )

    df_mf["{}_std".format(regression_features.split()[0])] = np.std(features1_)
    df_mf["{}_std".format(regression_features.split()[1])] = np.std(features2_)
    df_mf.to_csv((moleculer_field_dir + "/" + "moleculer_field.csv"))
    df_unfold = df_mf
    if True:  # fold:
        df_unfold = zunfolding(df_mf, regression_features)

        # df_unfold["Dt_std"] =np.std(features1_unfold)
        # df_unfold["ESP_std"] = np.std(features2_unfold)
    DtR1s = []
    DtR2s = []

    ESPR1s = []
    ESPR2s = []

    for mol in df["mol"]:
        df1 = pd.read_csv("{}/{}/feature_y.csv".format(features_dir_name, mol.GetProp("InchyKey"))).sort_values(
            ['x', 'y', "z"], ascending=[True, True, True])
        df1["DtR1"] = df1["Dt"].values / df_unfold["Dt_std"].values * df_unfold["MF_Dt"].values
        df1["ESPR1"] = df1["{}".format(regression_features.split()[1])].values / df_unfold["ESP_std"].values * \
                       df_unfold["MF_{}".format(regression_features.split()[1])].values
        df1.to_csv("../errortest/df1R1R2.csv")
        DtR1s.append(df1[(df1["z"] > 0)]["DtR1"].sum())
        DtR2s.append(df1[(df1["z"] < 0)]["DtR1"].sum())
        ESPR1s.append(df1[(df1["z"] > 0)]["ESPR1"].sum())
        ESPR2s.append(df1[(df1["z"] < 0)]["ESPR1"].sum())
    df["DtR1"] = DtR1s
    df["DtR2"] = DtR2s
    df["DtR1R2"] = df["DtR1"] + df["DtR2"]
    df["ESPR1"] = ESPR1s
    df["ESPR2"] = ESPR2s
    df["ESPR1R2"] = df["ESPR1"] + df["ESPR2"]

    # ここからテストセットの実行

    kf = KFold(n_splits=len(df), shuffle=False)
    print(regression_type)
    if regression_type == "lassocv" or regression_type == "PLS" or regression_type == "ridgecv" or regression_type == "elasticnetcv":
        l = []

        for (train_index, test_index) in kf.split(df):
            features1 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                             "{}".format(regression_features.split()[0])].values
                         for
                         mol in df.iloc[train_index]["mol"]]
            features2 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                             "{}".format(regression_features.split()[1])].values
                         for
                         mol in df.iloc[train_index]["mol"]]
            features1_ = np.array(features1)
            features2_ = np.array(features2)
            features1 = features1 / np.std(features1_)
            features2 = features2 / np.std(features2_)
            features_train = np.concatenate([features1, features2], axis=1)
            features = features_train  # / np.std(features_train, axis=0)
            features = np.nan_to_num(features)
            Y = df.iloc[train_index]["ΔΔG.expt."].values
            if regression_type == "lassocv":
                model = linear_model.LassoCV(fit_intercept=False, cv=5).fit(features, Y)

            elif regression_type == "PLS":
                model = PLSRegression(n_components=3).fit(features, Y)
            elif regression_type == "ridgecv":
                model = RidgeCV(fit_intercept=False, cv=5).fit(features, Y)
            elif regression_type == "elasticnetcv":
                model = ElasticNetCV(fit_intercept=False, cv=5).fit(features, Y)
            else:
                print("regressionerror")
                # raise ValueError

            features1 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                             "{}".format(regression_features.split()[0])].values
                         for
                         mol in df.iloc[test_index]["mol"]]
            features2 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                             "{}".format(regression_features.split()[1])].values
                         for
                         mol in df.iloc[test_index]["mol"]]
            features1 = np.array(features1)
            features2 = np.array(features2)
            features1 = features1 / np.std(features1_)
            features2 = features2 / np.std(features2_)
            features_test = np.concatenate([features1, features2], axis=1)
            features = features_test  # / np.std(features_train, axis=0)
            features = np.nan_to_num(features)
            predict = model.predict(features)

            if regression_type == "PLS":
                l.extend([_[0] for _ in predict])
                # for i in range(len(predict)):
                #     # if maxmin == "True":
                #     #
                #     #     if predict[i] >= df.iloc[test_index]["ΔΔmaxG.expt."].values[i]:
                #     #         predict[i] = df.iloc[test_index]["ΔΔmaxG.expt."].values[i]
                #     #     if predict[i] <= df.iloc[test_index]["ΔΔminG.expt."].values[i]:
                #     #         predict[i] = df.iloc[test_index]["ΔΔminG.expt."].values[i]
                #     l.extend(predict[i])

            else:
                l.extend(predict)
                # for i in range(len(predict)):
                #     # if maxmin == "True":
                #     #     if predict[i] >= df.iloc[test_index]["ΔΔmaxG.expt."].values[i]:
                #     #         predict[i] = df.iloc[test_index]["ΔΔmaxG.expt."].values[i]
                #     #     if predict[i] <= df.iloc[test_index]["ΔΔminG.expt."].values[i]:
                #     #         predict[i] = df.iloc[test_index]["ΔΔminG.expt."].values[i]
                #
                #     l.extend([predict[i]])

    if regression_type in ["gaussian", "gaussianFP"]:
        l = []
        for (train_index, test_index) in kf.split(df):
            features1 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                             "{}".format(regression_features.split()[0])].values
                         for
                         mol in df.iloc[train_index]["mol"]]
            features2 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                             "{}".format(regression_features.split()[1])].values
                         for
                         mol in df.iloc[train_index]["mol"]]
            features1_ = np.array(features1)
            features2_ = np.array(features2)
            features1 = features1 / np.std(features1_)
            features2 = features2 / np.std(features2_)

            hparam1 = p['{}param'.format(regression_features.split()[0])].values
            hparam2 = p['{}param'.format(regression_features.split()[1])].values
            features_train = np.concatenate([features1, features2], axis=1)
            features = features_train  # / np.std(features_train, axis=0)
            penalty1 = np.concatenate([hparam1 * penalty, np.zeros(penalty.shape)], axis=1)
            penalty2 = np.concatenate([np.zeros(penalty.shape), hparam2 * penalty], axis=1)
            X = np.concatenate([features, penalty1, penalty2], axis=0)

            Y = np.concatenate([df.iloc[train_index]["ΔΔG.expt."], np.zeros(penalty.shape[0] * 2)], axis=0)
            model = linear_model.Ridge(alpha=0, fit_intercept=False).fit(X, Y)

            # ここから、テストセットの特徴量計算
            features1 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                             "{}".format(regression_features.split()[0])].values for
                         mol in df.iloc[test_index]["mol"]]
            features2 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                             "{}".format(regression_features.split()[1])].values for
                         mol in df.iloc[test_index]["mol"]]
            features1 = np.array(features1)
            features2 = np.array(features2)
            features1 = features1 / np.std(features1_)
            features2 = features2 / np.std(features2_)
            features_test = np.concatenate([features1, features2], axis=1)
            features = features_test  # / np.std(features_train, axis=0)

            predict = model.predict(features)
            # if maxmin == "True":
            #     for i in range(len(predict)):
            #         if predict[i] >= df.iloc[test_index]["ΔΔmaxG.expt."].values[i]:
            #             predict[i] = df.iloc[test_index]["ΔΔmaxG.expt."].values[i]
            #         if predict[i] <= df.iloc[test_index]["ΔΔminG.expt."].values[i]:
            #             predict[i] = df.iloc[test_index]["ΔΔminG.expt."].values[i]

            l.extend(predict)

    df["ΔΔG.loo"] = l
    print(l)
    r2 = r2_score(df["ΔΔG.expt."], l)
    print("r2", r2)
    df["error"] = l - df["ΔΔG.expt."]
    df["inchikey"] = df["mol"].apply(lambda mol: mol.GetProp("InchyKey"))
    os.makedirs("../errortest/", exist_ok=True)
    # df.to_csv("../errortest/df.csv")
    PandasTools.AddMoleculeColumnToFrame(df, "smiles")

    # df=df.replace('0', np.nan)

    try:
        df = df.drop(['level_0', 'Unnamed: 0', 'mol'])
    except:
        None
    # df=df[["smiles","ROMol","inchikey","er.","ΔΔG.expt.","ΔΔminG.expt.","ΔΔmaxG.expt.", 'mol','level_0']]
    df.to_excel("../errortest/test.xlsx")
    PandasTools.SaveXlsxFromFrame(df, out_file_name, size=(100, 100), molCol='ROMol')

    # if param["cat"] == "cbs":
    #     dfp = pd.read_csv("../result/old /cbs_gaussian_nomax/result_grid_search.csv")
    #     dfp=pd.read_csv(gridsearch_file_name)
    # elif param["cat"] == "dip":
    #     dfp = pd.read_csv("../result/old /dip-chloride_gaussian_nomax/result_grid_search.csv")
    # elif param["cat"] == "RuSS":
    #     dfp = pd.read_csv("../result/old /RuSS_gaussian_nomax/result_grid_search.csv")
    # print(dfp)
    # min_index = dfp['RMSE'].idxmin()
    # min_row = dfp.loc[min_index, :]
    # p = pd.DataFrame([min_row], index=[min_index])
    # print(p)
    # p.to_csv(param["out_dir_name"] + "/hyperparamloocv.csv")

    return model


# def train_testfold(fold, features_dir_name, regression_features, feature_number, df, gridsearch_file_name,
#                    looout_file_name, testout_file_name, param, fplist, regression_type, maxmin, dfp):
#
#
#     # df_train, df_test = train_test_split(df, train_size=0.7, random_state=1)
#
#     if fold:
#         grid_features_name = "{}/{}/feature_yz.csv"
#     else:
#         grid_features_name = "{}/{}/feature_y.csv"
#     df_train, df_test = train_test_split(df, train_size=0.8, random_state=0)
#     if param["Regression_type"] in "gaussian":
#         grid_search(fold, features_dir_name, regression_features, feature_number, df_train, dfp, gridsearch_file_name,
#                     fplist, regression_type, maxmin)
#     p = []
#     if param["cat"] == "cbs":
#         p = pd.read_csv("../result/cbs_gaussian/hyperparam.csv")
#     elif param["cat"] == "dip":
#         p = pd.read_csv("../result/dip-chloride_gaussian/hyperparam.csv")
#     elif param["cat"] == "RuSS":
#         p = pd.read_csv("../result/RuSS_gaussian/hyperparam.csv")
#     else:
#         print("Not exist gridsearch result")
#
#     if param["Regression_type"] in "gaussian":
#         model = leave_one_out(fold, features_dir_name, regression_features, feature_number, df_train, looout_file_name,
#                               param, fplist, regression_type, maxmin, p)
#     else:
#         model = leave_one_out(fold, features_dir_name, regression_features, feature_number, df_train, looout_file_name,
#                               param,
#                               fplist, regression_type, maxmin, p=None)
#
#     features1 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
#                      "{}".format(regression_features.split()[0])].values
#                  for
#                  mol in df_test["mol"]]
#     features2 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
#                      "{}".format(regression_features.split()[1])].values
#                  for
#                  mol in df_test["mol"]]
#     features = np.concatenate([features1, features2], axis=1)
#     testpredict = model.predict(features)
#     l = []
#     print(testpredict)
#
#     if maxmin == "True":
#         if regression_type == "PLS":
#             for i in range(len(testpredict)):
#                 ans = testpredict[i][0]
#                 print(ans)
#
#                 if ans >= df_test["ΔΔmaxG.expt."].values[i]:
#                     ans = df_test["ΔΔmaxG.expt."].values[i]
#                 if ans <= df_test["ΔΔminG.expt."].values[i]:
#                     ans = df_test["ΔΔminG.expt."].values[i]
#                 l.append(ans)
#         else:
#             for i in range(len(testpredict)):
#                 ans = testpredict[i]
#                 print(ans)
#
#                 if ans >= df_test["ΔΔmaxG.expt."].values[i]:
#                     ans = df_test["ΔΔmaxG.expt."].values[i]
#                 if ans <= df_test["ΔΔminG.expt."].values[i]:
#                     ans = df_test["ΔΔminG.expt."].values[i]
#                 l.append(ans)
#     df_test["ΔΔG.test"] = l
#     r2 = r2_score(df_test["ΔΔG.expt."], df_test["ΔΔG.test"])
#     print(r2)
#     df_test["error"] = df_test["ΔΔG.test"] - df_test["ΔΔG.expt."]
#     df_test["inchikey"] = df_test["mol"].apply(lambda mol: mol.GetProp("InchyKey"))
#     PandasTools.AddMoleculeColumnToFrame(df_test, "smiles")
#     try:
#         df_test = df_test.drop(['level_0', 'Unnamed: 0'])
#     except:
#         None
#     df_test = df_test.round(5)
#     df_test = df_test.fillna(0)
#     PandasTools.SaveXlsxFromFrame(df_test, testout_file_name, size=(100, 100))


def doublecrossvalidation(fold, features_dir_name, regression_features, feature_number, df, gridsearch_file_name,
                          looout_file_name, testout_file_name, param, regression_type, maxmin, dfp, file_name):
    # if fold:
    grid_features_name = "{}/{}/feature_yz.csv"
    # else:
    # grid_features_name = "{}/{}/feature_y.csv"
    df = df.sample(frac=1, random_state=0)
    df.to_csv("../errortest/dfrandom.csv")
    df_train = df

    # p = None
    if regression_type in "gaussian":
        grid_search(fold, features_dir_name, regression_features, feature_number, df_train, dfp,
                    gridsearch_file_name,
                    None, maxmin)
        # if True:#param["cat"] in ["cbs","dip","RuSS"]:
        p = pd.read_csv(param["out_dir_name"] + "/hyperparam.csv")
        # leave_one_out(fold, features_dir_name, regression_features, df_train,
        #               looout_file_name, param, regression_type, dfp, p)  # 分子場の出力　looの出力
    else:
        p = None
    leave_one_out(fold, features_dir_name, regression_features, df_train,  # 分子場の出力　looの出力
                  looout_file_name, param["moleculer_field_dir"] + "/" + file_name,
                  regression_type, dfp, p)
    df = df.sample(frac=1, random_state=0)

    kf = KFold(n_splits=5)
    penalty = np.load("../old/penalty/penalty.npy")

    testlist = []
    for (train_index, test_index) in kf.split(df):
        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]
        features1 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                         "{}".format(regression_features.split()[0])].values for mol
                     in df_train["mol"]]
        features2 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                         "{}".format(regression_features.split()[1])].values for mol
                     in df_train["mol"]]
        features1_ = np.array(features1)
        features1 = features1 / np.std(features1)
        features2_ = np.array(features2)
        features2 = features2 / np.std(features2)
        features_train = np.concatenate([features1, features2], axis=1)
        features = features_train  # / np.std(features_train, axis=0)
        print(features.shape)
        # if regression_type == "lassocv" or regression_type == "PLS" or regression_type == "ridgecv" or regression_type == "elasticnetcv":
        Y = df_train["ΔΔG.expt."].values

        if regression_type == "lassocv":
            model = linear_model.LassoCV(fit_intercept=False, cv=5).fit(features, Y)
        elif regression_type == "PLS":
            model = PLSRegression(n_components=3).fit(features, Y)
        elif regression_type == "ridgecv":
            model = RidgeCV(fit_intercept=False, cv=5).fit(features, Y)
        elif regression_type == "elasticnetcv":
            model = ElasticNetCV(fit_intercept=False, cv=5).fit(features, Y)
        elif regression_type in ["gaussian"]:
            hparam1 = p['{}param'.format(regression_features.split()[0])].values
            hparam2 = p['{}param'.format(regression_features.split()[1])].values
            penalty1 = np.concatenate([hparam1 * penalty, np.zeros(penalty.shape)], axis=1)
            penalty2 = np.concatenate([np.zeros(penalty.shape), hparam2 * penalty], axis=1)

            X = np.concatenate([features, penalty1, penalty2], axis=0)
            Y = np.concatenate([df_train["ΔΔG.expt."], np.zeros(penalty.shape[0] * 2)], axis=0)
            model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)

        features1 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                         "{}".format(regression_features.split()[0])].values
                     for
                     mol in df_test["mol"]]
        features2 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                         "{}".format(regression_features.split()[1])].values
                     for
                     mol in df_test["mol"]]
        features1 = np.array(features1)
        features2 = np.array(features2)
        features1 = features1 / np.std(features1_)
        features2 = features2 / np.std(features2_)
        features_test = np.concatenate([features1, features2], axis=1)
        features = features_test  # / np.std(features_train, axis=0)
        testpredict = model.predict(features)

        if regression_type == "PLS":
            testlist.extend([_[0] for _ in testpredict])
        else:
            testlist.extend(testpredict)

    df["ΔΔG.crosstest"] = testlist
    r2 = r2_score(df["ΔΔG.expt."], df["ΔΔG.crosstest"])
    # resultfile=param["out_dir_name"]

    # with open("{}/r2result.txt".format(resultfile), mode='w') as f:
    #     f.write(r2)

    print(r2)
    df["testerror"] = df["ΔΔG.crosstest"] - df["ΔΔG.expt."]
    df["inchikey"] = df["mol"].apply(lambda mol: mol.GetProp("InchyKey"))
    PandasTools.AddMoleculeColumnToFrame(df, "smiles")
    try:
        df = df.drop(['level_0', 'Unnamed: 0'])
    except:
        None
    df = df.round(5)
    df = df.fillna(0)
    print("dflen")
    print(len(df))
    PandasTools.SaveXlsxFromFrame(df, testout_file_name, size=(100, 100))


def energy_to_Boltzmann_distribution(mol, RT=1.99e-3 * 273):
    energies = np.array([float(conf.GetProp("energy")) for conf in mol.GetConformers()])
    energies = energies - np.min(energies)
    rates = np.exp(-energies / RT)
    rates = rates / sum(rates)
    for conf, rate in zip(mol.GetConformers(), rates):
        conf.SetProp("Boltzmann_distribution", str(rate))


if __name__ == '__main__':
    # time.sleep(60*100)
    start = time.perf_counter()  # 計測開始

    # for param_file_name in [
    #
    #     # "../parameter_nomax/parameter_cbs_gaussian.txt",
    #     # "../parameter_nomax/parameter_cbs_ridgecv.txt",
    #     # "../parameter_nomax/parameter_cbs_PLS.txt",
    #     # "../parameter_nomax/parameter_cbs_lassocv.txt",
    #     # "../parameter_nomax/parameter_cbs_elasticnetcv.txt",
    #     # "../parameter_nomax/parameter_dip-chloride_PLS.txt",
    #     # "../parameter_nomax/parameter_dip-chloride_lassocv.txt",
    #     # "../parameter_nomax/parameter_dip-chloride_gaussian.txt",
    #     # "../parameter_nomax/parameter_dip-chloride_elasticnetcv.txt",
    #     # "../parameter_nomax/parameter_dip-chloride_ridgecv.txt",
    #     # "../parameter_nomax/parameter_RuSS_gaussian.txt",
    #     # "../parameter_nomax/parameter_RuSS_PLS.txt",
    #     # "../parameter_nomax/parameter_RuSS_lassocv.txt",
    #     # "../parameter_nomax/parameter_RuSS_elasticnetcv.txt",
    #     # "../parameter_nomax/parameter_RuSS_ridgecv.txt",
    #
    #     "../parameter_0227/parameter_cbs_gaussian.txt",
    #     "../parameter_0227/parameter_dip-chloride_gaussian.txt",
    #     "../parameter_0227/parameter_RuSS_gaussian.txt",
    #
    #
    #     "../parameter_0227/parameter_cbs_PLS.txt",
    #     "../parameter_0227/parameter_dip-chloride_PLS.txt",
    #     "../parameter_0227/parameter_RuSS_PLS.txt",
    #
    #     "../parameter_0227/parameter_cbs_ridgecv.txt",
    #     "../parameter_0227/parameter_dip-chloride_ridgecv.txt",
    #     "../parameter_0227/parameter_RuSS_ridgecv.txt",
    #
    #     "../parameter_0227/parameter_cbs_elasticnetcv.txt",
    #     "../parameter_0227/parameter_cbs_lassocv.txt",
    #
    #
    #     "../parameter_0227/parameter_RuSS_lassocv.txt",
    #     "../parameter_0227/parameter_RuSS_elasticnetcv.txt",
    #     "../parameter_0227/parameter_dip-chloride_lassocv.txt",
    #
    #     "../parameter_0227/parameter_dip-chloride_elasticnetcv.txt",
    #
    # ]:
    for file in glob.glob("../arranged_dataset/*.xlsx"):
        with open("../parameter/cube_to_grid/cube_to_grid.txt", "r") as f:
            param = json.loads(f.read())
        print(param)
        df = pd.read_excel(file).dropna(subset=['smiles']).reset_index(drop=True)  # [:50]

        # print(param_file_name)
        # with open(param_file_name, "r") as f:
        #     param = json.loads(f.read())
        file_name = os.path.splitext(os.path.basename(file))[0]

        features_dir_name = param[
                                "grid_coordinates"] + file_name  # + "/"  # param["grid_dir_name"] + "/[{}]/".format(param["grid_sizefile"])
        print(features_dir_name)
        # fparranged_dataname = param["fpdata_file_path"]
        # df_fplist = pd.read_csv(fparranged_dataname).dropna(subset=['smiles']).reset_index(drop=True)
        # fplist = pd.read_csv(param["fplist"]).columns.values.tolist()
        xyz_dir_name = param["cube_dir_name"]
        # df = pd.read_excel(param["data_file_path"]).dropna(subset=['smiles']).reset_index(drop=True)  # [:10]
        print("dflen")
        print(len(df))
        # df = pd.concat([df, df_fplist], axis=1)
        # df = df.dropna(subset=['smiles']).reset_index(drop=True)
        df["mol"] = df["smiles"].apply(calculate_conformation.get_mol)
        print("dfbefore")
        print(len(df))
        # df = df[[os.path.isdir(features_dir_name + "/" + mol.GetProp("InchyKey")) for mol in df["mol"]]]

        df = df[
            [os.path.isdir("{}/{}".format(param["grid_coordinates"], mol.GetProp("InchyKey"))) for mol in df["mol"]]]
        df["mol"].apply(lambda mol: calculate_conformation.read_xyz(mol, xyz_dir_name + "/" + mol.GetProp("InchyKey")))

        dfp = pd.read_csv(param["grid_coordinates"] + "/penalty_param.csv")  # [:1]
        # dfp=np.load(param["grid_coordinates"]+"/penalty_param.npy")
        Dts = []
        for mol, RT in df[["mol", "RT"]].values:

            energy_to_Boltzmann_distribution(mol, RT)
            Dt = []
            weights = []
            for conf in mol.GetConformers():
                data = pd.read_pickle(
                    "{}/{}/data_yz{}.pkl".format(param["grid_coordinates"], mol.GetProp("InchyKey"), conf.GetId()))
                Dt.append(data["Dt"].values)
                weight = float(conf.GetProp("Boltzmann_distribution"))
                weights.append(weight)
            Dt_ = np.average(Dt, weights=weights, axis=0)
            Dts.append(Dt_)
        df["Dt"] = Dts

        for _ in range(5):
            df_ = df.sample(frac=1, random_state=_)
            save_path = param["out_dir_name"] + "/" + file_name + "/" + str(_)
            os.makedirs(save_path, exist_ok=True)
            Gaussian_penalized(features_dir_name, df_, dfp, param["grid_coordinates"], save_path)

        # looout_file_name = param["out_dir_name"] +file_name+ "/result_loonondfold.xlsx"
        # testout_file_name = param["out_dir_name"] +file_name+ "/result_train_test.xlsx"
        # crosstestout_file_name = param["out_dir_name"] +file_name+ "/result_5crossvalidnonfold.xlsx"
        # if fold:

        # if traintest:
        #     train_testfold(fold, features_dir_name, param["Regression_features"], param["feature_number"], df,
        #                    gridsearch_file_name
        #                    , looout_file_name, testout_file_name, param, fplist, param["Regression_type"],
        #                    param["maxmin"], dfp)
        # for regression_type in ["gaussian", "ridgecv", "PLS", "lassocv", "elasticnetcv"]:
        #     dir_name = param["out_dir_name"] + "/" + file_name + regression_type
        #     os.makedirs(dir_name, exist_ok=True)
        #     gridsearch_file_name = dir_name + "/result_grid_search.csv"
        #     looout_file_name = dir_name + "/result_loo.xlsx"
        #     crosstestout_file_name = dir_name + "/result_5crossvalid.xlsx"
        #     doublecrossvalidation(True, features_dir_name, param["Regression_features"], None, df,
        #                           gridsearch_file_name, looout_file_name,
        #                           crosstestout_file_name, param, regression_type,
        #                           False, dfp, file_name)

    end = time.perf_counter()  # 計測終了
    print("Finish")

    print('{:.2f}'.format(end - start))
    # path_w = '../errortest/alltime.txt'
    #
    # s = end - start
    #
    # with open(path_w, mode='w') as f:
    #     f.write(str(s))

    # elif practice:
    #     if param["Regression_type"] in "gaussian":
    #         grid_search(fold, features_dir_name, param["Regression_features"], param["feature_number"], df, dfp,
    #                     gridsearch_file_name,
    #                     fplist, param["Regression_type"], param["maxmin"])
    #     p = []
    #     if param["cat"] == "cbs":
    #         p = pd.read_csv("../errortest/hyperparam.csv")
    #
    #     else:
    #         print("Not exist gridsearch result")
    #
    #     if param["Regression_type"] in "gaussian":
    #         model = leave_one_out(fold, features_dir_name, param["Regression_features"], param["feature_number"],
    #                               df, looout_file_name,
    #                               param, fplist, param["Regression_type"], param["maxmin"], p)
    #
    # else:
    #     if param["Regression_type"] in ["gaussian", "gaussianFP"]:
    #         if param["Regression_type"] in "gaussian":
    #             grid_search(fold, features_dir_name, param["Regression_features"], param["feature_number"], df, dfp,
    #                         param["out_dir_name"] + "/result_grid_search.csv", fplist, param["Regression_type"],
    #                         param["maxmin"])
    #
    #     if param["Regression_type"] in ["gaussian", "gaussianFP"]:
    #
    #         if param["cat"] == "cbs":
    #             p = pd.read_csv("../result/cbs_gaussian/hyperparam.csv")
    #         elif param["cat"] == "dip":
    #             p = pd.read_csv("../result/dip-chloride_gaussian/hyperparam.csv")
    #         elif param["cat"] == "RuSS":
    #             p = pd.read_csv("../result/RuSS_gaussian/hyperparam.csv")
    #         else:
    #             raise ValueError
    #         leave_one_out(fold, features_dir_name, param["Regression_features"], param["feature_number"], df,
    #                       param["out_dir_name"] + "/result_loo.xls", param, fplist, param["Regression_type"],
    #                       param["maxmin"],
    #                       p)
    #
    #     else:
    #         leave_one_out(fold, features_dir_name, param["Regression_features"], param["feature_number"], df,
    #                       param["out_dir_name"] + "/result_loo.xls", param, fplist, param["Regression_type"],
    #                       param["maxmin"], p=None)
