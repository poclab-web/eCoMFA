import pandas as pd
import calculate_conformation
import os
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from rdkit.Chem import PandasTools
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
import time
import warnings
import json
import csv

"""
feature_num
"""
warnings.simplefilter('ignore')


def grid_search(fold, features_dir_name, regression_features, feature_number, df, dfp, out_file_name, fplist,
                regression_type, maxmin):
    os.makedirs("../errortest/", exist_ok=True)

    if fold:
        xyz = pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))[
            ["x", "y", "z"]].values
        print("yzfold")
    else:
        xyz = pd.read_csv("{}/{}/feature_y.csv".format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))[
            ["x", "y", "z"]].values
        print("zfold")

    d = np.array([np.linalg.norm(xyz - _, axis=1) for _ in xyz])
    d_y = np.array([np.linalg.norm(xyz - _ * np.array([1, -1, 1]), axis=1) for _ in xyz])
    d_z = np.array([np.linalg.norm(xyz - _ * np.array([1, 1, -1]), axis=1) for _ in xyz])
    d_yz = np.array([np.linalg.norm(xyz - _ * np.array([1, -1, -1]), axis=1) for _ in xyz])

    def gauss_func(d):
        sigma = 0.5
        leng = 1
        ans = 1 / (2 * np.pi * np.sqrt(2 * np.pi) * sigma ** 3) * leng ** 3 \
              * np.exp(-d ** 2 / (2 * sigma ** 2))
        return ans

    penalty = np.where(d < 0.5 * 3, gauss_func(d), 0)
    penalty_y = np.where(d_y < 0.5 * 3, gauss_func(d_y), 0)
    penalty_z = np.where(d_z < 0.5 * 3, gauss_func(d_z), 0)
    penalty_yz = np.where(d_yz < 0.5 * 3, gauss_func(d_yz), 0)
    if fold:
        penalty = penalty + penalty_y + penalty_z + penalty_yz
        grid_features_name = "{}/{}/feature_yz.csv"
    else:
        penalty = penalty + penalty_y
        grid_features_name = "{}/{}/feature_y.csv"

    penalty = penalty / np.max(np.sum(penalty, axis=0))
    # np.fill_diagonal(penalty, -1)
    penalty = penalty - np.identity(penalty.shape[0])/np.sum(penalty,axis=0)
    os.makedirs("../penalty", exist_ok=True)
    np.save('../penalty/penalty.npy', penalty)
    r2_list = []
    RMSE_list = []

    if False and feature_number == "1":
        hyperparam = list(dict.fromkeys(dfp["{}param".format(regression_features)]))
        q = []
        for L1 in hyperparam:
            a = []
            a.append(L1)
            q.append(a)
            print(L1)
            penalty1 = L1 * penalty
            l = []
            kf = KFold(n_splits=5, shuffle=False)
            for (train_index, test_index) in kf.split(df):

                feature = [
                    pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                        "{}".format(regression_features)].values
                    for
                    mol in df.iloc[train_index]["mol"]]

                if regression_type in ["gaussianFP"]:
                    X = np.concatenate([feature, penalty1], axis=0)
                    zeroweight = np.zeros(int(penalty.shape[1])).reshape(1, penalty.shape[1])
                    train_if = []
                    for weight in df.loc[:, fplist].columns:
                        if (train_tf := not df.iloc[train_index][weight].to_list()
                                                    .count(df.iloc[train_index][weight].to_list()[0]) == len(
                            df.iloc[train_index][weight])):
                            S = df.iloc[train_index][weight].values.reshape(1, -1)
                            Q = np.concatenate([S, zeroweight], axis=1)
                            X = np.concatenate([X, Q.T], axis=1)
                        train_if.append(train_tf)
                    Y = np.concatenate([df.iloc[train_index]["ΔΔG.expt."], np.zeros(penalty.shape[0])], axis=0)
                    model = linear_model.Ridge(alpha=0, fit_intercept=False).fit(X, Y)  # ただの線形回帰
                if regression_type in ["gaussian"]:
                    X = np.concatenate([feature, penalty1], axis=0)
                    Y = np.concatenate([df.iloc[train_index]["ΔΔG.expt."], np.zeros(penalty.shape[0])], axis=0)
                    model = linear_model.Ridge(alpha=0, fit_intercept=False).fit(X, Y)  # Ridgeじゃないただの線形回帰

                feature = [
                    pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                        "{}".format(regression_features)].values
                    for
                    mol in df.iloc[test_index]["mol"]]

                if regression_type in ["FP", "gaussianFP"]:
                    for weight in df.loc[:, fplist].columns:
                        S = np.array(df.iloc[test_index][weight].values).reshape(1, np.array(
                            df.iloc[test_index][weight].values).shape[0]).T
                        feature = np.concatenate([feature, S], axis=1)
                predict = model.predict(feature)
                if maxmin == "True":
                    for i in range(len(predict)):

                        if predict[i] >= df.iloc[test_index]["ΔΔmaxG.expt."].values[i]:
                            predict[i] = df.iloc[test_index]["ΔΔmaxG.expt."].values[i]
                        if predict[i] <= df.iloc[test_index]["ΔΔminG.expt."].values[i]:
                            predict[i] = df.iloc[test_index]["ΔΔminG.expt."].values[i]
                l.extend(predict)
            r2 = r2_score(df["ΔΔG.expt."], l)
            RMSE = np.sqrt(mean_squared_error(df["ΔΔG.expt."], l))
            print("r2", r2)
            r2_list.append(r2)
            RMSE_list.append(RMSE)

        paramlist = pd.DataFrame(q)
        print(regression_features.split()[0])
        paramlist.rename(columns={0: "{}param".format(regression_features.split()[0])}, inplace=True)


    elif feature_number == "2":
        feature1param = list(dict.fromkeys(dfp["{}param".format(regression_features.split()[0])]))
        feature2param = list(dict.fromkeys(dfp["{}param".format(regression_features.split()[1])]))
        #q = []
        # for L1 in feature1param:
        #     for L2 in feature2param:
        for L1,L2 in zip(dfp["Dtparam"],dfp["ESP_cutoffparam"]):
            print([L1, L2])
            # a = []
            # a.append(L1)
            # a.append(L2)
            # q.append(a)
            penalty1 = np.concatenate([L1 * penalty, np.zeros(penalty.shape)], axis=1)
            penalty2 = np.concatenate([np.zeros(penalty.shape), L2 * penalty], axis=1)
            l = []
            kf = KFold(n_splits=5, shuffle=False)
            for (train_index, test_index) in kf.split(df):

                feature1 = [
                    pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                        "{}".format(regression_features.split()[0])].values
                    for
                    mol in df.iloc[train_index]["mol"]]
                feature2 = [
                    pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                        "{}".format(regression_features.split()[1])].values
                    for
                    mol in df.iloc[train_index]["mol"]]

                features = np.concatenate([feature1, feature2], axis=1)
                if regression_type in ["gaussianFP"]:
                    X = np.concatenate([features, penalty1, penalty2], axis=0)
                    zeroweight = np.zeros(int(penalty.shape[1])).reshape(1, penalty.shape[1])
                    train_if = []
                    for weight in df.loc[:, fplist].columns:
                        if (train_tf := not df.iloc[train_index][weight].to_list()
                                                    .count(df.iloc[train_index][weight].to_list()[0]) == len(
                            df.iloc[train_index][weight])):
                            S = df.iloc[train_index][weight].values.reshape(1, -1)
                            Q = np.concatenate([S, zeroweight, zeroweight], axis=1)
                            X = np.concatenate([X, Q.T], axis=1)
                        train_if.append(train_tf)
                    Y = np.concatenate([df.iloc[train_index]["ΔΔG.expt."], np.zeros(penalty.shape[0] * 2)], axis=0)
                    model = linear_model.Ridge(alpha=0, fit_intercept=False).fit(X, Y)  # ただの線形回帰
                if regression_type in ["gaussian"]:
                    X = np.concatenate([features, penalty1, penalty2], axis=0)
                    Y = np.concatenate([df.iloc[train_index]["ΔΔG.expt."], np.zeros(penalty.shape[0] * 2)], axis=0)
                    model = linear_model.Ridge(alpha=0, fit_intercept=False).fit(X, Y)  # Ridgeじゃないただの線形回帰

                # ここから、テストセットの特徴量計算
                feature1 = [
                    pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                        "{}".format(regression_features.split()[0])].values
                    for
                    mol in df.iloc[test_index]["mol"]]
                feature2 = [
                    pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                        "{}".format(regression_features.split()[1])].values
                    for
                    mol in df.iloc[test_index]["mol"]]

                features = np.concatenate([feature1, feature2], axis=1)

                if regression_type in ["FP", "gaussianFP"]:
                    for weight in df.loc[:, fplist].columns:
                        S = np.array(df.iloc[test_index][weight].values).reshape(1, np.array(
                            df.iloc[test_index][weight].values).shape[0]).T
                        features = np.concatenate([features, S], axis=1)
                predict = model.predict(features)
                if maxmin == "True":
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
        #print(q)
        #paramlist = pd.DataFrame(q)
        paramlist=dfp
        print(regression_features.split()[0])
        print(regression_features.split()[1])
        paramlist.rename(columns={0: "{}param".format(regression_features.split()[0]),
                                  1: "{}param".format(regression_features.split()[1])}, inplace=True)




    elif False and feature_number == "3":
        feature1param = list(dict.fromkeys(dfp["{}param".format(regression_features.split()[0])]))
        feature2param = list(dict.fromkeys(dfp["{}param".format(regression_features.split()[1])]))
        feature3param = list(dict.fromkeys(dfp["{}param".format(regression_features.split()[2])]))
        q = []
        for L1 in feature1param:
            for L2 in feature2param:
                for L3 in feature3param:
                    a = []
                    a.append(L1)
                    a.append(L2)
                    a.append(L3)
                    q.append(a)
                    penalty1 = np.concatenate([L1 * penalty, np.zeros(penalty.shape), np.zeros(penalty.shape)], axis=1)
                    penalty2 = np.concatenate([np.zeros(penalty.shape), L2 * penalty, np.zeros(penalty.shape)], axis=1)
                    penalty3 = np.concatenate([np.zeros(penalty.shape), np.zeros(penalty.shape), L3 * penalty], axis=1)
                    l = []
                    kf = KFold(n_splits=5, shuffle=False)
                    for (train_index, test_index) in kf.split(df):
                        feature1 = [
                            pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                "{}".format(regression_features.split()[0])].values
                            for
                            mol in df.iloc[train_index]["mol"]]
                        feature2 = [
                            pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                "{}".format(regression_features.split()[1])].values
                            for
                            mol in df.iloc[train_index]["mol"]]
                        feature3 = [
                            pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                "{}".format(regression_features.split()[2])].values
                            for
                            mol in df.iloc[train_index]["mol"]]
                        features = np.concatenate([feature1, feature2, feature3], axis=1)
                        if regression_type in ["gaussianFP"] or ["gaussian"]:
                            X = np.concatenate([features, penalty1, penalty2, penalty3], axis=0)
                            if regression_type in ["gaussianFP"]:
                                zeroweight = np.zeros(int(penalty.shape[1])).reshape(1, penalty.shape[1])
                                train_if = []
                                for weight in df.loc[:, fplist].columns:
                                    if (train_tf := not df.iloc[train_index][weight].to_list()
                                                                .count(
                                        df.iloc[train_index][weight].to_list()[0]) == len(
                                        df.iloc[train_index][weight])):
                                        S = df.iloc[train_index][weight].values.reshape(1, -1)
                                        Q = np.concatenate([S, zeroweight, zeroweight], axis=1)
                                        X = np.concatenate([X, Q.T], axis=1)
                                    train_if.append(train_tf)
                            Y = np.concatenate([df.iloc[train_index]["ΔΔG.expt."], np.zeros(penalty.shape[0] * 3)],
                                               axis=0)
                            model = linear_model.Ridge(alpha=0, fit_intercept=False).fit(X, Y)  # ただの線形回帰

                        # ここから、テストセットの特徴量計算
                        feature1 = [
                            pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                "{}".format(regression_features.split()[0])].values
                            for
                            mol in df.iloc[test_index]["mol"]]
                        feature2 = [
                            pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                "{}".format(regression_features.split()[1])].values
                            for
                            mol in df.iloc[test_index]["mol"]]
                        feature3 = [
                            pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                "{}".format(regression_features.split()[2])].values
                            for
                            mol in df.iloc[test_index]["mol"]]
                        features = np.concatenate([feature1, feature2, feature3], axis=1)

                        if regression_type in ["FP", "gaussianFP"]:
                            for weight in df.loc[:, fplist].columns:
                                S = np.array(df.iloc[test_index][weight].values).reshape(1, np.array(
                                    df.iloc[test_index][weight].values).shape[0]).T

                                features = np.concatenate([features, S], axis=1)
                        predict = model.predict(features)
                        if maxmin == "True":
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

        paramlist = pd.DataFrame(q)
        paramlist.rename(columns={0: "{}param".format(regression_features.split()[0]),
                                  1: "{}param".format(regression_features.split()[1]),
                                  2: "{}param".format(regression_features.split()[2])}, inplace=True)
    # else:
    #     raise ValueError
    paramlist["r2"] = r2_list
    paramlist["RMSE"] = RMSE_list
    print(paramlist)
    print(paramlist.columns)
    paramlist.to_csv(out_file_name)

    min_index = paramlist['RMSE'].idxmin()
    min_row = paramlist.loc[min_index, :]
    p = pd.DataFrame([min_row], index=[min_index])
    print(p)
    p.to_csv(param["out_dir_name"] + "/hyperparam.csv")


def leave_one_out(fold, features_dir_name, regression_features, feature_number, df, out_file_name, param, fplist,
                  regression_type, maxmin, p=None):
    penalty = np.load("../penalty/penalty.npy")
    if fold:
        grid_features_name = "{}/{}/feature_yz.csv"
    else:
        grid_features_name = "{}/{}/feature_y.csv"

    # if regression_features in ["LUMO"]:
    if False and feature_number == "1":

        feature = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                       "{}".format(regression_features)].values
                   for mol
                   in df["mol"]]
        if regression_type == "lassocv" or regression_type == "PLS" or regression_type == "ridgecv" or regression_type == "elasticnetcv":

            Y = df["ΔΔG.expt."].values
            if regression_type == "lassocv":
                model = linear_model.LassoCV(fit_intercept=False, cv=5).fit(feature, Y)
            elif regression_type == "PLS":
                model = PLSRegression(n_components=5).fit(feature, Y)
            elif regression_type == "ridgecv":
                model = RidgeCV(fit_intercept=False, cv=5).fit(feature, Y)
            elif regression_type == "elasticnetcv":
                model = ElasticNetCV(fit_intercept=False, cv=5).fit(feature, Y)

            df["ΔΔG.train"] = model.predict(feature)
            df_mf = pd.read_csv(grid_features_name.format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))
            os.makedirs(param["moleculer_field_dir"], exist_ok=True)
            df_mf["MF_{}".format(regression_features)] = model.coef_[:penalty.shape[0]]
            if regression_type == "lassocv":
                df_mf.to_csv(
                    (param["moleculer_field_dir"] + "/" + "{}lassocvmoleculer_field.csv".format(regression_features)))
            elif regression_type == "PLS":
                df_mf.to_csv(
                    (param["moleculer_field_dir"] + "/" + "{}PLSmoleculer_field.csv".format(regression_features)))
            elif regression_type == "ridgecv":
                df_mf.to_csv(
                    (param["moleculer_field_dir"] + "/" + "{}ridgecvmoleculer_field.csv".format(regression_features)))
            elif regression_type == "elasticnetcv":
                df_mf.to_csv((param["moleculer_field_dir"] + "/" + "{}elasticnetcvmoleculer_field.csv".format(
                    regression_features)))
            else:
                print("regression type error")
                raise ValueError

        if regression_type in ["gaussian"]:
            hparam = p['{}param'.format(regression_features)].values
            p = float(hparam[0])
            penalty3 = p * penalty

            X = np.concatenate([feature, penalty3], axis=0)
            Y = np.concatenate([df["ΔΔG.expt."], np.zeros(penalty.shape[0])], axis=0)
            model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)
            df["ΔΔG.train"] = model.predict(feature)
            df_mf = pd.read_csv(grid_features_name.format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))
            print(df["mol"].iloc[0].GetProp("InchyKey"))
            os.makedirs(param["moleculer_field_dir"], exist_ok=True)

            df_mf["MF_{}".format(regression_features)] = model.coef_[:penalty.shape[0]]

            df_mf.to_csv((param["moleculer_field_dir"] + "/" + "{}moleculer_field.csv".format(regression_features)))

        if regression_type in ["gaussianFP"]:
            penalty3 = p * penalty
            X = np.concatenate([feature, penalty3], axis=0)
            zeroweight = np.zeros(int(penalty.shape[1])).reshape(1, penalty.shape[1])
            for weight in df.loc[:, fplist].columns:
                S = np.array(df[weight].values).reshape(1, -1)
                Q = np.concatenate([S, zeroweight], axis=1)
                X = np.concatenate([X, Q.T], axis=1)
            Y = np.concatenate([df["ΔΔG.expt."], np.zeros(penalty.shape[0])], axis=0)
            model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)
            for weight in df.loc[:, fplist].columns:
                S = df[weight].values.reshape(-1, 1)
                features = np.concatenate([features, S], axis=1)
            df["ΔΔG.train"] = model.predict(features)
            df_mf = pd.read_csv(grid_features_name.format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))

            print("model.coef_[penalty.shape[0]*2:]")
            # print(model.coef_)
            print(model.coef_[penalty.shape[0]:])
            df_fpv = str(model.coef_[penalty.shape[0]:])
            path_w = param["out_dir_name"] + "/" + "{}fpvalue".format(regression_features)
            with open(path_w, mode='w') as f:
                f.write(df_fpv)

        if regression_type in ["FP"]:
            X = np.concatenate([features], axis=0)
            for weight in df.loc[:, fplist].columns:
                S = df[weight].values.reshape(-1, 1)
                X = np.concatenate([X, S], axis=1)
            Y = df["ΔΔG.expt."].values
            # model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)
            model = PLSRegression(n_components=5).fit(features, Y)
            for weight in df.loc[:, fplist].columns:
                S = np.array(df[weight].values).reshape(-1, 1)
                features = np.concatenate([features, S], axis=1)
            df["ΔΔG.train"] = model.predict(features)
            df_mf = pd.read_csv(grid_features_name.format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))
            print(model.coef_.shape)
            df_mf["{}".format(regression_features)] = model.coef_[:penalty.shape[0]]
            print("model.coef_[penalty.shape[0]*2:]")
            # print(model.coef_)
            print(model.coef_[penalty.shape[0]:])
            df_fpv = str(model.coef_[penalty.shape[0]:])
            path_w = param["out_dir_name"] + "/" + "{}fpvalue".format(regression_features)
            with open(path_w, mode='w') as f:
                f.write(df_fpv)

    elif feature_number == "2":
        features1 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                         "{}".format(regression_features.split()[0])].values for mol
                     in df["mol"]]
        features2 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                         "{}".format(regression_features.split()[1])].values for mol
                     in df["mol"]]
        features = np.concatenate([features1, features2], axis=1)
        print(features.shape)
        if regression_type == "lassocv" or regression_type == "PLS" or regression_type == "ridgecv" or regression_type == "elasticnetcv":
            Y = df["ΔΔG.expt."].values
            print(Y.shape)
            if regression_type == "lassocv":
                model = linear_model.LassoCV(fit_intercept=False, cv=5).fit(features, Y)
            elif regression_type == "PLS":
                model = PLSRegression(n_components=5).fit(features, Y)
            elif regression_type == "ridgecv":
                model = RidgeCV(fit_intercept=False, cv=5).fit(features, Y)
            elif regression_type == "elasticnetcv":
                model = ElasticNetCV(fit_intercept=False, cv=5).fit(features, Y)
            df["ΔΔG.train"] = model.predict(features)
            df_mf = pd.read_csv(grid_features_name.format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))
            os.makedirs(param["moleculer_field_dir"], exist_ok=True)

            if regression_type == "lassocv":
                df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[:int(model.coef_.shape[0] / 2)]
                df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[int(model.coef_.shape[0] / 2):int(
                    model.coef_.shape[0])]
            elif regression_type == "PLS":
                df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[0][
                                                                        :int(model.coef_[0].shape[0] / 2)]
                df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[0][
                                                                        int(model.coef_[0].shape[0] / 2):int(
                                                                            model.coef_[0].shape[0])]
            elif regression_type == "ridgecv":
                df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[:int(model.coef_.shape[0] / 2)]
                df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[int(model.coef_.shape[0] / 2):int(
                    model.coef_.shape[0])]
            elif regression_type == "elasticnetcv":
                df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[:int(model.coef_.shape[0] / 2)]
                df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[int(model.coef_.shape[0] / 2):int(
                    model.coef_.shape[0])]

            df_mf.to_csv((param["moleculer_field_dir"] + "/" + "moleculer_field.csv"))

        if regression_type in ["gaussian"]:
            print(p)
            hparam1 = p['{}param'.format(regression_features.split()[0])].values
            hparam2 = p['{}param'.format(regression_features.split()[1])].values
            penalty1 = np.concatenate([hparam1 * penalty, np.zeros(penalty.shape)], axis=1)
            penalty2 = np.concatenate([np.zeros(penalty.shape), hparam2 * penalty], axis=1)
            X = np.concatenate([features, penalty1, penalty2], axis=0)
            Y = np.concatenate([df["ΔΔG.expt."], np.zeros(penalty.shape[0] * 2)], axis=0)
            model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)
            df["ΔΔG.train"] = model.predict(features)
            df_mf = pd.read_csv(grid_features_name.format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))
            print(df["mol"].iloc[0].GetProp("InchyKey"))
            os.makedirs(param["moleculer_field_dir"], exist_ok=True)

            df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[:penalty.shape[0]]
            df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[penalty.shape[0]:penalty.shape[0] * 2]
            df_mf.to_csv((param["moleculer_field_dir"] + "/" + "moleculer_field.csv"))
            df["{}_contribution".format(regression_features.split()[0])] = np.sum(
                features1 * df_mf["MF_{}".format(regression_features.split()[0])].values, axis=1)
            df["{}_contribution".format(regression_features.split()[1])] = np.sum(
                features2 * df_mf["MF_{}".format(regression_features.split()[1])].values, axis=1)

        if regression_type in ["gaussianFP"]:
            hparam1 = p['{}param'.format(regression_features.split()[0])].values
            hparam2 = p['{}param'.format(regression_features.split()[1])].values
            penalty1 = np.concatenate([hparam1 * penalty, np.zeros(penalty.shape)], axis=1)
            penalty2 = np.concatenate([np.zeros(penalty.shape), hparam2 * penalty], axis=1)
            X = np.concatenate([features, penalty1, penalty2], axis=0)
            zeroweight = np.zeros(int(penalty.shape[1])).reshape(1, penalty.shape[1])
            for weight in df.loc[:, fplist].columns:
                S = np.array(df[weight].values).reshape(1, -1)
                Q = np.concatenate([S, zeroweight, zeroweight], axis=1)
                X = np.concatenate([X, Q.T], axis=1)
            Y = np.concatenate([df["ΔΔG.expt."], np.zeros(penalty.shape[0] * 2)], axis=0)
            model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)
            for weight in df.loc[:, fplist].columns:
                S = df[weight].values.reshape(-1, 1)
                features = np.concatenate([features, S], axis=1)
            df["ΔΔG.train"] = model.predict(features)
            df_mf = pd.read_csv(grid_features_name.format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))

            df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[:penalty.shape[0]]
            df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[penalty.shape[0]:penalty.shape[0] * 2]

            df["{}_contribution".format(regression_features.split()[0])] = np.sum(
                features1 * df_mf["MF_{}".format(regression_features.split()[0])].values, axis=1)
            df["{}_contribution".format(regression_features.split()[1])] = np.sum(
                features2 * df_mf["MF_{}".format(regression_features.split()[1])].values, axis=1)
            print("model.coef_[penalty.shape[0]*2:]")
            # print(model.coef_)
            print(model.coef_[penalty.shape[0] * 2:])
            df_fpv = str(model.coef_[penalty.shape[0] * 2:])
            path_w = param["out_dir_name"] + "/" + "fpvalue"
            with open(path_w, mode='w') as f:
                f.write(df_fpv)
        if regression_type in ["FP"]:
            X = np.concatenate([features], axis=0)
            for weight in df.loc[:, fplist].columns:
                S = df[weight].values.reshape(-1, 1)
                X = np.concatenate([X, S], axis=1)
            Y = df["ΔΔG.expt."].values
            # model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)
            model = PLSRegression(n_components=5).fit(X, Y)
            for weight in df.loc[:, fplist].columns:
                S = np.array(df[weight].values).reshape(-1, 1)
                features = np.concatenate([features, S], axis=1)
            df["ΔΔG.train"] = model.predict(features)
            df_mf = pd.read_csv(grid_features_name.format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))
            print(model.coef_.shape)
            df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[0][:penalty.shape[0]]
            df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[0][
                                                                    penalty.shape[0]:penalty.shape[0] * 2]
            df["{}_contribution".format(regression_features.split()[0])] = np.sum(
                features1 * df_mf["MF_{}".format(regression_features.split()[0])].values, axis=1)
            df["{}_contribution".format(regression_features.split()[1])] = np.sum(
                features2 * df_mf["MF_{}".format(regression_features.split()[1])].values, axis=1)
            print("model.coef_[penalty.shape[0]*2:]")
            # print(model.coef_)
            print(model.coef_[0][penalty.shape[0] * 2:])
            df_fpv = str(model.coef_[0][penalty.shape[0] * 2:])
            path_w = param["out_dir_name"] + "/" + "fpvalue"
            with open(path_w, mode='w') as f:
                f.write(df_fpv)

    elif False and feature_number == "3":
        features1 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                         "{}".format(regression_features.split()[0])].values for mol
                     in df["mol"]]
        features2 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                         "{}".format(regression_features.split()[1])].values for mol
                     in df["mol"]]
        features3 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                         "{}".format(regression_features.split()[2])].values for mol
                     in df["mol"]]

        features = np.concatenate([features1, features2, features3], axis=1)
        if regression_type == "lassocv" or regression_type == "PLS" or regression_type == "ridgecv" or regression_type == "elasticnetcv":
            Y = df["ΔΔG.expt."].values
            if regression_type == "lassocv":
                model = linear_model.LassoCV(fit_intercept=False, cv=5).fit(features, Y)
            elif regression_type == "PLS":
                model = PLSRegression(n_components=5).fit(features, Y)
            elif regression_type == "ridgecv":
                model = RidgeCV(fit_intercept=False, cv=5).fit(features, Y)
            elif regression_type == "elasticnetcv":
                model = ElasticNetCV(fit_intercept=False, cv=5).fit(features, Y)
            df["ΔΔG.train"] = model.predict(features)
            df_mf = pd.read_csv(grid_features_name.format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))

            os.makedirs(param["moleculer_field_dir"], exist_ok=True)

            print("aaa")

            if regression_type == "lassocv":
                df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[:int(model.coef_.shape[0] / 3)]
                df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[
                                                                        int(model.coef_.shape[0] / 3):int(
                                                                            model.coef_.shape[0] / 3) * 2]
                df_mf["MF_{}".format(regression_features.split()[2])] = model.coef_[
                                                                        int(
                                                                            model.coef_.shape[0] / 3) * 2:int(
                                                                            model.coef_.shape[0])]
            elif regression_type == "PLS":
                df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[0][:penalty.shape[0]]
                df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[0][
                                                                        penalty.shape[0]:penalty.shape[0] * 2]
                df_mf["MF_{}".format(regression_features.split()[2])] = model.coef_[0][
                                                                        penalty.shape[0] * 2:penalty.shape[0] * 3]
            elif regression_type == "ridgecv":
                df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[0][:penalty.shape[0]]
                df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[0][
                                                                        penalty.shape[0]:penalty.shape[0] * 2]
                df_mf["MF_{}".format(regression_features.split()[2])] = model.coef_[0][
                                                                        penalty.shape[0] * 2:penalty.shape[0] * 3]
            elif regression_type == "elasticnetcv":
                df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[0][:penalty.shape[0]]
                df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[0][
                                                                        penalty.shape[0]:penalty.shape[0] * 2]
                df_mf["MF_{}".format(regression_features.split()[2])] = model.coef_[0][
                                                                        penalty.shape[0] * 2:penalty.shape[0] * 3]

            df_mf.to_csv((param["moleculer_field_dir"] + "/" + "moleculer_field.csv"))

        if regression_type in ["gaussian"]:
            hparam1 = p['{}param'.format(regression_features.split()[0])].values
            hparam2 = p['{}param'.format(regression_features.split()[1])].values
            hparam3 = p['{}param'.format(regression_features.split()[2])].values
            penalty1 = np.concatenate([hparam1 * penalty, np.zeros(penalty.shape), np.zeros(penalty.shape)], axis=1)
            penalty2 = np.concatenate([np.zeros(penalty.shape), hparam2 * penalty, np.zeros(penalty.shape)], axis=1)
            penalty3 = np.concatenate([np.zeros(penalty.shape), np.zeros(penalty.shape), hparam3 * penalty], axis=1)
            X = np.concatenate([features, penalty1, penalty2, penalty3], axis=0)
            Y = np.concatenate([df["ΔΔG.expt."], np.zeros(penalty.shape[0] * 3)], axis=0)
            model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)
            df["ΔΔG.train"] = model.predict(features)
            df_mf = pd.read_csv(grid_features_name.format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))
            print(df["mol"].iloc[0].GetProp("InchyKey"))
            os.makedirs(param["moleculer_field_dir"], exist_ok=True)

            df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[:penalty.shape[0]]
            df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[penalty.shape[0]:penalty.shape[0] * 2]
            df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[
                                                                    penalty.shape[0] * 2:penalty.shape[0] * 3]
            df_mf.to_csv((param["moleculer_field_dir"] + "/" + "moleculer_field.csv"))
            df["{}_contribution".format(regression_features.split()[0])] = np.sum(
                features1 * df_mf["MF_{}".format(regression_features.split()[0])].values, axis=1)
            df["{}_contribution".format(regression_features.split()[1])] = np.sum(
                features2 * df_mf["MF_{}".format(regression_features.split()[1])].values, axis=1)
            df["{}_contribution".format(regression_features.split()[2])] = np.sum(
                features3 * df_mf["MF_{}".format(regression_features.split()[2])].values, axis=1)

        if regression_type in ["gaussianFP"]:
            penalty1 = np.concatenate([p[0] * penalty, np.zeros(penalty.shape), np.zeros(penalty.shape)], axis=1)
            penalty2 = np.concatenate([np.zeros(penalty.shape), p[1] * penalty, np.zeros(penalty.shape)], axis=1)
            penalty3 = np.concatenate([np.zeros(penalty.shape), np.zeros(penalty.shape), p[2] * penalty], axis=1)
            X = np.concatenate([features, penalty1, penalty2, penalty3], axis=0)
            zeroweight = np.zeros(int(penalty.shape[1])).reshape(1, penalty.shape[1])
            for weight in df.loc[:, fplist].columns:
                S = np.array(df[weight].values).reshape(1, -1)
                Q = np.concatenate([S, zeroweight, zeroweight, zeroweight], axis=1)
                X = np.concatenate([X, Q.T], axis=1)
            Y = np.concatenate([df["ΔΔG.expt."], np.zeros(penalty.shape[0] * 3)], axis=0)
            model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)
            for weight in df.loc[:, fplist].columns:
                S = df[weight].values.reshape(-1, 1)
                features = np.concatenate([features, S], axis=1)
            df["ΔΔG.train"] = model.predict(features)
            df_mf = pd.read_csv(
                grid_features_name.format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))

            df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[:penalty.shape[0]]
            df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[
                                                                    penalty.shape[0]:penalty.shape[0] * 2]
            df_mf["MF_{}".format(regression_features.split()[2])] = model.coef_[
                                                                    penalty.shape[0] * 2:penalty.shape[0] * 3]

            df["{}_contribution".format(regression_features.split()[0])] = np.sum(
                features1 * df_mf["MF_{}".format(regression_features.split()[0])].values, axis=1)
            df["{}_contribution".format(regression_features.split()[1])] = np.sum(
                features2 * df_mf["MF_{}".format(regression_features.split()[1])].values, axis=1)
            df["{}_contribution".format(regression_features.split()[2])] = np.sum(
                features3 * df_mf["MF_{}".format(regression_features.split()[2])].values, axis=1)
            print("model.coef_[penalty.shape[0]*2:]")
            # print(model.coef_)
            print(model.coef_[penalty.shape[0] * 3:])
            df_fpv = str(model.coef_[penalty.shape[0] * 3:])
            path_w = param["out_dir_name"] + "/" + "fpvalue"
            with open(path_w, mode='w') as f:
                f.write(df_fpv)
        if regression_type in ["FP"]:
            X = np.concatenate([features], axis=0)
            for weight in df.loc[:, fplist].columns:
                S = df[weight].values.reshape(-1, 1)
                X = np.concatenate([X, S], axis=1)
            Y = df["ΔΔG.expt."].values
            # model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)
            model = PLSRegression(n_components=5).fit(X, Y)
            for weight in df.loc[:, fplist].columns:
                S = np.array(df[weight].values).reshape(-1, 1)
                features = np.concatenate([features, S], axis=1)
            df["ΔΔG.train"] = model.predict(features)
            df_mf = pd.read_csv(
                grid_features_name.format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))
            print(model.coef_.shape)
            df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[0][:penalty.shape[0]]
            df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[0][
                                                                    penalty.shape[0]:penalty.shape[0] * 2]
            df_mf["MF_{}".format(regression_features.split()[2])] = model.coef_[0][
                                                                    penalty.shape[0] * 2:penalty.shape[0] * 3]
            df["{}_contribution".format(regression_features.split()[0])] = np.sum(
                features1 * df_mf["MF_{}".format(regression_features.split()[0])].values, axis=1)
            df["{}_contribution".format(regression_features.split()[1])] = np.sum(
                features2 * df_mf["MF_{}".format(regression_features.split()[1])].values, axis=1)
            df["{}_contribution".format(regression_features.split()[2])] = np.sum(
                features3 * df_mf["MF_{}".format(regression_features.split()[2])].values, axis=1)
            print("model.coef_[penalty.shape[0]*2:]")
            # print(model.coef_)
            print(model.coef_[0][penalty.shape[0] * 3:])
            df_fpv = str(model.coef_[0][penalty.shape[0] * 3:])
            path_w = param["out_dir_name"] + "/" + "fpvalue"
            with open(path_w, mode='w') as f:
                f.write(df_fpv)

    # ここからテストセットの実行

    if False and feature_number == "1":
        if regression_type == "lassocv" or regression_type == "PLS" or regression_type == "ridgecv" or regression_type == "elasticnetcv":
            l = []
            kf = KFold(n_splits=len(df), shuffle=False)
            for (train_index, test_index) in kf.split(df):
                features = [
                    pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                        "{}".format(regression_features)].values
                    for
                    mol in df.iloc[train_index]["mol"]]

                Y = df.iloc[train_index]["ΔΔG.expt."].values
                if regression_type == "lassocv":
                    model = linear_model.LassoCV(fit_intercept=False, cv=5).fit(features, Y)
                elif regression_type == "PLS":
                    model = PLSRegression(n_components=5).fit(features, Y)
                elif regression_type == "ridgecv":
                    model = RidgeCV(fit_intercept=False, cv=5).fit(features, Y)
                elif regression_type == "elasticnetcv":
                    model = ElasticNetCV(fit_intercept=False, cv=5).fit(features, Y)

                features = [
                    pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                        "{}".format(regression_features)].values
                    for
                    mol in df.iloc[test_index]["mol"]]

                predict = model.predict(features)

                l.extend(predict)
        if regression_type in ["gaussian", "gaussianFP"]:
            l = []
            kf = KFold(n_splits=len(df), shuffle=False)
            for (train_index, test_index) in kf.split(df):
                features = [
                    pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                        "{}".format(regression_features)].values
                    for
                    mol in df.iloc[train_index]["mol"]]

                penalty1 = p * penalty

                X = np.concatenate([features, penalty1], axis=0)
                if regression_type in ["gaussianFP"]:
                    zeroweight = np.zeros(int(penalty.shape[1])).reshape(1, penalty.shape[1])
                    for weight in df.loc[:, fplist].columns:
                        S = df.iloc[train_index][weight].values.reshape(1, -1)
                        Q = np.concatenate([S, zeroweight], axis=1)
                        X = np.concatenate([X, Q.T], axis=1)
                Y = np.concatenate([df.iloc[train_index]["ΔΔG.expt."], np.zeros(penalty.shape[0])], axis=0)
                model = linear_model.Ridge(alpha=0, fit_intercept=False).fit(X, Y)

                features = [
                    pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                        "{}".format(regression_features)].values
                    for
                    mol in df.iloc[test_index]["mol"]]

                if regression_type in ["gaussianFP"]:
                    for weight in df.loc[:, fplist].columns:
                        S = df.iloc[test_index][weight].values.reshape(-1, 1)
                        features = np.concatenate([features, S], axis=1)
                predict = model.predict(features)

                l.extend(predict)

        if regression_type in "FP":
            l = []
            kf = KFold(n_splits=len(df), shuffle=False)
            for (train_index, test_index) in kf.split(df):
                features = [
                    pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                        "{}".format(regression_features)].values
                    for
                    mol in df.iloc[train_index]["mol"]]
                X = np.concatenate([features], axis=0)

                for weight in df.loc[:, fplist].columns:
                    S = df.iloc[train_index][weight].values.reshape(-1, 1)
                    X = np.concatenate([X, S], axis=1)
                Y = df.iloc[train_index]["ΔΔG.expt."]
                model = PLSRegression(n_components=5).fit(X, Y)
                # model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)

                features = [
                    pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))["LUMO"].values
                    for
                    mol in df.iloc[test_index]["mol"]]

                for weight in df.loc[:, fplist].columns:
                    S = df.iloc[test_index][weight].values.reshape(-1, 1)
                    features = np.concatenate([features, S], axis=1)
                predict = model.predict(features)

                l.extend(predict)


    elif feature_number == "2":
        kf = KFold(n_splits=len(df), shuffle=False)
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
                features = np.concatenate([features1, features2], axis=1)

                Y = df.iloc[train_index]["ΔΔG.expt."].values
                if regression_type == "lassocv":
                    model = linear_model.LassoCV(fit_intercept=False, cv=5).fit(features, Y)
                elif regression_type == "PLS":
                    model = PLSRegression(n_components=5).fit(features, Y)
                elif regression_type == "ridgecv":
                    model = RidgeCV(fit_intercept=False, cv=5).fit(features, Y)
                elif regression_type == "elasticnetcv":
                    model = ElasticNetCV(fit_intercept=False, cv=5).fit(features, Y)
                else:
                    print("regressionerror")
                    raise ValueError

                features1 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[0])].values
                             for
                             mol in df.iloc[test_index]["mol"]]
                features2 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[1])].values
                             for
                             mol in df.iloc[test_index]["mol"]]

                features = np.concatenate([features1, features2], axis=1)
                predict = model.predict(features)

                if regression_type == "PLS":
                    for i in range(len(predict)):
                        if maxmin == "True":

                            if predict[i] >= df.iloc[test_index]["ΔΔmaxG.expt."].values[i]:
                                predict[i] = df.iloc[test_index]["ΔΔmaxG.expt."].values[i]
                            if predict[i] <= df.iloc[test_index]["ΔΔminG.expt."].values[i]:
                                predict[i] = df.iloc[test_index]["ΔΔminG.expt."].values[i]
                        l.extend(predict[i])

                else:
                    for i in range(len(predict)):
                        if maxmin == "True":
                            if predict[i] >= df.iloc[test_index]["ΔΔmaxG.expt."].values[i]:
                                predict[i] = df.iloc[test_index]["ΔΔmaxG.expt."].values[i]
                            if predict[i] <= df.iloc[test_index]["ΔΔminG.expt."].values[i]:
                                predict[i] = df.iloc[test_index]["ΔΔminG.expt."].values[i]

                        l.extend([predict[i]])

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
                features = np.concatenate([features1, features2], axis=1)
                penalty1 = np.concatenate([hparam1 * penalty, np.zeros(penalty.shape)], axis=1)
                penalty2 = np.concatenate([np.zeros(penalty.shape), hparam2 * penalty], axis=1)
                X = np.concatenate([features, penalty1, penalty2], axis=0)
                if regression_type in ["gaussianFP"]:
                    zeroweight = np.zeros(int(penalty.shape[1])).reshape(1, penalty.shape[1])
                    for weight in df.loc[:, fplist].columns:
                        S = df.iloc[train_index][weight].values.reshape(1, -1)
                        Q = np.concatenate([S, zeroweight, zeroweight], axis=1)
                        X = np.concatenate([X, Q.T], axis=1)
                Y = np.concatenate([df.iloc[train_index]["ΔΔG.expt."], np.zeros(penalty.shape[0] * 2)], axis=0)
                model = linear_model.Ridge(alpha=0, fit_intercept=False).fit(X, Y)

                # ここから、テストセットの特徴量計算
                features1 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[0])].values for
                             mol in df.iloc[test_index]["mol"]]
                features2 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[1])].values for
                             mol in df.iloc[test_index]["mol"]]
                features = np.concatenate([features1, features2], axis=1)
                if regression_type in ["FP", "gaussianFP"]:
                    for weight in df.loc[:, fplist].columns:
                        S = df.iloc[test_index][weight].values.reshape(-1, 1)
                        features = np.concatenate([features, S], axis=1)
                predict = model.predict(features)
                if maxmin == "True":
                    for i in range(len(predict)):
                        if predict[i] >= df.iloc[test_index]["ΔΔmaxG.expt."].values[i]:
                            predict[i] = df.iloc[test_index]["ΔΔmaxG.expt."].values[i]
                        if predict[i] <= df.iloc[test_index]["ΔΔminG.expt."].values[i]:
                            predict[i] = df.iloc[test_index]["ΔΔminG.expt."].values[i]

                l.extend(predict)
        if regression_type in "FP":
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
                features = np.concatenate([features1, features2], axis=1)
                X = np.concatenate([features], axis=0)

                for weight in df.loc[:, fplist].columns:
                    S = df.iloc[train_index][weight].values.reshape(-1, 1)
                    X = np.concatenate([X, S], axis=1)
                Y = df.iloc[train_index]["ΔΔG.expt."]
                # model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)
                model = PLSRegression(n_components=5).fit(X, Y)

                features1 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[0])].values
                             for
                             mol in df.iloc[test_index]["mol"]]
                features2 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[1])].values
                             for
                             mol in df.iloc[test_index]["mol"]]
                features = np.concatenate([features1, features2], axis=1)

                for weight in df.loc[:, fplist].columns:
                    S = df.iloc[test_index][weight].values.reshape(-1, 1)
                    features = np.concatenate([features, S], axis=1)
                predict = model.predict(features)
                if regression_type == "PLS":
                    if maxmin == "True":
                        print("kouzicyuu maxmin")
                        raise ValueError

                        for i in range(len(predict)):
                            if predict[i] >= df.iloc[test_index]["ΔΔmaxG.expt."].values[i]:
                                predict[i] = df.iloc[test_index]["ΔΔmaxG.expt."].values[i]
                            if predict[i] <= df.iloc[test_index]["ΔΔminG.expt."].values[i]:
                                predict[i] = df.iloc[test_index]["ΔΔminG.expt."].values[i]
                    l.extend(predict[i])

    elif False and feature_number == "3":
        if regression_type == "lassocv" or regression_type == "PLS" or regression_type == "ridgecv" or regression_type == "elasticnetcv":
            l = []
            kf = KFold(n_splits=len(df), shuffle=False)
            for (train_index, test_index) in kf.split(df):
                features1 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[0])].values
                             for
                             mol in df.iloc[train_index]["mol"]]
                features2 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[1])].values
                             for
                             mol in df.iloc[train_index]["mol"]]
                features3 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[2])].values
                             for
                             mol in df.iloc[train_index]["mol"]]
                features = np.concatenate([features1, features2, features3], axis=1)

                Y = df.iloc[train_index]["ΔΔG.expt."].values
                if regression_type == "lassocv":
                    model = linear_model.LassoCV(fit_intercept=False, cv=5,tol=0.01).fit(features, Y)
                elif regression_type == "PLS":
                    model = PLSRegression(n_components=5).fit(features, Y)
                elif regression_type == "ridgecv":
                    model = RidgeCV(fit_intercept=False, cv=5).fit(features, Y)
                elif regression_type == "elasticnetcv":
                    model = ElasticNetCV(fit_intercept=False, cv=5).fit(features, Y)

                features1 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[0])].values
                             for
                             mol in df.iloc[test_index]["mol"]]
                features2 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[1])].values
                             for
                             mol in df.iloc[test_index]["mol"]]
                features3 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[1])].values
                             for
                             mol in df.iloc[test_index]["mol"]]

                features = np.concatenate([features1, features2, features3], axis=1)
                predict = model.predict(features)
                if regression_type == "PLS":
                    if maxmin == "True":
                        print("PLS")
                        for i in range(len(predict)):
                            if predict[i] >= df.iloc[test_index]["ΔΔmaxG.expt."].values[i]:
                                predict[i] = df.iloc[test_index]["ΔΔmaxG.expt."].values[i]
                            if predict[i] <= df.iloc[test_index]["ΔΔminG.expt."].values[i]:
                                predict[i] = df.iloc[test_index]["ΔΔminG.expt."].values[i]
                    l.extend(predict[i])

                else:
                    if maxmin == "True":
                        for i in range(len(predict)):
                            if predict >= df.iloc[test_index]["ΔΔmaxG.expt."].values:
                                predict = df.iloc[test_index]["ΔΔmaxG.expt."].values
                            if predict <= df.iloc[test_index]["ΔΔminG.expt."].values:
                                predict = df.iloc[test_index]["ΔΔminG.expt."].values
                    l.extend(predict)

        if regression_type in ["gaussian", "gaussianFP"]:
            l = []
            kf = KFold(n_splits=len(df), shuffle=False)
            for (train_index, test_index) in kf.split(df):
                features1 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[0])].values
                             for
                             mol in df.iloc[train_index]["mol"]]
                features2 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[1])].values
                             for
                             mol in df.iloc[train_index]["mol"]]
                features3 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[2])].values
                             for
                             mol in df.iloc[train_index]["mol"]]
                features = np.concatenate([features1, features2, features3], axis=1)
                penalty1 = np.concatenate([hparam1 * penalty, np.zeros(penalty.shape)], axis=1)
                penalty2 = np.concatenate([np.zeros(penalty.shape), hparam2 * penalty, np.zeros(penalty.shape)], axis=1)
                penalty3 = np.concatenate([np.zeros(penalty.shape), np.zeros(penalty.shape), hparam3 * penalty], axis=1)
                X = np.concatenate([features, penalty1, penalty2, penalty3], axis=0)
                if regression_type in ["gaussianFP"]:
                    zeroweight = np.zeros(int(penalty.shape[1])).reshape(1, penalty.shape[1])
                    for weight in df.loc[:, fplist].columns:
                        S = df.iloc[train_index][weight].values.reshape(1, -1)
                        Q = np.concatenate([S, zeroweight, zeroweight, zeroweight], axis=1)
                        X = np.concatenate([X, Q.T], axis=1)
                Y = np.concatenate([df.iloc[train_index]["ΔΔG.expt."], np.zeros(penalty.shape[0] * 3)], axis=0)
                model = linear_model.Ridge(alpha=0, fit_intercept=False).fit(X, Y)

                features1 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[0])].values for
                             mol in df.iloc[test_index]["mol"]]
                features2 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[1])].values for
                             mol in df.iloc[test_index]["mol"]]
                features3 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[2])].values for
                             mol in df.iloc[test_index]["mol"]]
                features = np.concatenate([features1, features2, features3], axis=1)
                if regression_type in ["FP", "gaussianFP"]:
                    for weight in df.loc[:, fplist].columns:
                        S = df.iloc[test_index][weight].values.reshape(-1, 1)
                        features = np.concatenate([features, S], axis=1)
                predict = model.predict(features)
                if maxmin == "True":

                    for i in range(len(predict)):
                        if predict >= df.iloc[test_index]["ΔΔmaxG.expt."].values:
                            predict = df.iloc[test_index]["ΔΔmaxG.expt."].values
                        if predict <= df.iloc[test_index]["ΔΔminG.expt."].values:
                            predict = df.iloc[test_index]["ΔΔminG.expt."].values
                l.extend(predict)
        if regression_type in "FP":
            l = []
            kf = KFold(n_splits=len(df), shuffle=False)
            for (train_index, test_index) in kf.split(df):
                features1 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[0])].values
                             for
                             mol in df.iloc[train_index]["mol"]]
                features2 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[1])].values
                             for
                             mol in df.iloc[train_index]["mol"]]
                features3 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[2])].values
                             for
                             mol in df.iloc[train_index]["mol"]]
                features = np.concatenate([features1, features2, features3], axis=1)
                X = np.concatenate([features], axis=0)

                for weight in df.loc[:, fplist].columns:
                    S = df.iloc[train_index][weight].values.reshape(-1, 1)
                    X = np.concatenate([X, S], axis=1)
                Y = df.iloc[train_index]["ΔΔG.expt."]
                # model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)
                model = PLSRegression(n_components=5).fit(X, Y)

                features1 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[0])].values
                             for
                             mol in df.iloc[test_index]["mol"]]
                features2 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[1])].values
                             for
                             mol in df.iloc[test_index]["mol"]]
                features3 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[2])].values
                             for
                             mol in df.iloc[test_index]["mol"]]
                features = np.concatenate([features1, features2, features3], axis=1)

                for weight in df.loc[:, fplist].columns:
                    S = df.iloc[test_index][weight].values.reshape(-1, 1)
                    features = np.concatenate([features, S], axis=1)
                predict = model.predict(features)
                if maxmin == "True":
                    for i in range(len(predict)):
                        if predict[i] >= df.iloc[test_index]["ΔΔmaxG.expt."].values[i]:
                            predict[i] = df.iloc[test_index]["ΔΔmaxG.expt."].values[i]
                        if predict[i] <= df.iloc[test_index]["ΔΔminG.expt."].values[i]:
                            predict[i] = df.iloc[test_index]["ΔΔminG.expt."].values[i]
                l.extend(predict[i])

    print(len(l), l)
    df["ΔΔG.loo"] = l
    r2 = r2_score(df["ΔΔG.expt."], l)
    print("r2", r2)
    df["error"] = l - df["ΔΔG.expt."]
    df["inchikey"] = df["mol"].apply(lambda mol: mol.GetProp("InchyKey"))
    os.makedirs("../errortest/", exist_ok=True)
    df.to_csv("../errortest/df.csv")
    PandasTools.AddMoleculeColumnToFrame(df, "smiles")

    # df=df.replace('0', np.nan)

    df = df.round(5)
    df = df.fillna(0)
    df.to_csv("../errortest/df3.csv")
    df = pd.read_csv("../errortest/df3.csv")
    try:
        df = df.drop(['level_0', 'Unnamed: 0', 'mol'])
    except:
        None
    # df=df[["smiles","ROMol","inchikey","er.","ΔΔG.expt.","ΔΔminG.expt.","ΔΔmaxG.expt.", 'mol','level_0']]
    PandasTools.SaveXlsxFromFrame(df, out_file_name, size=(100, 100), molCol='ROMol')

    if param["cat"] == "cbs":
        dfp = pd.read_csv("../result/cbs_gaussian/result_grid_search.csv")
    else:
        dfp = pd.read_csv("../result/dip-chloride_gaussian/result_grid_search.csv")
    print(dfp)
    min_index = dfp['RMSE'].idxmin()
    min_row = dfp.loc[min_index, :]
    p = pd.DataFrame([min_row], index=[min_index])
    print(p)
    p.to_csv(param["out_dir_name"] + "/hyperparam.csv")

    return model


def train_testfold(fold, features_dir_name, regression_features, feature_number, df, gridsearch_file_name,
                   looout_file_name, testout_file_name, param, fplist, regression_type, maxmin, dfp):
    """


    :param fold: yzでおりかえすならtrue,yのみならFalse
    :param features_dir_name: 特徴量(grid_features)のディレクトリ
    :param regression_features: 　
    :param feature_number:
    :param df: dataframe
    :param gridsearch_file_name: グリッドサーチの保存先
    :param looout_file_name:looの保存先
    :param testout_file_name: testデータの保存先
    :param param: param
    :param fplist:
    :param regression_type:
    :param maxmin:
    :param dfp:
    :return:
    """

    # df_train, df_test = train_test_split(df, train_size=0.7, random_state=1)

    if fold:
        grid_features_name = "{}/{}/feature_yz.csv"
    else:
        grid_features_name = "{}/{}/feature_y.csv"
    df_train, df_test = train_test_split(df, train_size=0.8, random_state=0)
    if param["Regression_type"] in "gaussian":
        grid_search(fold, features_dir_name, regression_features, feature_number, df_train, dfp, gridsearch_file_name,
                    fplist, regression_type, maxmin)
    p = []
    if param["cat"] == "cbs":
        p = pd.read_csv("../result/cbs_gaussian/hyperparam.csv")
    elif param["cat"] == "dip":
        p = pd.read_csv("../result/dip-chloride_gaussian/hyperparam.csv")
    elif param["cat"] == "RuSS":
        p = pd.read_csv("../result/RuSS_gaussian/hyperparam.csv")
    else:
        print("Not exist gridsearch result")

    if param["Regression_type"] in "gaussian":
        model = leave_one_out(fold, features_dir_name, regression_features, feature_number, df_train, looout_file_name,
                              param, fplist, regression_type, maxmin, p)
    else:
        model = leave_one_out(fold, features_dir_name, regression_features, feature_number, df_train, looout_file_name,
                              param,
                              fplist, regression_type, maxmin, p=None)

    features1 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                     "{}".format(regression_features.split()[0])].values
                 for
                 mol in df_test["mol"]]
    features2 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                     "{}".format(regression_features.split()[1])].values
                 for
                 mol in df_test["mol"]]
    features = np.concatenate([features1, features2], axis=1)
    testpredict = model.predict(features)
    l = []
    print(testpredict)

    if maxmin == "True":
        if regression_type == "PLS":
            for i in range(len(testpredict)):
                ans = testpredict[i][0]
                print(ans)

                if ans >= df_test["ΔΔmaxG.expt."].values[i]:
                    ans = df_test["ΔΔmaxG.expt."].values[i]
                if ans <= df_test["ΔΔminG.expt."].values[i]:
                    ans = df_test["ΔΔminG.expt."].values[i]
                l.append(ans)
        else:
            for i in range(len(testpredict)):
                ans = testpredict[i]
                print(ans)

                if ans >= df_test["ΔΔmaxG.expt."].values[i]:
                    ans = df_test["ΔΔmaxG.expt."].values[i]
                if ans <= df_test["ΔΔminG.expt."].values[i]:
                    ans = df_test["ΔΔminG.expt."].values[i]
                l.append(ans)
    df_test["ΔΔG.test"] = l
    r2 = r2_score(df_test["ΔΔG.expt."], df_test["ΔΔG.test"])
    print(r2)
    df_test["error"] = df_test["ΔΔG.test"] - df_test["ΔΔG.expt."]
    df_test["inchikey"] = df_test["mol"].apply(lambda mol: mol.GetProp("InchyKey"))
    PandasTools.AddMoleculeColumnToFrame(df_test, "smiles")
    try:
        df_test = df_test.drop(['level_0', 'Unnamed: 0'])
    except:
        None
    df_test = df_test.round(5)
    df_test = df_test.fillna(0)
    PandasTools.SaveXlsxFromFrame(df_test, testout_file_name, size=(100, 100))


def doublecrossvalidation(fold, features_dir_name, regression_features, feature_number, df, gridsearch_file_name,
                          looout_file_name, testout_file_name, param, fplist, regression_type, maxmin, dfp):
    if fold:
        grid_features_name = "{}/{}/feature_yz.csv"
    else:
        grid_features_name = "{}/{}/feature_y.csv"
    df = df.sample(frac=1, random_state=0)
    kf = KFold(n_splits=5)
    df.to_csv("../errortest/dfrandom.csv")
    testlist = []
    for (train_index, test_index) in kf.split(df):
        # df.iloc[train_index].to_csv("../errortest/dftrainrandom.csv")
        # df.iloc[test_index].to_csv("../errortest/dftestrandom.csv")
        print("test_index")
        print(test_index)
        df_train, df_test = df.iloc[train_index], df.iloc[test_index]
        p = None
        if param["Regression_type"] in "gaussian":
            grid_search(fold, features_dir_name, regression_features, feature_number, df_train, dfp,
                        gridsearch_file_name,
                        fplist, regression_type, maxmin)
            if param["cat"] == "cbs":
                p = pd.read_csv("../result/cbs_gaussian/hyperparam.csv")
            elif param["cat"] == "dip":
                p = pd.read_csv("../result/dip-chloride_gaussian/hyperparam.csv")
            elif param["cat"] == "RuSS":
                p = pd.read_csv("../result/RuSS_gaussian/hyperparam.csv")
            else:
                print("Not exist gridsearch result")


        if param["Regression_type"] in "gaussian":
            model = leave_one_out(fold, features_dir_name, regression_features, feature_number, df_train,
                                  looout_file_name, param, fplist, regression_type, maxmin, p)
        else:
            model = leave_one_out(fold, features_dir_name, regression_features, feature_number, df_train,
                                  looout_file_name, param,
                                  fplist, regression_type, maxmin, p=None)

        features1 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                         "{}".format(regression_features.split()[0])].values
                     for
                     mol in df_test["mol"]]
        features2 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                         "{}".format(regression_features.split()[1])].values
                     for
                     mol in df_test["mol"]]
        features = np.concatenate([features1, features2], axis=1)
        testpredict = model.predict(features)

        if maxmin == "True":
            if regression_type == "PLS":  # PLSだけ出力形式が少し変
                for i in range(len(testpredict)):
                    ans = testpredict[i][0]
                    print(ans)

                    if ans >= df_test["ΔΔmaxG.expt."].values[i]:
                        ans = df_test["ΔΔmaxG.expt."].values[i]
                    if ans <= df_test["ΔΔminG.expt."].values[i]:
                        ans = df_test["ΔΔminG.expt."].values[i]
                    testlist.append(ans)
            else:
                for i in range(len(testpredict)):
                    ans = testpredict[i]
                    print(ans)

                    if ans >= df_test["ΔΔmaxG.expt."].values[i]:
                        ans = df_test["ΔΔmaxG.expt."].values[i]
                    if ans <= df_test["ΔΔminG.expt."].values[i]:
                        ans = df_test["ΔΔminG.expt."].values[i]
                    testlist.append(ans)
        else:
            if regression_type == "PLS":  # PLSだけ出力形式が少し変
                for i in range(len(testpredict)):
                    ans = testpredict[i][0]
                    testlist.append(ans)
            else:
                for i in range(len(testpredict)):
                    ans = testpredict[i]
                    testlist.append(ans)




    df["ΔΔG.crosstest"] = testlist
    r2 = r2_score(df["ΔΔG.expt."], df["ΔΔG.crosstest"])
    resultfile=param["out_dir_name"]


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
    PandasTools.SaveXlsxFromFrame(df, testout_file_name, size=(100, 100))


if __name__ == '__main__':
    for param_file_name in [
        "../parameter_nomax/parameter_cbs_gaussian.txt",
        "../parameter_nomax/parameter_cbs_ridgecv.txt",
        "../parameter_nomax/parameter_cbs_PLS.txt",
        "../parameter_nomax/parameter_cbs_lassocv.txt",
        "../parameter_nomax/parameter_cbs_elasticnetcv.txt",
        "../parameter_nomax/parameter_RuSS_gaussian.txt",
        "../parameter_nomax/parameter_RuSS_lassocv.txt",
        "../parameter_nomax/parameter_RuSS_PLS.txt",
        "../parameter_nomax/parameter_RuSS_elasticnetcv.txt",
        "../parameter_nomax/parameter_RuSS_ridgecv.txt",
        "../parameter_nomax/parameter_dip-chloride_PLS.txt",
        "../parameter_nomax/parameter_dip-chloride_lassocv.txt",
        "../parameter_nomax/parameter_dip-chloride_gaussian.txt",
        "../parameter_nomax/parameter_dip-chloride_elasticnetcv.txt",
        "../parameter_nomax/parameter_dip-chloride_ridgecv.txt",

        # "../parameter/parameter_dip-chloride_gaussian.txt",
        # "../parameter/parameter_RuSS_gaussian.txt",
        # "../parameter/parameter_cbs_gaussian_practice.txt",
        # "../parameter/parameter_cbs_gaussian.txt",
        # "../parameter/parameter_cbs_ridgecv.txt",
        # "../parameter/parameter_cbs_PLS.txt",
        # "../parameter/parameter_cbs_lassocv.txt",
        # "../parameter/parameter_cbs_elasticnetcv.txt",
        # "../parameter/parameter_RuSS_gaussian.txt",
        # "../parameter/parameter_RuSS_lassocv.txt",
        # "../parameter/parameter_RuSS_PLS.txt",
        # "../parameter/parameter_RuSS_elasticnetcv.txt",
        # "../parameter/parameter_RuSS_ridgecv.txt",
        # "../parameter/parameter_dip-chloride_PLS.txt",
        # "../parameter/parameter_dip-chloride_lassocv.txt",
        # # "../parameter/parameter_dip-chloride_gaussian.txt",
        # "../parameter/parameter_dip-chloride_elasticnetcv.txt",
        # "../parameter/parameter_dip-chloride_ridgecv.txt",

        # "../parameter/parameter_cbs_gaussian_FP.txt",
        # "../parameter/parameter_dip-chloride_gaussian_FP.txt",
        # "../parameter/parameter_RuSS_gaussian_FP.txt",
    ]:

        fold = True
        traintest = False
        doublecrossvalid = True
        practice = False
        print(param_file_name)
        with open(param_file_name, "r") as f:
            param = json.loads(f.read())

        features_dir_name = param["grid_dir_name"] + "/[{}]/".format(param["grid_sizefile"])

        fparranged_dataname = param["fpdata_file_path"]
        df_fplist = pd.read_csv(fparranged_dataname).dropna(subset=['smiles']).reset_index(drop=True)
        fplist = pd.read_csv(param["fplist"]).columns.values.tolist()
        xyz_dir_name = param["cube_dir_name"]
        df = pd.read_excel(param["data_file_path"]).dropna(subset=['smiles']).reset_index(drop=True)  # [:10]
        df = pd.concat([df, df_fplist], axis=1)
        df.to_csv("../errortest/df.csv")
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.dropna(subset=['smiles']).reset_index(drop=True)
        df["mol"] = df["smiles"].apply(calculate_conformation.get_mol)
        print(len(df))
        print(df["smiles"][[os.path.isdir(features_dir_name + mol.GetProp("InchyKey")) for mol in df["mol"]]])
        df = df[[os.path.isdir(features_dir_name + mol.GetProp("InchyKey")) for mol in df["mol"]]]
        print(len(df))
        df["mol"].apply(lambda mol: calculate_conformation.read_xyz(mol, xyz_dir_name + "/" + mol.GetProp("InchyKey")))
        dfp = pd.read_csv(param["penalty_param_dir"])  # [:1]
        print(dfp)
        os.makedirs(param["out_dir_name"], exist_ok=True)
        gridsearch_file_name = param["out_dir_name"] + "/result_grid_search.csv"
        looout_file_name = param["out_dir_name"] + "/result_loo.xlsx"
        testout_file_name = param["out_dir_name"] + "/result_train_test.xlsx"
        crosstestout_file_name = param["out_dir_name"] + "/result_5crossvalid.xlsx"
        if traintest:
            train_testfold(fold, features_dir_name, param["Regression_features"], param["feature_number"], df,
                           gridsearch_file_name
                           , looout_file_name, testout_file_name, param, fplist, param["Regression_type"],
                           param["maxmin"], dfp)

        elif doublecrossvalid:
            doublecrossvalidation(fold, features_dir_name, param["Regression_features"], param["feature_number"], df,
                                  gridsearch_file_name, looout_file_name,
                                  crosstestout_file_name, param, fplist, param["Regression_type"],
                                  param["maxmin"], dfp)
        elif practice:
            if param["Regression_type"] in "gaussian":
                grid_search(fold, features_dir_name, param["Regression_features"], param["feature_number"], df, dfp,
                            gridsearch_file_name,
                            fplist, param["Regression_type"], param["maxmin"])
            p = []
            if param["cat"] == "cbs":
                p = pd.read_csv("../errortest/hyperparam.csv")

            else:
                print("Not exist gridsearch result")

            if param["Regression_type"] in "gaussian":
                model = leave_one_out(fold, features_dir_name, param["Regression_features"], param["feature_number"],
                                      df, looout_file_name,
                                      param, fplist, param["Regression_type"], param["maxmin"], p)

        else:
            if param["Regression_type"] in ["gaussian", "gaussianFP"]:
                if param["Regression_type"] in "gaussian":
                    grid_search(fold, features_dir_name, param["Regression_features"], param["feature_number"], df, dfp,
                                param["out_dir_name"] + "/result_grid_search.csv", fplist, param["Regression_type"],
                                param["maxmin"])

            if param["Regression_type"] in ["gaussian", "gaussianFP"]:

                if param["cat"] == "cbs":
                    p = pd.read_csv("../result/cbs_gaussian/hyperparam.csv")
                elif param["cat"] == "dip":
                    p = pd.read_csv("../result/dip-chloride_gaussian/hyperparam.csv")
                elif param["cat"] == "RuSS":
                    p = pd.read_csv("../result/RuSS_gaussian/hyperparam.csv")
                else:
                    raise ValueError
                leave_one_out(fold, features_dir_name, param["Regression_features"], param["feature_number"], df,
                              param["out_dir_name"] + "/result_loo.xls", param, fplist, param["Regression_type"],
                              param["maxmin"],
                              p)

            else:
                leave_one_out(fold, features_dir_name, param["Regression_features"], param["feature_number"], df,
                              param["out_dir_name"] + "/result_loo.xls", param, fplist, param["Regression_type"],
                              param["maxmin"], p=None)
