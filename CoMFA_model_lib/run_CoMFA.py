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


import time
import warnings
import json
import csv
"""
feature_num
"""
warnings.simplefilter('ignore')


def grid_search(features_dir_name,regression_features ,regression_number,df, dfp, out_file_name,fplist,reguression_type):
    xyz = pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))[
        ["x", "y", "z"]].values
    d = np.array([np.linalg.norm(xyz - _, axis=1) for _ in xyz])
    d_y = np.array([np.linalg.norm(xyz - _ * np.array([1, -1, 1]), axis=1) for _ in xyz])
    d_z = np.array([np.linalg.norm(xyz - _ * np.array([1, 1, -1]), axis=1) for _ in xyz])
    d_yz = np.array([np.linalg.norm(xyz - _ * np.array([1, -1, -1]), axis=1) for _ in xyz])

    def gauss_func(d):
        sigma = 0.5
        leng = 0.5
        ans = 1 / (2 * np.pi * np.sqrt(2 * np.pi) * sigma ** 3) * leng ** 3 \
              * np.exp(-d ** 2 / (2 * sigma ** 2))
        return ans

    penalty = np.where(d < 0.5 * 3, gauss_func(d), 0)
    penalty_y = np.where(d_y < 0.5 * 3, gauss_func(d_y), 0)
    penalty_z = np.where(d_z < 0.5 * 3, gauss_func(d_z), 0)
    penalty_yz = np.where(d_yz < 0.5 * 3, gauss_func(d_yz), 0)
    penalty = penalty + penalty_y + penalty_z + penalty_yz
    penalty = penalty / np.max(np.sum(penalty, axis=0))
    np.fill_diagonal(penalty, -1)
    os.makedirs("../penalty", exist_ok=True)
    np.save('../penalty/penalty.npy', penalty)
    r2_list = []
    RMSE_list = []

    if regression_number=="1":
        hyperparam=list(dict.fromkeys(dfp["{}param".format(regression_features)]))
        q = []
        for L1 in hyperparam:
            a = []
            a.append(L1)
            q.append(a)
            print(L1)
            penalty1 = L1 * penalty
            l=[]
            kf = KFold(n_splits=5, shuffle=False)
            for (train_index, test_index) in kf.split(df):
                feature = [
                    pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))[
                        "{}".format(regression_features)].values
                    for
                    mol in df.iloc[train_index]["mol"]]
                if reguression_type in ["gaussianFP"]:
                    X = np.concatenate([feature, penalty1], axis=0)
                    zeroweight = np.zeros(int(penalty.shape[1])).reshape(1, penalty.shape[1])
                    train_if = []
                    for weight in df.loc[:, fplist].columns:
                        if (train_tf := not df.iloc[train_index][weight].to_list()
                                                         .count(df.iloc[train_index][weight].to_list()[0]) == len(df.iloc[train_index][weight])):
                            S = df.iloc[train_index][weight].values.reshape(1, -1)
                            Q = np.concatenate([S, zeroweight], axis=1)
                            X = np.concatenate([X, Q.T], axis=1)
                        train_if.append(train_tf)
                    Y = np.concatenate([df.iloc[train_index]["ΔΔG.expt."], np.zeros(penalty.shape[0] )], axis=0)
                    model = linear_model.Ridge(alpha=0, fit_intercept=False).fit(X, Y)#ただの線形回帰
                if reguression_type in["gaussian"]:
                    X = np.concatenate([feature, penalty1], axis=0)
                    Y = np.concatenate([df.iloc[train_index]["ΔΔG.expt."], np.zeros(penalty.shape[0] )], axis=0)
                    model = linear_model.Ridge(alpha=0, fit_intercept=False).fit(X, Y)#Ridgeじゃないただの線形回帰

                feature = [
                    pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["LUMO"].values
                    for
                    mol in df.iloc[test_index]["mol"]]

                if reguression_type in["FP","gaussianFP"]:
                    for weight in df.loc[:, fplist].columns:
                        S = np.array(df.iloc[test_index][weight].values).reshape(1, np.array(df.iloc[test_index][weight].values).shape[0]).T
                        feature = np.concatenate([feature, S], axis=1)
                predict = model.predict(feature)
                l.extend(predict)
            r2 = r2_score(df["ΔΔG.expt."], l)
            RMSE = np.sqrt(mean_squared_error(df["ΔΔG.expt."], l))
            print("r2", r2)
            r2_list.append(r2)
            RMSE_list.append(RMSE)

        paramlist = pd.DataFrame(q)
        print(regression_features.split()[0])
        paramlist.rename(columns={0: "{}param".format(regression_features.split()[0])},inplace=True)


    elif regression_number == "2":
        feature1param =list(dict.fromkeys(dfp["{}param".format(regression_features.split()[0])]))
        feature2param =list(dict.fromkeys(dfp["{}param".format(regression_features.split()[1])]))
        q=[]
        for L1 in feature1param:
            for L2 in feature2param:
                a=[]
                a.append(L1)
                a.append(L2)
                q.append(a)
                penalty1 = np.concatenate([L1 * penalty, np.zeros(penalty.shape)], axis=1)
                penalty2 = np.concatenate([np.zeros(penalty.shape), L2 * penalty], axis=1)
                l = []
                kf = KFold(n_splits=5, shuffle=False)
                for (train_index, test_index) in kf.split(df):
                    feature1 = [
                        pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features.split()[0])].values
                        for
                        mol in df.iloc[train_index]["mol"]]
                    feature2 = [
                        pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features.split()[1])].values
                        for
                        mol in df.iloc[train_index]["mol"]]
                    features = np.concatenate([feature1, feature2], axis=1)
                    if reguression_type in ["gaussianFP"]:
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
                    if reguression_type in ["gaussian"]:
                        X = np.concatenate([features, penalty1, penalty2], axis=0)
                        Y = np.concatenate([df.iloc[train_index]["ΔΔG.expt."], np.zeros(penalty.shape[0] * 2)], axis=0)
                        model = linear_model.Ridge(alpha=0, fit_intercept=False).fit(X, Y)  # Ridgeじゃないただの線形回帰

                    # ここから、テストセットの特徴量計算
                    feature1 = [
                        pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features.split()[0])].values
                        for
                        mol in df.iloc[test_index]["mol"]]
                    feature2 = [
                        pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features.split()[1])].values
                        for
                        mol in df.iloc[test_index]["mol"]]
                    features = np.concatenate([feature1, feature2], axis=1)

                    if reguression_type in ["FP", "gaussianFP"]:
                        for weight in df.loc[:, fplist].columns:
                            S = np.array(df.iloc[test_index][weight].values).reshape(1, np.array(
                                df.iloc[test_index][weight].values).shape[0]).T
                            features = np.concatenate([features, S], axis=1)
                    predict = model.predict(features)
                    l.extend(predict)
                print(l)
                r2 = r2_score(df["ΔΔG.expt."], l)
                RMSE = np.sqrt(mean_squared_error(df["ΔΔG.expt."], l))
                print("r2", r2)
                r2_list.append(r2)
                RMSE_list.append(RMSE)
        print(q)
        paramlist=pd.DataFrame(q)

        paramlist.rename(columns={0: "{}param".format(regression_features.split()[0]),
                                              1: "{}param".format(regression_features.split()[1])},inplace=True)




    elif regression_number == "3":
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
                    penalty1 = np.concatenate([L1 * penalty, np.zeros(penalty.shape),np.zeros(penalty.shape)], axis=1)
                    penalty2 = np.concatenate([np.zeros(penalty.shape), L2 * penalty ,np.zeros(penalty.shape)], axis=1)
                    penalty3 = np.concatenate([np.zeros(penalty.shape),np.zeros(penalty.shape), L3 *penalty], axis=1)
                    l = []
                    kf = KFold(n_splits=5, shuffle=False)
                    for (train_index, test_index) in kf.split(df):
                        feature1 = [
                            pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))[
                                "{}".format(regression_features.split()[0])].values
                            for
                            mol in df.iloc[train_index]["mol"]]
                        feature2 = [
                            pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))[
                                "{}".format(regression_features.split()[1])].values
                            for
                            mol in df.iloc[train_index]["mol"]]
                        feature3 = [
                            pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))[
                                "{}".format(regression_features.split()[2])].values
                            for
                            mol in df.iloc[train_index]["mol"]]
                        features = np.concatenate([feature1, feature2,feature3], axis=1)
                        if reguression_type in ["gaussianFP"] or ["gaussian"]:
                            X = np.concatenate([features, penalty1, penalty2,penalty3], axis=0)
                            if reguression_type in ["gaussianFP"]:
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
                            Y = np.concatenate([df.iloc[train_index]["ΔΔG.expt."], np.zeros(penalty.shape[0] * 3)],
                                               axis=0)
                            model = linear_model.Ridge(alpha=0, fit_intercept=False).fit(X, Y)  # ただの線形回帰

                        # ここから、テストセットの特徴量計算
                        feature1 = [
                            pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))[
                                "{}".format(regression_features.split()[0])].values
                            for
                            mol in df.iloc[test_index]["mol"]]
                        feature2 = [
                            pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))[
                                "{}".format(regression_features.split()[1])].values
                            for
                            mol in df.iloc[test_index]["mol"]]
                        feature3 = [
                            pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))[
                                "{}".format(regression_features.split()[2])].values
                            for
                            mol in df.iloc[test_index]["mol"]]
                        features = np.concatenate([feature1, feature2 ,feature3], axis=1)

                        if reguression_type in ["FP", "gaussianFP"]:
                            for weight in df.loc[:, fplist].columns:
                                S = np.array(df.iloc[test_index][weight].values).reshape(1, np.array(
                                    df.iloc[test_index][weight].values).shape[0]).T

                                features = np.concatenate([features, S], axis=1)
                        predict = model.predict(features)
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
                                              2: "{}param".format(regression_features.split()[2])},inplace=True)
    else :
        raise ValueError
    paramlist["r2"] = r2_list
    paramlist["RMSE"] = RMSE_list
    print(paramlist)
    print(paramlist.columns)
    paramlist.to_csv(out_file_name)



def leave_one_out(features_dir_name, regression_features,feature_number, df, out_file_name, param, fplist, regression_type, p=None):
    # Dts = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["Dt"].values for mol
    #        in df["mol"]]
    # ESPs = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["ESP"].values for mol
    #         in df["mol"]]
    # LUMOs = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["LUMO"].values for mol
    #         in df["mol"]]
    penalty = np.load("../penalty/penalty.npy")



    # if regression_features in ["LUMO"]:
    if feature_number == "1":

        feature = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))[
                     "{}".format(regression_features)].values
                 for mol
                 in df["mol"]]
        if regression_type == "lassocv" or regression_type=="PLS" or regression_type=="ridgecv" or regression_type=="elasticnetcv":


            Y = df["ΔΔG.expt."].values
            if regression_type=="lassocv":
                model = linear_model.LassoCV(fit_intercept=False).fit(feature, Y)
            elif regression_type=="PLS" :
                model = PLSRegression(n_components=5).fit(feature, Y)
            elif regression_type=="ridgecv":
                model = RidgeCV(fit_intercept=False).fit(feature, Y)
            elif regression_type=="elasticnetcv":
                model = ElasticNetCV(fit_intercept=False).fit(feature, Y)

            df["ΔΔG.train"] = model.predict(feature)
            df_mf = pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))
            os.makedirs(param["moleculer_field_dir"], exist_ok=True)
            df_mf["MF_{}".format(regression_features)] = model.coef_[:penalty.shape[0]]
            if regression_type=="lassocv":
                df_mf.to_csv((param["moleculer_field_dir"] + "/" + "{}lassocvmoleculer_field.csv".format(regression_features)))
            elif regression_type=="PLS" :
                df_mf.to_csv((param["moleculer_field_dir"] + "/" + "{}PLSmoleculer_field.csv".format(regression_features)))
            elif regression_type=="ridgecv":
                df_mf.to_csv((param["moleculer_field_dir"] + "/" + "{}ridgecvmoleculer_field.csv".format(regression_features)))
            elif regression_type=="elasticnetcv":
                df_mf.to_csv((param["moleculer_field_dir"] + "/" + "{}elasticnetcvmoleculer_field.csv".format(regression_features)))
            else:
                print("regression type error")
                raise ValueError

        if regression_type in ["gaussian"]:
            hparam =p['{}param'.format(regression_features)].values
            p=float(hparam[0])
            penalty3 = p * penalty

            X = np.concatenate([feature, penalty3], axis=0)
            Y = np.concatenate([df["ΔΔG.expt."], np.zeros(penalty.shape[0])], axis=0)
            model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)
            df["ΔΔG.train"] = model.predict(feature)
            df_mf = pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))
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
            Y = np.concatenate([df["ΔΔG.expt."], np.zeros(penalty.shape[0] )], axis=0)
            model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)
            for weight in df.loc[:, fplist].columns:
                S = df[weight].values.reshape(-1, 1)
                features = np.concatenate([features, S], axis=1)
            df["ΔΔG.train"] = model.predict(features)
            df_mf=pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))

            print("model.coef_[penalty.shape[0]*2:]")
        # print(model.coef_)
            print(model.coef_[penalty.shape[0]:])
            df_fpv = str(model.coef_[penalty.shape[0] :])
            path_w = param["out_dir_name"] + "/" + "{}fpvalue".format(regression_features)
            with open(path_w, mode='w') as f:
                f.write(df_fpv)

        if regression_type in ["FP"]:
            X = np.concatenate([features], axis=0)
            for weight in df.loc[:, fplist].columns:
                S = df[weight].values.reshape(-1, 1)
                X = np.concatenate([X, S], axis=1)
            Y = df["ΔΔG.expt."].values
            #model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)
            model = PLSRegression(n_components=5).fit(features, Y)
            for weight in df.loc[:, fplist].columns:
                S = np.array(df[weight].values).reshape(-1, 1)
                features = np.concatenate([features, S], axis=1)
            df["ΔΔG.train"] = model.predict(features)
            df_mf=pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))
            print(model.coef_.shape)
            df_mf["{}".format(regression_features)]=model.coef_[:penalty.shape[0]]
            print("model.coef_[penalty.shape[0]*2:]")
        # print(model.coef_)
            print(model.coef_[penalty.shape[0]:])
            df_fpv=str(model.coef_[penalty.shape[0]:])
            path_w = param["out_dir_name"]+"/"+"{}fpvalue".format(regression_features)
            with open(path_w, mode='w') as f:
                f.write(df_fpv)

    elif feature_number == "2" :
        features1 =[pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features.split()[0])].values for mol
           in df["mol"]]
        features2 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))[
                         "{}".format(regression_features.split()[1])].values for mol
                     in df["mol"]]
        features = np.concatenate([features1, features2], axis=1)
        if regression_type == "lassocv" or regression_type=="PLS" or regression_type=="ridgecv" or regression_type=="elasticnetcv":
            Y = df["ΔΔG.expt."].values
            if regression_type == "lassocv":
                model = linear_model.LassoCV(fit_intercept=False).fit(features, Y)
            elif regression_type == "PLS":
                model = PLSRegression(n_components=5).fit(features, Y)
            elif regression_type == "ridgecv":
                model = RidgeCV(fit_intercept=False).fit(features, Y)
            elif regression_type == "elasticnetcv":
                model = ElasticNetCV(fit_intercept=False).fit(features, Y)
            df["ΔΔG.train"] = model.predict(features)
            df_mf = pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))

            os.makedirs(param["moleculer_field_dir"], exist_ok=True)


            print("aaa")

            if regression_type=="lassocv":
                df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[:penalty.shape[0]]
                df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[penalty.shape[0]:penalty.shape[0] * 2]
            elif regression_type=="PLS" :
                df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[0][:penalty.shape[0]]
                df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[0][penalty.shape[0]:penalty.shape[0] * 2]
            elif regression_type=="ridgecv" :
                df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[0][:penalty.shape[0]]
                df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[0][penalty.shape[0]:penalty.shape[0] * 2]
            elif regression_type=="elasticnetcv" :
                df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[0][:penalty.shape[0]]
                df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[0][penalty.shape[0]:penalty.shape[0] * 2]

            df_mf.to_csv((param["moleculer_field_dir"] + "/" + "moleculer_field.csv"))

        if regression_type in ["gaussian"]:

            hparam1 = p['{}param'.format(regression_features.split()[0])].values
            hparam2 = p['{}param'.format(regression_features.split()[1])].values
            penalty1 = np.concatenate([hparam1 * penalty, np.zeros(penalty.shape)], axis=1)
            penalty2 = np.concatenate([np.zeros(penalty.shape), hparam2 * penalty], axis=1)
            X = np.concatenate([features, penalty1, penalty2], axis=0)
            Y = np.concatenate([df["ΔΔG.expt."], np.zeros(penalty.shape[0] * 2)], axis=0)
            model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)
            df["ΔΔG.train"] = model.predict(features)
            df_mf = pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))
            print(df["mol"].iloc[0].GetProp("InchyKey"))
            os.makedirs(param["moleculer_field_dir"], exist_ok=True)

            df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[:penalty.shape[0]]
            df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[penalty.shape[0]:penalty.shape[0] * 2]
            df_mf.to_csv((param["moleculer_field_dir"] + "/" + "moleculer_field.csv"))
            df["{}_contribution".format(regression_features.split()[0])] = np.sum(features1 * df_mf["MF_{}".format(regression_features.split()[0])].values, axis=1)
            df["{}_contribution".format(regression_features.split()[1])] = np.sum(features2 * df_mf["MF_{}".format(regression_features.split()[1])].values, axis=1)


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
            df_mf=pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))

            df_mf["MF_{}".format(regression_features.split()[0])]=model.coef_[:penalty.shape[0]]
            df_mf["MF_{}".format(regression_features.split()[1])]=model.coef_[penalty.shape[0]:penalty.shape[0]*2]

            df["{}_contribution".format(regression_features.split()[0])] = np.sum(features1*df_mf["MF_{}".format(regression_features.split()[0])].values,axis=1)
            df["{}_contribution".format(regression_features.split()[1])] = np.sum(features2*df_mf["MF_{}".format(regression_features.split()[1])].values,axis=1)
            print("model.coef_[penalty.shape[0]*2:]")
        # print(model.coef_)
            print(model.coef_[penalty.shape[0]*2:])
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
            #model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)
            model = PLSRegression(n_components=5).fit(X, Y)
            for weight in df.loc[:, fplist].columns:
                S = np.array(df[weight].values).reshape(-1, 1)
                features = np.concatenate([features, S], axis=1)
            df["ΔΔG.train"] = model.predict(features)
            df_mf=pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))
            print(model.coef_.shape)
            df_mf["MF_{}".format(regression_features.split()[0])]=model.coef_[0][:penalty.shape[0]]
            df_mf["MF_{}".format(regression_features.split()[1])]=model.coef_[0][penalty.shape[0]:penalty.shape[0]*2]
            df["{}_contribution".format(regression_features.split()[0])] = np.sum(features1*df_mf["MF_{}".format(regression_features.split()[0])].values,axis=1)
            df["{}_contribution".format(regression_features.split()[1])] = np.sum(features2*df_mf["MF_{}".format(regression_features.split()[1])].values,axis=1)
            print("model.coef_[penalty.shape[0]*2:]")
        # print(model.coef_)
            print(model.coef_[0][penalty.shape[0]*2:])
            df_fpv=str(model.coef_[0][penalty.shape[0]*2:])
            path_w = param["out_dir_name"]+"/"+"fpvalue"
            with open(path_w, mode='w') as f:
                f.write(df_fpv)

    elif feature_number == "3":
        features1 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))[
                         "{}".format(regression_features.split()[0])].values for mol
                     in df["mol"]]
        features2 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))[
                         "{}".format(regression_features.split()[1])].values for mol
                     in df["mol"]]
        features3 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))[
                         "{}".format(regression_features.split()[2])].values for mol
                     in df["mol"]]

        features = np.concatenate([features1, features2,features3], axis=1)
        if regression_type == "lassocv" or regression_type=="PLS" or regression_type=="ridgecv" or regression_type=="elasticnetcv":
            Y = df["ΔΔG.expt."].values
            if regression_type == "lassocv":
                model = linear_model.LassoCV(fit_intercept=False).fit(features, Y)
            elif regression_type == "PLS":
                model = PLSRegression(n_components=5).fit(features, Y)
            elif regression_type == "ridgecv":
                model = RidgeCV(fit_intercept=False).fit(features, Y)
            elif regression_type == "elasticnetcv":
                model = ElasticNetCV(fit_intercept=False).fit(features, Y)
            df["ΔΔG.train"] = model.predict(features)
            df_mf = pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))

            os.makedirs(param["moleculer_field_dir"], exist_ok=True)

            print("aaa")

            if regression_type == "lassocv":
                df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[:penalty.shape[0]]
                df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[
                                                                        penalty.shape[0]:penalty.shape[0] * 2]
                df_mf["MF_{}".format(regression_features.split()[2])] = model.coef_[
                                                                        penalty.shape[0] *2:penalty.shape[0] * 3]
            elif regression_type == "PLS":
                df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[0][:penalty.shape[0]]
                df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[0][
                                                                        penalty.shape[0]:penalty.shape[0] * 2]
                df_mf["MF_{}".format(regression_features.split()[2])] = model.coef_[0][
                                                                        penalty.shape[0]*2:penalty.shape[0] * 3]
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
            penalty1 = np.concatenate([hparam1 * penalty, np.zeros(penalty.shape),np.zeros(penalty.shape)], axis=1)
            penalty2 = np.concatenate([np.zeros(penalty.shape), hparam2 * penalty,np.zeros(penalty.shape)], axis=1)
            penalty3 = np.concatenate([np.zeros(penalty.shape), np.zeros(penalty.shape), hparam3 * penalty], axis=1)
            X = np.concatenate([features, penalty1, penalty2,penalty3], axis=0)
            Y = np.concatenate([df["ΔΔG.expt."], np.zeros(penalty.shape[0] * 3)], axis=0)
            model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)
            df["ΔΔG.train"] = model.predict(features)
            df_mf = pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))
            print(df["mol"].iloc[0].GetProp("InchyKey"))
            os.makedirs(param["moleculer_field_dir"], exist_ok=True)

            df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[:penalty.shape[0]]
            df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[penalty.shape[0]:penalty.shape[0] * 2]
            df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[penalty.shape[0]*2:penalty.shape[0] * 3]
            df_mf.to_csv((param["moleculer_field_dir"] + "/" + "moleculer_field.csv"))
            df["{}_contribution".format(regression_features.split()[0])] = np.sum(features1 * df_mf["MF_{}".format(regression_features.split()[0])].values, axis=1)
            df["{}_contribution".format(regression_features.split()[1])] = np.sum(features2 * df_mf["MF_{}".format(regression_features.split()[1])].values, axis=1)
            df["{}_contribution".format(regression_features.split()[2])] = np.sum(features3 * df_mf["MF_{}".format(regression_features.split()[2])].values, axis=1)

        if regression_type in ["gaussianFP"]:
            penalty1 = np.concatenate([p[0] * penalty, np.zeros(penalty.shape),np.zeros(penalty.shape)], axis=1)
            penalty2 = np.concatenate([np.zeros(penalty.shape), p[1] * penalty,np.zeros(penalty.shape)], axis=1)
            penalty3 = np.concatenate([np.zeros(penalty.shape), np.zeros(penalty.shape),p[2] * penalty], axis=1)
            X = np.concatenate([features, penalty1, penalty2,penalty3], axis=0)
            zeroweight = np.zeros(int(penalty.shape[1])).reshape(1, penalty.shape[1])
            for weight in df.loc[:, fplist].columns:
                S = np.array(df[weight].values).reshape(1, -1)
                Q = np.concatenate([S, zeroweight, zeroweight,zeroweight], axis=1)
                X = np.concatenate([X, Q.T], axis=1)
            Y = np.concatenate([df["ΔΔG.expt."], np.zeros(penalty.shape[0] * 3)], axis=0)
            model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)
            for weight in df.loc[:, fplist].columns:
                S = df[weight].values.reshape(-1, 1)
                features = np.concatenate([features, S], axis=1)
            df["ΔΔG.train"] = model.predict(features)
            df_mf = pd.read_csv(
                "{}/{}/feature_yz.csv".format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))

            df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[:penalty.shape[0]]
            df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[
                                                                    penalty.shape[0]:penalty.shape[0] * 2]
            df_mf["MF_{}".format(regression_features.split()[2])] = model.coef_[
                                                                    penalty.shape[0]*2:penalty.shape[0] * 3]

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
                "{}/{}/feature_yz.csv".format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))
            print(model.coef_.shape)
            df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[0][:penalty.shape[0]]
            df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[0][
                                                                    penalty.shape[0]:penalty.shape[0] * 2]
            df_mf["MF_{}".format(regression_features.split()[2])] = model.coef_[0][
                                                                    penalty.shape[0]*2:penalty.shape[0] * 3]
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


    #ここからテストセットの実行

    if feature_number == "1":
        if regression_type == "lassocv" or regression_type=="PLS" or regression_type=="ridgecv" or regression_type=="elasticnetcv":
            l=[]
            kf = KFold(n_splits=len(df), shuffle=False)
            for (train_index, test_index) in kf.split(df):
                features = [
                    pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features)].values
                    for
                    mol in df.iloc[train_index]["mol"]]

                Y=df.iloc[train_index]["ΔΔG.expt."].values
                if regression_type == "lassocv":
                    model = linear_model.LassoCV(fit_intercept=False).fit(features, Y)
                elif regression_type == "PLS":
                    model = PLSRegression(n_components=5).fit(features, Y)
                elif regression_type == "ridgecv":
                    model = RidgeCV(fit_intercept=False).fit(features, Y)
                elif regression_type == "elasticnetcv":
                    model = ElasticNetCV(fit_intercept=False).fit(features, Y)

                features = [
                    pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features)].values
                    for
                    mol in df.iloc[test_index]["mol"]]

                predict = model.predict(features)
                l.extend(predict)
        if regression_type in ["gaussian", "gaussianFP"]:
            l = []
            kf = KFold(n_splits=len(df), shuffle=False)
            for (train_index, test_index) in kf.split(df):
                features = [
                    pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features)].values
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
                Y = np.concatenate([df.iloc[train_index]["ΔΔG.expt."], np.zeros(penalty.shape[0] )], axis=0)
                model = linear_model.Ridge(alpha=0, fit_intercept=False).fit(X, Y)

                features = [
                    pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features)].values
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
                    pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features)].values
                    for
                    mol in df.iloc[train_index]["mol"]]
                X = np.concatenate([features], axis=0)

                for weight in df.loc[:, fplist].columns:
                    S = df.iloc[train_index][weight].values.reshape(-1, 1)
                    X = np.concatenate([X, S], axis=1)
                Y = df.iloc[train_index]["ΔΔG.expt."]
                model = PLSRegression(n_components=5).fit(X, Y)
                #model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)


                features = [
                    pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["LUMO"].values
                    for
                    mol in df.iloc[test_index]["mol"]]

                for weight in df.loc[:, fplist].columns:
                    S = df.iloc[test_index][weight].values.reshape(-1, 1)
                    features = np.concatenate([features, S], axis=1)
                predict = model.predict(features)
                l.extend(predict)


    elif feature_number == "2":
        if regression_type == "lassocv" or regression_type=="PLS" or regression_type=="ridgecv" or regression_type=="elasticnetcv":
            l = []
            kf = KFold(n_splits=len(df), shuffle=False)
            for (train_index, test_index) in kf.split(df):
                features1 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features.split()[0])].values
                       for
                       mol in df.iloc[train_index]["mol"]]
                features2 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features.split()[1])].values
                        for
                        mol in df.iloc[train_index]["mol"]]
                features = np.concatenate([features1, features2], axis=1)

                Y=df.iloc[train_index]["ΔΔG.expt."].values
                if regression_type == "lassocv":
                    model = linear_model.LassoCV(fit_intercept=False).fit(features, Y)
                elif regression_type == "PLS":
                    model = PLSRegression(n_components=5).fit(features, Y)
                elif regression_type == "ridgecv":
                    model = RidgeCV(fit_intercept=False).fit(features, Y)
                elif regression_type == "elasticnetcv":
                    model = ElasticNetCV(fit_intercept=False).fit(features, Y)

                features1 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features.split()[0])].values
                       for
                       mol in df.iloc[test_index]["mol"]]
                features2 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features.split()[1])].values
                        for
                        mol in df.iloc[test_index]["mol"]]


                features = np.concatenate([features1, features2], axis=1)
                predict = model.predict(features)
                if regression_type == "lassocv":
                    l.extend(predict)
                else:
                    l.extend(predict[0])

        if regression_type in ["gaussian", "gaussianFP"]:
            l = []
            kf = KFold(n_splits=len(df), shuffle=False)
            for (train_index, test_index) in kf.split(df):
                features1 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[0])].values
                             for
                             mol in df.iloc[train_index]["mol"]]
                features2 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))[
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
                features1 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features.split()[0])].values for
                       mol in df.iloc[test_index]["mol"]]
                features2 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features.split()[1])].values for
                        mol in df.iloc[test_index]["mol"]]
                features = np.concatenate([features1, features2], axis=1)
                if regression_type in ["FP", "gaussianFP"]:
                    for weight in df.loc[:, fplist].columns:
                        S = df.iloc[test_index][weight].values.reshape(-1, 1)
                        features = np.concatenate([features, S], axis=1)
                predict = model.predict(features)
                l.extend(predict)
        if regression_type in "FP":
            l = []
            kf = KFold(n_splits=len(df), shuffle=False)
            for (train_index, test_index) in kf.split(df):
                features1 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features.split()[0])].values
                       for
                       mol in df.iloc[train_index]["mol"]]
                features2 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features.split()[1])].values
                        for
                        mol in df.iloc[train_index]["mol"]]
                features = np.concatenate([features1, features2], axis=1)
                X = np.concatenate([features], axis=0)

                for weight in df.loc[:, fplist].columns:
                    S = df.iloc[train_index][weight].values.reshape(-1, 1)
                    X = np.concatenate([X, S], axis=1)
                Y=df.iloc[train_index]["ΔΔG.expt."]
                #model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)
                model = PLSRegression(n_components=5).fit(X, Y)

                features1 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features.split()[0])].values
                      for
                      mol in df.iloc[test_index]["mol"]]
                features2 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features.split()[1])].values
                        for
                        mol in df.iloc[test_index]["mol"]]
                features = np.concatenate([features1, features2], axis=1)

                for weight in df.loc[:, fplist].columns:
                    S = df.iloc[test_index][weight].values.reshape(-1, 1)
                    features = np.concatenate([features, S], axis=1)
                predict = model.predict(features)
                l.extend(predict[0])

    elif feature_number == "3":
        if regression_type == "lassocv" or regression_type=="PLS" or regression_type=="ridgecv" or regression_type=="elasticnetcv":
            l = []
            kf = KFold(n_splits=len(df), shuffle=False)
            for (train_index, test_index) in kf.split(df):
                features1 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features.split()[0])].values
                       for
                       mol in df.iloc[train_index]["mol"]]
                features2 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features.split()[1])].values
                        for
                        mol in df.iloc[train_index]["mol"]]
                features3 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[2])].values
                             for
                             mol in df.iloc[train_index]["mol"]]
                features = np.concatenate([features1, features2 ,features3], axis=1)

                Y=df.iloc[train_index]["ΔΔG.expt."].values
                if regression_type == "lassocv":
                    model = linear_model.LassoCV(fit_intercept=False).fit(features, Y)
                elif regression_type == "PLS":
                    model = PLSRegression(n_components=5).fit(features, Y)
                elif regression_type == "ridgecv":
                    model = RidgeCV(fit_intercept=False).fit(features, Y)
                elif regression_type == "elasticnetcv":
                    model = ElasticNetCV(fit_intercept=False).fit(features, Y)

                features1 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features.split()[0])].values
                       for
                       mol in df.iloc[test_index]["mol"]]
                features2 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features.split()[1])].values
                        for
                        mol in df.iloc[test_index]["mol"]]
                features3 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[1])].values
                             for
                             mol in df.iloc[test_index]["mol"]]



                features = np.concatenate([features1, features2,features3], axis=1)
                predict = model.predict(features)
                if regression_type == "lassocv":
                    l.extend(predict)
                else:
                    l.extend(predict[0])

        if regression_type in ["gaussian", "gaussianFP"]:
            l = []
            kf = KFold(n_splits=len(df), shuffle=False)
            for (train_index, test_index) in kf.split(df):
                features1 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[0])].values
                             for
                             mol in df.iloc[train_index]["mol"]]
                features2 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[1])].values
                             for
                             mol in df.iloc[train_index]["mol"]]
                features3 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[2])].values
                             for
                             mol in df.iloc[train_index]["mol"]]
                features = np.concatenate([features1, features2,features3], axis=1)
                penalty1 = np.concatenate([hparam1 * penalty, np.zeros(penalty.shape)], axis=1)
                penalty2 = np.concatenate([np.zeros(penalty.shape), hparam2 * penalty,np.zeros(penalty.shape)], axis=1)
                penalty3 = np.concatenate([np.zeros(penalty.shape), np.zeros(penalty.shape),hparam3 * penalty], axis=1)
                X = np.concatenate([features, penalty1, penalty2,penalty3], axis=0)
                if regression_type in ["gaussianFP"]:
                    zeroweight = np.zeros(int(penalty.shape[1])).reshape(1, penalty.shape[1])
                    for weight in df.loc[:, fplist].columns:
                        S = df.iloc[train_index][weight].values.reshape(1, -1)
                        Q = np.concatenate([S, zeroweight, zeroweight,zeroweight], axis=1)
                        X = np.concatenate([X, Q.T], axis=1)
                Y = np.concatenate([df.iloc[train_index]["ΔΔG.expt."], np.zeros(penalty.shape[0] * 3)], axis=0)
                model = linear_model.Ridge(alpha=0, fit_intercept=False).fit(X, Y)


                features1 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features.split()[0])].values for
                       mol in df.iloc[test_index]["mol"]]
                features2 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features.split()[1])].values for
                        mol in df.iloc[test_index]["mol"]]
                features3 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[2])].values for
                             mol in df.iloc[test_index]["mol"]]
                features = np.concatenate([features1, features2,features3], axis=1)
                if regression_type in ["FP", "gaussianFP"]:
                    for weight in df.loc[:, fplist].columns:
                        S = df.iloc[test_index][weight].values.reshape(-1, 1)
                        features = np.concatenate([features, S], axis=1)
                predict = model.predict(features)
                l.extend(predict)
        if regression_type in "FP":
            l = []
            kf = KFold(n_splits=len(df), shuffle=False)
            for (train_index, test_index) in kf.split(df):
                features1 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features.split()[0])].values
                       for
                       mol in df.iloc[train_index]["mol"]]
                features2 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features.split()[1])].values
                        for
                        mol in df.iloc[train_index]["mol"]]
                features3 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[2])].values
                             for
                             mol in df.iloc[train_index]["mol"]]
                features = np.concatenate([features1, features2,features3], axis=1)
                X = np.concatenate([features], axis=0)

                for weight in df.loc[:, fplist].columns:
                    S = df.iloc[train_index][weight].values.reshape(-1, 1)
                    X = np.concatenate([X, S], axis=1)
                Y=df.iloc[train_index]["ΔΔG.expt."]
                #model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)
                model = PLSRegression(n_components=5).fit(X, Y)

                features1 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features.split()[0])].values
                      for
                      mol in df.iloc[test_index]["mol"]]
                features2 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["{}".format(regression_features.split()[1])].values
                        for
                        mol in df.iloc[test_index]["mol"]]
                features3 = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))[
                                 "{}".format(regression_features.split()[2])].values
                             for
                             mol in df.iloc[test_index]["mol"]]
                features = np.concatenate([features1, features2,features3], axis=1)

                for weight in df.loc[:, fplist].columns:
                    S = df.iloc[test_index][weight].values.reshape(-1, 1)
                    features = np.concatenate([features, S], axis=1)
                predict = model.predict(features)
                l.extend(predict[0])



    print(len(l), l)
    df["ΔΔG.loo"] = l
    r2 = r2_score(df["ΔΔG.expt."], l)
    print("r2", r2)
    df["error"] = l - df["ΔΔG.expt."]
    df["inchikey"]=df["mol"].apply(lambda  mol:mol.GetProp("InchyKey"))
    PandasTools.AddMoleculeColumnToFrame(df, "smiles")
    PandasTools.SaveXlsxFromFrame(df, out_file_name, size=(100, 100),)

if __name__ == '__main__':
    for param_file_name in [
        "../parameter/parameter_cbs_PLS.txt",
    "../parameter/parameter_cbs_lassocv.txt",
    "../parameter/parameter_cbs_ridgecv.txt",
    "../parameter/parameter_cbs_elasticnetcv.txt",
    "../parameter/parameter_cbs_gaussian.txt"]:
    # for param_file_name in [
    #     "../parameter/parameter_cbs_PLS.txt",
    #     "../parameter/parameter_cbs_FP.txt",
    #     "../parameter/parameter_dip-chloride_PLS.txt",
    #     "../parameter/parameter_dip-chloride_FP.txt",
    #     "../parameter/parameter_cbs_gaussian.txt",
    #     "../parameter/parameter_cbs_gaussian_FP.txt",
    #     "../parameter/parameter_dip-chloride_gaussian.txt",
    #     "../parameter/parameter_dip-chloride_gaussian_FP.txt"]:
    # for param_file_name in [
    #     "../parameter/parameter_RuSS_gaussian.txt",
    #     "../parameter/parameter_RuSS_gaussian_FP.txt",
    #     "../parameter/parameter_RuSS_PLS.txt",
    #     "../parameter/parameter_RuSS_FP.txt"
    # ]:
        print(param_file_name)
        with open(param_file_name, "r") as f:
            param = json.loads(f.read())
        features_dir_name = param["grid_dir_name"]
        fparranged_dataname=param["fpdata_file_path"]
        df_fplist =pd.read_csv(fparranged_dataname).dropna(subset=['smiles']).reset_index()
        fplist = pd.read_csv(param["fplist"]).columns.values.tolist()
        xyz_dir_name = param["cube_dir_name"]
        df = pd.read_excel(param["data_file_path"]).dropna(subset=['smiles']).reset_index()  # [:10]
        df= pd.concat([df,df_fplist],axis=1)
        df=df.loc[:, ~df.columns.duplicated()]
        df["mol"] = df["smiles"].apply(calculate_conformation.get_mol)
        print(len(df))
        df = df[[os.path.isdir(features_dir_name + "/" + mol.GetProp("InchyKey")) for mol in df["mol"]]]
        print(len(df))
        df["mol"].apply(lambda mol: calculate_conformation.read_xyz(mol, xyz_dir_name + "/" + mol.GetProp("InchyKey")))
        dfp = pd.read_csv(param["penalty_param_dir"])  # [:1]
        print(dfp)
        os.makedirs(param["out_dir_name"], exist_ok=True)

        if param["Regression_type"] in ["gaussian","gaussianFP"]:
            if param["Regression_type"] in "gaussian":
                grid_search(features_dir_name,param["Regression_features"] ,param["feature_number"],df, dfp, param["out_dir_name"] + "/result_grid_search.csv",fplist,param["Regression_type"])
            if param["cat"]=="cbs":
                dfp=pd.read_csv("../result/cbs_gaussian/result_grid_search.csv")
            else :
                dfp = pd.read_csv("../result/dip-chloride_gaussian/result_grid_search.csv")
            print(dfp)
            min_index = dfp['RMSE'].idxmin()
            min_row = dfp.loc[min_index, :]
            p = pd.DataFrame([min_row], index=[min_index])
            print(p)

            p.to_csv(param["out_dir_name"]+"/hyperparam.csv")

        # with open(param["out_dir_name"]+"/hyperparam.csv", "w") as f:
        #         writer = csv.writer(f)
        #         writer.writerow(min_df)


        if param["Regression_type"] in ["gaussian", "gaussianFP"]:

            # leave_one_out(features_dir_name, param["Regression_features"], param["feature_number"], df,
            #                    param["out_dir_name"] + "/result_loo.xls", param, fplist, param["Regression_type"],
            #                    dfp[["λ1", "λ2","λ3"]].values[dfp["RMSE"].idxmin()])
            leave_one_out(features_dir_name, param["Regression_features"], param["feature_number"], df,
                          param["out_dir_name"] + "/result_loo.xls", param, fplist, param["Regression_type"],
                          p)
            # if param["Regression_features"] in ["LUMO"]:
            #     leave_one_out(features_dir_name, param["Regression_features"],param["feature_number"], df,
            #                   param["out_dir_name"] + "/result_loo.xls", param, fplist, param["Regression_type"],
            #                   dfp["λ3"].values[dfp["RMSE"].idxmin()])
        #         # else:
        #         #     leave_one_out(features_dir_name,param["Regression_features"],param["feature_number"],df,
        #         #       param["out_dir_name"] + "/result_loo.xls",param,fplist,param["Regression_type"],dfp[["λ1", "λ2"]].values[dfp["RMSE"].idxmin()])
        else:
            leave_one_out(features_dir_name, param["Regression_features"],param["feature_number"],df,
                      param["out_dir_name"] + "/result_loo.xls", param, fplist, param["Regression_type"],p=None)
        # raise ValueError