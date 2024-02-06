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


def grid_search(fold, features_dir_name, regression_features, feature_number, df, dfp, out_file_name,
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
    penalty = penalty - np.identity(penalty.shape[0]) / np.sum(penalty, axis=0)
    os.makedirs("../penalty", exist_ok=True)
    np.save('../penalty/penalty.npy', penalty)
    r2_list = []
    RMSE_list = []

    if feature_number == "2":
        feature1param = list(dict.fromkeys(dfp["{}param".format(regression_features.split()[0])]))
        feature2param = list(dict.fromkeys(dfp["{}param".format(regression_features.split()[1])]))
        # q = []
        # for L1 in feature1param:
        #     for L2 in feature2param:
        for L1, L2 in zip(dfp["Dtparam"], dfp["ESP_cutoffparam"]):
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

                features1 = [
                    pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                        "{}".format(regression_features.split()[0])].values
                    for
                    mol in df.iloc[train_index]["mol"]]
                features2 = [
                    pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                        "{}".format(regression_features.split()[1])].values
                    for
                    mol in df.iloc[train_index]["mol"]]
                features1_ = np.array(features1)
                features2_ = np.array(features2)
                features1 = features1 / np.std(features1_)
                features2 = features2 / np.std(features2_)
                features_train = np.concatenate([features1, features2], axis=1)
                features = features_train #/ np.std(features_train, axis=0)

                if regression_type in ["gaussian"]:
                    X = np.concatenate([features, penalty1, penalty2], axis=0)
                    Y = np.concatenate([df.iloc[train_index]["ΔΔG.expt."], np.zeros(penalty.shape[0] * 2)], axis=0)
                    model = linear_model.Ridge(alpha=0, fit_intercept=False).fit(X, Y)  # Ridgeじゃないただの線形回帰

                # ここから、テストセットの特徴量計算
                features1 = [
                    pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                        "{}".format(regression_features.split()[0])].values
                    for
                    mol in df.iloc[test_index]["mol"]]
                features2 = [
                    pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                        "{}".format(regression_features.split()[1])].values
                    for
                    mol in df.iloc[test_index]["mol"]]
                features1 = np.array(features1)
                features2 = np.array(features2)
                features1 = features1 / np.std(features1_)
                features2 = features2 / np.std(features2_)
                features_test = np.concatenate([features1, features2], axis=1)
                features = features_test #/ np.std(features_train, axis=0)

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


import copy

def df1unfolding(df):


    df_y = copy.deepcopy(df)
    df_y["y"]=-df_y["y"]

    df_z = copy.deepcopy(df)
    df_z["z"]=-df_z["z"]
    df_z["Dt"]=-df_z["Dt"]
    df_z["ESP_cutoff"] = -df_z["ESP_cutoff"]
    df_yz= copy.deepcopy(df)
    df_yz["y"]=-df_yz["y"]
    df_yz["z"]=-df_yz["z"]
    df_yz["Dt"]=-df_yz["Dt"]
    df_yz["ESP_cutoff"] = -df_yz["ESP_cutoff"]

    df = pd.concat([df, df_y, df_z, df_yz]).sort_values(by=["x", "y", "z"])

    return df
def zunfolding(df,regression_features):

    df.to_csv("../errortest/dfbefore.csv")
    df_z = copy.deepcopy(df)
    df_z["z"]=-df_z["z"]
    df_z["MF_Dt"]=-df_z["MF_Dt"]

    df_z["MF_{}".format(regression_features.split()[1])]=-df_z["MF_{}".format(regression_features.split()[1])]


    df = pd.concat([df ,df_z ]).sort_values(['x', 'y',"z"], ascending=[True, True,True])#.sort_values(by=["x", "y", "z"])
    df.to_csv("../errortest/dfafter.csv")
    return df
# def unfolding(df):
#
#     df.to_csv("../errortest/dfbefore.csv")
#     df_y = copy.deepcopy(df)
#     df_y["y"]=-df_y["y"]
#     df_y.to_csv("../errortest/df_y.csv")
#     df_z = copy.deepcopy(df)
#     df_z["z"]=-df_z["z"]
#     df_z["MF_Dt"]=-df_z["MF_Dt"]
#     df_z["MF_ESP_cutoff"] = -df_z["MF_ESP_cutoff"]
#     df_z.to_csv("../errortest/df_z.csv")
#     df_yz= copy.deepcopy(df)
#     df_yz["y"]=-df_yz["y"]
#     df_yz["z"]=-df_yz["z"]
#     df_yz["MF_Dt"]=-df_yz["MF_Dt"]
#     df_yz["MF_ESP_cutoff"] = -df_yz["MF_ESP_cutoff"]
#     df_yz.to_csv("../errortest/df_yz.csv")

    # df_y["y"] = -df_y["y"]
    # df_y = df_y[(df_y["z"] > 0) & (df_y["y"] < 0)]
    # df_z = df_z[(df_z["y"] != 0) & (df_z["z"] > 0)]
    # df_z["z"] = -df_z["z"]
    # df_z["MF_Dt"] = -df_z["MF_Dt"]
    # df_z["MF_ESP_cutoff"] = -df_z["MF_ESP_cutoff"]
    # df_yz = copy.deepcopy(df)
    # df_yz["y"] = -df_yz["y"]
    # df_yz["z"] = -df_yz["z"]
    # df_z0 = copy.deepcopy(df[df["z"] == 1])
    # df_z0["MF_Dt"] = 0
    # df_z0["MF_ESP_cutoff"] = 0
    # df_z01 = copy.deepcopy(df_z0)
    # df_z01["y"] = -df_z01["y"]
    # df_z01 = df_z01[df_z01["y"] != 0]
    # df_yz1 = copy.deepcopy(df_yz)
    # df_yz["y"] = df_yz1[df_yz1["z"] <= 0]["y"]
    # df_yz["MF_Dt"] = -df_yz["MF_Dt"]
    # df_yz["MF_ESP_cutoff"] = -df_yz["MF_ESP_cutoff"]
    # df = pd.concat([df, df_y, df_z, df_yz]).sort_values(by=["x", "y", "z"])
    # df.to_csv("../errortest/dfafter.csv")
    #
    # return df

def leave_one_out(fold, features_dir_name, regression_features, df, out_file_name, param,
                  regression_type, p=None):
    penalty = np.load("../penalty/penalty.npy")
    if fold:
        grid_features_name = "{}/{}/feature_yz.csv"
    else:
        grid_features_name = "{}/{}/feature_y.csv"

    # if regression_features in ["LUMO"]:

    features1 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                     "{}".format(regression_features.split()[0])].values for mol
                 in df["mol"]]
    features2 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                     "{}".format(regression_features.split()[1])].values for mol
                 in df["mol"]]
    features1unfold = [pd.read_csv("{}/{}/feature_y.csv".format(features_dir_name, mol.GetProp("InchyKey")))[
                     "{}".format(regression_features.split()[0])].values for mol
                 in df["mol"]]
    features2unfold = [pd.read_csv("{}/{}/feature_y.csv".format(features_dir_name, mol.GetProp("InchyKey")))[
                     "{}".format(regression_features.split()[1])].values for mol
                 in df["mol"]]

    features1_=np.array(features1)


    features1=features1/np.std(features1_)
    features2_=np.array(features2)
    features1_unfold = np.array(features1unfold)
    features2_unfold = np.array(features2unfold)
    features2=features2/np.std(features2_)
    features_train = np.concatenate([features1, features2], axis=1)
    # np.save(param["moleculer_field_dir"]+"/trainstd", np.std(features1, axis=0))
    features = features_train #/ np.std(features_train, axis=0)
    features=np.nan_to_num(features)

    df_mf = pd.read_csv(grid_features_name.format(features_dir_name, "KWOLFJPFCHCOCG-UHFFFAOYSA-N"))
    df_mf.to_csv("../errortest/dfmf.csv")
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

        os.makedirs(param["moleculer_field_dir"], exist_ok=True)
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
        # df_mf.to_csv((param["moleculer_field_dir"] + "/" + "moleculer_field.csv"))

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


        os.makedirs(param["moleculer_field_dir"], exist_ok=True)

        df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[:penalty.shape[0]]
        df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[penalty.shape[0]:penalty.shape[0] * 2]
        print("writemoleculerfield")


    df["{}_contribution".format(regression_features.split()[0])] = np.sum(
        features1 * df_mf["MF_{}".format(regression_features.split()[0])].values, axis=1)#/ np.std(features1, axis=0)

    df["{}_contribution".format(regression_features.split()[1])] = np.sum(
        features2 * df_mf["MF_{}".format(regression_features.split()[1])].values, axis=1)#/ (np.std(features2, axis=0)) )

    df_mf["Dt_std"] = np.std(features1_)
    df_mf["ESP_std"] = np.std(features2_)
    df_unfold = df_mf
    if fold :
        df_unfold = zunfolding(df_mf,regression_features)

    # df_unfold["Dt_std"] =np.std(features1_unfold)
    # df_unfold["ESP_std"] = np.std(features2_unfold)
    DtR1s = []
    DtR2s = []

    ESPR1s = []
    ESPR2s = []

    for mol in df["mol"]:
        # df1 = pd.read_csv("{}/{}/feature.csv".format(features_dir_name, mol.GetProp("InchyKey"))).sort_values(by=["x", "y", "z"])
        #df1 = pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey"))).sort_values(by=["x", "y", "z"])
        df1 = pd.read_csv("{}/{}/feature_y.csv".format(features_dir_name, mol.GetProp("InchyKey"))).sort_values(['x', 'y',"z"], ascending=[True, True,True])#.sort_values(by=["x", "y", "z"])
        # df1 = df1unfolding(df1)
            #["{}".format(regression_features.split()[0])].values
        #df1=df1[df1["z"]!=0]
        #df_unfold=df_unfold.drop_duplicates(subset=['x', 'y',"z"])
        #df1[df1["y"]==0]["Dt"]=df1[df1["y"]==0]["Dt"]**2
        #df1[df1["y"] == 0]["ESP_cutoff"] = df1[df1["y"] == 0]["ESP_cutoff"] ** 2
        df1["DtR1"] = df1["Dt"].values / df_unfold["Dt_std"].values * df_unfold["MF_Dt"].values  # dfdtnumpy

        df1["ESPR1"] = df1["{}".format(regression_features.split()[1])].values / df_unfold["ESP_std"].values * df_unfold["MF_{}".format(regression_features.split()[1])].values  # dfespnumpy

        #ここまではあっている
        df1.to_csv("../errortest/df1R1R2.csv")


        DtR1s.append(df1[(df1["z"] >0)  ]["DtR1"].sum())
        DtR2s.append(df1[(df1["z"] < 0) ]["DtR1"].sum())

        ESPR1s.append(df1[(df1["z"] > 0) ]["ESPR1"].sum())
        ESPR2s.append(df1[(df1["z"] < 0) ]["ESPR1"].sum())
    df["DtR1"] = DtR1s
    df["DtR2"] = DtR2s

    df["DtR1R2"]=df["DtR1"]+df["DtR2"]
    df["ESPR1"] = ESPR1s
    df["ESPR2"] = ESPR2s

    df["ESPR1R2"] = df["ESPR1"] + df["ESPR2"]
    df_mf.to_csv((param["moleculer_field_dir"] + "/" + "moleculer_field.csv"))
    param["grid_dir_name"] + "/[{}]/".format(param["grid_sizefile"])


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
            features1_=np.array(features1)
            features2_=np.array(features2)
            features1=features1/np.std(features1_)
            features2=features2/np.std(features2_)
            features_train = np.concatenate([features1, features2], axis=1)
            features = features_train #/ np.std(features_train, axis=0)
            features=np.nan_to_num(features)
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
            features1=np.array(features1)
            features2=np.array(features2)
            features1=features1/np.std(features1_)
            features2=features2/np.std(features2_)
            features_test = np.concatenate([features1, features2], axis=1)
            features = features_test# / np.std(features_train, axis=0)
            features=np.nan_to_num(features)
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
            features = features_train #/ np.std(features_train, axis=0)
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
            features = features_test #/ np.std(features_train, axis=0)

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

    if param["cat"] == "cbs":
        dfp = pd.read_csv("../result/old /cbs_gaussian_nomax/result_grid_search.csv")
    elif param["cat"] == "dip":
        dfp = pd.read_csv("../result/old /dip-chloride_gaussian_nomax/result_grid_search.csv")
    elif param["cat"] == "RuSS":
        dfp = pd.read_csv("../result/old /RuSS_gaussian_nomax/result_grid_search.csv")
    print(dfp)
    min_index = dfp['RMSE'].idxmin()
    min_row = dfp.loc[min_index, :]
    p = pd.DataFrame([min_row], index=[min_index])
    print(p)
    p.to_csv(param["out_dir_name"] + "/hyperparam.csv")

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
                          looout_file_name, testout_file_name, param, regression_type, maxmin, dfp):
    if fold:
        grid_features_name = "{}/{}/feature_yz.csv"
    else:
        grid_features_name = "{}/{}/feature_y.csv"
    df = df.sample(frac=1, random_state=0)

    df.to_csv("../errortest/dfrandom.csv")

    df_train = df


    p = None
    if param["Regression_type"] in "gaussian":
        grid_search(fold, features_dir_name, regression_features, feature_number, df_train, dfp,
                    gridsearch_file_name,
                    regression_type, maxmin)
        if param["cat"] == "cbs":
            p = pd.read_csv("../result/old /cbs_gaussian_nomax/hyperparam.csv")
            p=pd.read_csv(param["out_dir_name"] + "/hyperparam.csv")
        elif param["cat"] == "dip":
            p = pd.read_csv("../result/old /dip-chloride_gaussian_nomax/hyperparam.csv")
            p = pd.read_csv(param["out_dir_name"] + "/hyperparam.csv")
        elif param["cat"] == "RuSS":
            p = pd.read_csv("../result/old /RuSS_gaussian_nomax/hyperparam.csv")
            p = pd.read_csv(param["out_dir_name"] + "/hyperparam.csv")
        else:
            print("Not exist gridsearch result")
        leave_one_out(fold, features_dir_name, regression_features, df_train,
                      looout_file_name, param, regression_type, p)  # 分子場の出力　looの出力
    else:
        leave_one_out(fold, features_dir_name, regression_features, df_train,  # 分子場の出力　looの出力
                      looout_file_name, param,
                      regression_type, p=None)
    df = df.sample(frac=1, random_state=0)

    kf = KFold(n_splits=5)
    penalty = np.load("../penalty/penalty.npy")

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
        features1_=np.array(features1)
        features1=features1/np.std(features1)
        features2_=np.array(features2)
        features2=features2/np.std(features2)
        features_train = np.concatenate([features1, features2], axis=1)
        features = features_train# / np.std(features_train, axis=0)
        print(features.shape)
        # if regression_type == "lassocv" or regression_type == "PLS" or regression_type == "ridgecv" or regression_type == "elasticnetcv":
        Y = df_train["ΔΔG.expt."].values

        if regression_type == "lassocv":
            model = linear_model.LassoCV(fit_intercept=False, cv=5).fit(features, Y)
        elif regression_type == "PLS":
            model = PLSRegression(n_components=5).fit(features, Y)
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
        features1=np.array(features1)
        features2=np.array(features2)
        features1=features1/np.std(features1_)
        features2=features2/np.std(features2_)
        features_test = np.concatenate([features1, features2], axis=1)
        features = features_test# / np.std(features_train, axis=0)
        testpredict = model.predict(features)

        # if maxmin == "True":
        #     if regression_type == "PLS":  # PLSだけ出力形式が少し変
        #         for i in range(len(testpredict)):
        #             ans = testpredict[i][0]
        #             print(ans)
        #
        #             if ans >= df_test["ΔΔmaxG.expt."].values[i]:
        #                 ans = df_test["ΔΔmaxG.expt."].values[i]
        #             if ans <= df_test["ΔΔminG.expt."].values[i]:
        #                 ans = df_test["ΔΔminG.expt."].values[i]
        #             testlist.append(ans)
        #     else:
        #         for i in range(len(testpredict)):
        #             ans = testpredict[i]
        #             print(ans)
        #
        #             if ans >= df_test["ΔΔmaxG.expt."].values[i]:
        #                 ans = df_test["ΔΔmaxG.expt."].values[i]
        #             if ans <= df_test["ΔΔminG.expt."].values[i]:
        #                 ans = df_test["ΔΔminG.expt."].values[i]
        #             testlist.append(ans)
        if regression_type == "PLS":
            testlist.extend([_[0] for _ in testpredict])
        else:
            testlist.extend(testpredict)
        # for i in range(len(testpredict)):
        #     if regression_type == "PLS":  # PLSだけ出力形式が少し変
        #         ans = testpredict[i][0]
        #         testlist.append(ans)
        #     else:
        #         ans = testpredict[i]
        #         testlist.append(ans)

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


if __name__ == '__main__':
    for param_file_name in [

        # "../parameter_nomax/parameter_cbs_gaussian.txt",
        # "../parameter_nomax/parameter_cbs_ridgecv.txt",
        # "../parameter_nomax/parameter_cbs_PLS.txt",
        # "../parameter_nomax/parameter_cbs_lassocv.txt",
        # "../parameter_nomax/parameter_cbs_elasticnetcv.txt",
        # "../parameter_nomax/parameter_dip-chloride_PLS.txt",
        # "../parameter_nomax/parameter_dip-chloride_lassocv.txt",
        # "../parameter_nomax/parameter_dip-chloride_gaussian.txt",
        # "../parameter_nomax/parameter_dip-chloride_elasticnetcv.txt",
        # "../parameter_nomax/parameter_dip-chloride_ridgecv.txt",
        # "../parameter_nomax/parameter_RuSS_gaussian.txt",
        # "../parameter_nomax/parameter_RuSS_PLS.txt",
        # "../parameter_nomax/parameter_RuSS_lassocv.txt",
        # "../parameter_nomax/parameter_RuSS_elasticnetcv.txt",
        # "../parameter_nomax/parameter_RuSS_ridgecv.txt",

        "../parameter_0206/parameter_cbs_gaussian.txt",
        "../parameter_0206/parameter_cbs_ridgecv.txt",
        "../parameter_0206/parameter_cbs_PLS.txt",
        "../parameter_0206/parameter_cbs_lassocv.txt",
        "../parameter_0206/parameter_cbs_elasticnetcv.txt",
        "../parameter_0206/parameter_dip-chloride_PLS.txt",
        "../parameter_0206/parameter_dip-chloride_lassocv.txt",
        "../parameter_0206/parameter_dip-chloride_gaussian.txt",
        "../parameter_0206/parameter_dip-chloride_elasticnetcv.txt",
        "../parameter_0206/parameter_dip-chloride_ridgecv.txt",
        "../parameter_0206/parameter_RuSS_gaussian.txt",
        "../parameter_0206/parameter_RuSS_PLS.txt",
        "../parameter_0206/parameter_RuSS_lassocv.txt",
        "../parameter_0206/parameter_RuSS_elasticnetcv.txt",
        "../parameter_0206/parameter_RuSS_ridgecv.txt",


    ]:

        fold = True
        traintest = False
        doublecrossvalid = True
        practice = False
        print(param_file_name)
        with open(param_file_name, "r") as f:
            param = json.loads(f.read())

        features_dir_name = param["grid_dir_name"] + "/[{}]/".format(param["grid_sizefile"])
        print(features_dir_name)
        fparranged_dataname = param["fpdata_file_path"]
        # df_fplist = pd.read_csv(fparranged_dataname).dropna(subset=['smiles']).reset_index(drop=True)
        # fplist = pd.read_csv(param["fplist"]).columns.values.tolist()
        xyz_dir_name = param["cube_dir_name"]
        df = pd.read_excel(param["data_file_path"]).dropna(subset=['smiles']).reset_index(drop=True)  # [:10]
        print("dflen")
        print(len(df))
        # df = pd.concat([df, df_fplist], axis=1)
        df = df.dropna(subset=['smiles']).reset_index(drop=True)
        df["mol"] = df["smiles"].apply(calculate_conformation.get_mol)
        print("dfbefore")
        print(len(df))
        df = df[[os.path.isdir(features_dir_name + mol.GetProp("InchyKey")) for mol in df["mol"]]]
        print("dfafter")
        print(len(df))
        df["mol"].apply(lambda mol: calculate_conformation.read_xyz(mol, xyz_dir_name + "/" + mol.GetProp("InchyKey")))

        dfp = pd.read_csv(param["penalty_param_dir"])  # [:1]
        os.makedirs(param["out_dir_name"], exist_ok=True)
        gridsearch_file_name = param["out_dir_name"] + "/result_grid_search.csv"
        looout_file_name = param["out_dir_name"] + "/result_loonondfold.xlsx"
        testout_file_name = param["out_dir_name"] + "/result_train_test.xlsx"
        crosstestout_file_name = param["out_dir_name"] + "/result_5crossvalidnonfold.xlsx"
        if  fold:

            looout_file_name = param["out_dir_name"] + "/result_loo.xlsx"
            crosstestout_file_name = param["out_dir_name"] + "/result_5crossvalid.xlsx"
        # if traintest:
        #     train_testfold(fold, features_dir_name, param["Regression_features"], param["feature_number"], df,
        #                    gridsearch_file_name
        #                    , looout_file_name, testout_file_name, param, fplist, param["Regression_type"],
        #                    param["maxmin"], dfp)

        if doublecrossvalid:
            doublecrossvalidation(fold, features_dir_name, param["Regression_features"], param["feature_number"], df,
                                  gridsearch_file_name, looout_file_name,
                                  crosstestout_file_name, param, param["Regression_type"],
                                  param["maxmin"], dfp)

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
