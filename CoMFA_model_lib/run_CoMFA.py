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
    penalty = penalty - np.identity(penalty.shape[0])/np.sum(penalty,axis=0)
    os.makedirs("../penalty", exist_ok=True)
    np.save('../penalty/penalty.npy', penalty)
    r2_list = []
    RMSE_list = []




    if feature_number == "2":
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
                features_train = np.concatenate([feature1, feature2], axis=1)
                features = features_train / np.std(features_train, axis=0)

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

                features_test = np.concatenate([feature1, feature2], axis=1)
                features = features_test / np.std(features_train, axis=0)


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





    paramlist["r2"] = r2_list
    paramlist["RMSE"] = RMSE_list

    paramlist.to_csv(out_file_name)

    min_index = paramlist['RMSE'].idxmin()
    min_row = paramlist.loc[min_index, :]
    p = pd.DataFrame([min_row], index=[min_index])
    print(p)
    p.to_csv(param["out_dir_name"] + "/hyperparam.csv")


def leave_one_out(fold, features_dir_name, regression_features,  df, out_file_name, param,
                  regression_type,  p=None):
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

    features_train = np.concatenate([features1, features2], axis=1)
    #np.save(param["moleculer_field_dir"]+"/trainstd", np.std(features1, axis=0))
    features=features_train/np.std(features_train,axis=0)
    print("features.shape")
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
        if regression_type == "PLS":
            df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[0][
                                                                    :int(model.coef_[0].shape[0] / 2)]
            df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[0][
                                                                    int(model.coef_[0].shape[0] / 2):int(
                                                                        model.coef_[0].shape[0])]
        else:# regression_type == "lassocv":
            df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[:int(model.coef_.shape[0] / 2)]
            df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[int(model.coef_.shape[0] / 2):int(
                model.coef_.shape[0])]

        # elif regression_type == "ridgecv":
        #     df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[:int(model.coef_.shape[0] / 2)]
        #     df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[int(model.coef_.shape[0] / 2):int(
        #         model.coef_.shape[0])]
        # elif regression_type == "elasticnetcv":
        #     df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[:int(model.coef_.shape[0] / 2)]
        #     df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[int(model.coef_.shape[0] / 2):int(
        #         model.coef_.shape[0])]
        print("writemoleculerfield")
        df_mf.to_csv((param["moleculer_field_dir"] + "/" + "moleculer_field.csv"))

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
        df_mf = pd.read_csv(grid_features_name.format(features_dir_name, "KWOLFJPFCHCOCG-UHFFFAOYSA-N"))
        print(df["mol"].iloc[0].GetProp("InchyKey"))
        os.makedirs(param["moleculer_field_dir"], exist_ok=True)

        df_mf["MF_{}".format(regression_features.split()[0])] = model.coef_[:penalty.shape[0]]
        df_mf["MF_{}".format(regression_features.split()[1])] = model.coef_[penalty.shape[0]:penalty.shape[0] * 2]
        print("writemoleculerfield")

    df["{}_contribution".format(regression_features.split()[0])] = np.sum(
        features1/ np.std(features1,axis=0)* df_mf["MF_{}".format(regression_features.split()[0])].values, axis=1)
    df["{}_contribution".format(regression_features.split()[1])] = np.sum(
        features2 /np.std(features2,axis=0)* df_mf["MF_{}".format(regression_features.split()[1])].values, axis=1)
    df_mf["Dt_std"]=  np.std(features1, axis=0)
    df_mf["ESP_std"] = np.std(features2, axis=0)
    df_mf.to_csv((param["moleculer_field_dir"] + "/" + "moleculer_field.csv"))

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
            features_train = np.concatenate([features1, features2], axis=1)
            features = features_train / np.std(features_train, axis=0)

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

            features_test = np.concatenate([features1, features2], axis=1)
            features= features_test / np.std(features_train, axis=0)
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
            hparam1 = p['{}param'.format(regression_features.split()[0])].values
            hparam2 = p['{}param'.format(regression_features.split()[1])].values
            features_train = np.concatenate([features1, features2], axis=1)
            features = features_train / np.std(features_train, axis=0)
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
            features_test = np.concatenate([features1, features2], axis=1)
            features =features_test / np.std(features_train, axis=0)

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
    df.to_csv("../errortest/df.csv")
    PandasTools.AddMoleculeColumnToFrame(df, "smiles")

    # df=df.replace('0', np.nan)


    try:
        df = df.drop(['level_0', 'Unnamed: 0', 'mol'])
    except:
        None
    # df=df[["smiles","ROMol","inchikey","er.","ΔΔG.expt.","ΔΔminG.expt.","ΔΔmaxG.expt.", 'mol','level_0']]
    PandasTools.SaveXlsxFromFrame(df, out_file_name, size=(100, 100), molCol='ROMol')

    if param["cat"] == "cbs":
        dfp = pd.read_csv("../result/cbs_gaussian_nomax/result_grid_search.csv")
    elif param["cat"] =="dip":
        dfp = pd.read_csv("../result/dip-chloride_gaussian_nomax/result_grid_search.csv")
    elif param["cat"] == "RuSS":
        dfp = pd.read_csv("../result/RuSS_gaussian_nomax/result_grid_search.csv")
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
    penalty = np.load("../penalty/penalty.npy")
    df = df.sample(frac=1, random_state=0)

    df.to_csv("../errortest/dfrandom.csv")

    df_train =df
    p = None
    if param["Regression_type"] in "gaussian":
        grid_search(fold, features_dir_name, regression_features, feature_number, df_train, dfp,
                    gridsearch_file_name,
                     regression_type, maxmin)
        if param["cat"] == "cbs":
            p = pd.read_csv("../result/cbs_gaussian_nomax/hyperparam.csv")
        elif param["cat"] == "dip":
            p = pd.read_csv("../result/dip-chloride_gaussian_nomax/hyperparam.csv")
        elif param["cat"] == "RuSS":
            p = pd.read_csv("../result/RuSS_gaussian_nomax/hyperparam.csv")
        else:
            print("Not exist gridsearch result")
        leave_one_out(fold, features_dir_name, regression_features,  df_train,
                              looout_file_name, param,  regression_type, p)#分子場の出力　looの出力
    else:
        leave_one_out(fold, features_dir_name, regression_features,  df_train,#分子場の出力　looの出力
                              looout_file_name, param,
                               regression_type,  p=None)
    df = df.sample(frac=1, random_state=0)
    kf = KFold(n_splits=5)

    testlist = []
    for (train_index, test_index) in kf.split(df):
        df_train=df.iloc[train_index]
        df_test=df.iloc[test_index]
        features1 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                         "{}".format(regression_features.split()[0])].values for mol
                     in df_train["mol"]]
        features2 = [pd.read_csv(grid_features_name.format(features_dir_name, mol.GetProp("InchyKey")))[
                         "{}".format(regression_features.split()[1])].values for mol
                     in df_train["mol"]]
        features_train = np.concatenate([features1, features2], axis=1)
        features = features_train / np.std(features_train, axis=0)
        print(features.shape)
        #if regression_type == "lassocv" or regression_type == "PLS" or regression_type == "ridgecv" or regression_type == "elasticnetcv":
        Y = df_train["ΔΔG.expt."].values
        print(Y.shape)
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
            print(penalty1.shape)
            print(penalty2.shape)
            print(features.shape)
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

        features_test = np.concatenate([features1, features2], axis=1)
        features = features_test / np.std(features_train, axis=0)
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
    #resultfile=param["out_dir_name"]


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
        # "../parameter_nomax/parameter_RuSS_lassocv.txt",
        # "../parameter_nomax/parameter_RuSS_PLS.txt",
        # "../parameter_nomax/parameter_RuSS_elasticnetcv.txt",
        # "../parameter_nomax/parameter_RuSS_ridgecv.txt",


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
        # df_fplist = pd.read_csv(fparranged_dataname).dropna(subset=['smiles']).reset_index(drop=True)
        # fplist = pd.read_csv(param["fplist"]).columns.values.tolist()
        xyz_dir_name = param["cube_dir_name"]
        df = pd.read_excel(param["data_file_path"]).dropna(subset=['smiles']).reset_index(drop=True)  # [:10]
        # df = pd.concat([df, df_fplist], axis=1)
        df = df.dropna(subset=['smiles']).reset_index(drop=True)
        df["mol"] = df["smiles"].apply(calculate_conformation.get_mol)
        df = df[[os.path.isdir(features_dir_name + mol.GetProp("InchyKey")) for mol in df["mol"]]]
        df["mol"].apply(lambda mol: calculate_conformation.read_xyz(mol, xyz_dir_name + "/" + mol.GetProp("InchyKey")))
        dfp = pd.read_csv(param["penalty_param_dir"])  # [:1]
        os.makedirs(param["out_dir_name"], exist_ok=True)
        gridsearch_file_name = param["out_dir_name"] + "/result_grid_search.csv"
        looout_file_name = param["out_dir_name"] + "/result_loo.xlsx"
        testout_file_name = param["out_dir_name"] + "/result_train_test.xlsx"
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
