import pandas as pd
import calculate_conformation
import os
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from rdkit.Chem import PandasTools
from sklearn.model_selection import LeaveOneOut, KFold
import time
import warnings
import json
import csv
"""
main1109
"""

warnings.simplefilter('ignore')


def grid_search(features_dir_name,regression_features ,df, dfp, out_file_name,fplist,reguression_type):
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
    for L1, L2 in zip(dfp["λ1"], dfp["λ2"]):
        penalty1 = np.concatenate([L1 * penalty, np.zeros(penalty.shape)], axis=1)
        penalty2 = np.concatenate([np.zeros(penalty.shape), L2 * penalty], axis=1)
        penalty3 = L1 * penalty
        l = []
        kf = KFold(n_splits=5, shuffle=False)
        for (train_index, test_index) in kf.split(df):
            Dts = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["Dt"].values
                   for
                   mol in df.iloc[train_index]["mol"]]

            ESPs = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["ESP"].values
                    for
                    mol in df.iloc[train_index]["mol"]]
            LUMOs = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["ESP"].values
                    for
                    mol in df.iloc[train_index]["mol"]]

            # else:
            #
            if reguression_type in["gaussianFP"]:
                zeroweight = np.zeros(int(penalty.shape[1])).reshape(1, penalty.shape[1])
                if regression_features in["LUMO"]:
                    features = LUMOs.values
                    X= np.concatenate([features,penalty3],axis=0)
                    train_if = []
                    for weight in df.loc[:, fplist].columns:
                        if (train_tf := not df.iloc[train_index][weight].to_list()
                                                    .count(df.iloc[train_index][weight].to_list()[0]) == len(
                            df.iloc[train_index][weight])):
                            S = df.iloc[train_index][weight].values.reshape(1, -1)
                            Q = np.concatenate([S, zeroweight], axis=1)
                            X = np.concatenate([X, Q.T], axis=1)
                        train_if.append(train_tf)
                    Y = np.concatenate([df.iloc[train_index]["ΔΔG.expt."], np.zeros(penalty.shape[0] * 2)], axis=0)
                    model = linear_model.Ridge(alpha=0, fit_intercept=False).fit(X, Y)  # ただの線形回帰
                else:
                    features = np.concatenate([Dts, ESPs], axis=1)
                    X = np.concatenate([features, penalty1, penalty2], axis=0)
                    train_if = []
                    for weight in df.loc[:, fplist].columns:
                        if (train_tf := not df.iloc[train_index][weight].to_list()
                                                         .count(df.iloc[train_index][weight].to_list()[0]) == len(df.iloc[train_index][weight])):
                            S = df.iloc[train_index][weight].values.reshape(1, -1)
                            Q = np.concatenate([S, zeroweight, zeroweight], axis=1)
                            X = np.concatenate([X, Q.T], axis=1)
                        train_if.append(train_tf)
                    Y = np.concatenate([df.iloc[train_index]["ΔΔG.expt."], np.zeros(penalty.shape[0] * 2)], axis=0)
                    model = linear_model.Ridge(alpha=0, fit_intercept=False).fit(X, Y)#ただの線形回帰
            if reguression_type in["gaussian"]:
                if regression_features in["LUMO"]:
                    features = LUMOs.values
                    X= np.concatenate([features,penalty3],axis=0)
                    Y = np.concatenate([df.iloc[train_index]["ΔΔG.expt."], np.zeros(penalty.shape[0] )], axis=0)
                else:
                    features = np.concatenate([Dts, ESPs], axis=1)
                    X = np.concatenate([features, penalty1, penalty2], axis=0)
                    Y = np.concatenate([df.iloc[train_index]["ΔΔG.expt."], np.zeros(penalty.shape[0] * 2)], axis=0)

                model = linear_model.Ridge(alpha=0, fit_intercept=False).fit(X, Y)#Ridgeじゃないただの線形回帰

            # ここから、テストセットの特徴量計算
            Dts = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["Dt"].values
                   for
                   mol in df.iloc[test_index]["mol"]]
            ESPs = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["ESP"].values
                    for
                    mol in df.iloc[test_index]["mol"]]
            LUMOs = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["LUMO"].values
                    for
                    mol in df.iloc[test_index]["mol"]]

            if  regression_features in ["LUMO"]:
                features = LUMOs.values

            else :
                features = np.concatenate([Dts, ESPs], axis=1)
            if reguression_type in ["FP", "gaussianFP"]:
                for weight in df.loc[:, fplist].columns:
                    S = np.array(df.iloc[test_index][weight].values).reshape(1, np.array(
                        df.iloc[test_index][weight].values).shape[0]).T
                    features = np.concatenate([features, S], axis=1)


            # if reguression_type in["FP","gaussianFP"]:
            #     if regression_features in ["LUMO"]:
            #         features = LUMOs.values
            #         for weight in df.loc[:, fplist].columns:
            #             S = np.array(df.iloc[test_index][weight].values).reshape(1, np.array(df.iloc[test_index][weight].values).shape[0]).T
            #             features = np.concatenate([features, S], axis=1)
            #     else:
            #         features = np.concatenate([Dts, ESPs], axis=1)
            #         for weight in df.loc[:, fplist].columns:
            #             S = np.array(df.iloc[test_index][weight].values).reshape(1, np.array(df.iloc[test_index][weight].values).shape[0]).T
            #             features = np.concatenate([features, S], axis=1)
            predict = model.predict(features)
            l.extend(predict)
        print(l)
        r2 = r2_score(df["ΔΔG.expt."], l)
        RMSE = np.sqrt(mean_squared_error(df["ΔΔG.expt."], l))
        print("r2", r2)
        r2_list.append(r2)
        RMSE_list.append(RMSE)
    dfp["r2"] = r2_list
    dfp["RMSE"] = RMSE_list
    # raise ValueError
    dfp.to_csv(out_file_name)



def leave_one_out(features_dir_name, regression_features, df, out_file_name, param, fplist, regression_type, p=None):
    Dts = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["Dt"].values for mol
           in df["mol"]]
    ESPs = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["ESP"].values for mol
            in df["mol"]]
    LUMOs = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["LUMO"].values for mol
            in df["mol"]]

    penalty = np.load("../penalty/penalty.npy")
    if regression_features in ["LUMO"]:
        features = LUMOs.values
    else:
        features = np.concatenate([Dts, ESPs], axis=1)
    if regression_type== "lassocv":#lassocv
        Y = df["ΔΔG.expt."].values
        model = linear_model.LassoCV(fit_intercept=False).fit(features, Y)
        df["ΔΔG.train"] = model.predict(features)
        df_mf = pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))
        os.makedirs(param["moleculer_field_dir"], exist_ok=True)
        if regression_features in["LUMO"]:
            None
        else:
            df_mf["MF_Dt"] = model.coef_[:penalty.shape[0]]
            df_mf["MF_ESP"] = model.coef_[penalty.shape[0]:penalty.shape[0] * 2]
            df_mf.to_csv((param["moleculer_field_dir"] + "/" + "moleculer_field.csv"))

    if regression_type in ["gaussian"]:
        if regression_features in ["LUMO"]:
            None
        else:
            penalty1 = np.concatenate([p[0] * penalty, np.zeros(penalty.shape)], axis=1)
            penalty2 = np.concatenate([np.zeros(penalty.shape), p[1] * penalty], axis=1)
            X = np.concatenate([features, penalty1, penalty2], axis=0)
            Y = np.concatenate([df["ΔΔG.expt."], np.zeros(penalty.shape[0] * 2)], axis=0)
        model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)
        df["ΔΔG.train"] = model.predict(features)
        df_mf = pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))
        print(df["mol"].iloc[0].GetProp("InchyKey"))
        os.makedirs(param["moleculer_field_dir"], exist_ok=True)
        if regression_features in ["LUMO"]:
            df["MF_LUMO"]=model.coef_[:penalty.shape[0]]
        else:
            df_mf["MF_Dt"] = model.coef_[:penalty.shape[0]]
            df_mf["MF_ESP"] = model.coef_[penalty.shape[0]:penalty.shape[0] * 2]
        df_mf.to_csv((param["moleculer_field_dir"] + "/" + "moleculer_field.csv"))
        if not regression_features in["LUMO"]:
            df["Dt_contribution"] = np.sum(Dts * df_mf["MF_Dt"].values, axis=1)
            df["ESP_contribution"] = np.sum(ESPs * df_mf["MF_ESP"].values, axis=1)


    if regression_type in ["gaussianFP"]:
        penalty1 = np.concatenate([p[0] * penalty, np.zeros(penalty.shape)], axis=1)
        penalty2 = np.concatenate([np.zeros(penalty.shape), p[1] * penalty], axis=1)
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

        if regression_features in ["LUMO"]:
            df_fpv = str(model.coef_[penalty.shape[0] :])
            path_w = param["out_dir_name"] + "/" + "fpvalue"
            with open(path_w, mode='w') as f:
                f.write(df_fpv)
        else:
            df_mf["MF_Dt"]=model.coef_[:penalty.shape[0]]
            df_mf["MF_ESP"]=model.coef_[penalty.shape[0]:penalty.shape[0]*2]
            df["Dt_contribution"] = np.sum(Dts*df_mf["MF_Dt"].values,axis=1)
            df["ESP_contribution"] = np.sum(ESPs*df_mf["MF_ESP"].values,axis=1)
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
        model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)
        for weight in df.loc[:, fplist].columns:
            S = np.array(df[weight].values).reshape(-1, 1)
            features = np.concatenate([features, S], axis=1)
        df["ΔΔG.train"] = model.predict(features)
        df_mf=pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))
        print(model.coef_.shape)
        if regression_features in ["LUMO"]:
            df_fpv = str(model.coef_[penalty.shape[0] :])
            path_w = param["out_dir_name"] + "/" + "fpvalue"
            with open(path_w, mode='w') as f:
                f.write(df_fpv)
        else:
            df_mf["MF_Dt"]=model.coef_[:penalty.shape[0]]
            df_mf["MF_ESP"]=model.coef_[penalty.shape[0]:penalty.shape[0]*2]
            df["Dt_contribution"] = np.sum(Dts*df_mf["MF_Dt"].values,axis=1)
            df["ESP_contribution"] = np.sum(ESPs*df_mf["MF_ESP"].values,axis=1)
            print("model.coef_[penalty.shape[0]*2:]")
        # print(model.coef_)
            print(model.coef_[penalty.shape[0]*2:])
            df_fpv=str(model.coef_[penalty.shape[0]*2:])
            path_w = param["out_dir_name"]+"/"+"fpvalue"
            with open(path_w, mode='w') as f:
                f.write(df_fpv)



    if regression_type in ["lassocv"]:
        l = []
        kf = KFold(n_splits=len(df), shuffle=False)
        for (train_index, test_index) in kf.split(df):
            Dts = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["Dt"].values
                   for
                   mol in df.iloc[train_index]["mol"]]
            ESPs = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["ESP"].values
                    for
                    mol in df.iloc[train_index]["mol"]]
            LUMOs = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["LUMO"].values
                    for
                    mol in df.iloc[train_index]["mol"]]
            if regression_features in ["LUMO"]:
                features= LUMOs.values
            else :
                features = np.concatenate([Dts, ESPs], axis=1)

            Y=df.iloc[train_index]["ΔΔG.expt."].values
            model = linear_model.LassoCV(fit_intercept=False).fit(features, Y)
            Dts = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["Dt"].values
                   for
                   mol in df.iloc[test_index]["mol"]]
            ESPs = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["ESP"].values
                    for
                    mol in df.iloc[test_index]["mol"]]
            LUMOs = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["LUMO"].values
                    for
                    mol in df.iloc[test_index]["mol"]]
            if regression_features in ["LUMO"]:
                features = LUMOs.values
            else :
                features = np.concatenate([Dts, ESPs], axis=1)
            predict = model.predict(features)
            l.extend(predict)
    if regression_type in ["gaussian", "gaussianFP"]:
        l = []
        kf = KFold(n_splits=len(df), shuffle=False)
        for (train_index, test_index) in kf.split(df):
            Dts = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["Dt"].values for
                   mol in df.iloc[train_index]["mol"]]
            ESPs = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["ESP"].values for
                    mol in df.iloc[train_index]["mol"]]
            LUMOs = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["LUMO"].values
                for
                mol in df.iloc[train_index]["mol"]]
            if regression_features in ["LUMO"]:
                features = LUMOs.values
            else :
                features = np.concatenate([Dts, ESPs], axis=1)


            penalty1 = np.concatenate([p[0] * penalty, np.zeros(penalty.shape)], axis=1)
            penalty2 = np.concatenate([np.zeros(penalty.shape), p[1] * penalty], axis=1)
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
            Dts = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["Dt"].values for
                   mol in df.iloc[test_index]["mol"]]
            ESPs = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["ESP"].values for
                    mol in df.iloc[test_index]["mol"]]
            features = np.concatenate([Dts, ESPs], axis=1)
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
            Dts = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["Dt"].values
                   for
                   mol in df.iloc[train_index]["mol"]]
            ESPs = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["ESP"].values
                    for
                    mol in df.iloc[train_index]["mol"]]
            features = np.concatenate([Dts, ESPs], axis=1)
            X = np.concatenate([features], axis=0)

            for weight in df.loc[:, fplist].columns:
                S = df.iloc[train_index][weight].values.reshape(-1, 1)
                X = np.concatenate([X, S], axis=1)
            Y=df.iloc[train_index]["ΔΔG.expt."]
            model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)

            Dts = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["Dt"].values
                  for
                  mol in df.iloc[test_index]["mol"]]
            ESPs = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["ESP"].values
                    for
                    mol in df.iloc[test_index]["mol"]]
            features = np.concatenate([Dts, ESPs], axis=1)

            for weight in df.loc[:, fplist].columns:
                S = df.iloc[test_index][weight].values.reshape(-1, 1)
                features = np.concatenate([features, S], axis=1)
            predict = model.predict(features)
            l.extend(predict)

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
        "../parameter/parameter_cbs_gaussian.txt",
        "../parameter/parameter_dip-chloride_gaussian.txt",
        "../parameter/parameter_dip-chloride_gaussian_FP.txt",
        "../parameter/parameter_cbs_FP.txt",
        "../parameter/parameter_cbs_gaussian_FP.txt",
        "../parameter/parameter_cbs_lassocv.txt",
        "../parameter/parameter_dip-chloride_lassocv.txt",
        "../parameter/parameter_dip-chloride_FP.txt"
    ]:
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
        # if param["Regression_type"] in ["gaussian", "gaussianFP"]:
        #     grid_search(features_dir_name, df, dfp, param["out_dir_name"] + "/result_grid_search.csv", fplist,
        #                 param["Regression_type"])
        #     print(dfp[["λ1", "λ2"]].values[dfp["RMSE"].idxmin()])
        #     leave_one_out(features_dir_name, df,param["out_dir_name"] + "/result_loo.xls",param,fplist,param["Regression_type"],dfp[["λ1", "λ2"]].values[dfp["RMSE"].idxmin()])

        if param["Regression_type"] in ["gaussian","gaussianFP"]:
            if param["Regression_type"] in "gaussian":
                grid_search(features_dir_name,param["Regression_features"] ,df, dfp, param["out_dir_name"] + "/result_grid_search.csv",fplist,param["Regression_type"])
            if param["cat"]=="cbs":
                dfp=pd.read_csv("../result/cbs_gaussian/result_grid_search.csv")
            else :
                dfp = pd.read_csv("../result/dip-chloride_gaussian/result_grid_search.csv")

            print(dfp)
            print(dfp[["λ1", "λ2"]].values[dfp["RMSE"].idxmin()])

            with open(param["out_dir_name"]+"/hyperparam.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(dfp[["λ1", "λ2"]].values[dfp["RMSE"].idxmin()])

            if param["Regression_type"] in ["gaussian", "gaussianFP"]:

                leave_one_out(features_dir_name,param["Regression_features"] ,df,
                      param["out_dir_name"] + "/result_loo.xls",param,fplist,param["Regression_type"],dfp[["λ1", "λ2"]].values[dfp["RMSE"].idxmin()])
        else:
            leave_one_out(features_dir_name, param["Regression_features"],df,
                      param["out_dir_name"] + "/result_loo.xls", param, fplist, param["Regression_type"],p=None)