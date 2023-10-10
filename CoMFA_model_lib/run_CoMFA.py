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

warnings.simplefilter('ignore')


def grid_search(features_dir_name, df, dfp, out_file_name):
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
        l = []
        kf = KFold(n_splits=5, shuffle=False)
        for (train_index, test_index) in kf.split(df):
            # for (train_index, test_index) in LeaveOneOut().split(df):
            start = time.time()
            Dts = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["Dt"].values
                   for
                   mol in df.iloc[train_index]["mol"]]

            ESPs = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["ESP"].values
                    for
                    mol in df.iloc[train_index]["mol"]]
            # print("csv",time.time()-start)
            features = np.concatenate([Dts, ESPs], axis=1)

            X = np.concatenate([features, penalty1, penalty2], axis=0)
            Y = np.concatenate([df.iloc[train_index]["ΔΔG.expt."], np.zeros(penalty.shape[0] * 2)], axis=0)
            model = linear_model.Ridge(alpha=0, fit_intercept=False).fit(X, Y)
            # print("reg",time.time()-start)
            # ここから、テストセットの特徴量計算
            Dts = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["Dt"].values
                   for
                   mol in df.iloc[test_index]["mol"]]
            ESPs = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["ESP"].values
                    for
                    mol in df.iloc[test_index]["mol"]]
            features = np.concatenate([Dts, ESPs], axis=1)
            predict = model.predict(features)
            # r2 = r2_score(df.iloc[test_index]["ΔΔG.expt."], predict)
            l.extend(predict)
        print(l)
        # X=np.concatenate([features,penalty1,penalty2],axis=0)
        # Y=np.concatenate([df["ΔΔG.expt."],np.zeros(penalty.shape[0]*2)],axis=0)
        # model=linear_model.LinearRegression(fit_intercept=False).fit(X,Y)
        # #model=linear_model.LassoCV(fit_intercept=False).fit(features,y)
        # #model = linear_model.RidgeCV(fit_intercept=False).fit(features, y)
        # predict=model.predict(features)
        r2 = r2_score(df["ΔΔG.expt."], l)
        RMSE = np.sqrt(mean_squared_error(df["ΔΔG.expt."], l))
        print("r2", r2)
        r2_list.append(r2)
        RMSE_list.append(RMSE)
    dfp["r2"] = r2_list
    dfp["RMSE"] = RMSE_list
    dfp.to_csv(out_file_name)


def leave_one_out(features_dir_name, df, p, out_file_name):
    Dts = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["Dt"].values for mol
           in df["mol"]]
    ESPs = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["ESP"].values for mol
            in df["mol"]]
    penalty = np.load("../penalty/penalty.npy")
    features = np.concatenate([Dts, ESPs], axis=1)
    penalty1 = np.concatenate([p[0] * penalty, np.zeros(penalty.shape)], axis=1)
    penalty2 = np.concatenate([np.zeros(penalty.shape), p[1] * penalty], axis=1)

    X = np.concatenate([features, penalty1, penalty2], axis=0)
    Y = np.concatenate([df["ΔΔG.expt."], np.zeros(penalty.shape[0] * 2)], axis=0)
    model = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)
    # model=linear_model.LassoCV(fit_intercept=False).fit(features,y)
    # model = linear_model.RidgeCV(fit_intercept=False).fit(features, y)
    predict = model.predict(features)
    df["ΔΔG.train"] = predict
    df_mf=pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))
    to_dir_name="../moleculer_field"
    to_file_name="moleculer_field.csv"
    os.makedirs(to_dir_name,exist_ok=True)
    df_mf["MF_Dt"]=model.coef_[:penalty.shape[0]]
    df_mf["MF_ESP"]=model.coef_[penalty.shape[0]:]
    df_mf.to_csv(to_dir_name+"/"+to_file_name)


    # ここから、loo
    l = []
    kf = KFold(n_splits=len(df), shuffle=False)
    for (train_index, test_index) in kf.split(df):
        # for (train_index, test_index) in LeaveOneOut().split(df):
        Dts = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["Dt"].values for
               mol in df.iloc[train_index]["mol"]]
        ESPs = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["ESP"].values for
                mol in df.iloc[train_index]["mol"]]
        features = np.concatenate([Dts, ESPs], axis=1)

        X = np.concatenate([features, penalty1, penalty2], axis=0)
        Y = np.concatenate([df.iloc[train_index]["ΔΔG.expt."], np.zeros(penalty.shape[0] * 2)], axis=0)
        model = linear_model.Ridge(alpha=0, fit_intercept=False).fit(X, Y)

        # ここから、テストセットの特徴量計算
        Dts = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["Dt"].values for
               mol in df.iloc[test_index]["mol"]]
        ESPs = [pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, mol.GetProp("InchyKey")))["ESP"].values for
                mol in df.iloc[test_index]["mol"]]
        features = np.concatenate([Dts, ESPs], axis=1)
        predict = model.predict(features)
        l.extend(predict)

    print(len(l), l)
    df["ΔΔG.loo"] = l
    r2 = r2_score(df["ΔΔG.expt."], l)
    print("r2", r2)
    df["error"] = l - df["ΔΔG.expt."]
    PandasTools.AddMoleculeColumnToFrame(df, "smiles")
    PandasTools.SaveXlsxFromFrame(df, out_file_name, size=(100, 100))


if __name__ == '__main__':
    #param_file_name = "../parameter/parameter_cbs.txt"
    param_file_name = "../parameter/parameter_dip-chloride.txt"
    with open(param_file_name, "r") as f:
        param = json.loads(f.read())
    features_dir_name = param["grid_dir_name"]
    xyz_dir_name = param["cube_dir_name"]

    df = pd.read_excel(param["data_file_path"]).dropna(subset=['smiles']).reset_index()  # [:10]
    df["mol"] = df["smiles"].apply(calculate_conformation.get_mol)
    print(len(df))
    df = df[[os.path.isdir(features_dir_name + "/" + mol.GetProp("InchyKey")) for mol in df["mol"]]]
    print(len(df))
    df["mol"].apply(lambda mol: calculate_conformation.read_xyz(mol, xyz_dir_name + "/" + mol.GetProp("InchyKey")))
    dfp = pd.read_csv(param["penalty_param_dir"])  # [:1]
    print(dfp)
    print(dfp)
    os.makedirs(param["out_dir_name"], exist_ok=True)
    grid_search(features_dir_name, df, dfp, param["out_dir_name"] + "/result_grid_search.csv")
    print(dfp)
    print(dfp[["λ1", "λ2"]].values[dfp["RMSE"].idxmin()])
    leave_one_out(features_dir_name, df, dfp[["λ1", "λ2"]].values[dfp["RMSE"].idxmin()],
                  param["out_dir_name"] + "/result_loo.xls")
