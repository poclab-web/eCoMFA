import glob
import json
import os
import time
import warnings
from multiprocessing import Pool

import cclib
import numpy as np
import pandas as pd
from numpy.linalg import solve
from rdkit.Chem import PandasTools
from sklearn import linear_model
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold

import calculate_conformation

warnings.simplefilter('ignore')


def Gaussian_penalized(df, dfp, gaussian_penalize, save_name):
    df_coord = pd.read_csv(gaussian_penalize + "/coordinates_yz.csv").sort_values(['x', 'y', "z"],
                                                                                  ascending=[True, True, True])
    features_all = np.array(df["Dt"].tolist()).reshape(len(df), -1, 1).transpose(2, 0, 1)
    std = np.std(features_all, axis=(1, 2)).reshape(features_all.shape[0], 1, 1)
    features = np.concatenate(features_all / std, axis=1)

    for n in range(1, 11):
        penalty = np.load(gaussian_penalize + "/penalty{}.npy".format(n))
        zeros = np.zeros(penalty.shape[0] * features_all.shape[0])
        gaussians = []
        predicts = []
        for L, n_num in zip(dfp["lambda"], range(1, len(dfp) + 1)):
            penalty_L = []
            for _ in range(features_all.shape[0]):
                penalty_L_ = []
                for __ in range(features_all.shape[0]):
                    if _ == __:
                        penalty_L_.append(np.sqrt(L) * penalty)
                    else:
                        penalty_L_.append(np.zeros(penalty.shape))
                penalty_L.append(penalty_L_)

            X = np.block([[features]] + penalty_L).astype('float32')
            Y = np.concatenate([df["ΔΔG.expt."], zeros], axis=0).astype('float32')

            start = time.time()
            gaussian_coef = np.linalg.solve(X.T @ X, X.T @ Y)
            x = np.sum(gaussian_coef * features, axis=1)
            a = np.dot(x, df["ΔΔG.expt."].values) / (x ** 2).sum()
            # print(a)
            # print(time.time() - start)

            gaussians.append(gaussian_coef * a.tolist())
            n_ = int(gaussian_coef.shape[0] / features_all.shape[0])
            df_coord["Gaussian_Dt"] = gaussian_coef[:n_]
            df_coord.to_csv(save_name + "/molecular_filed{}{}.csv".format(n, L))

            kf = KFold(n_splits=3, shuffle=False)
            gaussian_predicts = []

            for (train_index, test_index) in kf.split(df):
                features_training = features_all[:,
                                    train_index]
                std = np.std(features_training, axis=(1, 2)).reshape(features_all.shape[0], 1, 1)
                features_training = np.concatenate(features_training / std, axis=1)
                X = np.block([[features_training]] + penalty_L).astype('float32')
                Y = np.concatenate([df.iloc[train_index]["ΔΔG.expt."], zeros], axis=0).astype('float32')

                start = time.time()
                gaussian_coef_ = np.linalg.solve(X.T @ X, X.T @ Y)
                x = np.sum(gaussian_coef_ * features_training, axis=1)
                a = np.dot(x, df.iloc[train_index]["ΔΔG.expt."].values) / (x ** 2).sum()
                features_test = features_all[:, test_index]
                features_test = np.concatenate(features_test / std, axis=1)
                predict = np.sum(gaussian_coef_ * a * features_test, axis=1).tolist()
                gaussian_predicts.extend(predict)

            predicts.append([gaussian_predicts])
        gaussians = np.array(gaussians)
        gaussians = gaussians.reshape(gaussians.shape[0], 1, -1)
        dfp["Gaussian_regression_predict{}".format(n)] = np.sum(gaussians * features.reshape(1, features.shape[0], -1),
                                                                axis=2).tolist()
        dfp[["Gaussian_regression_r2{}".format(n)]] = dfp[
            ["Gaussian_regression_predict{}".format(n)]].applymap(
            lambda predict: r2_score(df["ΔΔG.expt."], predict))
        dfp[["Gaussian_regression_RMSE{}".format(n)]] = \
            dfp[[
                "Gaussian_regression_predict{}".format(n)]].applymap(
                lambda predict: np.sqrt(mean_squared_error(df["ΔΔG.expt."], predict)))

        dfp[["Gaussian_test_predict{}".format(n)]] = predicts
        dfp[["Gaussian_test_r2{}".format(n)]] = dfp[
            ["Gaussian_test_predict{}".format(n)]].applymap(
            lambda predict: r2_score(df["ΔΔG.expt."], predict))
        dfp[["Gaussian_test_r{}".format(n)]] = dfp[
            ["Gaussian_test_predict{}".format(n)]].applymap(
            lambda predict: np.corrcoef(df["ΔΔG.expt."], predict)[1, 0])
        dfp[["Gaussian_test_RMSE{}".format(n)]] = dfp[
            ["Gaussian_test_predict{}".format(n)]].applymap(
            lambda predict: np.sqrt(mean_squared_error(df["ΔΔG.expt."], predict)))
        df["Gaussian_predict{}".format(n)] = \
            dfp[dfp["Gaussian_test_r{}".format(n)] == dfp["Gaussian_test_r{}".format(n)].max()].iloc[0][
                "Gaussian_test_predict{}".format(n)]
        df["Gaussian_error{}".format(n)] = df["Gaussian_predict{}".format(n)] - df["ΔΔG.expt."]
        print(dfp[["Gaussian_test_r2{}".format(n)]].max())

    df = df.sort_values(by='Gaussian_error2', key=abs, ascending=[False])
    PandasTools.AddMoleculeColumnToFrame(df, "smiles")
    PandasTools.SaveXlsxFromFrame(df, save_name + "/n_comparison.xlsx", size=(100, 100))
    dfp.to_csv(save_name + "/n_comparison.csv")
    print(save_name)


def regression_comparison(df, dfp, gaussian_penalize, save_name, n):
    df_coord = pd.read_csv(gaussian_penalize + "/coordinates_yz.csv").sort_values(['x', 'y', "z"],
                                                                                  ascending=[True, True, True])
    penalty = np.load(gaussian_penalize + "/penalty{}.npy".format(str(n)))
    features_all = np.array(df["Dt"].tolist()).reshape(len(df), -1, 1).transpose(2, 0, 1)
    std = np.std(features_all, axis=(1, 2)).reshape(features_all.shape[0], 1, 1)
    features = np.concatenate(features_all / std, axis=1)
    zeros = np.zeros(penalty.shape[0] * features_all.shape[0])

    models = []
    gaussians = []
    predicts = []  # [[] for _ in range(4)]
    for L, n_num in zip(dfp["lambda"],dfp["n_components"]):
        penalty_L = []
        for _ in range(features_all.shape[0]):
            penalty_L_ = []
            for __ in range(features_all.shape[0]):
                if _ == __:
                    penalty_L_.append(np.sqrt(L) * penalty)
                else:
                    penalty_L_.append(np.zeros(penalty.shape))
            penalty_L.append(penalty_L_)

        X = np.block([[features]] + penalty_L).astype('float32')
        Y = np.concatenate([df["ΔΔG.expt."], zeros], axis=0).astype('float32')
        start = time.time()

        ridge = linear_model.Ridge(alpha=L, fit_intercept=False).fit(features, df["ΔΔG.expt."])
        # print(time.time()-start)
        lasso = linear_model.Lasso(alpha=L / 1000, fit_intercept=False).fit(features, df["ΔΔG.expt."])
        # print(time.time()-start)
        # features_norm = features / np.std(features, axis=0)
        # print(time.time()-start)
        # pls = PLSRegression(n_components=n_num).fit(features_norm, df["ΔΔG.expt."])
        pls = PLSRegression(n_components=n_num).fit(features, df["ΔΔG.expt."])
        # print(time.time()-start)
        models.append([ridge, lasso, pls])
        gaussian_coef = np.linalg.solve(X.T @ X, X.T @ Y)

        gaussians.append(gaussian_coef.tolist())
        n = int(gaussian_coef.shape[0] / features_all.shape[0])
        df_coord["Gaussian_Dt"] = gaussian_coef[:n] * std[0].reshape([1])
        # df_coord["Gaussian_ESP"]=gaussian_coef[n:]
        df_coord["Ridge_Dt"] = ridge.coef_[:n] * std[0].reshape([1])
        # df_coord["Ridge_ESP"]=ridge.coef_[n:]
        df_coord["Lasso_Dt"] = lasso.coef_[:n] * std[0].reshape([1])
        # df_coord["Lasso_ESP"]=lasso.coef_[n:]
        df_coord["PLS_Dt"] = pls.coef_[0][:n] * std[0].reshape([1])
        # df_coord["PLS_Dt"] = pls.coef_[0][:n] * np.std(features, axis=0) * std[0].reshape([1])

        kf = KFold(n_splits=3, shuffle=False)
        gaussian_predicts = []
        ridge_predicts = []
        lasso_predicts = []
        pls_predicts = []

        for i, (train_index, test_index) in enumerate(kf.split(df)):
            features_training = features_all[:, train_index]
            std_ = np.std(features_training, axis=(1, 2)).reshape(features_all.shape[0], 1, 1)
            features_training = features_training / std_
            features_training = np.concatenate(features_training, axis=1)
            X = np.block([[features_training]] + penalty_L).astype('float32')
            Y = np.concatenate([df.iloc[train_index]["ΔΔG.expt."], zeros], axis=0).astype('float32')
            start = time.time()
            gaussian_coef_ = np.linalg.solve(X.T @ X, X.T @ Y)
            # print(time.time()-start)
            ridge = linear_model.Ridge(alpha=L, fit_intercept=False).fit(features_training,
                                                                         df.iloc[train_index]["ΔΔG.expt."])
            lasso = linear_model.Lasso(alpha=L / 1000, fit_intercept=False).fit(features_training,
                                                                                df.iloc[train_index]["ΔΔG.expt."])
            # features_training_norm = features_training / np.std(features_training, axis=0)
            # pls = PLSRegression(n_components=n_num).fit(features_training_norm,
            #                                             df.iloc[train_index]["ΔΔG.expt."])
            pls = PLSRegression(n_components=n_num).fit(features_training,
                                                        df.iloc[train_index]["ΔΔG.expt."])
            features_test = features_all[:, test_index]  # np.array(features_all)[test_index].transpose(2, 0, 1)
            features_test = np.concatenate(features_test / std_, axis=1)
            predict = np.sum(gaussian_coef_ * features_test, axis=1).tolist()  # model.predict(features_test)
            gaussian_predicts.extend(predict)
            # predicts[0].extend(predict)
            ridge_predict = ridge.predict(features_test)
            # predicts[1].extend(ridge_predict)
            ridge_predicts.extend(ridge_predict)
            lasso_predict = lasso.predict(features_test)
            # predicts[2].extend(lasso_predict)
            lasso_predicts.extend(lasso_predict)
            # features_test_norm = features_test / np.std(features_training, axis=0)
            # pls_predict = pls.predict(features_test_norm)
            pls_predict = pls.predict(features_test)
            # predicts[3].extend([_[0] for _ in pls_predict])
            pls_predicts.extend([_[0] for _ in pls_predict])
            n = int(gaussian_coef_.shape[0] / features_all.shape[0])

            df_coord["Gaussian_Dt{}".format(i)] = gaussian_coef_[:n] * std_[0].reshape([1])
            # df_coord["Gaussian_ESP"]=gaussian_coef[n:]
            df_coord["Ridge_Dt{}".format(i)] = ridge.coef_[:n] * std_[0].reshape([1])
            # df_coord["Ridge_ESP"]=ridge.coef_[n:]
            df_coord["Lasso_Dt{}".format(i)] = lasso.coef_[:n] * std_[0].reshape([1])
            # df_coord["Lasso_ESP"]=lasso.coef_[n:]
            # df_coord["PLS_Dt{}".format(i)] = pls.coef_[0][:n] * np.std(features, axis=0) * std_[0].reshape([1])
            df_coord["PLS_Dt{}".format(i)] = pls.coef_[0][:n] * std_[0].reshape([1])

        predicts.append([gaussian_predicts, ridge_predicts, lasso_predicts, pls_predicts])
        df_coord.to_csv(save_name + "/molecular_filed{}.csv".format(L))
    print(np.array(predicts).shape)
    dfp[["Ridge_model", "Lasso_model", "PLS_model"]] = models
    dfp[["Ridge_regression_predict", "Lasso_regression_predict", "PLS_regression_predict"]] \
        = dfp[["Ridge_model", "Lasso_model", "PLS_model"]].applymap(lambda model: model.predict(features))
    gaussians = np.array(gaussians)
    gaussians = gaussians.reshape(gaussians.shape[0], 1, -1)
    features = features.reshape(1, features.shape[0], -1)
    dfp["Gaussian_regression_predict"] = np.sum(gaussians * features, axis=2).tolist()
    dfp[["Gaussian_regression_r2", "Ridge_regression_r2", "Lasso_regression_r2", "PLS_regression_r2"]] = dfp[
        ["Gaussian_regression_predict", "Ridge_regression_predict", "Lasso_regression_predict",
         "PLS_regression_predict"]].applymap(
        lambda predict: r2_score(df["ΔΔG.expt."], predict))
    dfp[["Gaussian_regression_RMSE", "Ridge_regression_RMSE", "Lasso_regression_RMSE", "PLS_regression_RMSE"]] = dfp[[
        "Gaussian_regression_predict", "Ridge_regression_predict", "Lasso_regression_predict",
        "PLS_regression_predict"]].applymap(
        lambda predict: np.sqrt(mean_squared_error(df["ΔΔG.expt."], predict)))

    dfp[["Gaussian_test_predict", "ridge_test_predict", "lasso_test_predict", "pls_test_predict"]] = predicts
    dfp[["Gaussian_test_r2", "ridge_test_r2", "lasso_test_r2", "pls_test_r2"]] = dfp[
        ["Gaussian_test_predict", "ridge_test_predict", "lasso_test_predict", "pls_test_predict"]].applymap(
        lambda predict: r2_score(df["ΔΔG.expt."], predict))
    dfp[["Gaussian_test_r", "ridge_test_r", "lasso_test_r", "pls_test_r"]] = dfp[
        ["Gaussian_test_predict", "ridge_test_predict", "lasso_test_predict", "pls_test_predict"]].applymap(
        lambda predict: np.corrcoef(df["ΔΔG.expt."], predict)[1, 0])
    dfp[["Gaussian_test_RMSE", "ridge_test_RMSE", "lasso_test_RMSE", "pls_test_RMSE"]] = dfp[
        ["Gaussian_test_predict", "ridge_test_predict", "lasso_test_predict", "pls_test_predict"]].applymap(
        lambda predict: np.sqrt(mean_squared_error(df["ΔΔG.expt."], predict)))

    features_R1 = np.array(df["DtR1"].tolist()).reshape(len(df), -1, 1).transpose(2, 0, 1)
    features_R1 = np.concatenate(features_R1 / std, axis=1)
    dfp["Gaussian_regression_predictR1"] = np.sum(gaussians * features_R1, axis=2).tolist()
    dfp[["Ridge_regression_predictR1", "Lasso_regression_predictR1", "PLS_regression_predictR1"]] \
        = dfp[["Ridge_model", "Lasso_model", "PLS_model"]].applymap(lambda model: model.predict(features_R1))

    features_R2 = np.array(df["DtR2"].tolist()).reshape(len(df), -1, 1).transpose(2, 0, 1)
    features_R2 = np.concatenate(features_R2 / std, axis=1)
    dfp["Gaussian_regression_predictR2"] = np.sum(gaussians * features_R2, axis=2).tolist()
    dfp[["Ridge_regression_predictR2", "Lasso_regression_predictR2", "PLS_regression_predictR2"]] \
        = dfp[["Ridge_model", "Lasso_model", "PLS_model"]].applymap(lambda model: model.predict(features_R2))
    # df["Gaussian_predict"] = dfp[dfp["Gaussian_test_r"] == dfp["Gaussian_test_r"].max()].iloc[0][
    #     "Gaussian_test_predict"]
    # df["Ridge_predict"] = dfp[dfp["ridge_test_r"] == dfp["ridge_test_r"].max()].iloc[0][
    #     "ridge_test_predict"]
    # df["Lasso_predict"] = dfp[dfp["lasso_test_r"] == dfp["lasso_test_r"].max()].iloc[0]["lasso_test_predict"]
    # df["PLS_predict"] = dfp[dfp["pls_test_r"] == dfp["pls_test_r"].max()].iloc[0]["pls_test_predict"]
    df[["Gaussian_test", "Gaussian_regression", "Gaussian_R1", "Gaussian_R2"]] = \
        dfp.loc[dfp["Gaussian_test_r"].idxmax()][
            ["Gaussian_test_predict", "Gaussian_regression_predict", "Gaussian_regression_predictR1",
             "Gaussian_regression_predictR2"]]
    df["Ridge_test"] = dfp.loc[dfp["ridge_test_r"].idxmax()]["ridge_test_predict"]
    df["Lasso_test"] = dfp.loc[dfp["lasso_test_r"].idxmax()]["lasso_test_predict"]
    df["PLS_test"] = dfp.loc[dfp["pls_test_r"].idxmax()]["pls_test_predict"]

    df["Gaussian_error"] = df["Gaussian_test"] - df["ΔΔG.expt."]
    # df[["Gaussian_error","Ridge_error","Lasso_error"]] = df[["Gaussian_predict","Ridge_predict","Lasso_predict",]].applymap(lambda test:test - df["ΔΔG.expt."].values)
    df = df.sort_values(by='Gaussian_error', key=abs, ascending=[False])
    PandasTools.AddMoleculeColumnToFrame(df, "smiles")
    # df = df[[ "smiles", "ROMol","inchikey","er.", "RT", "ΔΔG.expt."]].drop_duplicates(subset="inchikey")#,"ΔΔminG.expt.","ΔΔmaxG.expt."
    # df = df[["smiles", "ROMol", "er.", "RT"]]
    print(dfp[["Gaussian_test_r2", "ridge_test_r2", "lasso_test_r2", "pls_test_r2"]].max())
    PandasTools.SaveXlsxFromFrame(df, save_name + "/result_test.xlsx", size=(100, 100))
    dfp.to_csv(save_name + "/result.csv", index = False)
    print(save_name)


def energy_to_Boltzmann_distribution(mol, RT=1.99e-3 * 273):
    energies = np.array([float(conf.GetProp("energy")) for conf in mol.GetConformers()])
    energies = energies - np.min(energies)
    rates = np.exp(-energies / RT)
    rates = rates / sum(rates)
    for conf, rate in zip(mol.GetConformers(), rates):
        conf.SetProp("Boltzmann_distribution", str(rate))
if True:
    def energy_to_Boltzmann_distribution(mol, RT=1.99e-3 * 273):

        energies = []
        for conf in mol.GetConformers():
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
            for l in frequencies_lines:
                splited = l.split()
                values = splited[2:]
                values = [float(v) for v in values]
                for v in values:
                    if v < 0:
                        f.close()
                        # print(filename)
                        return False
            f.close()
        return True
    except:
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


def GP(input):
    df_, dfp, grid_coordinates, save_path = input
    Gaussian_penalized(df_, dfp, grid_coordinates, save_path)


def RC(input):
    df_, dfp, grid_coordinates, save_path, n = input
    regression_comparison(df_, dfp, grid_coordinates, save_path, n)


if __name__ == '__main__':
    # time.sleep(60*10)
    start = time.perf_counter()  # 計測開始
    for file in glob.glob("../arranged_dataset/*.xlsx"):
        with open("../parameter/cube_to_grid/cube_to_grid.txt", "r") as f:
            param = json.loads(f.read())
            print(param)
            df = pd.read_excel(file).dropna(subset=['smiles']).reset_index(drop=True)  # [:50]
            file_name = os.path.splitext(os.path.basename(file))[0]
            features_dir_name = param["grid_coordinates"] + file_name
            print(features_dir_name)
            print("dflen", len(df))
            df["mol"] = df["smiles"].apply(calculate_conformation.get_mol)
            # df = df[[os.path.isdir(features_dir_name + "/" + mol.GetProp("InchyKey")) for mol in df["mol"]]]
            df = df[
                [os.path.isdir("{}/{}".format(param["grid_coordinates"], mol.GetProp("InchyKey"))) for mol in
                 df["mol"]]]
            df=df[[os.path.isdir("{}/{}".format(param["freq_dir"], mol.GetProp("InchyKey"))) for mol in
                 df["mol"]]]
            freq=[]
            for mol in df["mol"]:
                freq_ = all([is_normal_frequencies(path) for path in
                            sorted(
                                glob.glob(
                                    param["freq_dir"] + "/" + mol.GetProp("InchyKey") + "/gaussianinput?.log"))])
                print(mol.GetProp("InchyKey"),freq_)
                freq.append(freq_)
            df=df[freq]
            df["mol"].apply(
                lambda mol: calculate_conformation.read_xyz(mol,
                                                            param["cube_dir_name"] + "/" + mol.GetProp("InchyKey")))
            for mol in df["mol"]:
                dirs_name_freq = param["freq_dir"] + "/" + mol.GetProp("InchyKey") + "/gaussianinput?.log"
                print(dirs_name_freq)
                for path, conf in zip(
                        sorted(glob.glob(dirs_name_freq)),
                        mol.GetConformers()):
                    # print(path)
                    data = cclib.io.ccread(path)
                    ent = data.enthalpy * 627.5095  # hartree
                    entr = data.entropy * 627.5095  # hartree
                    conf.SetProp("freq", json.dumps([ent, entr]))

            print("dflen", len(df))
            dfp = pd.read_csv(param["grid_coordinates"] + "/penalty_param.csv")
            Dts = []
            DtR1 = []
            DtR2 = []
            for mol, RT in df[["mol", "RT"]].values:
                print(mol.GetProp("InchyKey"))
                energy_to_Boltzmann_distribution(mol, RT)
                Dt = []
                we = []
                for conf in mol.GetConformers():
                    data = pd.read_pickle(
                        "{}/{}/data{}.pkl".format(param["grid_coordinates"], mol.GetProp("InchyKey"), conf.GetId()))
                    Bd = float(conf.GetProp("Boltzmann_distribution"))
                    # if False:
                    #     we.append(Bd)
                    #     Dt.append(data["Dt"].values.tolist())
                    # else:
                    we.extend([Bd, Bd])
                    Dt.extend([data[data["y"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])[
                                   "Dt"].values.tolist(),
                               data[data["y"] < 0].sort_values(['x', 'y', "z"], ascending=[True, False, True])[
                                   "Dt"].values.tolist()])
                Dt = np.array(Dt)
                w = np.exp(-Dt / np.sqrt(np.average(Dt ** 2, axis=0)).reshape(1, -1))
                # we = np.array([float(conf.GetProp("Boltzmann_distribution")) for conf in mol.GetConformers()]).reshape(
                #     -1, 1)
                we = np.array(we).reshape(-1, 1)
                Dt_ = np.average(Dt, weights=we * w, axis=0)
                # if True:
                data_y = data[data["y"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])
                data_y["Dt"] = np.nan_to_num(Dt_)
                data_yz = data_y[data_y["z"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])
                data_yz["Dt"] = data_y[data_y["z"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])[
                                    "Dt"].values - \
                                data_y[data_y["z"] < 0].sort_values(['x', 'y', "z"], ascending=[True, True, False])[
                                    "Dt"].values
                data_yz["DtR1"] = \
                    data_y[data_y["z"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])["Dt"].values
                data_yz["DtR2"] = \
                    data_y[data_y["z"] < 0].sort_values(['x', 'y', "z"], ascending=[True, True, False])["Dt"].values
                dfp_yz = data_yz.copy()
                # else:
                #     data["Dt"] = np.nan_to_num(Dt_)
                #     dfp_yz = data[(data["y"] > 0) & (data["z"] > 0)][["x", "y", "z"]].sort_values(['x', 'y', "z"],
                #                                                                                   ascending=[True, True,
                #                                                                                              True])
                #     # feature = ["Dt"]
                #     dfp_yz["Dt"] = \
                #         data[(data["y"] > 0) & (data["z"] > 0)].sort_values(['x', 'y', "z"],
                #                                                             ascending=[True, True, True])[
                #             "Dt"].values \
                #         + \
                #         data[(data["y"] < 0) & (data["z"] > 0)].sort_values(['x', 'y', "z"],
                #                                                             ascending=[True, False, True])[
                #             "Dt"].values \
                #         - \
                #         data[(data["y"] > 0) & (data["z"] < 0)].sort_values(['x', 'y', "z"],
                #                                                             ascending=[True, True, False])[
                #             "Dt"].values \
                #         - data[(data["y"] < 0) & (data["z"] < 0)].sort_values(['x', 'y', "z"],
                #                                                               ascending=[True, False, False])[
                #             "Dt"].values
                #
                #     dfp_yz["DtR1"] = \
                #         data[(data["y"] > 0) & (data["z"] > 0)].sort_values(['x', 'y', "z"],
                #                                                             ascending=[True, True, True])[
                #             "Dt"].values \
                #         + \
                #         data[(data["y"] < 0) & (data["z"] > 0)].sort_values(['x', 'y', "z"],
                #                                                             ascending=[True, False, True])[
                #             "Dt"].values
                #     dfp_yz["DtR2"] = \
                #         -data[(data["y"] > 0) & (data["z"] < 0)].sort_values(['x', 'y', "z"],
                #                                                              ascending=[True, True, False])[
                #             "Dt"].values \
                #         - data[(data["y"] < 0) & (data["z"] < 0)].sort_values(['x', 'y', "z"],
                #                                                               ascending=[True, False, False])[
                #             "Dt"].values
                Dts.append(dfp_yz["Dt"].values.tolist())
                DtR1.append(dfp_yz["DtR1"].values.tolist())
                DtR2.append(dfp_yz["DtR2"].values.tolist())

            df["Dt"] = Dts
            df["DtR1"] = DtR1
            df["DtR2"] = DtR2
            # if False:
            #     df_ = df.sort_values(by=["ΔΔG.expt."])
            #     save_path = param["out_dir_name"] + "/" + file_name + "/sorted"
            #     os.makedirs(save_path, exist_ok=True)
            #     Gaussian_penalized(features_dir_name, df_, dfp, param["grid_coordinates"], save_path)

            inputs = []
            for _ in range(0):
                df_ = df.sample(frac=1, random_state=_)
                save_path = param["out_dir_name"] + "/" + file_name + "/comparison" + str(_)
                os.makedirs(save_path, exist_ok=True)
                input = df_, dfp, param["grid_coordinates"], save_path
                inputs.append(input)
                # GP(input)
                # Gaussian_penalized(df_, dfp, param["grid_coordinates"], save_path)
            p = Pool(5)
            p.map(GP, inputs)
            # n = select_σ_n(file_name)
            n = 3
            inputs = []
            for _ in range(50):
                df_ = df.sample(frac=1, random_state=_)
                save_path = param["out_dir_name"] + "/" + file_name + "/" + str(_)
                os.makedirs(save_path, exist_ok=True)
                input = df_, dfp, param["grid_coordinates"], save_path, n
                inputs.append(input)
                # regression_comparison(df_, dfp, param["grid_coordinates"], save_path, n)
            p = Pool(5)
            p.map(RC, inputs)
    end = time.perf_counter()  # 計測終了
    print('Finish{:.2f}'.format(end - start))
