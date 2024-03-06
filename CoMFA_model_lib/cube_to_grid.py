import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import calculate_conformation

def pkl_to_featurevalue(dir_name, dfp, mol,out_name):#グリッド特徴量を計算　ボルツマン分布の加重はしないでデータセットの区別もなし。
    drop_dupl_x = dfp.drop_duplicates(subset="x").sort_values('x')["x"]
    drop_dupl_y = dfp.drop_duplicates(subset="y").sort_values('y')["y"]
    drop_dupl_z = dfp.drop_duplicates(subset="z").sort_values('z')["z"]

    d_x = (drop_dupl_x.iloc[1] - drop_dupl_x.iloc[0])/2
    d_y = (drop_dupl_y.iloc[1] - drop_dupl_y.iloc[0])/2
    d_z = (drop_dupl_z.iloc[1] - drop_dupl_z.iloc[0])/2
    for conf in mol.GetConformers():
        filename = "{}/{}/data{}.pkl".format(dir_name, mol.GetProp("InchyKey"), conf.GetId())
        print(filename)
        data = pd.read_pickle(filename)
        data["Dt"]=data["Dt"].where(data["Dt"]<1,1)
        # #data[["x","y","z"]]=data[["x","y","z"]]
        # cond_x=[]
        # cond_y=[]
        # for x in drop_dupl_x:
        #     cond_x_=x-d_x<data["x"] and x+d_x>data["x"]
        #     cond_y_=
        Dt=[]
        for x in drop_dupl_x:
            data_x=data[x-d_x<data["x"]]
            data_x=data_x[x+d_x>data_x["x"]]
            for y in drop_dupl_y:
                data_y=data_x[y-d_y<data_x["y"]]
                data_y = data_y[y + d_y > data_y["y"]]
                for z in drop_dupl_z:
                    data_z=data_y[z-d_z<data_y["z"]]
                    data_z=data_z[z+d_z>data_z["z"]]
                    Dt_=np.average(data_z["Dt"].values)
                    Dt.append(Dt_)
        # start=time.time()
        # for i,x in enumerate(dfp[["x","y","z"]].values):
        #     print("{}/{}".format(i,len(dfp)))
        #     data_=data[x[0]-d_x<data["x"]]
        #     data_=data_[x[0]+d_x>data_["x"]]
        #     dt=np.average(data_["Dt"].values)
        #     Dt.append(dt)
        #     if i==100:
        #         print(time.time()-start)
        #         sys.exit()
        dfp["Dt"]=np.nan_to_num(Dt)
        filename = "{}/{}/data{}.pkl".format(out_name, mol.GetProp("InchyKey"), conf.GetId())
        dfp.to_pickle(filename)

        dfp_yz = dfp[(dfp["y"] > 0) & (dfp["z"] > 0)][["x", "y", "z"]].sort_values(['x', 'y', "z"],
                                                                                   ascending=[True, True,
                                                                                              True])  # そとにでる
        feature=["Dt"]
        dfp_yz[feature] = \
                        dfp[(dfp["y"] > 0) & (dfp["z"] > 0)].sort_values(['x', 'y', "z"], ascending=[True, True, True])[
                            feature].values \
                        + dfp[(dfp["y"] < 0) & (dfp["z"] > 0)].sort_values(['x', 'y', "z"], ascending=[True, False, True])[
                            feature].values \
                        - dfp[(dfp["y"] > 0) & (dfp["z"] < 0)].sort_values(['x', 'y', "z"], ascending=[True, True, False])[
                            feature].values \
                        - dfp[(dfp["y"] < 0) & (dfp["z"] < 0)].sort_values(['x', 'y', "z"], ascending=[True, False, False])[
                            feature].values
        # dfp_yz.to_csv(out_name + "/feature_yz_{}.csv".format(conf.GetId()))
        print(dfp_yz)
        filename = "{}/{}/data_yz{}.pkl".format(out_name, mol.GetProp("InchyKey"), conf.GetId())
        dfp_yz.to_pickle(filename)

# def read_cube(dir_name, dfp, mol, out_name):
#     os.makedirs(out_name, exist_ok=True)
#     for conf in mol.GetConformers():
#         Dt_file = "{}/{}/Dt02_{}.cube".format(dir_name, mol.GetProp("InchyKey"), conf.GetId())
#         with open(Dt_file, "r", encoding="UTF-8") as f:
#             Dt = f.read().splitlines()
#         ESP_file = "{}/{}/ESP02_{}.cube".format(dir_name, mol.GetProp("InchyKey"), conf.GetId())
#         with open(ESP_file, "r", encoding="UTF-8") as f:
#             ESP = f.read().splitlines()
#         LUMO_file = "{}/{}/LUMO02_{}.cube".format(dir_name, mol.GetProp("InchyKey"), conf.GetId())
#         with open(LUMO_file, "r", encoding="UTF-8") as f:
#             LUMO = f.read().splitlines()
#         l = np.array([_.split() for _ in Dt[2:6]])
#         n_atom = int(l[0, 0])
#         x0 = l[0, 1:].astype(float)  # * 0.52917720859
#         size = l[1:, 0].astype(int)
#         # axis = l[1:, 1:].astype(float)*0.52917720859
#         grid = dfp[["x", "y", "z"]].values /0.52917720859 - x0
#         grid = np.round(grid/0.2).astype(int)
#         cond1 = np.all(grid < size, axis=1)
#         cond2 = np.all(np.zeros(3) <= grid, axis=1)
#         cond_all = np.all(np.stack([cond1, cond2]), axis=0)
#         n = grid[:, 0] * size[2] * size[1] + grid[:, 1] * size[2] + grid[:, 2]
#
#         def fdt(n, feature):
#             try:
#                 ans = float(feature[3 + 3 + n_atom + n // 6].split()[n % 6])
#                 if ans > 0.1:
#                     ans = 0.1
#             except:
#                 ans = 0
#             return ans
#
#         def fesp(n, feature):
#             try:
#                 ans = float(feature[3 + 3 + n_atom + n // 6].split()[n % 6])
#                 if ans > 0.1:
#                     ans = 0.1
#             except:
#                 ans = 0
#             return ans
#
#         # 20240105 坂口　作成
#         def fesp_new(n, Dt, feature):
#             try:
#                 if float(Dt[3 + 3 + n_atom + n // 6].split()[n % 6]) < 0.01:  # 0.05
#                     ans = float(feature[3 + 3 + n_atom + n // 6].split()[n % 6])
#                 else:
#                     ans = 0
#             except:
#                 ans = 0
#             return ans
#
#         def fesp_deletenucleo(n, feature):
#             try:
#                 ans = float(feature[3 + 3 + n_atom + n // 6].split()[n % 6])
#                 if ans > 0:
#                     ans = 0
#             except:
#                 ans = 0
#             return ans
#
#         ##
#         def fLUMO(n, feature):
#             try:
#                 ans = float(feature[3 + 3 + n_atom + n // 6].split()[n % 6])
#                 if ans > 0.05:
#                     ans = 0.05
#             except:
#                 ans = 0
#             return ans
#
#         dfp["Dt"] = np.where(cond_all, np.array([fdt(_, Dt) for _ in n]).astype(float), 0)
#         dfp["ESP"] = np.where(cond_all, np.array([fesp_deletenucleo(_, ESP) for _ in n]).astype(float), 0)
#         dfp["ESP_cutoff"] = np.where(cond_all, np.array([fesp_new(_, Dt, ESP) for _ in n]).astype(float), 0)
#         dfp["LUMO"] = np.where(cond_all, np.array([fLUMO(_, LUMO) for _ in n]).astype(float), 0)
#         os.makedirs(out_name, exist_ok=True)
#         dfp_y = dfp[dfp["y"] >= 0][["x", "y", "z"]].sort_values(['x', 'y', "z"], ascending=[True, True, True])  # そとにでる
#         dfp_z = dfp[dfp["z"] > 0][["x", "y", "z"]].sort_values(['x', 'y', "z"], ascending=[True, True, True])  # そとにでる
#         dfp_yz = dfp[(dfp["y"] > 0) & (dfp["z"] > 0)][["x", "y", "z"]].sort_values(['x', 'y', "z"],
#                                                                                    ascending=[True, True,
#                                                                                               True])  # そとにでる
#
#         for feature in ["Dt", "ESP", "ESP_cutoff", "LUMO"]:
#             # if False:
#             #     # dfp_y[feature]=dfp[dfp["y"]>=0][feature].values+dfp[dfp["y"]<=0][feature].values
#             #     dfp_y[feature] = dfp[dfp["y"] > 0][feature].values + dfp[dfp["y"] < 0][feature].values
#             #     # dfp_y[dfp_y["y"]==0] = dfp[dfp["y"] == 0]
#             #
#             #     dfp_z[feature] = dfp[dfp["z"] > 0][feature].values - \
#             #                      dfp[dfp["z"] < 0][feature].values
#             #
#             #     # dfp_yz[feature]=dfp[(dfp["y"]>=0)&(dfp["z"]>0)][feature].values+\
#             #     #                 dfp[(dfp["y"]<=0)&(dfp["z"]>0)][feature].values-\
#             #     #                 dfp[(dfp["y"]>=0)&(dfp["z"]<0)][feature].values-\
#             #     #                 dfp[(dfp["y"]<=0)&(dfp["z"]<0)][feature].values
#             #     dfp_yz[feature] = dfp[(dfp["y"] > 0) & (dfp["z"] > 0)][feature].values + \
#             #                       dfp[(dfp["y"] < 0) & (dfp["z"] > 0)][feature].values - \
#             #                       dfp[(dfp["y"] > 0) & (dfp["z"] < 0)][feature].values - \
#             #                       dfp[(dfp["y"] < 0) & (dfp["z"] < 0)][feature].values
#             dfp_y[feature] = dfp[dfp["y"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])[
#                                  feature].values \
#                              + dfp[dfp["y"] < 0].sort_values(['x', 'y', "z"], ascending=[True, False, True])[
#                                  feature].values
#             # dfp_z = dfp_z.sort_values(['x', 'y', "z"], ascending=[True, True, True])  # そとにでる
#             dfp_z[feature] = dfp[dfp["z"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])[
#                                  feature].values \
#                              - dfp[dfp["z"] < 0].sort_values(['x', 'y', "z"], ascending=[True, True, False])[
#                                  feature].values
#             # dfp_yz = dfp_yz.sort_values(['x', 'y', "z"], ascending=[True, True, True])  # そとにでる
#             dfp_yz[feature] = \
#                 dfp[(dfp["y"] > 0) & (dfp["z"] > 0)].sort_values(['x', 'y', "z"], ascending=[True, True, True])[
#                     feature].values \
#                 + dfp[(dfp["y"] < 0) & (dfp["z"] > 0)].sort_values(['x', 'y', "z"], ascending=[True, False, True])[
#                     feature].values \
#                 - dfp[(dfp["y"] > 0) & (dfp["z"] < 0)].sort_values(['x', 'y', "z"], ascending=[True, True, False])[
#                     feature].values \
#                 - dfp[(dfp["y"] < 0) & (dfp["z"] < 0)].sort_values(['x', 'y', "z"], ascending=[True, False, False])[
#                     feature].values
#             # if False:
#             #     l_y = []
#             #     l_z = []
#             #     l_yz = []
#             #     for x, y, z in zip(dfp_y["x"], dfp_y["y"], dfp_y["z"]):
#             #         ans = dfp[(dfp["x"] == x) & (dfp["y"] == y) & (dfp["z"] == z)][feature].iloc[0]
#             #         ans_y = dfp[(dfp["x"] == x) & (dfp["y"] == -y) & (dfp["z"] == z)][feature].iloc[0]
#             #         l_y.append(ans + ans_y)
#             #     dfp_y[feature] = l_y
#             #     for x, y, z in zip(dfp_z["x"], dfp_z["y"], dfp_z["z"]):
#             #         ans = dfp[(dfp["x"] == x) & (dfp["y"] == y) & (dfp["z"] == z)][feature].iloc[0]
#             #         ans_z = dfp[(dfp["x"] == x) & (dfp["y"] == y) & (dfp["z"] == -z)][feature].iloc[0]
#             #         l_z.append(ans - ans_z)
#             #     dfp_z[feature] = l_z
#             #     for x, y, z in zip(dfp_yz["x"], dfp_yz["y"], dfp_yz["z"]):
#             #         ans = dfp[(dfp["x"] == x) & (dfp["y"] == y) & (dfp["z"] == z)][feature].iloc[0]
#             #         ans_y = dfp[(dfp["x"] == x) & (dfp["y"] == -y) & (dfp["z"] == z)][feature].iloc[0]
#             #         ans_z = dfp[(dfp["x"] == x) & (dfp["y"] == y) & (dfp["z"] == -z)][feature].iloc[0]
#             #         ans_zy = dfp[(dfp["x"] == x) & (dfp["y"] == -y) & (dfp["z"] == -z)][feature].iloc[0]
#             #         l_yz.append(ans + ans_y - ans_z - ans_zy)
#             #     dfp_yz[feature] = l_yz
#
#         dfp_y.to_csv(out_name + "/feature_y_{}.csv".format(conf.GetId()))
#         dfp_z.to_csv(out_name + "/feature_z_{}.csv".format(conf.GetId()))
#         dfp_yz.to_csv(out_name + "/feature_yz_{}.csv".format(conf.GetId()))
#         dfp.to_csv(out_name + "/feature_{}.csv".format(conf.GetId()))
#
#     Dts = np.stack(
#         [pd.read_csv(out_name + "/feature_{}.csv".format(conf.GetId()))["Dt"].values for conf in mol.GetConformers()])
#     ESPs = [pd.read_csv(out_name + "/feature_{}.csv".format(conf.GetId()))["ESP"].values for conf in
#             mol.GetConformers()]
#     LUMOs = [pd.read_csv(out_name + "/feature_{}.csv".format(conf.GetId()))["LUMO"].values for conf in
#              mol.GetConformers()]
#     ESPs_cutoff = [pd.read_csv(out_name + "/feature_{}.csv".format(conf.GetId()))["ESP_cutoff"].values for conf in
#                    mol.GetConformers()]
#     weights = [float(conf.GetProp("Boltzmann_distribution")) for conf in mol.GetConformers()]
#     dfp["Dt"] = np.average(Dts, weights=weights, axis=0)
#     dfp["ESP"] = np.average(ESPs, weights=weights, axis=0)
#     dfp["ESP_cutoff"] = np.average(ESPs_cutoff, weights=weights, axis=0)
#     dfp["LUMO"] = np.average(LUMOs, weights=weights, axis=0)
#     dfp.to_csv(out_name + "/feature.csv".format(conf.GetId()))
#
#     Dts = np.stack(
#         [pd.read_csv(out_name + "/feature_y_{}.csv".format(conf.GetId()))["Dt"].values for conf in mol.GetConformers()])
#     ESPs = [pd.read_csv(out_name + "/feature_y_{}.csv".format(conf.GetId()))["ESP"].values for conf in
#             mol.GetConformers()]
#     LUMOs = [pd.read_csv(out_name + "/feature_y_{}.csv".format(conf.GetId()))["LUMO"].values for conf in
#              mol.GetConformers()]
#     ESPs_cutoff = [pd.read_csv(out_name + "/feature_y_{}.csv".format(conf.GetId()))["ESP_cutoff"].values for conf in
#                    mol.GetConformers()]
#     weights = [float(conf.GetProp("Boltzmann_distribution")) for conf in mol.GetConformers()]
#     dfp_y["Dt"] = np.average(Dts, weights=weights, axis=0)
#     dfp_y["ESP"] = np.average(ESPs, weights=weights, axis=0)
#     dfp_y["ESP_cutoff"] = np.average(ESPs_cutoff, weights=weights, axis=0)
#     dfp_y["LUMO"] = np.average(LUMOs, weights=weights, axis=0)
#     dfp_y.to_csv(out_name + "/feature_y.csv".format(conf.GetId()))
#
#     Dts = np.stack(
#         [pd.read_csv(out_name + "/feature_z_{}.csv".format(conf.GetId()))["Dt"].values for conf in mol.GetConformers()])
#     ESPs = [pd.read_csv(out_name + "/feature_z_{}.csv".format(conf.GetId()))["ESP"].values for conf in
#             mol.GetConformers()]
#     LUMOs = [pd.read_csv(out_name + "/feature_z_{}.csv".format(conf.GetId()))["ESP"].values for conf in
#              mol.GetConformers()]
#     ESPs_cutoff = [pd.read_csv(out_name + "/feature_z_{}.csv".format(conf.GetId()))["ESP_cutoff"].values for conf in
#                    mol.GetConformers()]
#     weights = [float(conf.GetProp("Boltzmann_distribution")) for conf in mol.GetConformers()]
#     dfp_z["Dt"] = np.average(Dts, weights=weights, axis=0)
#     dfp_z["ESP"] = np.average(ESPs, weights=weights, axis=0)
#     dfp_z["LUMO"] = np.average(LUMOs, weights=weights, axis=0)
#     dfp_z["ESP_cutoff"] = np.average(ESPs_cutoff, weights=weights, axis=0)
#
#     dfp_z.to_csv(out_name + "/feature_z.csv".format(conf.GetId()))
#
#     Dts = np.stack(
#         [pd.read_csv(out_name + "/feature_yz_{}.csv".format(conf.GetId()))["Dt"].values for conf in
#          mol.GetConformers()])
#     ESPs = [pd.read_csv(out_name + "/feature_yz_{}.csv".format(conf.GetId()))["ESP"].values for conf in
#             mol.GetConformers()]
#     LUMOs = [pd.read_csv(out_name + "/feature_yz_{}.csv".format(conf.GetId()))["LUMO"].values for conf in
#              mol.GetConformers()]
#     ESPs_cutoff = [pd.read_csv(out_name + "/feature_yz_{}.csv".format(conf.GetId()))["ESP_cutoff"].values for conf in
#                    mol.GetConformers()]
#     weights = [float(conf.GetProp("Boltzmann_distribution")) for conf in mol.GetConformers()]
#     dfp_yz["Dt"] = np.average(Dts, weights=weights, axis=0)
#     dfp_yz["ESP"] = np.average(ESPs, weights=weights, axis=0)
#     dfp_yz["LUMO"] = np.average(LUMOs, weights=weights, axis=0)
#     dfp_yz["ESP_cutoff"] = np.average(ESPs_cutoff, weights=weights, axis=0)
#     dfp_yz.to_csv(out_name + "/feature_yz.csv".format(conf.GetId()))
#     print("!!!")





if __name__ == '__main__':
    dfs = []
    for path in glob.glob("../arranged_dataset/*.xlsx"):
        df = pd.read_excel(path)
        dfs.append(df)
    df = pd.concat(dfs).dropna(subset=['smiles']).drop_duplicates(subset=["smiles"])
    with open("../parameter/cube_to_grid/cube_to_grid.txt", "r") as f:
        param = json.loads(f.read())
    print(param)
    df["mol"] = df["smiles"].apply(calculate_conformation.get_mol)
    df = df[[os.path.isdir(param["cube_dir_name"] + "/" + mol.GetProp("InchyKey")) for mol in df["mol"]]]
    df["mol"].apply(
        lambda mol: calculate_conformation.read_xyz(mol, param["cube_dir_name"] + "/" + mol.GetProp("InchyKey")))
    dfp = pd.read_csv(param["grid_coordinates"] + "/coordinates.csv")
    print(dfp)
    for mol in df["mol"]:
        os.makedirs(param["grid_coordinates"] + "/" + mol.GetProp("InchyKey"), exist_ok=True)
        pkl_to_featurevalue(param["cube_dir_name"], dfp, mol, param["grid_coordinates"])
    # for file in glob.glob("../arranged_dataset/*.xlsx"):
    #     # for param_file_name in ["../parameter_0227/parameter_cbs_gaussian.txt",
    #     #                         "../parameter_0227/parameter_RuSS_gaussian.txt",
    #     #                         "../parameter_0227/parameter_dip-chloride_gaussian.txt", ][::-1]:
    #
    #     with open("../parameter/cube_to_grid/cube_to_grid.txt", "r") as f:
    #         param = json.loads(f.read())
    #     print(param)
    #     df = pd.read_excel(file).dropna(subset=['smiles'])[::-1]
    #     df["mol"] = df["smiles"].apply(calculate_conformation.get_mol)
    #     df = df[[os.path.isdir(param["cube_dir_name"] + "/" + mol.GetProp("InchyKey")) for mol in df["mol"]]]
    #     df["mol"].apply(
    #         lambda mol: calculate_conformation.read_xyz(mol, param["cube_dir_name"] + "/" + mol.GetProp("InchyKey")))
    #
    #     # if False:
    #     #     print(param["grid_sizefile"].split(","))
    #     #     l = param["grid_sizefile"].split(",")  # 4.5
    #     #     xgrid, ygrid, zgrid, gridinterval = [float(_) for _ in
    #     #                                          l]
    #     #     y_up = np.arange(gridinterval / 2, ygrid, gridinterval)
    #     #     y_down = -y_up
    #     #     y = np.sort(np.concatenate([y_down, y_up]))
    #     #     z_up = np.arange(gridinterval / 2, zgrid, gridinterval)
    #     #     z_down = -z_up
    #     #     print(z_up[-1])
    #     #     z = np.sort(np.concatenate([z_down, z_up]))
    #     #     sr = {"x": np.round(np.arange(-xgrid, 2.1, gridinterval), 3),
    #     #           # "y": np.round(np.arange(-ygrid, ygrid+0.1, gridinterval), 2),
    #     #           "y": np.round(y, 5),
    #     #           "z": np.round(z, 5)}
    #     #     dfp = pd.DataFrame([dict(zip(sr.keys(), l)) for l in product(*sr.values())]).astype(float).sort_values(
    #     #         by=["x", "y", "z"])
    #     # #else:
    #     #     dfp = pd.read_csv(
    #     #         "../grid_coordinates" + param["grid_coordinates_dir"] + "/[{}].csv".format(param["grid_sizefile"]))
    #     dfp = pd.read_csv(param["grid_coordinates"] + "/coordinates.csv")
    #     print(dfp)
    #
    #     # grid_dir_name = param["grid_dir_name"]
    #     # グリッド情報に変換
    #     for mol in df["mol"]:
    #         os.makedirs(param["grid_coordinates"]+"/" + mol.GetProp("InchyKey"),exist_ok=True)
    #         pkl_to_featurevalue(param["cube_dir_name"], dfp, mol,param["grid_coordinates"])
    #     #
    #     # file_name = os.path.splitext(os.path.basename(file))[0]
    #     # for mol, RT in df[["mol", "RT"]].values:
    #     #     print(mol.GetProp("InchyKey"))
    #     #     energy_to_Boltzmann_distribution(mol, RT)  # 温度が対応できていない#ボルツマン分布は最後のほうが良さそう
    #     #     read_cube(param["cube_dir_name"], dfp, mol, param["grid_coordinates"] + file_name + "/" + mol.GetProp(
    #     #         "InchyKey"))  # grid_dir_name + "/[{}]".format(param["grid_sizefile"]) +
    #     # print("complete")
