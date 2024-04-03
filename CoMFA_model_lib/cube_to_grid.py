import copy
import glob
import json
import os
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

import calculate_conformation

warnings.simplefilter('ignore')


def pkl_to_featurevalue(dir_name, dfp, mol, out_name):  # グリッド特徴量を計算　ボルツマン分布の加重はしないでデータセットの区別もなし。
    drop_dupl_x = dfp.drop_duplicates(subset="x").sort_values('x')["x"]
    drop_dupl_y = dfp.drop_duplicates(subset="y").sort_values('y')["y"]
    drop_dupl_z = dfp.drop_duplicates(subset="z").sort_values('z')["z"]

    d_x = (drop_dupl_x.iloc[1] - drop_dupl_x.iloc[0]) / 2
    d_y = (drop_dupl_y.iloc[1] - drop_dupl_y.iloc[0]) / 2
    d_z = (drop_dupl_z.iloc[1] - drop_dupl_z.iloc[0]) / 2

    for conf in mol.GetConformers():
        # 入力：幅
        # dfpをdata.pklの最大・最小から決定
        outfilename = "{}/data{}.pkl".format(out_name, conf.GetId())
        if os.path.isfile(outfilename):
            continue
        filename = "{}/data{}.pkl".format(dir_name, conf.GetId())
        print(filename)
        data = pd.read_pickle(filename)
        data["Dt"] = data["Dt"].where(data["Dt"] < 10, 10)
        # data["ESP"]=data["ESP"].values*np.exp(-data["Dt"].values/np.sqrt(np.average(data["Dt"].values ** 2)))
        data["ESP"] = data["ESP"].where(data["ESP"] < 0, 0)

        # print(data)
        start = time.time()

        if True:
            D = 0.1 * 0.52917720859
            Dt = []
            ESP=[]
            for x in drop_dupl_x:
                data = data[x - d_x < data["x"] + D]
                data_x = data[x + d_x > data["x"] - D]
                for y in drop_dupl_y:
                    data_x = data_x[y - d_y < data_x["y"] + D]
                    data_y = data_x[y + d_y > data_x["y"] - D]
                    for z in drop_dupl_z:
                        data_y = data_y[z - d_z < data_y["z"] + D]
                        data_z = data_y[z + d_z > data_y["z"] - D]
                        Dt_ = np.sum(data_z["Dt"].values *
                                     (np.where(data_z["x"].values - (x - d_x) < D, data_z["x"].values - (x - d_x),
                                               D) + np.where(x + d_x - data_z["x"].values < D,
                                                             x + d_x - data_z["x"].values, D)) *
                                     (np.where(data_z["y"].values - (y - d_y) < D, data_z["y"].values - (y - d_y),
                                               D) + np.where(y + d_y - data_z["y"].values < D,
                                                             y + d_y - data_z["y"].values, D)) *
                                     (np.where(data_z["z"].values - (z - d_z) < D, data_z["z"].values - (z - d_z),
                                               D) + np.where(z + d_z - data_z["z"].values < D,
                                                             z + d_z - data_z["z"].values, D)))
                        ESP_ = np.sum(data_z["ESP"].values *
                                     (np.where(data_z["x"].values - (x - d_x) < D, data_z["x"].values - (x - d_x),
                                               D) + np.where(x + d_x - data_z["x"].values < D,
                                                             x + d_x - data_z["x"].values, D)) *
                                     (np.where(data_z["y"].values - (y - d_y) < D, data_z["y"].values - (y - d_y),
                                               D) + np.where(y + d_y - data_z["y"].values < D,
                                                             y + d_y - data_z["y"].values, D)) *
                                     (np.where(data_z["z"].values - (z - d_z) < D, data_z["z"].values - (z - d_z),
                                               D) + np.where(z + d_z - data_z["z"].values < D,
                                                             z + d_z - data_z["z"].values, D)))
                        # Dt_=np.average(data_z["Dt"].values)
                        Dt.append(Dt_)
                        ESP.append(ESP_)
        # else:
        #     leng = 1
        #     sigma = leng
        #
        #     def gauss_func(d):
        #
        #         ans = 1 / (2 * np.pi * np.sqrt(2 * np.pi) * sigma ** 3) * leng ** 3 \
        #               * np.exp(-d ** 2 / (2 * sigma ** 2))
        #         return ans
        #
        #     D = 3 * sigma
        #     Dt = []
        #     for x in drop_dupl_x:
        #         data = data[x < data["x"] + D]
        #         data_x = data[x > data["x"] - D]
        #         for y in drop_dupl_y:
        #             data_x = data_x[y < data_x["y"] + D]
        #             data_y = data_x[y > data_x["y"] - D]
        #             for z in drop_dupl_z:
        #                 data_y = data_y[z < data_y["z"] + D]
        #                 data_z = data_y[z > data_y["z"] - D]
        #                 d = np.linalg.norm(data_z[["x", "y", "z"]].values - np.array([x, y, z]), axis=1)
        #                 Dt_ = np.average(data_z["Dt"].values, weights=np.where(d < sigma * 3, gauss_func(d), 0))
        #                 # Dt_ = np.sum(data_z["Dt"].values *np.where(d<3,exp(-d**2),0))
        #                 Dt.append(Dt_)
        print(time.time() - start)

        # else:
        #     data["x"] = np.round((data["x"].values - min(drop_dupl_x)) / (d_x * 2)).astype(int)
        #     data["y"] = np.round((data["y"].values - min(drop_dupl_y)) / (d_y * 2)).astype(int)
        #     data["z"] = np.round((data["z"].values - min(drop_dupl_z)) / (d_z * 2)).astype(int)
        #     # data=data[min(drop_dupl_x)-d_x<data["x"]]
        #     # data["Dt"] = data["Dt"].where(data["Dt"] < 10, 10)
        #     # #data[["x","y","z"]]=data[["x","y","z"]]
        #     # cond_x=[]
        #     # cond_y=[]
        #     # for x in drop_dupl_x:
        #     #     cond_x_=x-d_x<data["x"] and x+d_x>data["x"]
        #     #     cond_y_=
        #     start = time.time()
        #     Dt = []
        #     for x in range(len(drop_dupl_x)):
        #         data["x{}".format(x)] = data["x"] == x
        #     for y in range(len(drop_dupl_y)):
        #         data["y{}".format(y)] = data["y"] == y
        #     for z in range(len(drop_dupl_z)):
        #         data["z{}".format(z)] = data["z"] == z
        #     for x in range(len(drop_dupl_x)):
        #         # data_x=data[data["x"]==x]
        #         data_x = data[data["x{}".format(x)]]
        #         for y in range(len(drop_dupl_y)):
        #             # data_xy = data_x[data_x["y"] == y]
        #             data_xy = data_x[data_x["y{}".format(y)]]
        #             for z in range(len(drop_dupl_z)):
        #                 # Dt_ = np.sum(data_xy[data_xy["z{}".format(z)]]["Dt"].values) * (0.2 * 0.52917721067) ** 3
        #                 Dt_ = np.average(data_xy[data_xy["z{}".format(z)]]["Dt"].values)
        #                 Dt.append(Dt_)
        print(time.time() - start)

        dfp["Dt"] = np.nan_to_num(Dt)
        dfp["ESP"] = np.nan_to_num(ESP)
        dfp.to_pickle(outfilename)


def PF(input):
    cube_dir_name, dfp, mol, grid_coordinates = input
    print(mol)
    pkl_to_featurevalue(cube_dir_name, dfp, mol, grid_coordinates)


if __name__ == '__main__':
    # time.sleep(60*60*6)
    dfs = []
    for path in glob.glob("../arranged_dataset/*.xlsx"):
        df = pd.read_excel(path)
        print(len(df))
        dfs.append(df)
    dfs = pd.concat(dfs).dropna(subset=['smiles']).drop_duplicates(subset=["smiles"])
    print("len=",len(dfs))
    dfs["mol"] = dfs["smiles"].apply(calculate_conformation.get_mol)

    for param_name in sorted(glob.glob("../parameter/cube_to_grid/cube_to_grid0.5004022.txt")):
        df = copy.deepcopy(dfs)
        with open(param_name, "r") as f:
            param = json.loads(f.read())
        print(param)
        df = df[[os.path.isdir(param["cube_dir_name"] + "/" + mol.GetProp("InchyKey")) for mol in df["mol"]]]
        df["mol"].apply(
            lambda mol: calculate_conformation.read_xyz(mol, param["cube_dir_name"] + "/" + mol.GetProp("InchyKey")))
        dfp = pd.read_csv(param["grid_coordinates"] + "/coordinates.csv")
        print(dfp)
        inputs = []
        for mol in df["mol"]:
            os.makedirs(param["grid_coordinates"] + "/" + mol.GetProp("InchyKey"), exist_ok=True)
            input = param["cube_dir_name"] + "/" + mol.GetProp("InchyKey"), dfp, mol, param[
                "grid_coordinates"] + "/" + mol.GetProp("InchyKey")
            inputs.append(input)
            # pkl_to_featurevalue(param["cube_dir_name"]+"/"+mol.GetProp("InchyKey"), dfp, mol, param["grid_coordinates"]+"/"+mol.GetProp("InchyKey"))
        p = Pool(8)
        p.map(PF, inputs)
