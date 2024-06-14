import glob
import multiprocessing
import os
import time
import warnings
from multiprocessing import Pool
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import calculate_conformation

warnings.simplefilter('ignore')


def pkl_to_featurevalue(dir_name, dfp, mol, conf, out_name):  # グリッド特徴量を計算　ボルツマン分布の加重はしないでデータセットの区別もなし。
    conf=mol.GetConformer(conf)

    # for conf in mol.GetConformers():
    if True:
        drop_dupl_x = dfp.drop_duplicates(subset="x").sort_values('x')["x"]
        drop_dupl_y = dfp.drop_duplicates(subset="y").sort_values('y')["y"]
        drop_dupl_z = dfp.drop_duplicates(subset="z").sort_values('z')["z"]

        d_x = (drop_dupl_x.iloc[1] - drop_dupl_x.iloc[0]) / 2
        d_y = (drop_dupl_y.iloc[1] - drop_dupl_y.iloc[0]) / 2
        d_z = (drop_dupl_z.iloc[1] - drop_dupl_z.iloc[0]) / 2

        # 入力：幅
        # dfpをdata.pklの最大・最小から決定
        outfilename = "{}/data{}.pkl".format(out_name, conf.GetId())
        # if os.path.isfile(outfilename):
        #     return None
        filename = "{}/data{}.pkl".format(dir_name, conf.GetId())
        data = pd.read_pickle(filename)
        data["Dt"] = data["Dt"].where(data["Dt"] < 100, 100)
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
        print(outfilename,time.time() - start)

        dfp["Dt"] = np.nan_to_num(Dt)
        dfp["ESP"] = np.nan_to_num(ESP)
        dfp.to_pickle(outfilename)


def PF(input):
    cube_dir_name, dfp, mol, conf, grid_coordinates = input
    pkl_to_featurevalue(cube_dir_name, dfp, mol, conf, grid_coordinates)

def generate_grid_points(range,step):
    x_min,x_max,y_max,z_max=range
    x_step=y_step=z_step=step
    # y_max=(y_max-y_min)/2
    # z_max=(z_max-z_min)/2
    # x_max=((x_max // step+1).astype(int)-0.5) * step
    # y_max=((x_max // step+1).astype(int)-0.5) * step
    # z_max=((z_max // step+1).astype(int)-0.5) * step
    # x_min=((x_min // step-1).astype(int)+0.5) * step
    y_min=-y_max
    z_min=-z_max
    # それぞれの範囲内で格子点を生成
    x_values = np.arange(x_min, x_max + x_step, x_step)
    y_values = np.arange(y_min, y_max + y_step, y_step)
    z_values = np.arange(z_min, z_max + z_step, z_step)
    # 全ての組み合わせを作成
    points = np.array(np.meshgrid(x_values, y_values, z_values)).T.reshape(-1, 3)
    
    # DataFrameに変換
    df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    return df

def make_penalty(l, sigma, interval, out_dir_name):
    l = np.array(l)
    
    xyz = l[(l[:, 1] > 0) & (l[:, 2] > 0)]

    d = np.array([np.linalg.norm(xyz - _, axis=1) for _ in xyz])
    d_y = np.array([np.linalg.norm(xyz - _ * np.array([1, -1, 1]), axis=1) for _ in xyz])
    d_z = np.array([np.linalg.norm(xyz - _ * np.array([1, 1, -1]), axis=1) for _ in xyz])
    d_yz = np.array([np.linalg.norm(xyz - _ * np.array([1, -1, -1]), axis=1) for _ in xyz])

    def gauss_func(d):
        leng = interval
        ans = 1 / (2 * np.pi * np.sqrt(2 * np.pi) * sigma ** 3) * leng ** 3 \
              * np.exp(-d ** 2 / (2 * sigma ** 2))
        return ans

    penalty_ = np.where(d < sigma * 3, gauss_func(d), 0)
    penalty_y = np.where(d_y < sigma * 3, gauss_func(d_y), 0)
    penalty_z = np.where(d_z < sigma * 3, gauss_func(d_z), 0)
    penalty_yz = np.where(d_yz < sigma * 3, gauss_func(d_yz), 0)

    penalty = penalty_ + penalty_y - penalty_z - penalty_yz
    # penalty = penalty  / np.max(np.sum(penalty_, axis=0))
    # penalty = np.identity(penalty.shape[0])-penalty
    penalty = np.identity(penalty.shape[0])-2*penalty+penalty.T@penalty
    # penalty=np.concatenate([penalty, np.identity(penalty.shape[0])], 0)
    # print(np.sum(penalty,axis=1))
    # penalty=np.identity(penalty.shape[0])
    # filename = out_dir_name + "/penalty{}.npy".format(n)  # + "/" + param["grid_coordinates_dir"]
    # np.save(filename, penalty)
    filename = out_dir_name + "/1ptp{:.2f}.npy".format(sigma)
    ptp=penalty#.T@penalty
    np.save(filename,ptp.astype("float32"))
    print(filename)

    # penalty_L = []
    # for _ in range(2):
    #     penalty_L_ = []
    #     for __ in range(2):
    #         if _ == __:
    #             penalty_L_.append(penalty)
    #         else:
    #             penalty_L_.append(np.zeros(penalty.shape))
    #     penalty_L.append(penalty_L_)
    # penalty=np.block(penalty_L)
    # ptp=penalty.T@penalty

    # filename = out_dir_name + "/2ptp{:.2f}.npy".format(sigma)
    # np.save(filename, ptp.astype("float32"))
    # print(filename)

def read_pickle(dir):
    df=pd.read_pickle(dir)
    df=df[df["Dt"]>10e-3]
    ans=df[["x"]].min().to_list()+df[["x"]].max().to_list()+df[["y","z"]].abs().max().to_list()
    return ans


def histgram(cube_dir,name,out_dir_name):
    # rcParams['figure.figsize'] = 5, 5
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    for smiles in df["smiles"]:  # [:100]:
        print(smiles)
        mol = calculate_conformation.get_mol(smiles)
        input_dirs_name = cube_dir + "/" + mol.GetProp("InchyKey")
        i=0
        while os.path.isfile("{}/data{}.pkl".format(input_dirs_name, i)):
            data=pd.read_pickle("{}/data{}.pkl".format(input_dirs_name, i))[name].values
            i+=1
            axs[0].hist(data, bins=100, alpha=0.1, histtype='stepfilled', color='r')
            axs[1].hist(data, bins=np.logspace(np.log10(np.min(data)), np.log10(np.max(data)), 100), alpha=0.01, histtype='stepfilled', log=True, color='r')
    # plt.hist(df_[name], bins=np.logspace(-10, 4, 100), alpha=0.1, histtype='stepfilled', log=True, color='r')
    # axs[0].hist(data, bins=50, color='blue', edgecolor='black')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('electron density [e/$\mathrm{Bohr^3}$]')
    axs[0].set_ylabel('Frequency')

    # axs[1].hist(data, bins=np.logspace(np.log10(min(data)), np.log10(max(data)), 50), color='green', edgecolor='black')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_xlabel('electron density [e/$\mathrm{Bohr^3}$]')
    axs[1].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(out_dir_name, dpi=300)
    plt.show()

if __name__ == '__main__':

    # time.sleep(60*60*24*2)
    interval = 0.25
    dfs = []
    for path in glob.glob("../arranged_dataset/*.xlsx"):
        df = pd.read_excel(path)
        print(len(df))
        dfs.append(df)
    df = pd.concat(dfs).dropna(subset=['smiles']).drop_duplicates(subset=["smiles"])
    df["mol"] = df["smiles"].apply(calculate_conformation.get_mol)
    #cubeのディレクトリを指定。
    dir="F:/wB97X-D_def2-TZVP20240416"
    #読み込んでヒストグラムを出力。（メモリに注意）
    df = df[[os.path.isdir(dir + "/" + mol.GetProp("InchyKey")) for mol in df["mol"]]]
    df["mol"].apply(
        lambda mol: calculate_conformation.read_xyz(mol, dir + "/" + mol.GetProp("InchyKey")))
    
    # histgram(dir,"Dt","C:/Users/poclabws/result/histgram.png")
    print("histgram")
    l=[]
    for mol in df["mol"]:#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for conf in mol.GetConformers():
            l.append(dir+ "/" + mol.GetProp("InchyKey")+ "/data{}.pkl".format(conf.GetId()))
    p = multiprocessing.Pool(processes=50)
    l=p.map(read_pickle,l)

    # max_indices = np.argmax(np.abs(l), axis=0)
    # arr = np.array(l)[max_indices, range(np.array(l).shape[1])]
    
    arr=np.average(l,axis=0)
    rounded_arr = np.sign(arr) * np.ceil(np.abs(arr) /interval) * interval
    ans = rounded_arr - interval/2 * np.sign(rounded_arr)
    print(ans)
    dfp=generate_grid_points(ans,interval).sort_values(['x', 'y', "z"], ascending=[True, True, True])
    # out_dir_name="../../../penalty_20240606"
    out_dir_name="../../../grid_coordinates/20240606_"+str(interval).replace('.', '_')
    os.makedirs(out_dir_name,exist_ok=True)
    
    
    #if False:#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for n in range(5):
        sigma = interval *2** n
        make_penalty(dfp[["x","y","z"]].values, sigma, interval, out_dir_name)
    # raise ValueError
    # for param_name in sorted(glob.glob("../parameter/cube_to_grid/cube_to_grid0.500510.txt"),reverse=True):
    # df = copy.deepcopy(dfs)

    # df = df[[os.path.isdir(dir + "/" + mol.GetProp("InchyKey")) for mol in df["mol"]]]
    # df["mol"].apply(
    #     lambda mol: calculate_conformation.read_xyz(mol, dir + "/" + mol.GetProp("InchyKey")))
    # dfp = pd.read_csv(param["grid_coordinates"] + "/coordinates.csv")
    dfp_yz = dfp[(dfp["y"] > 0) & (dfp["z"] > 0)].sort_values(['x', 'y', "z"], ascending=[True, True, True])
    dfp_yz.to_csv((out_dir_name + "/coordinates_yz.csv"))
    print("len=", len(df))
    inputs = []
    for mol in df["mol"]:#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        os.makedirs(out_dir_name + "/" + mol.GetProp("InchyKey"), exist_ok=True)
        for conf in mol.GetConformers():
            conf =conf.GetId()
            input = dir + "/" + mol.GetProp("InchyKey"), dfp, mol, conf, out_dir_name + "/" + mol.GetProp("InchyKey")
            inputs.append(input)
        # pkl_to_featurevalue(param["cube_dir_name"]+"/"+mol.GetProp("InchyKey"), dfp, mol, param["grid_coordinates"]+"/"+mol.GetProp("InchyKey"))
    print(len(inputs))
    p = Pool(60)
    p.map(PF, inputs)
    print("END")
