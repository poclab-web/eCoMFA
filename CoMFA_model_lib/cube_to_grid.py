import glob
import multiprocessing
import os
import time
import warnings
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import numpy as np
import pandas as pd

import calculate_conformation

warnings.simplefilter('ignore')


def pkl_to_featurevalue(input):  # グリッド特徴量を計算　ボルツマン分布の加重はしないでデータセットの区別もなし。
    filename, dfp, outfilename=input
    drop_dupl_x = dfp.drop_duplicates(subset="x").sort_values('x')["x"].values
    drop_dupl_y = dfp.drop_duplicates(subset="y").sort_values('y')["y"].values
    drop_dupl_z = dfp.drop_duplicates(subset="z").sort_values('z')["z"].values

    d_x = (drop_dupl_x[1] - drop_dupl_x[0]) / 2
    d_y = (drop_dupl_y[1] - drop_dupl_y[0]) / 2
    d_z = (drop_dupl_z[1] - drop_dupl_z[0]) / 2

    data = pd.read_pickle(filename)[["x","y","z","Dt","ESP"]].astype(np.float32)
    data["Dt"] = data["Dt"].where(data["Dt"] < 10, 10)
    data["ESP"] = data["ESP"].where(data["Dt"] < 0.001, 0)

    start = time.time()

    D = 0.1 * 0.52917720859
    
    Dt_list = []
    ESP_list = []

    for x in drop_dupl_x:
        mask_x = (x - d_x < data["x"] + D) & (x + d_x > data["x"] - D)
        data_x = data[mask_x]
        for y in drop_dupl_y:
            mask_y = (y - d_y < data_x["y"] + D) & (y + d_y > data_x["y"] - D)
            data_y = data_x[mask_y]
            for z in drop_dupl_z:
                mask_z = (z - d_z < data_y["z"] + D) & (z + d_z > data_y["z"] - D)
                data_z = data_y[mask_z]

                if not data_z.empty:
                    x_diffs = np.minimum(np.abs(data_z["x"].values - (x - d_x)), D) + np.minimum(np.abs(x + d_x - data_z["x"].values), D)
                    y_diffs = np.minimum(np.abs(data_z["y"].values - (y - d_y)), D) + np.minimum(np.abs(y + d_y - data_z["y"].values), D)
                    z_diffs = np.minimum(np.abs(data_z["z"].values - (z - d_z)), D) + np.minimum(np.abs(z + d_z - data_z["z"].values), D)

                    Dt_ = np.sum(data_z["Dt"].values * x_diffs * y_diffs * z_diffs)
                    ESP_ = np.sum(data_z["ESP"].values * x_diffs * y_diffs * z_diffs)
                else:
                    Dt_ = 0
                    ESP_ = 0

                Dt_list.append(Dt_)
                ESP_list.append(ESP_)

    dfp["Dt"] = np.nan_to_num(Dt_list)
    dfp["ESP"] = np.nan_to_num(ESP_list)
    dfp.to_pickle(outfilename)
    print(outfilename, time.time() - start)


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
def gauss_func(d,interval,sigma):
    leng = interval
    ans=1 / (2 * np.pi * np.sqrt(2 * np.pi) * sigma ** 3) * leng ** 3 \
            * np.exp(-d ** 2 / (2 * sigma ** 2))
    return ans
def make_penalty(input):
    l, sigma, interval, out_dir_name=input
    
    l = np.array(l)
    
    xyz = l[(l[:, 1] > 0) & (l[:, 2] > 0)]

    # # Calculate distances using broadcasting
    # diff = xyz[:, np.newaxis, :] - xyz[np.newaxis, :, :]
    # d = np.linalg.norm(diff, axis=-1)

    # # Calculate distances with transformations
    # d_y = np.linalg.norm(diff * np.array([1, -1, 1]), axis=-1)
    # d_z = np.linalg.norm(diff * np.array([1, 1, -1]), axis=-1)
    # d_yz = np.linalg.norm(diff * np.array([1, -1, -1]), axis=-1)
    d = np.array([np.linalg.norm(xyz - _, axis=1) for _ in xyz])
    d_y = np.array([np.linalg.norm(xyz - _ * np.array([1, -1, 1]), axis=1) for _ in xyz])
    d_z = np.array([np.linalg.norm(xyz - _ * np.array([1, 1, -1]), axis=1) for _ in xyz])
    d_yz = np.array([np.linalg.norm(xyz - _ * np.array([1, -1, -1]), axis=1) for _ in xyz])
        
    penalty_ = np.where(d < sigma * 3, gauss_func(d,interval,sigma), 0)
    penalty_y = np.where(d_y < sigma * 3, gauss_func(d_y,interval,sigma), 0)
    penalty_z = np.where(d_z < sigma * 3, gauss_func(d_z,interval,sigma), 0)
    penalty_yz = np.where(d_yz < sigma * 3, gauss_func(d_yz,interval,sigma), 0)
    penalty = penalty_ + penalty_y - penalty_z - penalty_yz
    print(sigma,np.std(penalty.astype("float32")))
    # Compute PTP matrices and save them
    def save_ptp(penalty, sigma, out_dir_name, suffix):
        ptp = np.identity(penalty.shape[0]) - 2 * penalty + penalty.T @ penalty
        ptp=ptp.astype("float32")
        # ptp=np.eye(ptp.shape[0])
        filename = f"{out_dir_name}/{suffix}ptp{sigma:.2f}.npy"
        np.save(filename, ptp)
        eigenvalues,P=eigh(ptp, overwrite_a=True,check_finite=False)
        print((P/np.sqrt(eigenvalues)).dtype)
        np.save(f"{out_dir_name}/{suffix}eig{sigma:.2f}.npy",P/np.sqrt(eigenvalues))

    save_ptp(penalty, sigma, out_dir_name, "1")

    # Create penalty_2 and compute its PTP matrix
    # penalty_2 = np.block([[penalty, np.zeros_like(penalty)], [np.zeros_like(penalty), np.zeros_like(penalty)]])
    # save_ptp(penalty_2, sigma, out_dir_name, "2")
    # penalty = np.identity(penalty.shape[0])-2*penalty+penalty.T@penalty
    # filename = out_dir_name + "/1ptp{:.2f}.npy".format(sigma)
    # ptp=np.identity(penalty.shape[0])-2*penalty+penalty.T@penalty#.T@penalty
    # np.save(filename,ptp.astype("float32"))
    # print(filename)

    # # penalty_L = []
    # # for _ in range(2):
    # #     penalty_L_ = []
    # #     for __ in range(2):
    # #         if _ == __:
    # #             penalty_L_.append(penalty)
    # #         else:
    # #             penalty_L_.append(np.zeros_like(penalty))
    # #     penalty_L.append(penalty_L_)
    # penalty_2=np.block([[penalty,np.zeros_like(penalty)],[np.zeros_like(penalty),penalty]])
    
    # ptp=np.identity(penalty_2.shape[0])-2*penalty_2+penalty_2.T@penalty_2

    # filename = out_dir_name + "/2ptp{:.2f}.npy".format(sigma)
    # np.save(filename, ptp.astype("float32"))
    # print(filename)

def read_pickle(dir):
    df=pd.read_pickle(dir)[["Dt", "x", "y", "z"]]
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
    start = time.perf_counter()  # 計測開始
    # time.sleep(60*60*24*2)
    interval = 0.25
    # dfs = []
    # for path in glob.glob("../all_dataset/*.xlsx"):
    #     df = pd.read_excel(path)
    #     print(len(df))
    #     dfs.append(df)
    df = pd.concat([pd.read_excel(path) for path in glob.glob("../all_dataset/*.xlsx")]).dropna(subset=['smiles']).drop_duplicates(subset=["smiles"])
    df["mol"] = df["smiles"].apply(calculate_conformation.get_mol)
    #cubeのディレクトリを指定。
    dir="C:/Users/poclabws/calculation/wB97X-D_def2-TZVP20240416"#"F:/wB97X-D_def2-TZVP20240416"
    df = df[[os.path.isdir(dir + "/" + mol.GetProp("InchyKey")) for mol in df["mol"]]]
    df["mol"].apply(
        lambda mol: calculate_conformation.read_xyz(mol, dir + "/" + mol.GetProp("InchyKey")))
    
    # histgram(dir,"Dt","C:/Users/poclabws/result/histgram.png")
    print("histgram")
    l=[]
    for mol in df["mol"]:
        for conf in mol.GetConformers():
            l.append(dir+ "/" + mol.GetProp("InchyKey")+ "/data{}.pkl".format(conf.GetId()))
    p = multiprocessing.Pool(processes=10)
    l=p.map(read_pickle,l)

    arr=np.average(l,axis=0)
    
    ans=(np.round(arr / interval-0.5)+0.5) * interval
    print(ans)
    dfp=generate_grid_points(ans,interval).sort_values(['x', 'y', "z"], ascending=[True, True, True])
    out_dir_name="../../../grid_coordinates/20240606_"+str(interval).replace('.', '_')
    os.makedirs(out_dir_name,exist_ok=True)
    inputs=[]
    for n in range(5):
        sigma = interval *2** n
        input=dfp[["x","y","z"]].values, sigma, interval, out_dir_name
        inputs.append(input)
    p = Pool(1)
    p.map(make_penalty, inputs)
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
    for mol in df["mol"]:
        os.makedirs(out_dir_name + "/" + mol.GetProp("InchyKey"), exist_ok=True)
        for conf in mol.GetConformers():
            confid =conf.GetId()
            input = dir + "/" + mol.GetProp("InchyKey")+"/data{}.pkl".format(confid), dfp, out_dir_name + "/" + mol.GetProp("InchyKey")+"/data{}.pkl".format(confid)
            inputs.append(input)
        # pkl_to_featurevalue(param["cube_dir_name"]+"/"+mol.GetProp("InchyKey"), dfp, mol, param["grid_coordinates"]+"/"+mol.GetProp("InchyKey"))
    print(len(inputs))
    p = Pool(50)
    p.map(pkl_to_featurevalue, inputs)
    print("END")
    end = time.perf_counter()  # 計測終了
    print('Finish{:.2f}'.format(end - start))