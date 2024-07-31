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

    data = pd.read_pickle(filename)[["x","y","z","Dt","ESP","LUMO","LOL"]].astype(np.float32)
    data["ESP"] = data["ESP"].where(data["Dt"] < 0.001, 0)
    data[["DUAL"]] = data[["LUMO"]].applymap(np.abs).where(data["Dt"] < 0.001, 0)
    data[["Dt"]]=data[["Dt"]].applymap(np.sqrt)
    # data["Dt"] = data["Dt"].where(data["Dt"] < 10, 10)
    # data["DUAL"]=data["Dt"]*data["LOL"]
    # data["Dt"]=data["Dt"]*(1-data["LOL"])

    start = time.time()

    D = np.float32(0.1 * 0.52917720859)
    
    Dt_list = []
    ESP_list = []
    DUAL_list= []
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
                    DUAL_ = np.sum(data_z["DUAL"].values * x_diffs * y_diffs * z_diffs)
                else:
                    Dt_ = 0
                    ESP_ = 0
                    DUAL_=0

                Dt_list.append(Dt_)
                ESP_list.append(ESP_)
                DUAL_list.append(DUAL_)

    dfp["Dt"] = np.nan_to_num(Dt_list)
    dfp["ESP"] = np.nan_to_num(ESP_list)
    dfp["DUAL"]=np.nan_to_num(DUAL_list)
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
    df = pd.DataFrame(points, columns=['x', 'y', 'z']).astype(np.float32)
    return df

def gauss_func(d,interval,sigma):
    # leng = interval
    # ans=1 / (2 * np.pi * np.sqrt(2 * np.pi) * sigma ** 3) * leng ** 3 \
    #         * np.exp(-d ** 2 / (2 * sigma ** 2))
    ans=np.exp(-d**2/(2 * sigma ** 2))
    # ans=0.5**(d/sigma)
    # ans=np.where(d==0,1,0)
    return ans

def make_penalty(input):
    l, sigma, interval, out_dir_name=input
    
    # l = np.array(l)
    
    xyz = l[(l[:, 1] > 0) & (l[:, 2] > 0)]
    # print(xyz)
    # # Calculate distances using broadcasting
    # diff = xyz[:, np.newaxis, :] - xyz[np.newaxis, :, :]
    # d = np.linalg.norm(diff, axis=-1)

    # # Calculate distances with transformations
    # d_y = np.linalg.norm(diff * np.array([1, -1, 1]), axis=-1)
    # d_z = np.linalg.norm(diff * np.array([1, 1, -1]), axis=-1)
    # d_yz = np.linalg.norm(diff * np.array([1, -1, -1]), axis=-1)
    arr = np.array([np.linalg.norm(xyz - _, axis=1) for _ in xyz])
    arr_y = np.array([np.linalg.norm(xyz - _ * np.array([1, -1, 1]), axis=1) for _ in xyz])
    arr_z = np.array([np.linalg.norm(xyz - _ * np.array([1, 1, -1]), axis=1) for _ in xyz])
    arr_yz = np.array([np.linalg.norm(xyz - _ * np.array([1, -1, -1]), axis=1) for _ in xyz])
    cutoff=10
    arr = np.where(arr < sigma * cutoff, gauss_func(arr,interval,sigma), 0)
    arr_y=np.where(arr_y < sigma * cutoff, gauss_func(arr_y,interval,sigma), 0)
    arr_z=np.where(arr_z < sigma * cutoff, gauss_func(arr_z,interval,sigma), 0)
    arr_yz=np.where(arr_yz < sigma * cutoff, gauss_func(arr_yz,interval,sigma), 0)
    # arr=arr+arr_y-arr_z-arr_yz
    # arr=np.identity(xyz.shape[0])/(np.linalg.norm(xyz-np.average(xyz,axis=0)*np.array([1,0,1]),axis=1)+1)
    d=np.linalg.norm(xyz-np.average(xyz,axis=0)*np.array([1,0,1]),axis=1)
    print(d.shape)
    arr=np.exp(-d**2/(2*(sigma*5)**2))
    print(np.max(arr),np.min(arr),arr.shape)
    d_=np.array([np.linalg.norm(xyz - _, axis=1) for _ in xyz])
    arr=np.outer(arr,arr)*np.exp(-d_**2/(2*(sigma*5)**2))#+np.identity(arr.shape[0])
    # np.fill_diagonal(arr, np.diagonal(arr) * 2)
    print(arr.shape)
    print(np.max(arr),np.min(arr),arr.shape)
    
    # arr=np.array([np.exp(-d**2-_**2) for _ in d])
    # arr=arr+ np.eye(arr.shape[0]) * 1e-10#arr_y.T@arr_y-arr_z.T@arr_z-arr_yz.T@arr_yz+
    # arr=np.where((arr > 0.99*interval) & (arr < 1.01*interval),1,0)+np.where((arr_y > 0.99*interval) & (arr_y < 1.01*interval),1,0)-np.where((arr_z > 0.99*interval) & (arr_z < 1.01*interval),1,0)-np.where((arr_yz > 0.99*interval) & (arr_yz < 1.01*interval),1,0)
    # arr = arr + arr_y - arr_z - arr_yz
    print(np.sum(arr)/arr.shape[0])

    # ヒートマップを描画
    plt.figure(figsize=(10, 8))
    import seaborn as sns
    sns.heatmap(arr[:100,:100], annot=False, cmap='viridis')
    plt.title('Heatmap of Matrix')
    plt.savefig(f"{out_dir_name}/matrix.png")
    # Compute PTP matrices and save them
    def save_ptp(penalty, sigma, out_dir_name, suffix):
        # ptp = (6**2)*(1+sigma)*np.identity(penalty.shape[0]) - 2 * 6*penalty + penalty.T @ penalty
        # ptp = 36*np.identity(penalty.shape[0]) - 2 * 6*penalty + penalty.T @ penalty
        # penalty=penalty.astype(np.float32)
        # ptp=np.linalg.inv(penalty)#.astype(np.float32)
        # filename = f"{out_dir_name}/{suffix}ptp{sigma:.2f}.npy"
        # np.save(filename, ptp)
        # penalty = (penalty + penalty.T) / 2
        # penalty=penalty.T@penalty
        eigenvalues,eigenvectors=eigh(penalty, overwrite_a=True,check_finite=False)

        # 固有値を非負に調整
        print(np.sum(eigenvalues<0),np.shape(eigenvalues))
        # eigenvalues[eigenvalues < 0] = 0
        eigenvalues=np.abs(eigenvalues)
        # 非負固有値を持つ対角行列の構築
        D = np.diag(eigenvalues)

        # 調整後の行列の再構築
        eigenvectors = eigenvectors @ D @ eigenvectors.T

        if np.isnan(eigenvalues).any() or np.isnan(eigenvectors).any():
            raise ValueError("The matrix contains Inf values.")
        print(np.min(eigenvalues))
        # print((P/np.sqrt(eigenvalues)).dtype)
        P=eigenvectors*np.sqrt(eigenvalues)

        print(np.isnan(P).any())
        np.save(f"{out_dir_name}/{suffix}eig{sigma:.2f}.npy",P)
        np.save(f"{out_dir_name}/eigenvalues.npy",eigenvalues)
        np.save(f"{out_dir_name}/eigenvectors.npy",1/eigenvectors)

    save_ptp(arr, sigma, out_dir_name, "1")

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
    df=pd.read_pickle(dir)[["Dt", "x", "y", "z"]].astype(np.float32)
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
    interval = 0.50
    # dfs = []
    # for path in glob.glob("../all_dataset/*.xlsx"):
    #     df = pd.read_excel(path)
    #     print(len(df))
    #     dfs.append(df)
    df = pd.concat([pd.read_excel(path) for path in glob.glob("../all_dataset/*.xlsx")]).dropna(subset=['smiles']).drop_duplicates(subset=["smiles"])
    df["mol"] = df["smiles"].apply(calculate_conformation.get_mol)
    #cubeのディレクトリを指定。
    dir="D:/calculation/wB97X-D_def2-TZVP20240416"#"C:/Users/poclabws/calculation/wB97X-D_def2-TZVP20240416"#"F:/wB97X-D_def2-TZVP20240416"
    df = df[[os.path.isdir(dir + "/" + mol.GetProp("InchyKey")) for mol in df["mol"]]]
    df["mol"].apply(
        lambda mol: calculate_conformation.read_xyz(mol, dir + "/" + mol.GetProp("InchyKey")))
    
    # histgram(dir,"Dt","C:/Users/poclabws/result/histgram.png")
    print("histgram")
    if False:
        l=[]
        for mol in df["mol"]:
            for conf in mol.GetConformers():
                l.append(dir+ "/" + mol.GetProp("InchyKey")+ "/data{}.pkl".format(conf.GetId()))
        p = multiprocessing.Pool(processes=10)
        l=p.map(read_pickle,l)
        l=np.array(l)
        arr=np.average(l,axis=0)
        # arr=l[np.argmax(np.abs(l),axis=0),np.arange(l.shape[1])]
        
        ans=(np.round(arr / interval-0.5)+0.5) * interval
    else:
        ans=[-4.75,  3.25,  4.25,  6.75]
    print(ans)
    dfp=generate_grid_points(ans,interval).sort_values(['x', 'y', "z"], ascending=[True, True, True])
    out_dir_name="../../../grid_coordinates/20240606_"+str(interval).replace('.', '_')
    os.makedirs(out_dir_name,exist_ok=True)
    inputs=[]
    for n in range(-1,2):
        sigma = interval *2** n
        input=dfp[["x","y","z"]].values, sigma, interval, out_dir_name
        inputs.append(input)
    p = Pool(1)
    p.map(make_penalty, inputs)

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
    p = Pool(60)
    # p.map(pkl_to_featurevalue, inputs)
    print("END")
    end = time.perf_counter()  # 計測終了
    print('Finish{:.2f}'.format(end - start))