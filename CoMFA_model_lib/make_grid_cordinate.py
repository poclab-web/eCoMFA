import os
from itertools import product

import numpy as np
import pandas as pd


def make_grid_coordinate(orient, size, interval):
    out_dir_name = "../../../grid_coordinates" + "/{} {} {} 20240510".format(" ".join(map(str, orient)),
                                                                        " ".join(map(str, size)), interval)
    l = []
    for x in product(range(size[0]), range(size[1]), range(size[2])):
        ans = np.array(orient) + np.array(x) * interval
        ans = ans.tolist()
        l.append(ans)
    # ans=np.array(product(range(size[0]),range(size[1]),range(size[2])))
    # l = np.sort(np.meshgrid(range(size[0]), range(size[1]), range(size[2]))).T.reshape(-1, 3) * interval + np.array(
    #     orient)
    # print(l)
    dfp = pd.DataFrame(data=l, columns=["x", "y", "z"]).sort_values(['x', 'y', "z"], ascending=[True, True, True])
    # print(dfp)
    os.makedirs(out_dir_name, exist_ok=True)  # + "/" + param["grid_coordinates_dir"]

    filename = out_dir_name + "/coordinates.csv"  # + "/" + param["grid_coordinates_dir"]

    dfp.to_csv(filename)
    print(filename)
    dfp_yz = dfp[(dfp["y"] > 0) & (dfp["z"] > 0)].sort_values(['x', 'y', "z"], ascending=[True, True, True])

    dfp_yz.to_csv((out_dir_name + "/coordinates_yz.csv"))
    # print(dfp_yz)

    # list = np.logspace(0, 13, num=14, base=2).astype(int)  # 2**np.arange(10)#[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    list = np.logspace(0, 10, num=11, base=2)  # 2**np.arange(10)#[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    df = pd.DataFrame(list, columns=["lambda"])
    df["n_components"]=np.arange(1,12)
    df.to_csv(out_dir_name + "/penalty_param.csv")
    for n in range(5):
        sigma = interval *2** n
        make_penalty(l, sigma, interval, out_dir_name)
    print("END")

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
    penalty = penalty - np.identity(penalty.shape[0])  #* np.sum(penalty, axis=1)
    # print(np.sum(penalty,axis=1))
    # penalty=np.identity(penalty.shape[0])
    # filename = out_dir_name + "/penalty{}.npy".format(n)  # + "/" + param["grid_coordinates_dir"]
    # np.save(filename, penalty)
    filename = out_dir_name + "/1ptp{:.2f}.npy".format(sigma)
    ptp=penalty.T@penalty

    np.save(filename,ptp.astype("float32"))
    penalty_L = []
    for _ in range(2):
        penalty_L_ = []
        for __ in range(2):
            if _ == __:
                penalty_L_.append(penalty)
            else:
                penalty_L_.append(np.zeros(penalty.shape))
        penalty_L.append(penalty_L_)
    penalty=np.block(penalty_L)
    ptp=penalty.T@penalty

    filename = out_dir_name + "/2ptp{:.2f}.npy".format(sigma)
    print(filename)
    np.save(filename, ptp.astype("float32"))
    print(filename)

if __name__ == '__main__':
    # [-4.75 -2.75 -4.75] [14 12 20] 0.5
    # orient = [-4.75, -2.75, -4.75]
    # size = [10 + 4, 6 * 2, 10 * 2]
    # interval = 0.5
    # make_grid_coordinate(orient, size, interval)
    # [-5.0 -3.0 -5.0] [18 16 26] 0.4
    # orient = [-5.0, -3.0, -5.0]
    # size = [13 + 5, 8 * 2, 13 * 2]
    # interval = 0.4
    # make_grid_coordinate(orient, size, interval)
    # # [[-4.65, -2.85, -4.65],[22, 20, 32],0.3]
    # orient = [-4.65, -2.85, -4.65]
    # size = [16 + 6, 10 * 2, 16 * 2]
    # interval = 0.3
    # make_grid_coordinate(orient, size, interval)
    # # [-5.0 -1.8 -5.0] [18 10 26] 0.4
    # orient = [-5.0, -1.8, -5.0]
    # size = [13 + 5, 5 * 2, 13 * 2]
    # interval = 0.4
    # make_grid_coordinate(orient, size, interval)
    #
    # # [-5.8 -3.8 -5.8] [20 20 30] 0.4
    # orient = [-5.8, -3.8, -5.8]
    # size = [15 + 5, 10 * 2, 15 * 2]
    # interval = 0.4
    # make_grid_coordinate(orient, size, interval)
    #
    # # [-5.8 -3.4 -5.8] [20 18 30] 0.4
    # orient = [-5.8, -3.4, -5.8]
    # size = [15 + 5, 9 * 2, 15 * 2]
    # interval = 0.4
    # make_grid_coordinate(orient, size, interval)

    # # [-5.8 -3.0 -5.8] [20 16 30] 0.4
    # orient = [-5.8, -3.0, -5.8]
    # size = [15 + 5, 8 * 2, 15 * 2]
    # interval = 0.4
    # make_grid_coordinate(orient, size, interval)

    # orient = [-4.250, -2.750, -4.750]
    # size = [14, 12 , 20 ]
    # interval = 0.50
    # make_grid_coordinate(orient, size, interval)
    # orient = [-4.250, -3.750, -4.750]
    # size = [14, 16 , 20 ]
    # interval = 0.50
    # make_grid_coordinate(orient, size, interval)
    # orient = [-4.250, -4.750, -4.750]
    # size = [14, 20, 20]
    # interval = 0.50
    # make_grid_coordinate(orient, size, interval)
    orient = [-4.250, -4.750, -5.750]
    size = [14, 20, 24]
    interval = 0.50
    make_grid_coordinate(orient, size, interval)

    orient = [-4.750, -4.750, -9.750]
    size = [20, 20, 40]
    interval = 0.50
    make_grid_coordinate(orient, size, interval)
    if False:
        orient = [-4.375, -4.875, -5.875]
        size = [28, 20 * 2, 24 * 2]
        interval = 0.25
        make_grid_coordinate(orient, size, interval)
    # raise ValueError
    #
    # # [-5.8 -2.6 -5.8] [20 14 30] 0.4
    # orient = [-5.8, -2.6, -5.8]
    # size = [15 + 5, 7 * 2, 15 * 2]
    # interval = 0.4
    # make_grid_coordinate(orient, size, interval)