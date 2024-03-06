import os
from itertools import product

import numpy as np
import pandas as pd


def make_grid_coordinate(orient, size, interval):
    out_dir_name = "../../../grid_coordinates" + "/[{}] [{}] {}".format(" ".join(map(str, orient))," ".join(map(str, size)), interval)
    l = []
    for x in product(range(size[0]),range(size[1]),range(size[2])):
        ans = np.array(orient) + np.array(x) * interval
        ans = ans.tolist()
        l.append(ans)
    #ans=np.array(product(range(size[0]),range(size[1]),range(size[2])))
    # l = np.sort(np.meshgrid(range(size[0]), range(size[1]), range(size[2]))).T.reshape(-1, 3) * interval + np.array(
    #     orient)
    print(l)
    dfp = pd.DataFrame(data=l, columns=["x", "y", "z"]).sort_values(['x', 'y', "z"], ascending=[True, True, True])
    # print(dfp)
    os.makedirs(out_dir_name, exist_ok=True)  # + "/" + param["grid_coordinates_dir"]

    filename = out_dir_name + "/coordinates.csv"  # + "/" + param["grid_coordinates_dir"]

    dfp.to_csv(filename)
    print(filename)
    dfp_yz = dfp[(dfp["y"] > 0) & (dfp["z"] > 0)].sort_values(['x', 'y', "z"], ascending=[True, True, True])

    dfp_yz.to_csv((out_dir_name + "/coordinates_yz.csv"))
    # print(dfp_yz)
    l=np.array(l)
    xyz = l[(l[:, 1] > 0) & (l[:, 2] > 0)]

    d = np.array([np.linalg.norm(xyz - _, axis=1) for _ in xyz])
    d_y = np.array([np.linalg.norm(xyz - _ * np.array([1, -1, 1]), axis=1) for _ in xyz])
    d_z = np.array([np.linalg.norm(xyz - _ * np.array([1, 1, -1]), axis=1) for _ in xyz])
    d_yz = np.array([np.linalg.norm(xyz - _ * np.array([1, -1, -1]), axis=1) for _ in xyz])

    def gauss_func(d):
        sigma = interval  # *2
        leng = interval
        ans = 1 / (2 * np.pi * np.sqrt(2 * np.pi) * sigma ** 3) * leng ** 3 \
              * np.exp(-d ** 2 / (2 * sigma ** 2))
        return ans

    penalty = np.where(d < interval * 3, gauss_func(d), 0)
    penalty_y = np.where(d_y < interval * 3, gauss_func(d_y), 0)
    penalty_z = np.where(d_z < interval * 3, gauss_func(d_z), 0)
    penalty_yz = np.where(d_yz < interval * 3, gauss_func(d_yz), 0)

    penalty = penalty + penalty_y + penalty_z + penalty_yz
    penalty = penalty / np.max(np.sum(penalty, axis=0))
    penalty = penalty - np.identity(penalty.shape[0])/ np.sum(penalty, axis=0)


    filename = out_dir_name + "/penalty.npy"  # + "/" + param["grid_coordinates_dir"]
    np.save(filename, penalty)

    list = np.logspace(-5, 10, num=16, base=2)#2**np.arange(10)#[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    df = pd.DataFrame(list, columns=["lambda"])
    df.to_csv(out_dir_name + "/penalty_param.csv")


if __name__ == '__main__':
    # [[-4.75, -2.75, -4.75],[14, 12, 20],0.5]
    orient = [-4.75, -2.75, -4.75]
    size = [10 + 4, 6 * 2, 10 * 2]
    interval = 0.5
    make_grid_coordinate(orient, size, interval)
    # [[-5.0, -3.0, -5.0],[18, 16, 26],0.4]
    orient = [-5.0, -3.0, -5.0]
    size = [13 + 5, 8 * 2, 13 * 2]
    interval = 0.4
    make_grid_coordinate(orient, size, interval)
    # [[-4.65, -2.85, -4.65],[22, 20, 32],0.3]
    orient = [-4.65, -2.85, -4.65]
    size = [16 + 6, 10 * 2, 16 * 2]
    interval = 0.3
    make_grid_coordinate(orient, size, interval)
    # [-5.0 -1.8 -5.0] [18 10 26] 0.4
    orient = [-5.0, -1.8, -5.0]
    size = [13 + 5, 5 * 2, 13 * 2]
    interval = 0.4
    make_grid_coordinate(orient, size, interval)

