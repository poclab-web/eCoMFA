import os
import numpy as np
import pandas as pd



if __name__ == '__main__':
    # for param_file_name in ["../parameter_0227/parameter_cbs_gaussian.txt",
    #                         "../parameter_0227/parameter_dip-chloride_gaussian.txt",
    #                         "../parameter_0227/parameter_RuSS_gaussian.txt"]:
    #     with open(param_file_name, "r") as f:
    #         param = json.loads(f.read())
    if True:
        # print(param["grid_sizefile"].split(","))
        # l = param["grid_sizefile"].split(",")  # 4.5
        #[[-4.75, -2.75, -4.75],[14, 12, 20],0.5]
        orient = [-4.75, -2.75, -4.75]
        size = [10 + 4, 6 * 2, 10 * 2]
        interval = 0.5
        #[[-5.0, -3.0, -5.0],[18, 16, 26],0.4]
        orient = [-5.0, -3.0, -5.0]
        size = [13 + 5, 8 * 2, 13 * 2]
        interval = 0.4
        #[[-4.65, -2.85, -4.65],[22, 20, 32],0.3]
        orient = [-4.65, -2.85, -4.65]
        size = [16 + 6, 10 * 2, 16 * 2]
        interval = 0.3
        out_dir_name = "../../../grid_coordinates"+"/[{},{},{}]".format(orient, size,interval)
        # if False:
        #     xgrid, ygrid, zgrid, gridinterval = [5, 3, 5, 0.5]  # [float(_) for _ in l]
        #     y_up = np.arange(gridinterval / 2, ygrid, gridinterval)
        #     y_down = -y_up
        #     y = np.sort(np.concatenate([y_down, y_up]))
        #     z_up = np.arange(gridinterval / 2, zgrid, gridinterval)
        #     z_down = -z_up
        #     print(z_up[-1])
        #     z = np.sort(np.concatenate([z_down, z_up]))
        #     sr = {"x": np.round(np.arange(-xgrid, 2.1, gridinterval), 3),
        #           # "y": np.round(np.arange(-ygrid, ygrid+0.1, gridinterval), 2),
        #           "y": np.round(y, 5),
        #           "z": np.round(z, 5)}
        #     dfp = pd.DataFrame([dict(zip(sr.keys(), l)) for l in product(*sr.values())]).astype(float).sort_values(
        #         by=["x", "y", "z"])
        # else:
        l = []
        for x in range(size[0]):
            for y in range(size[1]):
                for z in range(size[2]):
                    ans = np.array(orient) + np.array([x, y, z]) * interval
                    ans = ans.tolist()
                    l.append(ans)
        dfp = pd.DataFrame(data=l, columns=["x", "y", "z"])
        print(dfp)
        os.makedirs(out_dir_name, exist_ok=True)  # + "/" + param["grid_coordinates_dir"]
        # dfp.to_csv(out_dir_name+"/"+param["grid_coordinates_dir"]+"/[{},{},{},{}].csv".format(xgrid,ygrid,zgrid,gridinterval))
        # filename = out_dir_name + "/[{},{},{},{}].csv".format(l[0], l[1], l[2],l[3])#+ "/" + param["grid_coordinates_dir"]
        filename = out_dir_name + "/coordinates.csv"  # + "/" + param["grid_coordinates_dir"]

        dfp.to_csv(filename)
        print(filename)
        dfp_yz=dfp[(dfp["y"]>0)&(dfp["z"]>0)].sort_values(['x', 'y', "z"], ascending=[True, True, True])
        print(dfp_yz)
        dfp_yz.to_csv((out_dir_name+"/coordinates_yz.csv"))
    if True:
        # if True:  # fold:
        #     # xyz = pd.read_csv("{}/{}/feature_yz.csv".format(features_dir_name, df["mol"].iloc[0].GetProp("InchyKey")))[
        #     #     ["x", "y", "z"]].values
        #     #print("yzfold")
        l=np.array(l)
        xyz = l[(l[:,1]>0)&(l[:,2]>0)]


        d = np.array([np.linalg.norm(xyz - _, axis=1) for _ in xyz])
        d_y = np.array([np.linalg.norm(xyz - _ * np.array([1, -1, 1]), axis=1) for _ in xyz])
        d_z = np.array([np.linalg.norm(xyz - _ * np.array([1, 1, -1]), axis=1) for _ in xyz])
        d_yz = np.array([np.linalg.norm(xyz - _ * np.array([1, -1, -1]), axis=1) for _ in xyz])

        def gauss_func(d):
            sigma = interval*2
            leng = interval
            ans = 1 / (2 * np.pi * np.sqrt(2 * np.pi) * sigma ** 3) * leng ** 3 \
                  * np.exp(-d ** 2 / (2 * sigma ** 2))
            return ans


        penalty = np.where(d < interval * 3, gauss_func(d), 0)
        penalty_y = np.where(d_y < interval * 3, gauss_func(d_y), 0)
        penalty_z = np.where(d_z < interval * 3, gauss_func(d_z), 0)
        penalty_yz = np.where(d_yz < interval * 3, gauss_func(d_yz), 0)
        # if True:  # fold:
        penalty = penalty + penalty_y + penalty_z + penalty_yz
        # grid_features_name = "{}/{}/feature_yz.csv"
        # else:
        #     penalty = penalty + penalty_y
        #     # grid_features_name = "{}/{}/feature_y.csv"

        penalty = penalty #/ np.max(np.sum(penalty, axis=0))
        # np.fill_diagonal(penalty, -1)
        penalty = penalty - np.identity(penalty.shape[0])# * np.sum(penalty, axis=0)
        print(penalty)
        #os.makedirs("../penalty", exist_ok=True)
        # np.save('../penalty/penalty.npy', penalty)
        filename = out_dir_name + "/penalty.npy"  # + "/" + param["grid_coordinates_dir"]
        np.save(filename,penalty)
    if True:
        list=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        df=pd.DataFrame(list,columns=["lambda"])
        df.to_csv(out_dir_name + "/penalty_param.csv")