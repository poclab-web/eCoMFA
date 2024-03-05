import copy
import json
import os
import pandas as pd
import glob


# def fesp_deletenucleo(n, feature):
#     try:
#         ans = float(feature[3 + 3 + n_atom + n // 6])
#         if ans > 0:
#             ans = 0
#     except:
#         ans = 0


def mfunfolding(df):
    df_y = copy.deepcopy(df)
    df_y["y"] = -df_y["y"]
    df_z = copy.deepcopy(df)
    df_z["z"] = -df_z["z"]
    df_yz = copy.deepcopy(df)
    df_yz["y"] = -df_yz["y"]
    df_yz["z"] = -df_yz["z"]

    # df_z["MF_Dt"] = -df_z["MF_Dt"]
    # df_yz["MF_Dt"] = -df_yz["MF_Dt"]
    # try:
    #     try:
    #         df_z["MF_ESP_cutoff"] = -df_z["MF_ESP_cutoff"]
    #         df_yz["MF_ESP_cutoff"] = -df_yz["MF_ESP_cutoff"]
    #     except:
    #         df_z["MF_ESP"] = -df_z["MF_ESP"]
    #         df_yz["MF_ESP"] = -df_yz["MF_ESP"]
    # except:
    #     None

    df = pd.concat([df, df_y, df_z, df_yz]).sort_values(by=["x", "y", "z"], ascending=[True, True, True])
    return df

def zunfolding(df,regression_features):

    df_z = copy.deepcopy(df)
    df_z["z"]=-df_z["z"]
    # 対象となる文字列
    feature1 = 'Dt'
    feature2 = "ESP"


    # 対象文字列を含む列名を取得
    column_inc_specific_feature = [column for column in df_z.columns if (feature1 or feature2) in column]
    df_z[column_inc_specific_feature]=-df_z[column_inc_specific_feature]
    # df_z["MF_{}".format(regression_features.split()[1])]=-df_z["MF_{}".format(regression_features.split()[1])]
    df = pd.concat([df ,df_z ]).sort_values(['x', 'y',"z"], ascending=[True, True,True])#.sort_values(by=["x", "y", "z"])

    return df

if __name__ == '__main__':
    # for param_file_name in [
    #     "../parameter_0227/parameter_cbs_gaussian.txt",
    #     "../parameter_0227/parameter_RuSS_gaussian.txt",
    #     "../parameter_0227/parameter_dip-chloride_gaussian.txt",
    # ]:
    for file in glob.glob("../result/*/molecular_filed*.csv"):
        df= pd.read_csv(file)


        df_unfold = df
        df_unfold = mfunfolding(df_unfold)
        df_unfold

        cubeinchikey = "../cube_aligned_b3lyp_6-31g(d)/KWOLFJPFCHCOCG-UHFFFAOYSA-N"
        for feature, cube_file_name in zip(["MF_ESP", "MF_Dt"], [
            cubeinchikey + "/ESP02_0.cube",
            cubeinchikey + "/Dt02_0.cube",
        ]):


            # inchykey = "WRYKPYJMRHQDBM-UHFFFAOYSA-N"
            with open(cube_file_name, "r", encoding="UTF-8") as f:
                cube = f.read().splitlines()
            n_atom = int(cube[2].split()[0])

            xyz = cube[6:6 + n_atom]
            dir_name = param["moleculer_field_dir"]
            os.makedirs(dir_name, exist_ok=True)
            to_cube_file_name = dir_name + "/" + feature + ".cube"
            # print(len(df))
            drop_dupl_x = df.drop_duplicates(subset="x").sort_values('x')["x"]
            print(drop_dupl_x)
            drop_dupl_y = df.drop_duplicates(subset="y").sort_values('y')["y"]
            drop_dupl_z = df.drop_duplicates(subset="z").sort_values('z')["z"]
            d_x = drop_dupl_x.iloc[1] - drop_dupl_x.iloc[0]
            d_y = drop_dupl_y.iloc[1] - drop_dupl_y.iloc[0]
            d_z = drop_dupl_z.iloc[1] - drop_dupl_z.iloc[0]

            # print(d_x,d_y,d_z)
            # print(drop_dupl_x.max(),drop_dupl_x.min())
            range_x = drop_dupl_x.max() - drop_dupl_x.min()
            range_y = drop_dupl_y.max() - drop_dupl_y.min()
            range_z = drop_dupl_z.max() - drop_dupl_z.min()
            print(range_x / d_x)
            print(range_y / d_y)
            print(range_z / d_z)

            count_x = round(range_x / d_x + 1)
            count_y = round(range_y / d_y + 1)
            count_z = round(range_z / d_z + 1)
            print(count_x, count_y, count_z)

            with open(to_cube_file_name, "w") as f:
                print(0, file=f)
                print(1, file=f)
                print("    {} {} {} {}".format(n_atom, drop_dupl_x.min() / 0.52917720859,
                                               drop_dupl_y.min() / 0.52917720859,
                                               drop_dupl_z.min() / 0.52917720859), file=f)
                print("    {}   {:.5f} {:.5f} {:.5f}".format(count_x, d_x / 0.52917720859, 0, 0), file=f)
                print("    {}   {:.5f} {:.5f} {:.5f}".format(count_y, 0, d_y / 0.52917720859, 0), file=f)
                print("    {}   {:.5f} {:.5f} {:.5f}".format(count_z, 0, 0, d_z / 0.52917720859), file=f)
                print("\n".join(xyz), file=f)
                for i in range(len(df) // 6 + 1):
                    line = df[feature].iloc[6 * i:6 * i + 6].values.tolist()
                    # line=list(map(str, line))
                    # line=" ".join(line)
                    line = ['{:.5E}'.format(n) for n in line]
                    line = " ".join(line)
                    print(line)
                    print(line, file=f)

            print(xyz)
            if False:
                if feature == "MF_ESP":
                    ESP0 = cube[0:6 + n_atom]
                    ESP1 = cube[6 + n_atom:]
                    ESP1list = []
                    for i in range(len(ESP1)):
                        espl = ESP1[i].split()
                        replaceespl = [str(0) if float(i2) > 0 else i2 for i2 in espl]
                        q = ' '.join(replaceespl)
                        ESP1list.append(q)
                    with open("../errortest/nonnucleaarespcube.cube", "w") as f:
                        print("\n".join(ESP0), file=f)
                        print("\n".join(ESP1list), file=f)

            if True:
                if feature == "MF_ESP_cutoff":
                    with open(cubeinchikey + "/Dt02_0.cube", "r", encoding="UTF-8") as f:
                        cubeDt = f.read().splitlines()
                    n_atomDt = int(cube[2].split()[0])
                    xyz = cube[6:6 + n_atom]
                    ESP0 = cube[0:6 + n_atom]
                    ESP1 = cube[6 + n_atom:]
                    ESP1list = []
                    xyzDt = cubeDt[6:6 + n_atomDt]
                    Dt0 = cubeDt[0:6 + n_atomDt]
                    Dt1 = cubeDt[6 + n_atomDt:]
                    Dt1list = []
                    for i in range(len(ESP1)):
                        espl = ESP1[i].split()
                        dtl = Dt1[i].split()
                        replaceespl = [espl1 if float(dtl1) < 0.01 else str(0) for dtl1, espl1 in zip(dtl, espl)]
                        q = ' '.join(replaceespl)
                        ESP1list.append(q)
                    with open("../errortest/cutoffespcube.cube", "w") as f:
                        print("\n".join(ESP0), file=f)
                        print("\n".join(ESP1list), file=f)
