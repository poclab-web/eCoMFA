import copy
import json
import os
import pandas as pd


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
    df_z["MF_Dt"] = -df_z["MF_Dt"]

    df_yz = copy.deepcopy(df)
    df_yz["y"] = -df_yz["y"]
    df_yz["z"] = -df_yz["z"]
    df_yz["MF_Dt"] = -df_yz["MF_Dt"]
    try:
        df_z["MF_ESP_cutoff"] = -df_z["MF_ESP_cutoff"]
        df_yz["MF_ESP_cutoff"] = -df_yz["MF_ESP_cutoff"]
    except:
        df_z["MF_ESP"] = -df_z["MF_ESP"]
        df_yz["MF_ESP"] = -df_yz["MF_ESP"]

    df = pd.concat([df, df_y, df_z, df_yz]).sort_values(by=["x", "y", "z"], ascending=[True, True, True])

    return df


if __name__ == '__main__':
    for param_file_name in [
        # "../parameter_nomax/parameter_cbs_gaussian.txt",
        # "../parameter_nomax/parameter_cbs_gaussian.txt",
        # "../parameter_nomax/parameter_cbs_ridgecv.txt",
        # "../parameter_nomax/parameter_cbs_PLS.txt",
        # "../parameter_nomax/parameter_cbs_lassocv.txt",
        # "../parameter_nomax/parameter_cbs_elasticnetcv.txt",
        # "../parameter_nomax/parameter_RuSS_gaussian.txt",
        # "../parameter_nomax/parameter_RuSS_lassocv.txt",
        # "../parameter_nomax/parameter_RuSS_PLS.txt",
        # "../parameter_nomax/parameter_RuSS_elasticnetcv.txt",
        # "../parameter_nomax/parameter_RuSS_ridgecv.txt",
        # "../parameter_nomax/parameter_dip-chloride_PLS.txt",
        # "../parameter_nomax/parameter_dip-chloride_lassocv.txt",
        # "../parameter_nomax/parameter_dip-chloride_gaussian.txt",
        # "../parameter_nomax/parameter_dip-chloride_elasticnetcv.txt",
        # "../parameter_nomax/parameter_dip-chloride_ridgecv.txt",

        "../parameter_0227/parameter_cbs_gaussian.txt",
        "../parameter_0227/parameter_RuSS_gaussian.txt",
        "../parameter_0227/parameter_dip-chloride_gaussian.txt",

        # "../parameter_0207/parameter_cbs_gaussian.txt",
        # "../parameter_0207/parameter_cbs_ridgecv.txt",
        # "../parameter_0207/parameter_cbs_PLS.txt",
        # "../parameter_0207/parameter_cbs_lassocv.txt",
        # "../parameter_0207/parameter_cbs_elasticnetcv.txt",
        # "../parameter_0227/parameter_RuSS_gaussian.txt",
        # "../parameter_0207/parameter_RuSS_lassocv.txt",
        # "../parameter_0207/parameter_RuSS_PLS.txt",
        # "../parameter_0207/parameter_RuSS_elasticnetcv.txt",
        # "../parameter_0207/parameter_RuSS_ridgecv.txt",
        # "../parameter_0207/parameter_dip-chloride_PLS.txt",
        # "../parameter_0207/parameter_dip-chloride_lassocv.txt",
        # "../parameter_0227/parameter_dip-chloride_gaussian.txt",
        # "../parameter_0207/parameter_dip-chloride_elasticnetcv.txt",
        # "../parameter_0207/parameter_dip-chloride_ridgecv.txt",
    ]:

        print(param_file_name)
        with open(param_file_name, "r") as f:
            param = json.loads(f.read())
        features_dir_name = param["moleculer_field_dir"]  # param["grid_dir_name"]

        # "../cube_aligned_b3lyp_6-31g(d)/ADCYRBXQAJXJTD-UHFFFAOYSA-N/ESP02_0.cube"
        # inchykey = "WRYKPYJMRHQDBM-UHFFFAOYSA-N","KWOLFJPFCHCOCG-UHFFFAOYSA-N"KWOLFJPFCHCOCG-UHFFFAOYSA-N　IMACFCSSMIZSPP-UHFFFAOYSA-N

        # feature="MF_Dt"
        fold = True
        dir_name = param["moleculer_field_dir"]
        os.makedirs(dir_name, exist_ok=True)
        df = pd.read_csv(
            dir_name + "/moleculer_field.csv")  # "{}/{}/feature_yz.csv".format(features_dir_name, inchykey))
        # for feature,cube_file_name in zip(["MF_Dt","MF_ESP","MF_ESP_cutoff"],["../cube_aligned_b3lyp_6-31g(d)/KWOLFJPFCHCOCG-UHFFFAOYSA-N/Dt02_0.cube",
        #                                                       "../cube_aligned_b3lyp_6-31g(d)/KWOLFJPFCHCOCG-UHFFFAOYSA-N/ESP02_0.cube",
        #                                                     "../cube_aligned_b3lyp_6-31g(d)/KWOLFJPFCHCOCG-UHFFFAOYSA-N/ESP02_0.cube"
        #                                                                       ]):
        # UPEUQDJSUFHFQP-UHFFFAOYSA-N ,WYJOVVXUZNRJQY-UHFFFAOYSA-N,PFIKCDNZZJYSMK-UHFFFAOYSA-N WYJOVVXUZNRJQY-UHFFFAOYSA-N,PFIKCDNZZJYSMK-UHFFFAOYSA-N,RIFKADJTWUGDOV-UHFFFAOYSA-N,
        # KRIOVPPHQSLHCZ-UHFFFAOYSA-N,CKGKXGQVRVAKEA-UHFFFAOYSA-N,AJKVQEKCUACUMD-UHFFFAOYSA-N,VRZSUVFVIIVLPV-UHFFFAOYSA-N
        cubeinchikey = "../cube_aligned_b3lyp_6-31g(d)/KWOLFJPFCHCOCG-UHFFFAOYSA-N"
        for feature, cube_file_name in zip(["MF_ESP", "MF_Dt"], [
            cubeinchikey + "/ESP02_0.cube",
            cubeinchikey + "/Dt02_0.cube",
        ]):
            df = pd.read_csv(dir_name + "/moleculer_field.csv")
            df = mfunfolding(df)
            # if fold:
            #     df = pd.read_csv(dir_name + "/moleculer_field.csv")
            #     df_y=copy.deepcopy(df)
            #     df_z = copy.deepcopy(df)
            #     df_y["y"]=-df_y["y"]
            #     df_y=df_y[(df_y["z"] > 0)&(df_y["y"] < 0)]
            #     #df_y.to_csv(dir_name + "/moleculer_fieldtesty.csv")
            #     df_z = df_z[(df_z["y"] !=0) & (df_z["z"] > 0) ]
            #     df_z["z"]=-df_z["z"]
            #     #df_z.to_csv(dir_name + "/moleculer_fieldtestz.csv")
            #     df_z[feature] = -df_z[feature]
            #     #df_z.to_csv(dir_name + "/moleculer_fieldtestz.csv")
            #     df_yz=copy.deepcopy(df)
            #     df_yz["y"] = -df_yz["y"]
            #     df_yz["z"] = -df_yz["z"]
            #     df_z0 = copy.deepcopy(df[df["z"]==1])
            #     df_z0[feature]=0
            #     df_z0["z"]=0
            #     #df_z0.to_csv(dir_name + "/moleculer_fieldtestz0.csv")
            #     #df_z0=df_z0[df_z0["y"]!=0]
            #     df_z01=copy.deepcopy(df_z0)
            #     df_z01["y"]=-df_z01["y"]
            #     df_z01 = df_z01[df_z01["y"] != 0]
            #     #df_z01.to_csv(dir_name + "/moleculer_fieldtestyz.csv")
            #     df_yz1 = copy.deepcopy(df_yz)
            #     # print(df_yz)
            #     # print(df_yz1)
            #     df_yz.to_csv(dir_name + "/moleculer_fieldtestyz1.csv")
            #     df_yz["y"]=df_yz1[df_yz1["z"]<=0]["y"]
            #     df_yz[feature]=-df_yz[feature]
            #     #df_yz.to_csv(dir_name + "/moleculer_fieldtestyz.csv")
            #     df=pd.concat([df_z0,df_z01,df,df_y,df_z,df_yz]).sort_values(by=["x","y","z"])
            #
            # else:
            #     None
            df.sort_values(by=["x", "y", "z"])
            df.to_csv(dir_name + "/moleculer_fieldunfold.csv")

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
