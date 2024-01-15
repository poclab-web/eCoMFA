import pandas as pd
import json
import os
import copy
import numpy as np

if __name__ == '__main__':
    for param_file_name in ["../parameter_nomax/parameter_cbs_gaussian.txt",]:


        print(param_file_name)
        with open(param_file_name, "r") as f:
            param = json.loads(f.read())
        features_dir_name = param["moleculer_field_dir"]#param["grid_dir_name"]
        dir_name=param["grid_dir_name"] + "/[{}]/".format(param["grid_sizefile"])#param["moleculer_field_dir"]
        os.makedirs(dir_name, exist_ok=True)
        # trainstd = np.load(param["moleculer_field_dir"] + "/trainstd.npy")
        # print(trainstd.shape)
        df = pd.read_csv(param["moleculer_field_dir"] + "/moleculer_field.csv")
        # print(len(df))


        # df["std"]=trainstd
        df_y=copy.deepcopy(df)
        df_z = copy.deepcopy(df)
        df_y["y"]=-df_y["y"]
        df_y=df_y[(df_y["z"] > 0)&(df_y["y"] < 0)]
        #df_y.to_csv(dir_name + "/moleculer_fieldtesty.csv")
        df_z = df_z[(df_z["y"] !=0) & (df_z["z"] > 0) ]
        df_z["z"]=-df_z["z"]
        #df_z.to_csv(dir_name + "/moleculer_fieldtestz.csv")
        df_z["MF_Dt"] = -df_z["MF_Dt"]
        df_z["MF_ESP_cutoff"] = -df_z["MF_ESP_cutoff"]
        #df_z.to_csv(dir_name + "/moleculer_fieldtestz.csv")
        df_yz=copy.deepcopy(df)
        df_yz["y"] = -df_yz["y"]
        df_yz["z"] = -df_yz["z"]
        df_z0 = copy.deepcopy(df[df["z"]==1])
        df_z0["MF_Dt"]=0
        df_z0["MF_ESP_cutoff"]=0
        #df_z0.to_csv(dir_name + "/moleculer_fieldtestz0.csv")
        #df_z0=df_z0[df_z0["y"]!=0]
        df_z01=copy.deepcopy(df_z0)
        df_z01["y"]=-df_z01["y"]
        df_z01 = df_z01[df_z01["y"] != 0]
        #df_z01.to_csv(dir_name + "/moleculer_fieldtestyz.csv")
        df_yz1 = copy.deepcopy(df_yz)
        # df_yz.to_csv(dir_name + "/moleculer_fieldtestyz1.csv")
        df_yz["y"]=df_yz1[df_yz1["z"]<=0]["y"]
        df_yz["MF_Dt"]=-df_yz["MF_Dt"]
        df_yz["MF_ESP_cutoff"] = -df_yz["MF_ESP_cutoff"]
        #df_yz.to_csv(dir_name + "/moleculer_fieldtestyz.csv")
        df=pd.concat([df_z0,df_z01,df,df_y,df_z,df_yz]).sort_values(by=["x","y","z"])

        df.to_csv(dir_name+"/KWOLFJPFCHCOCG-UHFFFAOYSA-N/moleculer_fieldtest.csv")







        df1=pd.read_csv(dir_name+"/KWOLFJPFCHCOCG-UHFFFAOYSA-N/feature.csv")
        print(dir_name+"/KWOLFJPFCHCOCG-UHFFFAOYSA-N/feature.csv")
        print(df1["Dt"].shape)
        df1["Dt"]#/trainstd
        df1["ESP_cutoff"]#/trainstd
        print(df1.index.is_unique)
        print(df.index.is_unique)
        # df1["DtR1"]=df1["Dt"]*df["MF_Dt"]
        print(df1.columns)
        print(len(df),len(df1))
        print(df)
        df1["DtR1"] = df1["Dt"] / df["Dt_std"].values*df["MF_Dt"].values#dfdtnumpy
        df1["ESPR1"] = df1["ESP_cutoff"] / df["ESP_std"].values*df["MF_ESP_cutoff"].values#dfespnumpy

        df1.to_csv("../grid_features/[4,2,5,0.5]/KWOLFJPFCHCOCG-UHFFFAOYSA-N/feature_R1.csv")
        ans=df1[df1["z"]>0].sum()
        print(df1[df1["z"]>=0]["DtR1"].sum(),df1[df1["z"]<0]["DtR1"].sum(),df1["DtR1"].sum())
        print(df1[df1["z"] >=0]["ESPR1"].sum(), df1[df1["z"] <0]["ESPR1"].sum(), df1["ESPR1"].sum())
        print(df1["DtR1"].sum()+df1["ESPR1"].sum())
        # print(df1numpy)
        # df2=dfnumpy * df1numpy
        # print(df2.shape)
        # np.savetxt(dir_name+"/numpy_df2.txt",df2)
        # df2=pd.DataFrame(df2)
        # df2=df2.rename(columns={"Dt": 0})
        #
        # df2["x"]=df1["x"]
        # df2["y"]=df1["y"]
        # df2["z"]=df1["z"]
        # df2.to_csv(dir_name+"/numpy_df2.csv")
        # print(df2.columns)
