import pandas as pd
import os
import calculate_conformation
import numpy as np
import json

def read_cube(dir_name,dfp,mol,out_name):
    os.makedirs(out_name, exist_ok=True)

    for conf in mol.GetConformers():
        Dt_file="{}/{}/Dt02_{}.cube".format(dir_name,mol.GetProp("InchyKey"),conf.GetId())
        with open(Dt_file,"r",encoding="UTF-8")as f:
            Dt=f.read().splitlines()
        ESP_file = "{}/{}/ESP02_{}.cube".format(dir_name, mol.GetProp("InchyKey"), conf.GetId())
        with open(ESP_file, "r", encoding="UTF-8") as f:
            ESP = f.read().splitlines()
        LUMO_file = "{}/{}/LUMO02_{}.cube".format(dir_name,mol.GetProp("InchyKey"),conf.GetId())
        with open(LUMO_file,"r",encoding="UTF-8")as f:
            LUMO=f.read().splitlines()
        l = np.array([_.split() for _ in Dt[2:6]])
        n_atom = int(l[0, 0])
        x0 = l[0, 1:].astype(float)*0.52917720859
        size = l[1:, 0].astype(int)
        #axis = l[1:, 1:].astype(float)*0.52917720859
        delta=0.2*0.52917720859
        grid=(dfp[["x", "y", "z"]].values-x0)/delta
        grid=np.round(grid).astype(int)
        cond1=np.all(grid < size,axis=1)
        cond2=np.all(np.zeros(3)<=grid,axis=1)
        cond_all=np.all(np.stack([cond1,cond2]),axis=0)
        n=grid[:,0]*size[2]*size[1]+grid[:,1]*size[2]+grid[:,2]

        def fdt(n,feature):
            try:
                ans=float(feature[3 + 3 + n_atom+n//6].split()[n%6])
                if ans>0.1:
                    ans=0.1
            except:
                ans=0
            return ans

        def fesp(n,feature):
            try:
                ans=float(feature[3 + 3 + n_atom+n//6].split()[n%6])
                if ans>0.1:
                    ans=0.1
            except:
                ans=0

            return ans
        # 20240105 坂口　作成
        def fesp_new(n,Dt,feature):
            try:
                if float(Dt[3 + 3 + n_atom+n//6].split()[n%6])<0.01:#0.05
                    ans=float(feature[3 + 3 + n_atom+n//6].split()[n%6])
                else:
                    ans=0
            except:
                ans=0
            return ans
        def fesp_deletenucleo(n,feature):
            try:
                ans=float(feature[3 + 3 + n_atom+n//6].split()[n%6])
                if ans>0:
                    ans=0
            except:
                ans=0


            return ans

        ##
        def fLUMO(n, feature):
            try:
                ans = float(feature[3 + 3 + n_atom + n // 6].split()[n % 6])
                if ans > 0.05:
                    ans = 0.05
            except:
                ans = 0
            return ans

        dfp["Dt"]=np.where(cond_all,np.array([fdt(_,Dt) for _ in n]).astype(float),0)
        dfp["ESP"]=np.where(cond_all,np.array([fesp_deletenucleo(_,ESP) for _ in n]).astype(float),0)
        dfp["ESP_cutoff"]=np.where(cond_all,np.array([fesp_new(_,Dt,ESP) for _ in n]).astype(float),0)
        dfp["LUMO"]=np.where(cond_all,np.array([fLUMO(_,LUMO) for _ in n]).astype(float),0)
        os.makedirs(out_name,exist_ok=True)
        dfp_y=dfp[dfp["y"]>=0][["x","y","z"]].sort_values(['x', 'y', "z"], ascending=[True, True, True])  # そとにでる
        dfp_z=dfp[dfp["z"]>0][["x","y","z"]].sort_values(['x', 'y', "z"], ascending=[True, True, True])  # そとにでる
        dfp_yz=dfp[(dfp["y"]>0)&(dfp["z"]>0)][["x","y","z"]].sort_values(['x', 'y', "z"], ascending=[True, True, True])  # そとにでる

        for feature in ["Dt","ESP","ESP_cutoff","LUMO"]:
            if False:
                #dfp_y[feature]=dfp[dfp["y"]>=0][feature].values+dfp[dfp["y"]<=0][feature].values
                dfp_y[feature] = dfp[dfp["y"] > 0][feature].values + dfp[dfp["y"] < 0][feature].values
                #dfp_y[dfp_y["y"]==0] = dfp[dfp["y"] == 0]

                dfp_z[feature]=dfp[dfp["z"]>0][feature].values-\
                                dfp[dfp["z"]<0][feature].values

                # dfp_yz[feature]=dfp[(dfp["y"]>=0)&(dfp["z"]>0)][feature].values+\
                #                 dfp[(dfp["y"]<=0)&(dfp["z"]>0)][feature].values-\
                #                 dfp[(dfp["y"]>=0)&(dfp["z"]<0)][feature].values-\
                #                 dfp[(dfp["y"]<=0)&(dfp["z"]<0)][feature].values
                dfp_yz[feature] = dfp[(dfp["y"] > 0) & (dfp["z"] > 0)][feature].values + \
                                  dfp[(dfp["y"] < 0) & (dfp["z"] > 0)][feature].values - \
                                  dfp[(dfp["y"] > 0) & (dfp["z"] < 0)][feature].values - \
                                  dfp[(dfp["y"] < 0) & (dfp["z"] < 0)][feature].values
            dfp_y[feature]=dfp[dfp["y"] > 0].sort_values(['x', 'y',"z"], ascending=[True, True,True])[feature].values\
                           +dfp[dfp["y"] < 0].sort_values(['x', 'y',"z"], ascending=[True, False,True])[feature].values
            # dfp_z = dfp_z.sort_values(['x', 'y', "z"], ascending=[True, True, True])  # そとにでる
            dfp_z[feature] = dfp[dfp["z"] > 0].sort_values(['x', 'y', "z"], ascending=[True, True, True])[feature].values \
                             - dfp[dfp["z"] < 0].sort_values(['x', 'y', "z"], ascending=[True, True, False])[feature].values
            # dfp_yz = dfp_yz.sort_values(['x', 'y', "z"], ascending=[True, True, True])  # そとにでる
            dfp_yz[feature]=dfp[(dfp["y"] > 0)&(dfp["z"] > 0)].sort_values(['x', 'y',"z"], ascending=[True, True,True])[feature].values\
                           +dfp[(dfp["y"] < 0)&(dfp["z"] > 0)].sort_values(['x', 'y',"z"], ascending=[True, False,True])[feature].values\
                            -dfp[(dfp["y"] > 0)&(dfp["z"] < 0)].sort_values(['x', 'y',"z"], ascending=[True, True,False])[feature].values\
                           -dfp[(dfp["y"] < 0)&(dfp["z"] < 0)].sort_values(['x', 'y',"z"], ascending=[True, False,False])[feature].values
            if False:
                l_y=[]
                l_z=[]
                l_yz=[]
                for x,y,z in zip(dfp_y["x"],dfp_y["y"],dfp_y["z"]):
                    ans=dfp[(dfp["x"] == x) & (dfp["y"] == y) & (dfp["z"] == z)][feature].iloc[0]
                    ans_y=dfp[(dfp["x"] == x) & (dfp["y"] == -y) & (dfp["z"] == z)][feature].iloc[0]
                    l_y.append(ans+ans_y)
                dfp_y[feature]=l_y
                for x,y,z in zip(dfp_z["x"],dfp_z["y"],dfp_z["z"]):
                    ans=dfp[(dfp["x"] == x) & (dfp["y"] == y) & (dfp["z"] == z)][feature].iloc[0]
                    ans_z=dfp[(dfp["x"] == x) & (dfp["y"] == y) & (dfp["z"] == -z)][feature].iloc[0]
                    l_z.append(ans-ans_z)
                dfp_z[feature]=l_z
                for x,y,z in zip(dfp_yz["x"],dfp_yz["y"],dfp_yz["z"]):
                    ans=dfp[(dfp["x"] == x) & (dfp["y"] == y) & (dfp["z"] == z)][feature].iloc[0]
                    ans_y=dfp[(dfp["x"] == x) & (dfp["y"] == -y) & (dfp["z"] == z)][feature].iloc[0]
                    ans_z=dfp[(dfp["x"] == x) & (dfp["y"] == y) & (dfp["z"] == -z)][feature].iloc[0]
                    ans_zy=dfp[(dfp["x"] == x) & (dfp["y"] == -y) & (dfp["z"] == -z)][feature].iloc[0]
                    l_yz.append(ans+ans_y-ans_z-ans_zy)
                dfp_yz[feature]=l_yz

        dfp_y.to_csv(out_name+"/feature_y_{}.csv".format(conf.GetId()))
        dfp_z.to_csv(out_name + "/feature_z_{}.csv".format(conf.GetId()))
        dfp_yz.to_csv(out_name + "/feature_yz_{}.csv".format(conf.GetId()))
        dfp.to_csv(out_name+"/feature_{}.csv".format(conf.GetId()))

    Dts=np.stack([pd.read_csv(out_name+"/feature_{}.csv".format(conf.GetId()))["Dt"].values for conf in mol.GetConformers()])
    ESPs=[pd.read_csv(out_name+"/feature_{}.csv".format(conf.GetId()))["ESP"].values for conf in mol.GetConformers()]
    LUMOs = [pd.read_csv(out_name + "/feature_{}.csv".format(conf.GetId()))["LUMO"].values for conf in
            mol.GetConformers()]
    ESPs_cutoff =[pd.read_csv(out_name + "/feature_{}.csv".format(conf.GetId()))["ESP_cutoff"].values for conf in
            mol.GetConformers()]
    weights= [float(conf.GetProp("Boltzmann_distribution")) for conf in mol.GetConformers()]
    dfp["Dt"]=np.average(Dts,weights=weights,axis=0)
    dfp["ESP"]=np.average(ESPs,weights=weights,axis=0)
    dfp["ESP_cutoff"]=np.average(ESPs_cutoff,weights=weights,axis=0)
    dfp["LUMO"]=np.average(LUMOs,weights=weights,axis=0)
    dfp.to_csv(out_name + "/feature.csv".format(conf.GetId()))

    Dts = np.stack(
        [pd.read_csv(out_name + "/feature_y_{}.csv".format(conf.GetId()))["Dt"].values for conf in mol.GetConformers()])
    ESPs = [pd.read_csv(out_name + "/feature_y_{}.csv".format(conf.GetId()))["ESP"].values for conf in mol.GetConformers()]
    LUMOs = [pd.read_csv(out_name + "/feature_y_{}.csv".format(conf.GetId()))["LUMO"].values for conf in
             mol.GetConformers()]
    ESPs_cutoff = [pd.read_csv(out_name + "/feature_y_{}.csv".format(conf.GetId()))["ESP_cutoff"].values for conf in
                   mol.GetConformers()]
    weights = [float(conf.GetProp("Boltzmann_distribution")) for conf in mol.GetConformers()]
    dfp_y["Dt"] = np.average(Dts, weights=weights, axis=0)
    dfp_y["ESP"] = np.average(ESPs, weights=weights, axis=0)
    dfp_y["ESP_cutoff"] = np.average(ESPs_cutoff, weights=weights, axis=0)
    dfp_y["LUMO"] = np.average(LUMOs, weights=weights, axis=0)
    dfp_y.to_csv(out_name + "/feature_y.csv".format(conf.GetId()))

    Dts = np.stack(
        [pd.read_csv(out_name + "/feature_z_{}.csv".format(conf.GetId()))["Dt"].values for conf in mol.GetConformers()])
    ESPs = [pd.read_csv(out_name + "/feature_z_{}.csv".format(conf.GetId()))["ESP"].values for conf in mol.GetConformers()]
    LUMOs = [pd.read_csv(out_name + "/feature_z_{}.csv".format(conf.GetId()))["ESP"].values for conf in
            mol.GetConformers()]
    ESPs_cutoff = [pd.read_csv(out_name + "/feature_z_{}.csv".format(conf.GetId()))["ESP_cutoff"].values for conf in
                   mol.GetConformers()]
    weights = [float(conf.GetProp("Boltzmann_distribution")) for conf in mol.GetConformers()]
    dfp_z["Dt"] = np.average(Dts, weights=weights, axis=0)
    dfp_z["ESP"] = np.average(ESPs, weights=weights, axis=0)
    dfp_z["LUMO"] = np.average(LUMOs, weights=weights, axis=0)
    dfp_z["ESP_cutoff"] = np.average(ESPs_cutoff, weights=weights, axis=0)

    dfp_z.to_csv(out_name + "/feature_z.csv".format(conf.GetId()))

    Dts = np.stack(
        [pd.read_csv(out_name + "/feature_yz_{}.csv".format(conf.GetId()))["Dt"].values for conf in mol.GetConformers()])
    ESPs = [pd.read_csv(out_name + "/feature_yz_{}.csv".format(conf.GetId()))["ESP"].values for conf in mol.GetConformers()]
    LUMOs = [pd.read_csv(out_name + "/feature_yz_{}.csv".format(conf.GetId()))["LUMO"].values for conf in
            mol.GetConformers()]
    ESPs_cutoff = [pd.read_csv(out_name + "/feature_yz_{}.csv".format(conf.GetId()))["ESP_cutoff"].values for conf in
                   mol.GetConformers()]
    weights = [float(conf.GetProp("Boltzmann_distribution")) for conf in mol.GetConformers()]
    dfp_yz["Dt"] = np.average(Dts, weights=weights, axis=0)
    dfp_yz["ESP"] = np.average(ESPs, weights=weights, axis=0)
    dfp_yz["LUMO"] = np.average(LUMOs, weights=weights, axis=0)
    dfp_yz["ESP_cutoff"] = np.average(ESPs_cutoff, weights=weights, axis=0)
    dfp_yz.to_csv(out_name + "/feature_yz.csv".format(conf.GetId()))
    print("!!!")

def energy_to_Boltzmann_distribution(mol, RT=1.99e-3 * 273):
    energies = np.array([float(conf.GetProp("energy")) for conf in mol.GetConformers()])
    energies = energies - np.min(energies)
    rates = np.exp(-energies / RT)
    rates = rates / sum(rates)
    for conf, rate in zip(mol.GetConformers(), rates):
        conf.SetProp("Boltzmann_distribution", str(rate))


if __name__ == '__main__':

    for param_file_name in ["../parameter_0125/parameter_cbs_gaussian.txt","../parameter_0125/parameter_dip-chloride_gaussian.txt","../parameter_0125/parameter_RuSS_gaussian.txt"]:
        with open(param_file_name, "r") as f:
            param = json.loads(f.read())

        #座標を読み込む
        # "data_file_path"
        # df1 = pd.read_excel( "../arranged_dataset/cbs.xls")
        # df2 = pd.read_excel("../arranged_dataset/DIP-chloride.xls")
        # df3 = pd.read_excel("../arranged_dataset/RuSS.xls")
        # df=pd.concat([df1,df2,df3]).dropna(subset=['smiles'])
        df = pd.read_excel(param["data_file_path"]).dropna(subset=['smiles'])
        df["mol"]=df["smiles"].apply(calculate_conformation.get_mol)
        df=df[[os.path.isdir(param["cube_dir_name"]+"/"+mol.GetProp("InchyKey"))for mol in df["mol"]]]
        df["mol"].apply(lambda mol: calculate_conformation.read_xyz(mol,param["cube_dir_name"]+"/"+mol.GetProp("InchyKey")))

        #gridのパラメータを読み込む

        # dfp=pd.read_csv(param["grid_coordinates_file"])
        print("../grid_coordinates"+param["grid_coordinates_dir"]+"/[{}]".format(param["grid_sizefile"]))
        dfp = pd.read_csv("../grid_coordinates"+param["grid_coordinates_dir"]+"/[{}].csv".format(param["grid_sizefile"]))
        print(dfp)

        grid_dir_name=param["grid_dir_name"]
        #グリッド情報に変換
        for mol in df["mol"]:
            energy_to_Boltzmann_distribution(mol, RT=0.54)
            read_cube(param["cube_dir_name"], dfp, mol,grid_dir_name+"/[{}]".format(param["grid_sizefile"])+"/"+mol.GetProp("InchyKey"))
