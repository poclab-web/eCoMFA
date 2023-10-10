import pandas as pd
import os
import calculate_conformation
import numpy as np
import json

def read_cube(dir_name,dfp,mol,out_name):
    for conf in mol.GetConformers():
        Dt_file="{}/{}/Dt02_{}.cube".format(dir_name,mol.GetProp("InchyKey"),conf.GetId())
        with open(Dt_file,"r",encoding="UTF-8")as f:
            Dt=f.read().splitlines()
        ESP_file = "{}/{}/ESP02_{}.cube".format(dir_name, mol.GetProp("InchyKey"), conf.GetId())
        with open(ESP_file, "r", encoding="UTF-8") as f:
            ESP = f.read().splitlines()
        l = np.array([_.split() for _ in Dt[2:6]])
        n_atom = int(l[0, 0])
        x0 = l[0, 1:].astype(float)*0.52917720859
        size = l[1:, 0].astype(int)
        axis = l[1:, 1:].astype(float)*0.52917720859
        delta=0.2*0.52917720859
        grid=(dfp[["x", "y", "z"]].values-x0)/delta
        grid=np.round(grid).astype(int)
        cond1=np.all(grid < size,axis=1)
        cond2=np.all(np.zeros(3)<=grid,axis=1)
        cond_all=np.all(np.stack([cond1,cond2]),axis=0)
        n=grid[:,0]*size[2]*size[1]+grid[:,1]*size[2]+grid[:,2]

        def f(n,feature):
            try:
                ans=float(feature[3 + 3 + n_atom+n//6].split()[n%6])
                if ans>0.01:
                    ans=0.01
            except:
                ans=0
            return ans

        dfp["Dt"]=np.where(cond_all,np.array([f(_,Dt) for _ in n]).astype(float),0)
        dfp["ESP"]=np.where(cond_all,np.array([f(_,ESP) for _ in n]).astype(float),0)
        os.makedirs(out_name,exist_ok=True)
        dfp_y=dfp[dfp["y"]>=0][["x","y","z"]]
        dfp_z=dfp[dfp["z"]>0][["x","y","z"]]
        dfp_yz=dfp[(dfp["y"]>=0)&(dfp["z"]>0)][["x","y","z"]]
        for feature in ["Dt","ESP"]:
            dfp_y[feature]=dfp[dfp["y"]>=0][feature].values#+dfp[dfp["y"]<=0][feature].values

            dfp_z[feature]=dfp[dfp["z"]>0][feature].values-\
                           dfp[dfp["z"]<0][feature].values

            dfp_yz[feature]=dfp[(dfp["y"]>=0)&(dfp["z"]>0)][feature].values+\
                            dfp[(dfp["y"]<=0)&(dfp["z"]>0)][feature].values-\
                            dfp[(dfp["y"]>=0)&(dfp["z"]<0)][feature].values-\
                            dfp[(dfp["y"]<=0)&(dfp["z"]<0)][feature].values
        dfp_y.to_csv(out_name+"/feature_y_{}.csv".format(conf.GetId()))
        dfp_z.to_csv(out_name + "/feature_z_{}.csv".format(conf.GetId()))
        dfp_yz.to_csv(out_name + "/feature_yz_{}.csv".format(conf.GetId()))
        dfp.to_csv(out_name+"/feature_{}.csv".format(conf.GetId()))
    Dts=np.stack([pd.read_csv(out_name+"/feature_{}.csv".format(conf.GetId()))["Dt"].values for conf in mol.GetConformers()])
    ESPs=[pd.read_csv(out_name+"/feature_{}.csv".format(conf.GetId()))["ESP"].values for conf in mol.GetConformers()]
    weights= [float(conf.GetProp("Boltzmann_distribution")) for conf in mol.GetConformers()]
    dfp["Dt"]=np.average(Dts,weights=weights,axis=0)
    dfp["ESP"]=np.average(ESPs,weights=weights,axis=0)
    dfp.to_csv(out_name + "/feature.csv".format(conf.GetId()))

    Dts = np.stack(
        [pd.read_csv(out_name + "/feature_y_{}.csv".format(conf.GetId()))["Dt"].values for conf in mol.GetConformers()])
    ESPs = [pd.read_csv(out_name + "/feature_y_{}.csv".format(conf.GetId()))["ESP"].values for conf in mol.GetConformers()]
    weights = [float(conf.GetProp("Boltzmann_distribution")) for conf in mol.GetConformers()]
    dfp_y["Dt"] = np.average(Dts, weights=weights, axis=0)
    dfp_y["ESP"] = np.average(ESPs, weights=weights, axis=0)
    dfp_y.to_csv(out_name + "/feature_y.csv".format(conf.GetId()))

    Dts = np.stack(
        [pd.read_csv(out_name + "/feature_z_{}.csv".format(conf.GetId()))["Dt"].values for conf in mol.GetConformers()])
    ESPs = [pd.read_csv(out_name + "/feature_z_{}.csv".format(conf.GetId()))["ESP"].values for conf in mol.GetConformers()]
    weights = [float(conf.GetProp("Boltzmann_distribution")) for conf in mol.GetConformers()]
    dfp_z["Dt"] = np.average(Dts, weights=weights, axis=0)
    dfp_z["ESP"] = np.average(ESPs, weights=weights, axis=0)
    dfp_z.to_csv(out_name + "/feature_z.csv".format(conf.GetId()))

    Dts = np.stack(
        [pd.read_csv(out_name + "/feature_yz_{}.csv".format(conf.GetId()))["Dt"].values for conf in mol.GetConformers()])
    ESPs = [pd.read_csv(out_name + "/feature_yz_{}.csv".format(conf.GetId()))["ESP"].values for conf in mol.GetConformers()]
    weights = [float(conf.GetProp("Boltzmann_distribution")) for conf in mol.GetConformers()]
    dfp_yz["Dt"] = np.average(Dts, weights=weights, axis=0)
    dfp_yz["ESP"] = np.average(ESPs, weights=weights, axis=0)
    dfp_yz.to_csv(out_name + "/feature_yz.csv".format(conf.GetId()))


def energy_to_Boltzmann_distribution(mol, RT=1.99e-3 * 273):
    energies = np.array([float(conf.GetProp("energy")) for conf in mol.GetConformers()])
    energies = energies - np.min(energies)
    rates = np.exp(-energies / RT)
    rates = rates / sum(rates)
    for conf, rate in zip(mol.GetConformers(), rates):
        conf.SetProp("Boltzmann_distribution", str(rate))


if __name__ == '__main__':
    param_file_name = "../parameter/parameter_cbs.txt"
    with open(param_file_name, "r") as f:
        param = json.loads(f.read())
    #座標を読み込む

    df1 = pd.read_excel( "../arranged_dataset/cbs.xls")
    df2 = pd.read_excel("../arranged_dataset/DIP-chloride.xls")
    df=pd.concat([df1,df2]).dropna(subset=['smiles'])
    df["mol"]=df["smiles"].apply(calculate_conformation.get_mol)
    df=df[[os.path.isdir(param["cube_dir_name"]+"/"+mol.GetProp("InchyKey"))for mol in df["mol"]]]
    df["mol"].apply(lambda mol: calculate_conformation.read_xyz(mol,param["cube_dir_name"]+"/"+mol.GetProp("InchyKey")))

    #gridのパラメータを読み込む

    dfp=pd.read_csv(param["grid_coordinates_file"])
    print(dfp)
    #グリッド情報に変換
    for mol in df["mol"]:
        energy_to_Boltzmann_distribution(mol, RT=0.54)
        read_cube(param["cube_dir_name"], dfp, mol,param["grid_dir_name"]+"/"+mol.GetProp("InchyKey"))
