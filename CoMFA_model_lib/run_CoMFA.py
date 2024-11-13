import glob
import json
import multiprocessing
import os
import time
import warnings
import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
from sklearn import linear_model
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import time
import calculate_conformation
from sklearn.pipeline import Pipeline
warnings.simplefilter('ignore')

# シャッフル前の順序に戻す関数
def unshuffle_array(shuffled_array, shuffled_indices):
    original_array = np.empty_like(shuffled_array)
    original_array[shuffled_indices] = shuffled_array
    return original_array

def calc_x(df):
    df_train=df[df["test"]==0]
    ste_train=np.array(df_train["Dt"].values.tolist())/np.sqrt(np.sum(np.array(df_train["Dt"].values.tolist())**2))
    esp_train=np.array(df_train["ESP"].values.tolist())/np.sqrt(np.sum(np.array(df_train["ESP"].values.tolist())**2))
    df_coord["Dt_std"]=np.average(np.sum(ste_train**2,axis=0))
    df_coord["ESP_std"]=np.average(np.sum(esp_train**2,axis=0))
    X_train=np.concatenate([ste_train,esp_train],1).astype("float32")
    return X_train

def ElasticNet(input):
    alpha,l1ratio,n_splits,n_repeats,df,df_coord,save_name=input
    df_train=df[df["test"]==0]
    df_test=df[df["test"]==1]
    ste_train=np.array(df_train["Dt"].values.tolist())/np.sqrt(np.sum(np.array(df_train["Dt"].values.tolist())**2))
    esp_train=np.array(df_train["ESP"].values.tolist())/np.sqrt(np.sum(np.array(df_train["ESP"].values.tolist())**2))
    df_coord["Dt_std"]=np.average(np.sum(ste_train**2,axis=0))
    df_coord["ESP_std"]=np.average(np.sum(esp_train**2,axis=0))
    X_train=np.concatenate([ste_train,esp_train],1).astype("float32")
    y_train=df_train["ΔΔG.expt."].values
    model = linear_model.ElasticNet(alpha=alpha,l1_ratio=l1ratio, fit_intercept=False).fit(X_train, y_train)
    df_train["regression"]=model.predict(X_train)
    # df_train["steric_cont"]=X_train[:,:len(df_coord)]@model.coef_[:len(df_coord)]
    # df_train["electrostatic_cont"]=X_train[:,len(df_coord):2*len(df_coord)]@model.coef_[len(df_coord):2*len(df_coord)]
    df_coord["coef_steric"]=model.coef_[:len(df_coord)]
    df_coord["coef_electric"]=model.coef_[len(df_coord):2*len(df_coord)]
    df_train["steric_cont"]=ste_train@df_coord["coef_steric"].values
    df_train["electrostatic_cont"]=esp_train@df_coord["coef_electric"].values
    df_coord[["coef_steric","coef_electric"]]=df_coord[["coef_steric","coef_electric"]].values*df_coord[["Dt_std","ESP_std"]].values
    df_coord.to_csv(save_name + "_coef.csv", index=False)

    result=[]
    for repeat in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=repeat)
        predict=[]
        sort_index=[]
        for train_index, test_index in kf.split(X_train):
            predict_cv = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1ratio, fit_intercept=False).fit(X_train[train_index], y_train[train_index]).predict(X_train[test_index])
            predict.extend(predict_cv.tolist())
            sort_index.extend(test_index.tolist())
        predict=unshuffle_array(np.clip(predict, np.min(y_train), np.max(y_train)),sort_index)
        RMSE=mean_squared_error(y_train,predict,squared=False)
        r2=r2_score(y_train,predict)
        result.append([RMSE,r2])
        df_train["validation{}".format(repeat)]=predict
    df_train["error"]=df_train["validation{}".format(repeat)]-df_train["ΔΔG.expt."]
    df_train=df_train.sort_values(by='error', ascending=False)
    ste_test=np.array(df_test["Dt"].values.tolist())/np.sqrt(np.sum(np.array(df_train["Dt"].values.tolist())**2))
    esp_test=np.array(df_test["ESP"].values.tolist())/np.sqrt(np.sum(np.array(df_train["ESP"].values.tolist())**2))
    X_test=np.concatenate([ste_test,esp_test],1).astype("float32")
    df_test["prediction"]=model.predict(X_test)
    df_test["steric_cont"]=X_test[:,:len(df_coord)]@model.coef_[:len(df_coord)]
    df_test["electrostatic_cont"]=X_test[:,len(df_coord):2*len(df_coord)]@model.coef_[len(df_coord):2*len(df_coord)]
    df_test["error"]=df_test["prediction"]-df_test["ΔΔG.expt."]
    df_test=df_test.sort_values(by='error', ascending=False)
    
    df_concat = pd.concat([df_train, df_test])
    df_concat["test"]=df_concat["test"].astype(str)
    df_concat.drop(columns=['Dt', 'DtR1',"DtR2",'ESP', 'ESPR1',"ESPR2"], inplace=True, errors='ignore')
    PandasTools.AddMoleculeColumnToFrame(df_concat, "smiles")
    df_concat.to_excel(save_name + "_prediction.xlsx")
    PandasTools.SaveXlsxFromFrame(df_concat.fillna("NaN"), save_name + "_prediction.xlsx", size=(100, 100))
    RMSE_train=mean_squared_error(df_train["ΔΔG.expt."],df_train["regression"],squared=False)
    r2_train=r2_score(df_train["ΔΔG.expt."],df_train["regression"])
    RMSE_test=mean_squared_error(df_test["ΔΔG.expt."],df_test["prediction"],squared=False)
    r2_test=r2_score(df_test["ΔΔG.expt."],df_test["prediction"])
    integ=np.sum(np.abs(df_coord[["coef_steric","coef_electric"]].values),axis=0).tolist()
    return [save_name,alpha,l1ratio,RMSE_train,r2_train]+np.average(result,axis=0).tolist()+[RMSE_test,r2_test]+integ

def PLS(input):
    alpha,n_splits,n_repeats,df,df_coord,save_name=input
    df_train=df[df["test"]==0]
    df_test=df[df["test"]==1]
    ste_train=np.array(df_train["Dt"].values.tolist())/np.sqrt(np.average(np.array(df_train["Dt"].values.tolist())**2))
    esp_train=np.array(df_train["ESP"].values.tolist())/np.sqrt(np.average(np.array(df_train["ESP"].values.tolist())**2))
    df_coord["Dt_std"]=np.sqrt(np.average(ste_train**2,axis=0))
    df_coord["ESP_std"]=np.sqrt(np.average(esp_train**2,axis=0))
    X_train=np.concatenate([ste_train,esp_train],1).astype("float32")
    y_train=df_train["ΔΔG.expt."].values
    model = PLSRegression(n_components=alpha).fit(X_train, y_train)
    df_train["regression"]=model.predict(X_train)
    df_train["steric_cont"]=X_train[:,:len(df_coord)]@model.coef_[0,:len(df_coord)]
    df_train["electrostatic_cont"]=X_train[:,len(df_coord):2*len(df_coord)]@model.coef_[0,len(df_coord):2*len(df_coord)]
    df_coord["coef_steric"]=model.coef_[0,:len(df_coord)]
    df_coord["coef_electric"]=model.coef_[0,len(df_coord):2*len(df_coord)]
    df_coord[["coef_steric","coef_electric"]]=df_coord[["coef_steric","coef_electric"]].values*df_coord[["Dt_std","ESP_std"]].values
    df_coord.to_csv(save_name + "_coef.csv", index=False)

    result=[]
    for repeat in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=repeat)
        predict=[]
        sort_index=[]
        for train_index, test_index in kf.split(X_train):
            predict_cv = PLSRegression(n_components=alpha).fit(X_train[train_index], y_train[train_index]).predict(X_train[test_index])
            predict.extend(predict_cv.tolist())
            sort_index.extend(test_index.tolist())
        predict=unshuffle_array(np.clip(predict, np.min(y_train), np.max(y_train)),sort_index)
        RMSE=mean_squared_error(y_train,predict,squared=False)
        r2=r2_score(y_train,predict)
        result.append([RMSE,r2])
        df_train["validation{}".format(repeat)]=predict
    
    ste_test=np.array(df_test["Dt"].values.tolist())/np.sqrt(np.average(np.array(df_train["Dt"].values.tolist())**2))
    esp_test=np.array(df_test["ESP"].values.tolist())/np.sqrt(np.average(np.array(df_train["ESP"].values.tolist())**2))
    X_test=np.concatenate([ste_test,esp_test],1).astype("float32")
    df_test["prediction"]=model.predict(X_test)
    df_test["steric_cont"]=X_test[:,:len(df_coord)]@model.coef_[0,:len(df_coord)]
    df_test["electrostatic_cont"]=X_test[:,len(df_coord):2*len(df_coord)]@model.coef_[0,len(df_coord):2*len(df_coord)]
    df_concat = pd.concat([df_train, df_test])
    df_concat["test"]=df_concat["test"].astype(str)
    df_concat.drop(columns=['Dt', 'DtR1',"DtR2",'ESP', 'ESPR1',"ESPR2"], inplace=True, errors='ignore')
    PandasTools.AddMoleculeColumnToFrame(df_concat, "smiles")
    df_concat.to_excel(save_name + "_prediction.xlsx")
    PandasTools.SaveXlsxFromFrame(df_concat.fillna("NaN"), save_name + "_prediction.xlsx", size=(100, 100))
    RMSE_train=mean_squared_error(y_train,df_train["regression"],squared=False)
    r2_train=r2_score(y_train,df_train["regression"])
    RMSE_test=mean_squared_error(df_test["ΔΔG.expt."],df_test["prediction"],squared=False)
    r2_test=r2_score(df_test["ΔΔG.expt."],df_test["prediction"])
    integ=np.sum(np.abs(df_coord[["coef_steric","coef_electric"]].values),axis=0).tolist()
    return [save_name,alpha,RMSE_train,r2_train]+np.average(result,axis=0).tolist()+[RMSE_test,r2_test]+integ

def PCR(input):
    alpha,n_splits,n_repeats,df,df_coord,save_name=input
    df_train=df[~df["test"]]
    df_test=df[df["test"]]
    ste_train=np.array(df_train["Dt"].values.tolist())/np.sqrt(np.average(np.array(df_train["Dt"].values.tolist())**2))
    esp_train=np.array(df_train["ESP"].values.tolist())/np.sqrt(np.average(np.array(df_train["ESP"].values.tolist())**2))
    X_train=np.concatenate([ste_train,esp_train],1).astype("float32")
    y_train=df_train["ΔΔG.expt."].values
    # PCR回帰のパイプラインを構築（切片をゼロに設定）
    pcr = Pipeline([
        ('pca', PCA(n_components=10)),  # 主成分数を指定
        ('regression', linear_model.LinearRegression(fit_intercept=False))
    ])
    model = pcr.fit(X_train, y_train)
    df_train["regression"]=model.predict(X_train)
    df_train["steric_cont"]=X_train[:,:len(df_coord)]@model.coef_[0,:len(df_coord)]
    df_train["electrostatic_cont"]=X_train[:,len(df_coord):2*len(df_coord)]@model.coef_[0,len(df_coord):2*len(df_coord)]
    df_coord["coef_steric"]=model.coef_[0,:len(df_coord)]
    df_coord["coef_electric"]=model.coef_[0,len(df_coord):2*len(df_coord)]
    df_coord[["coef_steric","coef_electric"]]=df_coord[["coef_steric","coef_electric"]].values*df_coord[["Dt_std","ESP_std"]].values
    df_coord.to_csv(save_name + "_coef.csv", index=False)

    result=[]
    for repeat in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=repeat)
        predict=[]
        sort_index=[]
        for train_index, test_index in kf.split(X_train):
            predict_cv = pcr.fit(X_train[train_index], y_train[train_index]).predict(X_train[test_index])
            predict.extend(predict_cv.tolist())
            sort_index.extend(test_index.tolist())
        predict=unshuffle_array(np.clip(predict, np.min(y_train), np.max(y_train)),sort_index)
        RMSE=mean_squared_error(y_train,predict,squared=False)
        r2=r2_score(y_train,predict)
        result.append([RMSE,r2])
        df_train["validation{}".format(repeat)]=predict
    
    ste_test=np.array(df_test["Dt"].values.tolist())/np.sqrt(np.average(np.array(df_train["Dt"].values.tolist())**2))
    esp_test=np.array(df_test["ESP"].values.tolist())/np.sqrt(np.average(np.array(df_train["ESP"].values.tolist())**2))
    X_test=np.concatenate([ste_test,esp_test],1).astype("float32")
    df_test["prediction"]=model.predict(X_test)
    df_test["steric_cont"]=X_test[:,:len(df_coord)]@model.coef_[0,:len(df_coord)]
    df_test["electrostatic_cont"]=X_test[:,len(df_coord):2*len(df_coord)]@model.coef_[0,len(df_coord):2*len(df_coord)]
    df_concat = pd.concat([df_train, df_test])
    df_concat["test"]=df_concat["test"].astype(str)
    df_concat.drop(columns=['Dt', 'DtR1',"DtR2",'ESP', 'ESPR1',"ESPR2"], inplace=True, errors='ignore')
    PandasTools.AddMoleculeColumnToFrame(df_concat, "smiles")
    df_concat.to_excel(save_name + "_prediction.xlsx")
    PandasTools.SaveXlsxFromFrame(df_concat.fillna("NaN"), save_name + "_prediction.xlsx", size=(100, 100))
    RMSE_train=mean_squared_error(y_train,df_train["regression"],squared=False)
    r2_train=r2_score(y_train,df_train["regression"])
    RMSE_test=mean_squared_error(df_test["ΔΔG.expt."],df_test["prediction"],squared=False)
    r2_test=r2_score(df_test["ΔΔG.expt."],df_test["prediction"])
    integ=np.sum(np.abs(df_coord[["coef_steric","coef_electric"]].values),axis=0).tolist()
    return [save_name,alpha,RMSE_train,r2_train]+np.average(result,axis=0).tolist()+[RMSE_test,r2_test]+integ


def energy_to_Boltzmann_distribution(mol, RT=1.99e-3 * 273):
    energies = []
    for conf in mol.GetConformers():
        # print(conf.GetId())
        line = json.loads(conf.GetProp("freq"))
        energies.append(float(line[0] - line[1] * RT / 1.99e-3))
    energies = np.array(energies)
    energies = energies - np.min(energies)
    rates = np.exp(-energies / RT)
    rates = rates / np.sum(rates)
    for conf, rate in zip(mol.GetConformers(), rates):
        conf.SetProp("Boltzmann_distribution", str(rate))


def is_normal_frequencies(filename):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            frequencies_lines = [line for line in lines if 'Frequencies' in line]
            if len(frequencies_lines)==0:
                f.close()
                print(filename, "fail_freq_calculation")
                return False
            for l in frequencies_lines:
                splited = l.split()
                values = splited[2:]
                values = [float(v)>0 for v in values]
                ans=all(values)
                if not ans:
                    f.close()
                    print(filename,"imaginary freq")
                    return ans
            f.close()
        return True
    except:
        print(filename," is bad conformer")
        return False

def get_free_energy(mol,dir):
    # dirs_name_freq = param["freq_dir"] + "/" + mol.GetProp("InchyKey") + "/gaussianinput?.log"
    del_list=[]

    for conf in mol.GetConformers():
        path=dir + "/" + mol.GetProp("InchyKey") + "/gaussianinput{}.log".format(conf.GetId())
        if is_normal_frequencies(path):
            try:
                with open(path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if 'Sum of electronic and thermal Enthalpies=' in line:
                            ent=float(line.split()[6])* 627.5095
                        if "Total     " in line:
                            entr=float(line.split()[3])/1000
                conf.SetProp("freq", json.dumps([ent, entr]))

            except:
                print("read error,",path)
        else:
            del_list.append(conf.GetId())
    for _ in del_list:
        mol.RemoveConformer(_)

def get_grid_feat(mol,RT,dir):
    energy_to_Boltzmann_distribution(mol, RT)
    feat = []
    # ESP = []
    features=["Dt","ESP"]
    we = []
    for conf in mol.GetConformers():
        data = pd.read_pickle(
            "{}/{}/data{}.pkl".format(dir, mol.GetProp("InchyKey"), conf.GetId()))
        Bd = float(conf.GetProp("Boltzmann_distribution"))
        we.append(Bd)
        feat.append(data[features].values.tolist())
    data[features] = np.nan_to_num(np.average(feat, weights=we, axis=0))
    # data.to_pickle( "{}/{}/dataBoltzmann.pkl".format(dir, mol.GetProp("InchyKey")))
    ans1=data[(data["y"] > 0)&(data["z"]>0)].sort_values(['x', 'y', "z"], ascending=[True, True, True])[features].values
    ans2=data[(data["y"] < 0)&(data["z"]>0)].sort_values(['x', 'y', "z"], ascending=[True, False, True])[features].values
    ans3=data[(data["y"] > 0)&(data["z"]<0)].sort_values(['x', 'y', "z"], ascending=[True, True, False])[features].values
    ans4=data[(data["y"] < 0)&(data["z"]<0)].sort_values(['x', 'y', "z"], ascending=[True, False, False])[features].values
    ans=ans1+ans2-ans3-ans4
    # ans.to_pickle( "{}/{}/dataBoltzmannfold.pkl".format(dir, mol.GetProp("InchyKey")))
    return ans.T.tolist()


if __name__ == '__main__':
    ridge_input=[]
    lasso_input=[]
    elasticnet_input=[]
    pls_input=[]
    pcr_input=[]
    for param_name in sorted(glob.glob("C:/Users/poclabws/PycharmProjects/CoMFA_model/parameter/cube_to_grid0.50.txt"),reverse=True):
        print(param_name)
        with open(param_name, "r") as f:
            param = json.loads(f.read())
        print(param)
        start = time.perf_counter()  # 計測開始
        for file in glob.glob("C:/Users/poclabws/PycharmProjects/CoMFA_model/arranged_dataset/*.xlsx"):
            df = pd.read_excel(file).dropna(subset=['smiles']).reset_index(drop=True)#[:10]
            file_name = os.path.splitext(os.path.basename(file))[0]
            features_dir_name = param["grid_coordinates"] + file_name
            print(len(df),features_dir_name)
            df["mol"] = df["smiles"].apply(calculate_conformation.get_mol)
            for mol in df["mol"]:
                if len(glob.glob(f'{param["grid_coordinates"]}/{mol.GetProp("InchyKey")}/*'))==0:
                    print(mol.GetProp("InchyKey"))
            # df = df[
            #     [len(glob.glob(f'{param["grid_coordinates"]}/{mol.GetProp("InchyKey")}/*'))>0 for mol in
            #      df["mol"]]]
            print(len(df),features_dir_name)
            df["mol"].apply(
                lambda mol: calculate_conformation.read_xyz(mol,
                                                            param["opt_structure"] + "/" + mol.GetProp("InchyKey")))
            df["mol"].apply(lambda mol : get_free_energy(mol,param["freq_dir"]))
            df=df[[mol.GetNumConformers()>0 for mol in df["mol"]]]
            print(len(df),features_dir_name)
            df[["Dt","ESP"]]=df[["mol", "RT"]].apply(lambda _: get_grid_feat(_[0], _[1],param["grid_coordinates"]), axis=1, result_type='expand')
            print("feature_calculated")
            # X=np.array(df["Dt"].values.tolist())/np.sqrt(np.sum(np.array(df["Dt"].values.tolist())**2))
            # X_esp=np.array(df["ESP"].values.tolist())/np.sqrt(np.sum(np.array(df["ESP"].values.tolist())**2))
            df_coord = pd.read_csv(param["grid_coordinates"] + "/coordinates_yz.csv").sort_values(['x', 'y', "z"],
                                                                                  ascending=[True, True, True])
            n_splits = int(param["n_splits"])
            n_repeats = int(param["n_repeats"])
            dir=param["out_dir_name"] + "/{}".format(file_name)
            os.makedirs(dir,exist_ok=True)

            alphas = np.logspace(-20,-10,11,base=2)
            l1ratios = np.linspace(0, 1, 11)

            for alpha in alphas:
                for l1ratio in l1ratios:
                    save_name=dir + "/ElasticNet_alpha_{}_l1ratio{}".format(alpha,l1ratio)
                    elasticnet_input.append([alpha,l1ratio,n_splits,n_repeats,df,df_coord,save_name])

            alphas = np.arange(1,10)
            for alpha in alphas:
                save_name=dir + "/PLS_alpha_{}".format(alpha)
                pls_input.append([alpha,n_splits,n_repeats,df,df_coord,save_name])
            # alphas = np.arange(1,10)
            # for alpha in alphas:
            #     save_name=dir + "/PCR_alpha_{}".format(alpha)
            #     pcr_input.append([alpha,n_splits,n_repeats,df,df_coord,save_name])
    p = multiprocessing.Pool(processes=int(param["processes"]))
    columns=["savefilename","alpha","RMSE_regression", "r2_regression","RMSE_validation", "r2_validation","RMSE_PCA_validation","r2_PCA_validation","integral_steric","integral_electric"]
    elasticnet_columns=["savefilename","alpha","l1ratio","RMSE_regression", "r2_regression","RMSE_validation", "r2_validation","RMSE_PCA_validation","r2_PCA_validation","integral_steric","integral_electric"]
    os.makedirs(param["out_dir_name"],exist_ok=True)
    pd.DataFrame(p.map(PLS,pls_input), columns=columns).apply(lambda x: x.round(4) if np.issubdtype(x.dtype, np.floating) else x).to_csv(param["out_dir_name"] + "/PLS.csv")
    pd.DataFrame(p.map(ElasticNet,elasticnet_input), columns=elasticnet_columns).apply(lambda x: x.round(4) if np.issubdtype(x.dtype, np.floating) else x).to_csv(param["out_dir_name"] + "/ElasticNet.csv")
    end = time.perf_counter()  # 計測終了
    print('Finish{:.2f}'.format(end - start))
    