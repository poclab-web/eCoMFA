import copy
import glob
import os

import pandas as pd

def coef_cube(file,mol_file,prop,column,to_cube_file_name):
    print(file)
    df = pd.read_csv(file)
    df_y = copy.deepcopy(df)
    df_y["y"] = -df_y["y"]
    df_z = copy.deepcopy(df)
    df_z["z"] = -df_z["z"]
    df_yz = copy.deepcopy(df)
    df_yz[["y"]] = -df_yz[["y"]]
    df_yz["z"] = -df_yz["z"]
    
    df_z[column] = -df_z[column]
    df_yz[column] = -df_yz[column]

    df = pd.concat([df,df_y,  df_z,df_yz])
    df = df.sort_values(by=["x", "y", "z"], ascending=[True, True, True])
    
    with open(mol_file, "r", encoding="UTF-8") as f:
        cube = f.read().splitlines()
    # try:
    n_atom = int(cube[2].split()[0])
    xyz = cube[6:6 + n_atom]
    # except:
    #     n_atom = int(cube[0].split()[0])
    #     xyz = cube[2:2 + n_atom]

    drop_dupl_x = df.drop_duplicates(subset="x").sort_values('x')["x"]
    drop_dupl_y = df.drop_duplicates(subset="y").sort_values('y')["y"]
    drop_dupl_z = df.drop_duplicates(subset="z").sort_values('z')["z"]

    d_x = drop_dupl_x.iloc[1] - drop_dupl_x.iloc[0]
    d_y = drop_dupl_y.iloc[1] - drop_dupl_y.iloc[0]
    d_z = drop_dupl_z.iloc[1] - drop_dupl_z.iloc[0]

    count_x = drop_dupl_x.count()  # round(range_x / d_x + 1)
    count_y = drop_dupl_y.count()  # round(range_y / d_y + 1)
    count_z = drop_dupl_z.count()  # round(range_z / d_z + 1)

    range_x = drop_dupl_x.max() - drop_dupl_x.min()
    range_y = drop_dupl_y.max() - drop_dupl_y.min()
    range_z = drop_dupl_z.max() - drop_dupl_z.min()

    count_x = round(range_x / d_x + 1)
    count_y = round(range_y / d_y + 1)
    count_z = round(range_z / d_z + 1)

    print(count_x, count_y, count_z, len(df))
      # os.path.splitext(os.path.basename(file))[0]
    print(to_cube_file_name)
    with open(to_cube_file_name, "w") as f:
        print(0, file=f)
        print(prop, file=f)
        print("    {} {} {} {}".format(n_atom, drop_dupl_x.min() / 0.52917720859,
                                        drop_dupl_y.min() / 0.52917720859,
                                        drop_dupl_z.min() / 0.52917720859), file=f)
        print("    {}   {:.5f} {:.5f} {:.5f}".format(count_x, d_x / 0.52917720859, 0, 0), file=f)
        print("    {}   {:.5f} {:.5f} {:.5f}".format(count_y, 0, d_y / 0.52917720859, 0), file=f)
        print("    {}   {:.5f} {:.5f} {:.5f}".format(count_z, 0, 0, d_z / 0.52917720859), file=f)
        print("\n".join(xyz), file=f)
        for i in range(len(df) // 6 + 1):
            line = df[column].iloc[6 * i:6 * i + 6].values.tolist()
            line = ['{:.5E}'.format(n) for n in line]
            line = " ".join(line)
            print(line, file=f)


if __name__ == '__main__':
    dir="../../../result/20240704_0_5_spl5"
    filename=dir+"/*/*_coef.csv"
    # mol_file = "D:/calculation/wB97X-D_def2-TZVP20240416/KWOLFJPFCHCOCG-UHFFFAOYSA-N/Dt02_0.cube"#"F:/cube_aligned/wB97X-D_def2-TZVP20240330/KWOLFJPFCHCOCG-UHFFFAOYSA-N/Dt02_0.cube"#RIFKADJTWUGDOV-UHFFFAOYSA-N
    mol_files=["C:/Users/poclabws/PycharmProjects/CoMFA_model/xyz_file/TS1R_B3LYP_6-31Gd_PCM_TS_new_4.xyz",
               "C:/Users/poclabws/PycharmProjects/CoMFA_model/xyz_file/dip_acetophenone_UFF_B3LYP_631Gd_PCM_IRC_new_4.xyz",
               "C:/Users/poclabws/PycharmProjects/CoMFA_model/xyz_file/Ru_acetophenone_TS_TS_TS_TS_new_4.xyz"]
    prop="Property: ALIE"
    # for file in glob.glob(filename):
    #     out_file = file[:-4]  + ".cube"
    #     coef_cube(file,mol_file,prop,out_file)
    
    for name in ["ElasticNet"]:
        filename=dir+"/{}.csv".format(name)
        df=pd.read_csv(filename,index_col = 'Unnamed: 0').sort_index()
        df["dataset"]=df.index*3//len(df)
        for _,mol_file in enumerate(mol_files):
            os.makedirs(dir+"/cube/dataset{}".format(_),exist_ok=True)
            df_=df[(df["dataset"]==_)]
            file=df_["savefilename"][df_["RMSE_validation"]==df_["RMSE_validation"].min()].iloc[0]+"_coef.csv"
            for prop in ["Property: ALIE"]:#,"Property: Default"
                for column in ["coef_steric","coef_electric"]:
                    out_file=dir+"/cube/dataset{}/{}{}{}.cube".format(_,file.replace(".","").replace('/', '_').replace("csv",""),prop.replace(": ",""),column)
                    coef_cube(file,mol_file,prop,column,out_file)
