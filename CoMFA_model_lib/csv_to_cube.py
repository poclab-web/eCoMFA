import pandas as pd
import json
import os
import copy

if __name__ == '__main__':
    param_file_name = "../parameter/parameter_cbs.txt"
    with open(param_file_name, "r") as f:
        param = json.loads(f.read())
    features_dir_name = "moleculer_field"#param["grid_dir_name"]
    inchykey="ADCYRBXQAJXJTD-UHFFFAOYSA-N"
    feature="MF_Dt"
    df=pd.read_csv("../moleculer_field/moleculer_field.csv")#"{}/{}/feature_yz.csv".format(features_dir_name, inchykey))
    df_y=copy.deepcopy(df)
    df_z=copy.deepcopy(df)
    df_y["y"]=-df["y"]
    df_z["z"]=-df["z"]
    df_z[feature]=-df[feature]
    df_yz=copy.deepcopy(df)
    df_yz["y"]=-df["y"]
    df_yz["z"]=-df["z"]
    df_yz[feature]=-df[feature]
    df=pd.concat([df,df_y,df_z,df_yz]).sort_values(by=["x","y","z"])



    cube_file_name="../cube_aligned_b3lyp_6-31g(d)/ADCYRBXQAJXJTD-UHFFFAOYSA-N/Dt02_0.cube"
    with open(cube_file_name, "r", encoding="UTF-8") as f:
        Dt = f.read().splitlines()
    n_atom=int(Dt[2].split()[0])
    xyz=Dt[6:6+n_atom]
    dir_name="../moleculer_field"
    os.makedirs(dir_name,exist_ok=True)
    to_cube_file_name=dir_name+"/test.cube"
    print(len(df))
    drop_dupl_x=df.drop_duplicates(subset="x").sort_values('x')["x"]
    drop_dupl_y = df.drop_duplicates(subset="y").sort_values('y')["y"]
    drop_dupl_z = df.drop_duplicates(subset="z").sort_values('z')["z"]
    d_x=drop_dupl_x.iloc[1]-drop_dupl_x.iloc[0]
    d_y=drop_dupl_y.iloc[1]-drop_dupl_y.iloc[0]
    d_z=drop_dupl_z.iloc[1]-drop_dupl_z.iloc[0]

    print(d_x,d_y,d_z)
    print(drop_dupl_x.max(),drop_dupl_x.min())
    range_x=drop_dupl_x.max()-drop_dupl_x.min()
    range_y=drop_dupl_y.max()-drop_dupl_y.min()
    range_z=drop_dupl_z.max()-drop_dupl_z.min()
    count_x=int(range_x/d_x+1)
    count_y = int(range_y / d_y + 1)
    count_z = int(range_z / d_z + 1)
    print(count_x,count_y,count_z)
    with open( to_cube_file_name, "w") as f:
        print(0,file=f)
        print(1,file=f)
        print("    {} {} {} {}".format(n_atom,drop_dupl_x.min(),drop_dupl_y.min(),drop_dupl_z.min()),file=f)
        print("    {}   {:.5f} {:.5f} {:.5f}".format(count_x,d_x,0,0),file=f)
        print("    {}   {:.5f} {:.5f} {:.5f}".format(count_y,0, d_y,  0), file=f)
        print("    {}   {:.5f} {:.5f} {:.5f}".format(count_z, 0,0,d_z), file=f)
        print("\n".join(xyz),file=f)
        for i in range(len(df)//6+1):
            line=df[feature].iloc[6*i:6*i+6].values.tolist()
            #line=list(map(str, line))
            #line=" ".join(line)
            line=['{:.5E}'.format(n) for n in line]
            line=" ".join(line)
            print(line)
            print(line,file=f)

    print(xyz)
