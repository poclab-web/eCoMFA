import copy
import glob

import pandas as pd

if __name__ == '__main__':
    for file in glob.glob("../../../result/*/*/comparison0/mo*1024.csv"):
        print(file)
        df = pd.read_csv(file)
        # df = mfunfolding(df)

        df_y = copy.deepcopy(df)
        df_y["y"] = -df_y["y"]
        df_z = copy.deepcopy(df)
        df_z["z"] = -df_z["z"]
        df_yz = copy.deepcopy(df)
        df_yz[["y"]] = -df_yz[["y"]]
        df_yz["z"] = -df_yz["z"]

        feature1 = 'Dt'
        feature2 = "ESP"

        # 対象文字列を含む列名を取得
        column_inc_specific_feature = [column for column in df_z.columns if
                                       (feature1 in column) or (feature2 in column)]

        df_z[column_inc_specific_feature] = -df_z[column_inc_specific_feature]
        df_yz[column_inc_specific_feature] = -df_yz[column_inc_specific_feature]

        df = pd.concat([df, df_y, df_z, df_yz])
        df = df.sort_values(by=["x", "y", "z"], ascending=[True, True, True])

        # to_cube_file_name = file[:-4] + "val" + ".csv"
        # df.to_csv(to_cube_file_name)
        # print(to_cube_file_name)
        print(column_inc_specific_feature)

        # df[(df["x"]>0)&(df["y"]>0)][column_inc_specific_feature]=1
        cubeinchikey = "../cube_aligned/wB97X-D_def2-TZVP20240308_1/RIFKADJTWUGDOV-UHFFFAOYSA-N/Dt02_0.cube"
        with open(cubeinchikey, "r", encoding="UTF-8") as f:
            cube = f.read().splitlines()
        n_atom = int(cube[2].split()[0])
        xyz = cube[6:6 + n_atom]

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
        print(range_x / d_x)
        print(range_y / d_y)
        print(range_z / d_z)

        count_x = round(range_x / d_x + 1)
        count_y = round(range_y / d_y + 1)
        count_z = round(range_z / d_z + 1)

        print(count_x, count_y, count_z, len(df))
        for column in column_inc_specific_feature:
            to_cube_file_name = file[:-4] + column + ".cube"  # os.path.splitext(os.path.basename(file))[0]
            print(to_cube_file_name)
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
                    line = df[column].iloc[6 * i:6 * i + 6].values.tolist()
                    # line=list(map(str, line))
                    # line=" ".join(line)
                    line = ['{:.5E}'.format(n) for n in line]
                    line = " ".join(line)
                    # print(line)
                    print(line, file=f)
