import pandas as pd
import numpy as np
import json
from itertools import product
import os

out_dir_name="../grid_coordinates"
out_file_name="/grid_coordinate_cbs"
out_file_name2="/grid_coordinate_dip-chloride"
out_file_name3="/grid_coordinate_RuSS"

if __name__ == '__main__':
    for param_file_name in ["../parameter_nomax/parameter_cbs_gaussian.txt",
                            "../parameter_nomax/parameter_dip-chloride_gaussian.txt",
                            "../parameter_nomax/parameter_RuSS_gaussian.txt"]:
        with open(param_file_name, "r") as f:
            param = json.loads(f.read())
        gridinterval = 0.4
        xgrid =5
        ygrid =2
        zgrid =6
        # sr = {"x":np.round(np.arange(-4.75,0,0.5),2),
        #       "y":np.round(np.arange(-2.75,3,0.5),2),
        #       "z":np.round(np.arange(-4.75,5,0.5),2)}
        # # sr ={"x": np.round(np.arange(-4, 1.11, 0.5), 2),
        # #  "y": np.round(np.arange(-2, 2.11, 0.5), 2),
        # #  "z": np.round(np.arange(-4, 4.11, 0.5), 2)}
        # sr = {"x": np.round(np.arange(-9, 1.1, 1), 2),
        #       "y": np.round(np.arange(-3, 3.1, 1), 2),
        #       "z": np.round(np.arange(-9, 9.1, 1), 2)}
        # # sr = {"x": np.round(np.arange(-5, 1.1, 1), 2),
        # #       "y": np.round(np.arange(-2, 2.1, 1), 2),
        # #       "z": np.round(np.arange(-5, 5.1, 1), 2)}
        # sr = {"x": np.round(np.arange(-8, 1.1, 0.5), 2),
        #       "y": np.round(np.arange(-3, 3.1, 0.5), 2),
        #       "z": np.round(np.arange(-9, 9.1, 0.5), 2)}
        sr = {"x": np.round(np.arange(-xgrid, 1.1, gridinterval), 2),
              "y": np.round(np.arange(-ygrid, ygrid+0.1, gridinterval), 2),
              "z": np.round(np.arange(-zgrid, zgrid+0.1, gridinterval), 2)}

        dfp = pd.DataFrame([dict(zip(sr.keys(), l)) for l in product(*sr.values())]).astype(float)
        os.makedirs(out_dir_name+"/"+param["grid_coordinates_dir"],exist_ok=True)
        dfp.to_csv(out_dir_name+"/"+param["grid_coordinates_dir"]+"/[{},{},{},{}].csv".format(xgrid,ygrid,zgrid,gridinterval))


        # dfp.to_csv(out_dir_name + "/" + out_file_name2)
        # dfp.to_csv(out_dir_name + "/" + out_file_name3)

