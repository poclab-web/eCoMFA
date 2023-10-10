import pandas as pd
import numpy as np
from itertools import product
import os

out_dir_name="../grid_coordinates"
out_file_name="/grid_coordinate_cbs.csv"
out_file_name2="/grid_coordinate_dip-chloride.csv"
if __name__ == '__main__':
    # sr = {"x":np.round(np.arange(-4.75,0,0.5),2),
    #       "y":np.round(np.arange(-2.75,3,0.5),2),
    #       "z":np.round(np.arange(-4.75,5,0.5),2)}
    sr ={"x": np.round(np.arange(-5.75, 0.2, 0.5), 2),
     "y": np.round(np.arange(-2.75, 3, 0.5), 2),
     "z": np.round(np.arange(-5.75, 6, 0.5), 2)}
    dfp = pd.DataFrame([dict(zip(sr.keys(), l)) for l in product(*sr.values())]).astype(float)
    os.makedirs(out_dir_name,exist_ok=True)
    dfp.to_csv(out_dir_name+"/"+out_file_name)
    dfp.to_csv(out_dir_name + "/" + out_file_name2)

