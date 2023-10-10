import pandas as pd
import numpy as np
from itertools import product
import os

out_dir_name="../penalty_param"
out_file_name="/penalty_param.csv"
if __name__ == '__main__':
    # sr = {"λ1":np.arange(0.05,0.2,0.05)*10,
    #       "λ2":np.arange(0.05,0.2,0.05)*10}
    sr = {"λ1": [0.01,0.1,1,10],
          "λ2": [0.01,0.1,1,10]}
    dfp = pd.DataFrame([dict(zip(sr.keys(), l)) for l in product(*sr.values())]).astype(float)
    os.makedirs(out_dir_name,exist_ok=True)
    dfp.to_csv(out_dir_name+"/"+out_file_name)