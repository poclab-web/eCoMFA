import pandas as pd
import numpy as np
from itertools import product
import os

out_dir_name="../penalty_param"
out_file_name="/penalty_param.csv"
if __name__ == '__main__':#cbsだいたい0.01あたりから,dip0.1あたりから
    # sr = {"λ1":np.arange(0.05,0.2,0.05)*10,
    #       "λ2":np.arange(0.05,0.2,0.05)*10}
    # sr = {"λ1": [0.01,0.05,0.1,0.5,1,5,10],
    #       "λ2": [0.01,0.05,0.1,0.5,1,5,10]}
    sr = {"Dtparam": [ 0.01,0.05,0.1, 0.5,1,5,10,50,100,500],
          # "ESPparam": [ 0.01,0.05,0.1,0.5, 1],
          # "LUMOparam":[0.001,0.005],
          # "ESP_cutoffparam": [ 0.01,0.05,0.1,0.5, 1]
          }
    # sr = {"Dtparam": [50, 100],
    #       # "ESPparam": [ 0.01,0.05,0.1,0.5, 1],
    #       # "LUMOparam":[0.001,0.005],
    #       # "ESP_cutoffparam": [ 0.01,0.05,0.1,0.5, 1]
    #       }
    sr = {"Dtparam": [0.1, 0.2,0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 102.4,204.8],
          # "ESPparam": [ 0.01,0.05,0.1,0.5, 1],
          # "LUMOparam":[0.001,0.005],
          # "ESP_cutoffparam": [ 0.01,0.05,0.1,0.5, 1]
          }
    sr = {"Dtparam": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
          # "ESPparam": [ 0.01,0.05,0.1,0.5, 1],
          # "LUMOparam":[0.001,0.005],
          # "ESP_cutoffparam": [ 0.01,0.05,0.1,0.5, 1]
          }
    # sr = {"λ1": [0.1, 0.5, 1, 5],
    #       "λ2": [ 0.1, 0.5, 1, 5]}
    dfp = pd.DataFrame([dict(zip(sr.keys(), l)) for l in product(*sr.values())]).astype(float)
    print(dfp)
    dfp["ESP_cutoffparam"]=dfp["Dtparam"]
    dfp["ESPparam"]=dfp["Dtparam"]

    os.makedirs(out_dir_name,exist_ok=True)
    dfp.to_csv(out_dir_name+"/"+out_file_name)