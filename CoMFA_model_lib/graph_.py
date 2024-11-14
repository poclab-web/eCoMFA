import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path="/Users/mac_poclab/PycharmProjects/CoMFA_model/arranged_dataset/cbs_regression.pkl"
    df=pd.read_pickle(path)
    plt.scatter(df["ΔΔG.expt."],df["predict 1 1"])
    path=path.replace(".pkl",".png")
    plt.savefig(path)