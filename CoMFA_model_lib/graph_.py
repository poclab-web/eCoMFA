import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def graph_(path):
    df=pd.read_pickle(path)
    columns=df.columns[df.columns.str.contains('cv ElasticNet')]
    print(columns)
    rmses=[]
    for column in columns:
        rmse=mean_squared_error(df["ΔΔG.expt."], df.loc[:,column],squared=False)
        rmses.append(rmse)
    print(rmses)
    column=columns[rmses.index(min(rmses))]
    plt.scatter(df["ΔΔG.expt."],df[column])

    path=path.replace(".pkl",".png")
    plt.savefig(path)

if __name__ == '__main__':
    path="/Users/mac_poclab/PycharmProjects/CoMFA_model/arranged_dataset/cbs_regression.pkl"
    graph_(path)
