import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score
import json
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error



# for i,param_file_name in enumerate([
# # "../parameter/parameter_cbs_gaussian.txt",
# # "../parameter/parameter_dip-chloride_gaussian.txt",
# # "../parameter/parameter_RuSS_gaussian.txt",
# #
# "../parameter_0227/parameter_cbs_gaussian.txt",
#  "../parameter_0227/parameter_cbs_PLS.txt",
#  "../parameter_0227/parameter_cbs_ridgecv.txt",
#  "../parameter_0227/parameter_cbs_elasticnetcv.txt",
# "../parameter_0227/parameter_cbs_lassocv.txt",
#
# "../parameter_0227/parameter_dip-chloride_gaussian.txt",
# "../parameter_0227/parameter_dip-chloride_PLS.txt",
# "../parameter_0227/parameter_dip-chloride_ridgecv.txt",
# "../parameter_0227/parameter_dip-chloride_elasticnetcv.txt",
# "../parameter_0227/parameter_dip-chloride_lassocv.txt",
#
# "../parameter_0227/parameter_RuSS_gaussian.txt",
# "../parameter_0227/parameter_RuSS_PLS.txt",
# "../parameter_0227/parameter_RuSS_ridgecv.txt",
# "../parameter_0227/parameter_RuSS_elasticnetcv.txt",
# "../parameter_0227/parameter_RuSS_lassocv.txt",
# # "../parameter_nomax/parameter_cbs_gaussian.txt",
# #  "../parameter_nomax/parameter_cbs_PLS.txt",
# #  "../parameter_nomax/parameter_cbs_ridgecv.txt",
# #  "../parameter_nomax/parameter_cbs_elasticnetcv.txt",
# # "../parameter_nomax/parameter_cbs_lassocv.txt",
# #
# # "../parameter_nomax/parameter_dip-chloride_gaussian.txt",
# # "../parameter_nomax/parameter_dip-chloride_PLS.txt",
# # "../parameter_nomax/parameter_dip-chloride_ridgecv.txt",
# # "../parameter_nomax/parameter_dip-chloride_elasticnetcv.txt",
# # "../parameter_nomax/parameter_dip-chloride_lassocv.txt",
# #
# # "../parameter_nomax/parameter_RuSS_gaussian.txt",
# # "../parameter_nomax/parameter_RuSS_PLS.txt",
# # "../parameter_nomax/parameter_RuSS_ridgecv.txt",
# # "../parameter_nomax/parameter_RuSS_elasticnetcv.txt",
# # "../parameter_nomax/parameter_RuSS_lassocv.txt",
#
#
#
#     ]):
#
#     with open(param_file_name, "r") as f:
#         param = json.loads(f.read())
with open("../parameter/cube_to_grid/cube_to_grid.txt", "r") as f:
    param = json.loads(f.read())
print(param)

fig =plt.figure(figsize=(3*3,1*3))
i=0
for file in glob.glob("../arranged_dataset/*.xlsx"):
    i+=1
    file_name = os.path.splitext(os.path.basename(file))[0]
    ax = fig.add_subplot(1, 3, i)
    ax.set_ylim(0.5, 1)
    ax.set_title(file_name)
    ax.set_yticks([0.5, 1])
    for _ in range(5):
        save_path = param["out_dir_name"] + "/" + file_name+"/"+str(_)
        dfp=pd.read_csv(save_path+"/result.csv")


        # ax.plot(dfp["lambda"],dfp["Gaussian_test_RMSE"],color="blue",label="Gaussian")
        # ax.plot(dfp["lambda"], dfp["lasso_test_RMSE"],color="red",label="lasso")
        # ax.plot(dfp["lambda"], dfp["ridge_test_RMSE"],color="green",label="ridge")

        ax.plot(dfp["lambda"],dfp["Gaussian_test_r"],color="blue",label="Gaussian",linewidth=1,alpha=0.5)
        ax.plot(dfp["lambda"], dfp["lasso_test_r"],color="red",label="lasso",linewidth=1,alpha=0.5)
        ax.plot(dfp["lambda"], dfp["ridge_test_r"],color="green",label="ridge",linewidth=1,alpha=0.5)
        ax.legend(loc='lower right', fontsize=6)
        #ax.set_xticks([-2, 0, 2])
        plt.xscale("log")

plt.savefig("../figs/rmse.png", transparent=False, dpi=300)

fig =plt.figure(figsize=(3*3,1*3))
i=0
for file in glob.glob("../arranged_dataset/*.xlsx"):
    i+=1
    file_name = os.path.splitext(os.path.basename(file))[0]
    ax = fig.add_subplot(1, 3, i)
    ax.set_ylim(-3, 3)
    ax.set_title(file_name)
    ax.set_yticks([-3,3])
    for _ in range(1):
        save_path = param["out_dir_name"] + "/" + file_name+"/"+str(_)
        dfp=pd.read_excel(save_path+"/result_test.xlsx")


        # ax.plot(dfp["lambda"],dfp["Gaussian_test_RMSE"],color="blue",label="Gaussian")
        # ax.plot(dfp["lambda"], dfp["lasso_test_RMSE"],color="red",label="lasso")
        # ax.plot(dfp["lambda"], dfp["ridge_test_RMSE"],color="green",label="ridge")

        ax.plot(dfp["ΔΔG.expt."],dfp["Gaussian_predict"],"o",color="blue",label="Gaussian",alpha=0.5)
        ax.plot(dfp["ΔΔG.expt."], dfp["Ridge_predict"],"o",color="red",label="Lasso",alpha=0.5)
        ax.plot(dfp["ΔΔG.expt."], dfp["Lasso_predict"],"o",color="green",label="Ridge",alpha=0.5)
        ax.legend(loc='lower right', fontsize=6)
        #ax.set_xticks([-2, 0, 2])

plt.savefig("../figs/test_predict.png", transparent=False, dpi=300)

fig =plt.figure(figsize=(3*3,3*3))
i=0
for file in glob.glob("../arranged_dataset/*.xlsx"):
    file_name = os.path.splitext(os.path.basename(file))[0]

    for _ in range(1):

        save_path = param["out_dir_name"] + "/" + file_name+"/"+str(_)
        dfp=pd.read_excel(save_path+"/result_test.xlsx")
        for name in ["Gaussian_predict","Ridge_predict","Lasso_predict"]:
            i += 1
            ax = fig.add_subplot(3, 3, i)
            ax.set_ylim(-3, 3)
            ax.set_title(file_name)
            ax.set_yticks([-3, 3])

            # ax.plot(dfp["lambda"],dfp["Gaussian_test_RMSE"],color="blue",label="Gaussian")
            # ax.plot(dfp["lambda"], dfp["lasso_test_RMSE"],color="red",label="lasso")
            # ax.plot(dfp["lambda"], dfp["ridge_test_RMSE"],color="green",label="ridge")

            ax.plot(dfp["ΔΔG.expt."],dfp[name],"o",color="blue",label=name,alpha=0.5)
            ax.legend(loc='lower right', fontsize=6)
            #ax.set_xticks([-2, 0, 2])

plt.savefig("../figs/test_predict_.png", transparent=False, dpi=300)

fig =plt.figure(figsize=(3*5, 3*3+1))
i=0
for file in glob.glob("../arranged_dataset/*.xlsx"):
    df = pd.read_excel(file).dropna(subset=['smiles']).reset_index(drop=True)
    for regression_type in ["gaussian","ridgecv","PLS","lassocv","elasticnetcv"]:
        i+=1
        file_name = os.path.splitext(os.path.basename(file))[0]

        input_dir_name=param["out_dir_name"]+"/" + file_name+regression_type
        save_dir="../figs"#param["fig_file_dir"]
        df = pd.read_excel("{}/result_loo.xlsx".format(input_dir_name))
        try:
            df_test = pd.read_excel("{}/result_5crossvalid.xlsx".format(input_dir_name))
            print(len(df_test))
            print("true")
        except:
            None
        # ax = fig.add_subplot(4, 2, i+1)
        ax = fig.add_subplot(3, 5, i)
        # ax =  subplot(2, 2, i)
        ax.plot([-3,3], [-3,3],color="Gray")
        #df_test["ΔΔG.crosstest"]=np.where(np.abs(df_test["ΔΔG.crosstest"].values) < 2.5, df_test["ΔΔG.crosstest"].values, 2.5 * np.sign(df_test["ΔΔG.crosstest"].values))

        print(df_test["ΔΔG.crosstest"].values)
        # ax.plot(df["ΔΔG.expt."], df["ΔΔG.loo"], "s", color="red",markersize=4 , alpha=0.5,label="loo $q^2$ = {:.2f}".format(r2_score(df["ΔΔG.expt."], df["ΔΔG.loo"])))
        # ax.plot(df["ΔΔG.expt."], df["ΔΔG.train"], "x",color="Black", markersize=4 ,alpha=0.5,label="train $r^2$ = {:.2f}".format(r2_score(df["ΔΔG.expt."], df["ΔΔG.train"])))
        try:
            ax.plot(df_test["ΔΔG.expt."], df_test["ΔΔG.crosstest"], "o",color="Blue",markersize=4 , alpha=0.5,label="test $r^2$ = {:.2f}".format(r2_score(df_test["ΔΔG.expt."], df_test["ΔΔG.crosstest"])))
        except:
            None
        ax.legend(loc = 'lower right',fontsize=6) #凡例

        # ax.text(-2.5, 2.0, "$\mathrm{N_{training}}$ = "+"{}\n".format(len(df))+"$\mathrm{N_{test}}$ = "+"{}".format(len(df_test)), fontsize=8)
        ax.text(0.8, 0.15, "N = "+"{}".format(len(df_test)), fontsize=8,transform=ax.transAxes)
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        ax.set_xlabel("ΔΔ${G_{expt}}$ [kcal/mol]")
        ax.set_ylabel("ΔΔ${G_{predict}}$ [kcal/mol]")
        ax.set_title(regression_type)
        # ax.set_title(param["title"]+"(N={})".format(len(df)))
        #RMSE = np.sqrt(mean_squared_error(df_test["ΔΔG.expt."], df_test["ΔΔG.crosstest"]))
        ax.set_xlim(-3,3)
        ax.set_ylim(-3,3)
        #np.savetxt("{}/RMSE.csv".format(param["out_dir_name"]), RMSE)
        #rmse=[]
        #rmse.append(RMSE)
        #p=pd.DataFrame(rmse)
        #p.to_csv("{}/RMSE.csv".format(param["out_dir_name"]))




save_dir="../figs"

fig.tight_layout()
plt.subplots_adjust(hspace=0.6)
#plt.text(-35, 10, "(A) (S)-CBS-Me", fontsize=12)
os.makedirs(save_dir,exist_ok=True)
plt.savefig(save_dir+"/plot.png",transparent=False, dpi=300)

fig =plt.figure(figsize=(3*5, 3*3))
margin = 0.2  # 0 <margin< 1
totoal_width = 1 - margin
x = np.arange(3)
fig, ax = plt.subplots(figsize=(6.4, 4))

labels=["CBS cat.","DIP-Chloride","Ru cat."]
labels=["(S)-CBS-Me","(-)-DIP-Chloride","$trans$-[RuC$\mathrm{l_2}$\n{(S)-XylBINAP}{(S)-DAIPEN}]"]
print(labels)
# filename=[["../parameter_nomax/parameter_cbs_gaussian.txt","../parameter_nomax/parameter_cbs_PLS.txt","../parameter_nomax/parameter_cbs_ridgecv.txt","../parameter_nomax/parameter_cbs_elasticnetcv.txt","../parameter_nomax/parameter_cbs_lassocv.txt"],
#            ["../parameter_nomax/parameter_dip-chloride_gaussian.txt","../parameter_nomax/parameter_dip-chloride_PLS.txt","../parameter_nomax/parameter_dip-chloride_ridgecv.txt","../parameter_nomax/parameter_dip-chloride_elasticnetcv.txt","../parameter_nomax/parameter_dip-chloride_lassocv.txt"],
#             ["../parameter_nomax/parameter_RuSS_gaussian.txt","../parameter_nomax/parameter_RuSS_PLS.txt","../parameter_nomax/parameter_RuSS_ridgecv.txt","../parameter_nomax/parameter_RuSS_elasticnetcv.txt","../parameter_nomax/parameter_RuSS_lassocv.txt"]]

filename=[["../parameter_0227/parameter_cbs_gaussian.txt","../parameter_0227/parameter_cbs_PLS.txt","../parameter_0227/parameter_cbs_ridgecv.txt","../parameter_0227/parameter_cbs_elasticnetcv.txt","../parameter_0227/parameter_cbs_lassocv.txt"],
           ["../parameter_0227/parameter_dip-chloride_gaussian.txt","../parameter_0227/parameter_dip-chloride_PLS.txt","../parameter_0227/parameter_dip-chloride_ridgecv.txt","../parameter_0227/parameter_dip-chloride_elasticnetcv.txt","../parameter_0227/parameter_dip-chloride_lassocv.txt"],
            ["../parameter_0227/parameter_RuSS_gaussian.txt","../parameter_0227/parameter_RuSS_PLS.txt","../parameter_0227/parameter_RuSS_ridgecv.txt","../parameter_0227/parameter_RuSS_elasticnetcv.txt","../parameter_0227/parameter_RuSS_lassocv.txt"]]


for i, param_file_names in enumerate(filename):
    ax.set_xlabel("ΔΔG.expt. [kcal/mol]")
    ax.set_ylabel("ΔΔG.predict [kcal/mol]")

    # ax.set_title(param["title"]+"(N={})".format(len(df)))
    RMSEs=[]
    for param_file_name in param_file_names:
        with open(param_file_name, "r") as f:
            param = json.loads(f.read())
        input_dir_name = param["out_dir_name"]
        save_dir = param["fig_file_dir"]
        df = pd.read_excel("{}/result_loo.xlsx".format(input_dir_name))
        try:
            df_test = pd.read_excel("{}/result_5crossvalid.xlsx".format(input_dir_name))

        except:
            None
        RMSE = np.sqrt(mean_squared_error(df_test["ΔΔG.expt."], df_test["ΔΔG.crosstest"]))
        RMSEs.append(RMSE)

    # np.savetxt("{}/RMSE.csv".format(param["out_dir_name"]), RMSE)
    lis=[i-0.3,i - 0.15, i, i + 0.15,i+0.3]


    if i==2:
        bars =ax.bar(lis,RMSEs,color="red", width=0.15,label =["Gaussian penalized","PLS","Ridge","Elastic Net","Lasso"])
        # bars =ax.bar(lis,RMSEs,color="red", width=0.15,label =["Gaussian penalized","PLS","Ridge","Elastic Net","Lasso"])
    else:
        bars = ax.bar(lis, RMSEs, color="red", width=0.15)
    ax.bar_label(bars, labels=["{:.2f}".format(_) for _ in RMSEs], rotation=60, label_type='center')  # ,fmt='%.2f'
    for i, b in enumerate(bars):
        print(i)
        b.set_alpha(i/5+0.2)#plt.cm.jet(1. * i / (3 - 1)))

        #ax.bar_label(bars, labels=["CBS","Ru cat.","DIP"],label_type='center')  # ,fmt='%.2f'
    #ax.bar_label(bar2, labels=["+{:.2f}".format(_) for _ in l[:, 0]], rotation=60, label_type='center')
ax.legend(loc='upper left',ncol=5, fontsize=8,)  # 凡例

ax.set_xticks(x)
ax.set_xlim(0-(1-margin)/3-margin, 2+(1-margin)/3+margin)
ax.set_ylim(0, 1.5)
ax.set_yticks([0,0.5,1,1.5])
ax.set_xticklabels(["{}".format(label) for label in labels])#, rotation=20
ax.set_xlabel("Data sets")
ax.set_ylabel("RMSE [kcal/mol]")
#ax.set_title(param["title"])
save_dir = "../figs"
fig.tight_layout()
os.makedirs(save_dir, exist_ok=True)
plt.savefig(save_dir + "/plot_bar.png", dpi=300)

