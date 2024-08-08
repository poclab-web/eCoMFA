import glob
import pandas as pd
from rdkit.Chem import PandasTools
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
# Define the file pattern
file_pattern = "C:/Users/poclabws/result/20240704_0_5_spl5/*/*prediction.xlsx"
# Get all files matching the pattern
files = glob.glob(file_pattern)
# Define the list of InChIKeys to extract


def get_subset(files,me_aryl,ph_aryl,me_hydrocarbon):
    for file in files:
        # Read the Excel file
        df = pd.read_excel(file)
        df['me_aryl_sigma'] = df['inchikey'].map(me_aryl)
        df['ph_aryl_sigma'] = df['inchikey'].map(ph_aryl)
        df['me_hydrocarbon_es'] = df['inchikey'].map(me_hydrocarbon)
        # NaN値を空文字列に置き換える
        df.fillna("", inplace=True)
        PandasTools.AddMoleculeColumnToFrame(df, "smiles")
        PandasTools.SaveXlsxFromFrame(df, file, size=(100, 100))
        # # Filter the dataframe to include only rows where the 'inchikey' column matches the given list
        # filtered_df = df[df['inchikey'].isin(inchikeys_to_extract.keys())]
        # filtered_df["feature"]
        
        # # Print the filtered dataframe (optional, for debugging)
        # print(filtered_df)
        
        # # Save the filtered dataframe to a new sheet in the same Excel file
        # with pd.ExcelWriter(file, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        #     filtered_df.to_excel(writer, sheet_name=sheet_name, index=False)

        # print(f"Filtered data saved to a new sheet in {file}")

me_aryl = {'KWOLFJPFCHCOCG-UHFFFAOYSA-N':0,#acetophenone
                         'FSPSELPMWGWDRY-UHFFFAOYSA-N':-0.07, #m-Me
                         "GNKZMNRKLCTJAY-UHFFFAOYSA-N":-0.17,#p-Me
                         "BAYUSCHCCGXLAY-UHFFFAOYSA-N":0.12,#m-MeO
                         "NTPLXRHDUXRPNE-UHFFFAOYSA-N":-0.27,#p-MeO
                         "ABXGMGUHGLQMAW-UHFFFAOYSA-N":0.43,#m-CF3
                         'HHAISVSEJFEWBZ-UHFFFAOYSA-N':0.54,#p-CF3
                         "QCZZSANNLWPGEA-UHFFFAOYSA-N":"",#p-Ph
                         "ZDPAWHACYDRYIW-UHFFFAOYSA-N":0.06,#p-F
                         "GPRYKVSEZCQIHD-UHFFFAOYSA-N":-0.66,#p-NH2
                         "JYAQYXOVOHJRCS-UHFFFAOYSA-N":0.39,#m-Br
                         "WYECURVXVYPVAT-UHFFFAOYSA-N":0.23,#p-Br
                         "YQYGPGKTNQNXMH-UHFFFAOYSA-N":0.78,#p-NO2
                         "IQZLUWLMQNGTIW-UHFFFAOYSA-N":0.12-0.27,#m,p-OMe
                         "VUGQIIQFXCXZJU-UHFFFAOYSA-N":0.12*2-0.27,#3,4,5OMe
}
ph_aryl={"WXPWZZHELZEVPO-UHFFFAOYSA-N":-0.17,#p-Me
         "SWFHGTMLYIBPPA-UHFFFAOYSA-N":-0.27,#p-MeO
         "OHTYZZYAMUVKQS-UHFFFAOYSA-N":0.54,#p-CF3
         "UGVRJVHOJNYEHR-UHFFFAOYSA-N":0.23,#p-Cl
         }
me_hydrocarbon={"ZWEHNKRNPOVVGH-UHFFFAOYSA-N":0.08,#Et
          "ZPVFWPFBNIEHGJ-UHFFFAOYSA-N":"",#hexyl
          "SYBYTAAJFKOIEJ-UHFFFAOYSA-N":0.48,#iPr
          "PJGSXYOJTGTZAV-UHFFFAOYSA-N":1.43,#tBu
          "HVCFCNAITDHQFX-UHFFFAOYSA-N":1.09,#cyclopropyl
          "LKENTYLPIUIMFG-UHFFFAOYSA-N":0.41,#cyclopentyl
          "RIFKADJTWUGDOV-UHFFFAOYSA-N":0.69,#cyclohexyl
          "SHOJXDKTYKFBRD-UHFFFAOYSA-N":"",#C=C(C)C
          "HDKLIZDXVUCLHQ-BQYQJAHWSA-N":"",#/C=C/CCCCC
          "LTYLUDGDHUEBGX-UHFFFAOYSA-N":"",#1-pentynyl
          "HDURLXYBKGWETC-UHFFFAOYSA-N":"",#C1=C(C)CCC1
          "CZSBBWHUCQLKNQ-UHFFFAOYSA-N":"",#C1=CCCCCC1
          "BWHOZHOGCMHOBV-BQYQJAHWSA-N":"",#/C=C/c1ccccc1
          "XRGPFNGLRSIPSA-UHFFFAOYSA-N":"",#ethynyl
        #   "UPEUQDJSUFHFQP-UHFFFAOYSA-N":1.97,#Cph
          "KWOLFJPFCHCOCG-UHFFFAOYSA-N":2.31,#Ph
          "YXWWHNCQZBVZPV-UHFFFAOYSA-N":2.82,#o-tolyl
          "FSPSELPMWGWDRY-UHFFFAOYSA-N":"",#m-tolyl
          "GNKZMNRKLCTJAY-UHFFFAOYSA-N":"",#p-tolyl
          "MQESVSITPLILCO-UHFFFAOYSA-N":"",#4-n-butylphenyl
          "HSDSKVWQTONQBJ-UHFFFAOYSA-N":"",#2,4-MethylPh
          "QCZZSANNLWPGEA-UHFFFAOYSA-N":"",#4-PhenylPh
          "XSAYZAUNJMRRIR-UHFFFAOYSA-N":"",#antracenyl
          }
#https://pubs.acs.org/doi/pdf/10.1021/cr00002a004
get_subset(files,me_aryl,ph_aryl,me_hydrocarbon)
filename="C:/Users/poclabws/result/20240704_0_5_spl5/Ridge.csv"
df=pd.read_csv(filename,index_col = 'Unnamed: 0').sort_index()
df["dataset"]=df.index*3//len(df)
# グラフを並べて描画
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for _ in range(3):
    df_=df[(df["dataset"]==_)]
    file=df_["savefilename"][df_["RMSE_validation"]==df_["RMSE_validation"].min()].iloc[0]+"_prediction.xlsx"
    print(file)
    df__ = pd.read_excel(file)
    # me_aryl_sigmaの数値行だけを用いる
    

    if _!=2:
        df__['me_aryl_sigma'] = pd.to_numeric(df__['me_aryl_sigma'], errors='coerce')
        df__ = df__.dropna(subset=['me_aryl_sigma'])
        # 単回帰分析
        slope, intercept, r_value, _, _ = linregress(df__['me_aryl_sigma'], df__['electrostatic_cont'])
        # 散布図と回帰直線のプロット
        axs[0].scatter(df__['me_aryl_sigma'], df__['electrostatic_cont'], label='Data Points')
        axs[0].plot(df__['me_aryl_sigma'], intercept + slope * df__['me_aryl_sigma'], 
                    label=f'y={slope:.2f}x+{intercept:.2f}\nR={r_value:.2f}')

    else:
        df__.loc[len(df__)] = [0 if col in ['ph_aryl_sigma', 'electrostatic_cont'] else np.nan for col in df__.columns]
        # me_aryl_sigmaの数値行だけを用いる
        df__['ph_aryl_sigma'] = pd.to_numeric(df__['ph_aryl_sigma'], errors='coerce')
        df__ = df__.dropna(subset=['ph_aryl_sigma'])
        # 単回帰分析
        slope, intercept, r_value, _, _ = linregress(df__['ph_aryl_sigma'], df__['electrostatic_cont'])
        # 散布図と回帰直線のプロット
        axs[0].scatter(df__['ph_aryl_sigma'], df__['electrostatic_cont'], label='Data Points')
        axs[0].plot(df__['ph_aryl_sigma'], intercept + slope * df__['ph_aryl_sigma'], 
                    label=f'y={slope:.2f}x+{intercept:.2f}\nR={r_value:.2f}')
        
    axs[0].set_xlabel('sigma')
    axs[0].set_ylabel('electrostatic_cont')
    axs[0].set_title('Hammett σ vs electrostatic_cont')
    axs[0].legend(loc='lower right')

    df__ = pd.read_excel(file)
    df__.loc[len(df__)] = [0 if col in ['me_hydrocarbon_es', 'steric_cont'] else np.nan for col in df__.columns]
    df__['me_hydrocarbon_es'] = pd.to_numeric(df__['me_hydrocarbon_es'], errors='coerce')
    df__ = df__.dropna(subset=['me_hydrocarbon_es'])
    # 単回帰分析
    slope, intercept, r_value, _, _ = linregress(df__['me_hydrocarbon_es'], df__['steric_cont'])
    # 散布図と回帰直線のプロット
    axs[1].scatter(df__['me_hydrocarbon_es'], df__['steric_cont'], label='Data Points')
    axs[1].plot(df__['me_hydrocarbon_es'], intercept + slope * df__['me_hydrocarbon_es'], 
                label=f'y={slope:.2f}x+{intercept:.2f}\nR={r_value:.2f}')
    axs[1].set_xlabel('es')
    axs[1].set_ylabel('steric_cont')
    axs[1].set_title('Taft Es vs steric cont')
    axs[1].legend(loc='lower right')
# グラフを表示
plt.tight_layout()
plt.savefig("C:/Users/poclabws/result/20240704_0_5_spl5/es_sigma.png")