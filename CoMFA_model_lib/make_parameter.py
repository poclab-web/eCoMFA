


# for param_file_name in [
#         "../parameter/parameter_cbs_gaussian.txt",
#         "../parameter/parameter_cbs_PLS.txt",
#         "../parameter/parameter_cbs_ridgecv.txt",
#         "../parameter/parameter_cbs_lassocv.txt",
#         "../parameter/parameter_cbs_elasticnetcv.txt",
#             "../parameter/parameter_dip-chloride_PLS.txt",
#         "../parameter/parameter_dip-chloride_lassocv.txt",
#         "../parameter/parameter_dip-chloride_gaussian.txt",
#         "../parameter/parameter_dip-chloride_elasticnetcv.txt",
#         "../parameter/parameter_dip-chloride_ridgecv.txt",
#             "../parameter/parameter_RuSS_gaussian.txt",
#             "../parameter/parameter_RuSS_lassocv.txt",
#             "../parameter/parameter_RuSS_PLS.txt",
#             "../parameter/parameter_RuSS_elasticnetcv.txt",
#             "../parameter/parameter_RuSS_ridgecv.txt",
#         "../parameter/parameter_cbs_gaussian_FP.txt",
#         "../parameter/parameter_dip-chloride_gaussian_FP.txt",
#         "../parameter/parameter_RuSS_gaussian_FP.txt",
#     ]:
for param_file_name in [
    "../parameter_nomax/parameter_cbs_gaussian.txt",
    "../parameter_nomax/parameter_cbs_PLS.txt",
    "../parameter_nomax/parameter_cbs_ridgecv.txt",
    "../parameter_nomax/parameter_cbs_lassocv.txt",
    "../parameter_nomax/parameter_cbs_elasticnetcv.txt",
    "../parameter_nomax/parameter_dip-chloride_PLS.txt",
    "../parameter_nomax/parameter_dip-chloride_lassocv.txt",
    "../parameter_nomax/parameter_dip-chloride_gaussian.txt",
    "../parameter_nomax/parameter_dip-chloride_elasticnetcv.txt",
    "../parameter_nomax/parameter_dip-chloride_ridgecv.txt",
    "../parameter_nomax/parameter_RuSS_gaussian.txt",
    "../parameter_nomax/parameter_RuSS_lassocv.txt",
    "../parameter_nomax/parameter_RuSS_PLS.txt",
    "../parameter_nomax/parameter_RuSS_elasticnetcv.txt",
    "../parameter_nomax/parameter_RuSS_ridgecv.txt",
    "../parameter_nomax/parameter_cbs_gaussian_FP.txt",
    "../parameter_nomax/parameter_dip-chloride_gaussian_FP.txt",
    "../parameter_nomax/parameter_RuSS_gaussian_FP.txt",
]:

    with open(param_file_name, encoding="cp932") as f:
        data_lines = f.read()
    try:
    # 文字列置換
        data_lines = data_lines.replace("5,3,5,0.5", "5,2,6,0.5")

    except:
        None
    # try:
    # # 文字列置換
    #     data_lines = data_lines.replace("../grid_features", "../../../grid_features")
    #
    # except:
    #     None
    # try:
    # # 文字列置換
    #     data_lines = data_lines.replace("../../../../grid_features", "../grid_features")
    #
    # except:
    #     None

    try:
    # 文字列置換
        data_lines = data_lines.replace("Dt ESP", "Dt ESP_cutoff")

    except:
        None
    try:
    # 文字列置換
        data_lines = data_lines.replace("Dt ESP_cutoff_cutoff", "Dt ESP_cutoff")

    except:
        None



    # 同じファイル名で保存
    with open(param_file_name, mode="w", encoding="cp932") as f:
        f.write(data_lines)