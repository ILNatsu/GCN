import os

# argo_trian_preprocessed_data_path = "/root/workspace/zj/LaneGCN/dataset/preprocess/train_huawei_dist6_angle90_theta_0829_big_curve_buchong_01.p"
argo_trian_preprocessed_data_path = "/home/work/zhn/ICA_CITY_ica_city_merge_baseline_lyj_v3_1_3XX.txt"
# argo_trian_preprocessed_data_path = "/home/work/zhn/LaneGCN/LaneGCN/dataset/preprocess/train_huawei_dist6_angle90_theta_0829_big_curve_buchong_01.p"
# argo_trian_preprocessed_data_path = "/root/workspace/zj/LaneGCN/dataset/preprocess/merged_file_0822_02.p"
# argo_trian_preprocessed_data_path2 = "/root/workspace/zj/LaneGCN/dataset/preprocess/train_huawei_dist6_angle90_theta_0821.p"
# argo_trian_preprocessed_data_path2 = "/root/workspace/zj/LaneGCN/dataset/preprocess/train_huawei_dist6_angle90_theta_0818.p"
argo_trian_preprocessed_data_path2 = "ICA_CITY_ica_city_merge_baseline_lyj_v3_1_3XX.txt"
# argo_trian_preprocessed_data_path2 = "/home/work/zhn/LaneGCN/LaneGCN/dataset/preprocess/train_huawei_dist6_angle90_theta_0818.p"
# test_result_data_path = "/root/workspace/zj/LaneGCN/results/lanegcn/results.pkl"
test_result_data_path = "/home/work/zhn/LaneGCN/LaneGCN/results/lanegcn/results_1.pkl"
# get_huawei_dataset_path = "/home/develop/yulei/"
get_huawei_dataset_path = "/home/work/zhn/"
# ckpt_lanegcn_path = "/root/workspace/zj/LaneGCN/results/lanegcn/160.000.ckpt"
# ckpt_lanegcn_path = "/root/workspace/zj/LaneGCN/results/lanegcn/320.000.ckpt"
ckpt_lanegcn_path = "/home/work/zhn/LaneGCN/LaneGCN/results/lanegcn/33_0816_03.000.ckpt"
# ckpt_lanegcn_path2 = "/root/workspace/zj/LaneGCN/save_dir/huawei_ckpt/36.000_0809.ckpt"
ckpt_lanegcn_path2 = "/home/work/zhn/LaneGCN/LaneGCN/results/lanegcn/36.000_0809.ckpt"


def get_test_result():
    return test_result_data_path


def get_huawei_dataset():
    return get_huawei_dataset_path


def get_ckpt_path():
    return ckpt_lanegcn_path


def get_ckpt_path2():
    return ckpt_lanegcn_path2


def get_argo_train_preprocessed():
    return argo_trian_preprocessed_data_path


def get_argo_train_preprocessed2():
    return argo_trian_preprocessed_data_path2


