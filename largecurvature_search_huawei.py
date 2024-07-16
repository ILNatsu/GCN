import os
import io
import argparse
import pickle5
import numpy as np
import torch
import time
import sys
import matplotlib.pyplot as plt
import numpy.linalg as LA
import copy
import glob
from scipy.interpolate import CubicSpline
# from scipy.ndimage.morphology import distance_transform_edt
from shapely.geometry import LineString
from shapely.affinity import affine_transform, rotate
from argparse import Namespace
from tqdm import tqdm
import math
from attack_functions import Combination
from scipy.interpolate import CubicSpline
# from data_huawei_attrack import calc_radius,calc_curvature,apply_transform_function,get_points_on_curve


def set_auth():
    # endpoint = 'http://7.183.83.207:80'
    endpoint='http://10.233.58.195:8080'
    os.environ['S3_ENDPOINT'] = endpoint
    os.environ['S3_USE_HTTPS'] = '0'
    os.environ['ACCESS_KEY_ID'] = 'y00612510'
    os.environ['SECRET_ACCESS_KEY'] = 'c7ebc922600e41a4b79cbf8665e85923'
    os.environ['MOX_SILENT_MODE'] = '0'


set_auth()

import moxing as mox
mox.file.set_auth(retry=20)


def get_samples(sample_file, data_root, sample_ratio, public_cloud=False):
    with mox.file.File(sample_file, 'r') as f:
        info_list = f.readlines()

    if sample_ratio < 1:
        step = int(1.0 / sample_ratio)
        info_list = info_list[::step]

    samples = []
    for info in info_list:
        pkl_name = info.split(',')[-1].strip()
        if "obs://" in pkl_name or data_root == "/":
            pkl_path = pkl_name
            if public_cloud:
                pkl_path = "obs://alluxio-131/data/" + str(pkl_path).split('/data/')[1]
        else:
            pkl_path = os.path.join(data_root, pkl_name)
        samples.append(pkl_path)

    return samples

def PJcurvature(x,y):
    """
    input  : the coordinate of the three point
    output : the curvature and norm direction
    refer to https://github.com/Pjer-zhang/PJCurvature for detail
    """
    t_a = LA.norm([x[1]-x[0],y[1]-y[0]])
    t_b = LA.norm([x[2]-x[1],y[2]-y[1]])
    # print(t_a,t_b)
    try:
        M = np.array([
            [1, -t_a, t_a**2],
            [1, 0,    0     ],
            [1,  t_b, t_b**2]
        ])

        a = np.matmul(LA.inv(M),x)
        b = np.matmul(LA.inv(M),y)
        kappa = 2*(a[2]*b[1]-b[2]*a[1])/(a[1]**2.+b[1]**2.)**(1.5)
    except:
        kappa = 0
    return kappa

default_params = {"smooth-turn": {"attack_power": 9, "pow": 3, "border": 5},
                "double-turn": {"attack_power": 0, "pow": 3, "l": 10, "border": 5},
                "ripple-road": {"attack_power": 0, "l": 60, "border": 5}}
attack_functions = ["smooth-turn", "double-turn", "ripple-road"]
correct_speed = True
scale_factor = 1
 #对历史轨迹做一个限速处理
def correct_history( history):  # inputs history points in the agents coordinate system and corrects its speed   
    # calculating the minimum r of the attacking turn
    border1 = attack_params["smooth-turn"]["border"]
    border2 = attack_params["double-turn"]["border"]
    border3 = attack_params["ripple-road"]["border"]
    l_range = min(border1, border2, border3)
    r_range = max(border1 + 10, border2 + 20 + attack_params["double-turn"]["l"],
                    border3 + attack_params["ripple-road"]["l"])

    search_points = np.linspace(l_range, r_range, 100)
    search_point_rs = calc_radius(search_points)
    min_r = search_point_rs.min()
    g = 9.8
    miu_s = 0.7
    max_speed = np.sqrt(miu_s * g * min_r)
    current_speed = np.sqrt(((history[-1] - history[-2]) ** 2).sum()) * 10
    if current_speed <= max_speed:
        return True
    return False

def get_obj_feats(df):
    object_feat = df['object_feat'].values[0].copy().astype(np.float32)
    label = df['full_label'].values[0].copy().astype(np.float32)
    # if object_feat.shape[0] <= 1:
    #     pred_id = 0
    # else:
    #     pred_id = 1
    pred_id = 0
    orig = object_feat[pred_id][-1,:2].copy().astype(np.float32) #agent最后一秒数据
    theta = object_feat[pred_id][-1,2]  #agent theta
    label_mask = df['label_mask'].values[0]
    object_mask = df['object_mask'].values[0]
    rot = np.asarray([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]], np.float32) 

    feats, ctrs, gt_preds, has_preds = [], [], [], []
    object_num = object_feat.shape[0]
    #添加feat_pred
    feat_pred = np.zeros((20, 3), np.float32)
    feat_pred[:,:2] = np.matmul(rot,(object_feat[pred_id][11:51:2,:2]-orig.reshape(-1, 2)).T).T
    # feat_pred[1:, :2] -= feat_pred[:-1, :2] #做增量
    feat_pred[:,2] = object_mask[pred_id][11:51:2]
    feats.append(feat_pred)

    has_preds.append(label_mask[pred_id][0:15])
    # feats.append(np.matmul(rot,(object_feat[pred_id][:, :2].copy()-orig.reshape(-1, 2)).T).T)
    for index in range(object_num):
        if index != pred_id and object_feat[index][-1,0]<=0:  #筛选在自车后方车辆
            feat = np.zeros((20, 3), np.float32)       
            feat[:,:2] = np.matmul(rot,(object_feat[index][11:51:2,:2]-orig.reshape(-1, 2)).T).T
            # feat[1:, :2] -= feat[:-1, :2] #做增量
            feat[:,2] = object_mask[index][11:51:2]
            feats.append(feat)

    feats = np.asarray(feats, np.float32)

    #数据增强--判断历史轨迹是否合理
    for i, feat in enumerate(feats):
        feat[:, :2] = apply_transform_function(feat[:, :2])  #获取变换后的特征（feats表示历史轨迹）
        if i == 0 and correct_speed:  # apply speed correction on the history of agent if the correct_speed flag is on
            if(correct_history(feat[:, :2])):
                return True
            return False        

def calc_curvature(x):
    """
    given any set of points x, outputs the curvature of the attack_function on those points
    """
    # print(params["smooth-turn"]["attack_power"])
    numerator = attack_function.f_zegond(x)
    denominator = (1 + attack_function.f_prime(x) ** 2) ** 1.5
    return numerator / denominator

def calc_radius(x):
    """
    given any set of points x, outputs the radius of a circle fitting to the attack_function on those points
    """
    curv = calc_curvature(x)
    ret = np.zeros_like(x)
    ret[curv == 0] = 1000_000_000_000  # inf
    ret[curv != 0] = 1 / np.abs(curv[curv != 0])
    return ret

def apply_transform_function(points):
    """
    Applies attack_function on the input points
    :param points: np array of points that we want to apply the transformation on
    :return: transformed points
    """
    points = points.copy()
    points[:, 1] += attack_function.f(points[:, 0])
    return points


def progress_bar(finish_tasks_number, tasks_number):
    """
    进度条
    :param finish_tasks_number: int, 已完成的任务数
    :param tasks_number: int, 总的任务数
    :return:
    """
    percentage = round(finish_tasks_number / tasks_number * 100)
    print("\r进度: {}%: ".format(percentage), "▓" * (percentage // 2), end="")
    sys.stdout.flush()

def point_calculate(x_list,y_list):
    points_x = []
    points_y = []
    # print(len(x_list)//4,len(x_list)//2,len(x_list)-3)
    points_x.append(x_list[1]+2)
    points_x.append(x_list[len(x_list)//2])
    points_x.append(x_list[len(x_list)-2])
    points_y.append(y_list[1]+2)
    points_y.append(y_list[len(y_list)//2])
    points_y.append(y_list[len(y_list)-2])

    points_x2 = []
    points_y2 = []
    points_x2.append(x_list[len(x_list)//4])
    points_x2.append(x_list[len(x_list)//2])
    points_x2.append(x_list[len(x_list)//2+len(x_list)//4])
    points_y2.append(y_list[len(x_list)//4])
    points_y2.append(y_list[len(x_list)//2])
    points_y2.append(y_list[len(x_list)//2+len(x_list)//4])
    return points_x,points_y,points_x2,points_y2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw vectorized BEV figure.')
    parser.add_argument('--sample_file', metavar='PATH', type=str, help='directory to pickle files.')
    args = parser.parse_args()
    # sample_file_path = 'ncasd_rcapi_train_data_pd_v1_1_sample_keep_13344030.txt'
    sample_file_path = 'big_curve_train_data_0829_02.txt'
    # sample_file_path = 'big_curve_train_data_0824_straight_history_true.txt'
    large_kappa = []
    single_car = []
    feat_not_full = []
    label_not_full = []
    # sample_files = get_samples(sample_file_path, "/", sample_ratio=0.1, public_cloud=False)
    sample_files = get_samples(sample_file_path, "/", sample_ratio=1.0, public_cloud=False)
    sample_files.sort()
    # for id in range(len(sample_files)):
    file = open("big_curve_train_data_0829_03.txt","a")
    print(len(sample_files))
    # for id in range(1200000,len(sample_files)):
    for id in range(len(sample_files)):
        sample_file = sample_files[id]
        with io.BytesIO(mox.file.read(sample_file, binary=True)) as fb1:
            df = pickle5.load(fb1)
        staticRG_feat = df['staticRG_feat'].values[0]
        staticRG_mask = df['staticRG_mask'].values[0]
        object_feat = df['object_feat'].values[0]
        object_mask = df['object_mask'].values[0]
        label_mask = df['label_mask'].values[0]
        label = df['full_label'].values[0][:, :, :2]
        time = df['time_stamp'].values[0]
        vel_class = df['object_class'].values[0]
        vel_ids = df['object_ids'].values[0]
        #lane（第几条车道、点、16特征）    lane重复点位置     编号、帧数（每帧0.025s）、维度特征   标签10s       label(10s)每帧0.2s 50帧   
        # print(staticRG_feat.shape, staticRG_mask.shape, object_feat.shape, object_mask.shape, label.shape,label_mask.shape)  #label_mask
        agt_ts = df["time_stamp"].values[0]
        kappas = []
        # 直道、车少筛选
        kappa = 0
        kappa2 = 0
        kappa3 = 0
        kappa4 = 0
        if staticRG_feat.shape[0] <= 1:
            idx = 0
            lane = staticRG_feat[idx, staticRG_mask[idx]]
            x_list = lane[:, 0]
            y_list = lane[:, 1]
            points_x,points_y,points_x2,points_y2 = point_calculate(x_list, y_list)
            kappa = PJcurvature(points_x,points_y)
            kappa2 = PJcurvature(points_x2,points_y2)
        else:
            idx = 0
            lane = staticRG_feat[idx, staticRG_mask[idx]]
            x_list = lane[:, 0]
            y_list = lane[:, 1]
            points_x,points_y,points_x2,points_y2 = point_calculate(x_list, y_list)
            kappa = PJcurvature(points_x,points_y)
            kappa2 = PJcurvature(points_x2,points_y2)

            idx = 1
            lane = staticRG_feat[idx, staticRG_mask[idx]]
            x_list = lane[:, 0]
            y_list = lane[:, 1]
            points_x3,points_y3,points_x4,points_y4 = point_calculate(x_list, y_list)
            kappa3 = PJcurvature(points_x3,points_y3)
            kappa4 = PJcurvature(points_x4,points_y4)
        num = 0
        for i in range(staticRG_feat.shape[0]):
            lane = staticRG_feat[i,staticRG_mask[i]]
            x_list = lane[:, 0]
            y_list = lane[:, 1]
            points_x,points_y,points_x2,points_y2 = point_calculate(x_list, y_list)
            kappa = PJcurvature(points_x,points_y)
            kappa2 = PJcurvature(points_x2,points_y2)
            if kappa <0.01 and kappa2 <0.01:
                num += 1
        distance = 0
        lane = staticRG_feat[0,staticRG_mask[0]]
        distance = math.sqrt(lane[-1,0]*lane[-1,0]+lane[-1,1]*lane[-1,1])

        x_list2 = label[idx][0:15,0]
        y_list2 = label[idx][0:15,1]
        points_x3,points_y3,points_x4,points_y4 = point_calculate(x_list2, y_list2)
        kappa5 = PJcurvature(points_x3,points_y3)
        kappa6 = PJcurvature(points_x4,points_y4)
        # print(kappa5,kappa6)
        # if staticRG_feat.shape[0] <= 8 and num == staticRG_feat.shape[0]:
        # if distance > 30 and num == staticRG_feat.shape[0]:
        # if label[0,-1,0]>20 and lane[0,0] > 20:
        # if label[0,-1,0]>20:
        if num == staticRG_feat.shape[0]:
        # if lane[0,0]*lane[-1,0] <= 0 and distance > 20 and distance < 80 and label[0,-1,0] >10 and  distance > 20 and distance < 100:
            single_car.append(id)
            print(id)
            file.write(sample_file + '\n')
        #挑选历史轨迹合理数据
        # params = copy.deepcopy(default_params)
        # global attack_params,attack_function
        # attack_params = params
        # attack_function = Combination(params)
        # if(get_obj_feats(df)):
        #    single_car.append(id)
        #    print(id)
        #    file.write(sample_file + '\n')
        # file.close()
        
        #初筛条件
        # if abs(kappa) <0.01 and abs(kappa2) <0.01 and abs(kappa3) <0.01 and abs(kappa4) <0.01 and abs(kappa5) <0.03 and abs(kappa6) <0.03 and lane[0,0]*lane[-1,0] <= 0 and  distance > 30 and distance < 100 and abs(label[0,-1,0])>20:
        #    single_car.append(id)
        #    print(kappa,kappa2,kappa3,kappa4,kappa5,kappa5,id)
        #    file.write(sample_file + '\n')
        # file.close()
        #对车辆轨迹进行筛选（要求直行）
        # if staticRG_feat.shape[0] <= 1:
        # idx = 0
       
        # print(x_list,y_list)
        # if abs(kappa) <0.03 and abs(kappa2) <0.03:
        #    single_car.append(id)
        #    print(id)
            

        progress_bar(id, len(sample_files))
        # progress_bar(id, 3000)

    # file.close()
    # np.savetxt('single_car_index',single_car)
    print(single_car)



      


        
                    


        


        
