import os
endpoint = 'http://10.233.58.195:8080' #'http://7.183.83.207:80'
os.environ['S3_ENDPOINT'] = endpoint
os.environ['S3_USE_HTTPS'] = '0'
os.environ['ACCESS_KEY_ID'] = 'y00612510'
os.environ['SECRET_ACCESS_KEY'] = 'c7ebc922600e41a4b79cbf8665e85923'
os.environ['MOX_SILENT_MODE'] = '1'
import lanegcn
import moxing as mox
mox.file.set_auth(retry=20)
import multiprocessing
import glob
import math
import argparse
import torch
from pathlib import Path
import random
import logging
import io
import pickle5
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import colors as mcolors
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from importlib import import_module
from lanegcn import get_model
from config import get_ckpt_path,get_ckpt_path2
from utils import Logger, load_pretrain
import copy
from pred_function import get_obj_feats,get_lane_graph,collate_fn

# from data_huawei import get_obj_feats,get_lane_graph

# from config import get_argo_train_preprocessed,get_argo_train_preprocessed2

color_dict = {"AGENT_PRED": "#DA4749"}
object_behavior_color_map = {0: (0.5, 0.5, 0.5),  # unknown: gray
                             1: (0, 1.0, 0),  # on lane green
                             2: (1.0, 0.0, 0),  # change lane
                             3: (1.0, 1.0, 0),  # not on lane yellow
                             4: (0.0, 0.0, 1.00),  # on lane 1000 blue
                             5: (0.0, 0.0, 0.0),  # not on lane 1000 black
                             }


def get_samples(sample_file, data_root, sample_ratio):
    with mox.file.File(sample_file, 'r') as f:
        info_list = f.readlines()

    if sample_ratio < 1:
        step = int(1.0 / sample_ratio)
        info_list = info_list[::step]

    samples = []
    for info in info_list:
        pkl_name = info.strip()
        pkl_path = os.path.join(data_root, pkl_name)
        samples.append(pkl_path)
    return samples


def draw_background(pkl):
    staticRG_feat = pkl['staticRG_feat'].values[0]
    staticRG_mask = pkl['staticRG_mask'].values[0]
    for idx in range(staticRG_feat.shape[0]):
    # idx = 0
        lane = staticRG_feat[idx, staticRG_mask[idx]]
        length = np.linalg.norm(lane[-1, :2] - lane[0, :2])
        x_list = lane[:, 0]
        y_list = lane[:, 1]
        plt.plot(x_list, y_list, 'y', zorder=0, linewidth=1.0)
        plt.arrow(x_list[-1], y_list[-1], (x_list[-1] - x_list[0]) / length, (y_list[-1] - y_list[0]) / length,
                    length_includes_head=True, head_width=1.0, head_length=1.0, fc='y', ec='y', zorder=0)
        rotation = math.atan2(y_list[1] - y_list[0], x_list[1] - x_list[0]) / math.pi * 180
        plt.text(x_list[0], y_list[0], f"{idx}", zorder=0, fontsize=2.5, color='darkorange', rotation=rotation,wrap=True)


def draw_object_state(pkl, idx):
    object_feat = pkl['object_feat'].values[0][idx]
    object_mask = pkl['object_mask'].values[0][idx]
    object_id = pkl['object_ids'].values[0][idx]
    behavior_label = pkl['label_object_behavior'].values[0][idx]
    if not object_mask.any():  #如果有一个为true返回true   
        return
    plt.plot(object_feat[object_mask, 0], object_feat[object_mask, 1], 'b--', linewidth=0.7, zorder=1)  #plot历史轨迹
    position_x, position_y = object_feat[-1, :2]
    bbox_corners_x = object_feat[-1, 13:21:2]
    bbox_corners_y = object_feat[-1, 14:21:2]
    line = plt.Polygon([[bbox_corners_x[0], bbox_corners_y[0]], [bbox_corners_x[1], bbox_corners_y[1]],
                        [bbox_corners_x[2], bbox_corners_y[2]], [bbox_corners_x[3], bbox_corners_y[3]]],
                       closed=True, fill=True, edgecolor=object_behavior_color_map[behavior_label],
                       facecolor=object_behavior_color_map[behavior_label], zorder=4)
    plt.text(position_x, position_y, f"{object_id}", zorder=1, fontsize=2.0, color='m', wrap=True)
    plt.gca().add_line(line)


def draw_fut_traj(pkl, idx):
    object_feat = pkl['object_feat'].values[0][idx]
    object_mask = pkl['object_mask'].values[0][idx]
    label = pkl['full_label'].values[0][idx]
    label_mask = pkl['label_mask'].values[0][idx]
    if not object_mask.any():
        return
    position_x, position_y = object_feat[-1, :2]
    label_x = label[label_mask, 0] + position_x
    label_y = label[label_mask, 1] + position_y
    plt.plot(label_x[::5], label_y[::5], 'g', zorder=3, linewidth=0.7, ms=1.0, alpha=0.5)

def draw_pred_fut_traj(pkl):
    data = get_obj_feats(pkl)
    data['graph'] = get_lane_graph(pkl,data)
    # print(len(data["feats"]),len(data["feats"][0]),len(data["feats"][0][0]))
    # print(data["feats"])
    data = dict(collate_fn([copy.deepcopy(data)]))



    with torch.no_grad():
        output = net(data)
    pred_data = output["reg"][0][0:1].detach().cpu().numpy().squeeze()[0]
        # results = [x[0:1].detach().cpu().numpy() for x in output["reg"]]
        # pred_data = result[idx]['gt_preds'][idx][0] if 'gt_preds' in data else None
    plt.plot(pred_data[:,0],pred_data[:,1],'r--',label='预测值')


    # plt.plot(
    #     pred_data[:, 0],
    #     pred_data[:, 1],
    #     "-",
    #     color=color_dict["AGENT_PRED"],
    #     # label='Prediction',
    #     alpha=1,
    #     linewidth=2,
    #     zorder=0,
    # )
    # plt.arrow(pred_data[-2, 0], pred_data[-2, 1], pred_data[-1, 0] - pred_data[-2, 0],
    #           pred_data[-1, 1] - pred_data[-2, 1], color=color_dict["AGENT_PRED"], width=0.4)
    # position_x, position_y = pred_data["preds"][idx][1][-1, :2]
    # label_x = pred_data["preds"][idx][1][:,0] + position_x
    # label_y = pred_data["preds"][idx][1][:,1] + position_y
    # plt.plot(label_x[::5], label_y[::5], 'r', zorder=3, linewidth=0.7, ms=1.0, alpha=0.5)

def draw_object(args, pkl):
    object_num = pkl['object_feat'].values[0].shape[0]
    object_mask = pkl['object_mask'].values[0]
    obj_exit_seq_label = pkl["label_exit_lane_index"].values[0][:, 0]
    obj_lane_seq_label = pkl["label_lane_seq_index"].values[0][:, 0]
    behavior_label = pkl['label_object_behavior'].values[0]

    draw_pred_fut_traj(pkl)

    for idx in range(object_num):
        if not object_mask[idx].any():
            continue
        draw_object_state(pkl, idx)
        draw_fut_traj(pkl, idx)
        if not args.draw_scene:
            draw_background(pkl)
            object_id = pkl['object_ids'].values[0][idx]
            veh_id = pkl['obs_path'].values[0].split('/')[-1].split('@')[1]
            output_dir = os.path.join(args.output_dir, "object")
            os.makedirs(output_dir, exist_ok=True)
            time_stamp = pkl["time_stamp"].values[0]
            plt.gca().set_aspect('equal')
            plt.savefig(os.path.join(output_dir, str(object_id) + "@" + veh_id + "@" + str(time_stamp) + ".png"))
            logging.info(os.path.join(output_dir, str(object_id) + "@" + veh_id + "@" + str(time_stamp) + ".png"))
            plt.cla()


def worker(args, sample_file):
    plt.cla()
    plt.clf()
    fig = plt.figure(dpi=600)
    # CROP_SIZE = 300
    # BACK_SIZE = 150
    CROP_SIZE = 200
    BACK_SIZE = 100
    xmin = -CROP_SIZE / 2 - 10
    xmax = CROP_SIZE / 2 + 10
    ymin = -BACK_SIZE - 10
    ymax = CROP_SIZE - BACK_SIZE + 10
    plt.gca().set_xlim(xmin=xmin, xmax=xmax)
    plt.gca().set_ylim(ymin=ymin, ymax=ymax)
    with io.BytesIO(mox.file.read(sample_file, binary=True)) as fb:
        pkl = pickle5.load(fb)

    draw_object(args, pkl)
    if args.draw_scene:
        draw_background(pkl)
        veh_id = pkl['obs_path'].values[0].split('/')[-1].split('@')[1]
        output_dir = os.path.join(args.output_dir, "scene")
        os.makedirs(output_dir, exist_ok=True)
        time_stamp = pkl["time_stamp"].values[0]
        plt.gca().set_aspect('equal')
        plt.savefig(os.path.join(output_dir, veh_id + "@" + str(time_stamp) + ".png"))
        logging.info(os.path.join(output_dir, veh_id + "@" + str(time_stamp) + ".png"))
        fig.clear()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw vectorized BEV figure.')
    parser.add_argument('--root_dir', metavar='PATH', type=str, help='directory to pickle files.')
    parser.add_argument('--sample_file', metavar='PATH', type=str, help='list to pickle files.')
    parser.add_argument('--filter_file', metavar='PATH', type=str, help='directory to pickle files.')
    parser.add_argument('--output_dir', metavar='PATH', type=str, help='directory to png files.')
    parser.add_argument('--worker_nums', type=int, default=16, help='sample step')
    parser.add_argument("--draw_scene", action="store_true", help="whether to draw scene.")
    parser.add_argument("-m", "--model", default="angle90", type=str, metavar="MODEL", help="model name")

    args = parser.parse_args()
    sample_files = []
    sample_files.extend(get_samples(args.sample_file,  args.root_dir, sample_ratio=0.01))
    # sample_files.extend(get_samples(args.sample_file,  args.root_dir, sample_ratio=1.0))

    #预测数据绘制
    model = import_module(args.model)
    _, _, _, net, _, _, _ = model.get_model()

    # ckpt_path = get_ckpt_path2()  ## load pretrain model 36.00ckpt
    ckpt_path = get_ckpt_path()  ## load pretrain model 160.00ckpt
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    load_pretrain(net, ckpt["state_dict"])
    # agent_pred = get_lanegcn_agent_pred()
    net.eval()



    if len(sample_files) == 0:
        sample_files = mox.file.list_directory(args.root_dir)
        sample_files.extend([os.path.join(args.root_dir, sample_file) for sample_file in sample_files])
    sample_files.sort()
    pool = multiprocessing.Pool(processes=args.worker_nums)


    for sample_file in sample_files:
    # print(len(ids))
    # for id in ids:
    #     sample_file = sample_files[id]
        worker(args, sample_file)
        pool.apply_async(worker, args=(args, sample_file))


    pool.close()
    pool.join()


   
