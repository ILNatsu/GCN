# Copyright (c) 2020 Uber Technologies, Inc.
# Please check LICENSE for more detail
import os
endpoint = 'http://10.233.58.195:8080' #'http://7.183.83.207:80'
os.environ['S3_ENDPOINT'] = endpoint
os.environ['S3_USE_HTTPS'] = '0'
os.environ['ACCESS_KEY_ID'] = 'y00612510'
os.environ['SECRET_ACCESS_KEY'] = 'c7ebc922600e41a4b79cbf8665e85923'
os.environ['MOX_SILENT_MODE'] = '1'

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import sparse
import moxing as mox
mox.file.set_auth(retry=20)
import pickle5
import os
import io
import copy
from skimage.transform import rotate
# from tf_general_vis import get_samples
from read_pkl import get_samples
import moxing as mox
mox.file.set_auth(retry=20)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')    


# from pred_function import get_obj_feats,get_lane_graph

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
    feat_pred[1:, :2] -= feat_pred[:-1, :2] #做增量
    feat_pred[:,2] = object_mask[pred_id][11:51:2]
    feats.append(feat_pred)
    ctrs.append(feat_pred[-1,:2].copy())
    # feats.append(np.matmul(rot,(object_feat[pred_id][:, :2].copy()-orig.reshape(-1, 2)).T).T)
    
    for index in range(object_num):
        if index != pred_id:
            feat = np.zeros((20, 3), np.float32)       
            feat[:,:2] = np.matmul(rot,(object_feat[index][11:51:2,:2]-orig.reshape(-1, 2)).T).T
            feat[1:, :2] -= feat[:-1, :2] #做增量
            feat[:,2] = object_mask[index][11:51:2]
            feats.append(feat)
            ctrs.append(feat[-1,:2].copy())


    gt_preds.append(label[pred_id][0:15,:2]-orig)
    has_preds.append(label_mask[pred_id][0:15])
    for index_pred in range(object_num):
        if index_pred != pred_id:
            gt_preds.append(label[index][0:15,:2]-orig)
            has_preds.append(label_mask[index][0:15])

    feats = np.asarray(feats, np.float32)
    ctrs = np.asarray(ctrs, np.float32)
    gt_preds = np.asarray(gt_preds, np.float32)
    has_preds = np.asarray(has_preds, np.bool_)
    

    data = dict()
    data['feats'] = feats  #（n,20,3）矩阵，历史轨迹
    data['ctrs'] = ctrs  #
    data['orig'] = orig  #agent第2s轨迹
    data['theta'] = theta  #两帧之间车辆位置转角
    data['rot'] = rot #两帧之间角度矩阵
    data['gt_preds'] = gt_preds #预测轨迹真值
    data['has_preds'] = has_preds  #bool类型 
    # print(len(data["feats"]),len(data["feats"][0]),len(data["feats"][0][0]))
    return data


def get_lane_graph(df,data):
    """Get a rectangle area defined by pred_range.""" #获取指定范围的所有高精地图lane
    # x_min, x_max, y_min, y_max = config['pred_range']
    # radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
    #get_lane_ids_in_xy_bbox：获取xy平面中具有曼哈顿距离搜索半径的所有车道ID
    # lane_ids = self.am.get_lane_ids_in_xy_bbox(data['orig'][0], data['orig'][1], data['city'], radius)
    # lane_ids = copy.deepcopy(lane_ids)

    #坐标系调整。将高精地图Lane的坐标系原点移动、旋转到与Agent一致。
    staticRG_feat = df['staticRG_feat'].values[0]
    staticRG_mask = df['staticRG_mask'].values[0]

    centerlines = []
    lanes = dict()
    lane_ids = []
    lane_num = staticRG_feat.shape[0]
    for i in range(lane_num):
        lane_ids.append(i)
    for lane_id in lane_ids:
        centerline = staticRG_feat[lane_id][:,:2]
        lane = []
        lane = np.matmul(data['rot'], (centerline - data['orig'].reshape(-1, 2)).T).T
        # x, y = centerline[:, 0], centerline[:, 1]
        centerlines.append(lane)
    ctrs, feats, turn, control, intersect = [], [], [], [], []
    for lane_id in lane_ids:
        centerline = centerlines[lane_id]
        ctrln = centerline
        num_segs = len(ctrln) - 1
        ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32))
        feats.append(np.asarray(ctrln[1:] - ctrln[:-1], np.float32))

    node_idcs = []
    count = 0
    for i, ctr in enumerate(ctrs):
        node_idcs.append(range(count, count + len(ctr)))
        count += len(ctr)
    num_nodes = count
    
                
    graph = dict()
    graph['centerlines'] = centerlines
    graph['ctrs'] = np.concatenate(ctrs, 0)
    graph['num_nodes'] = num_nodes  #节点数
    graph['feats'] = np.concatenate(feats, 0)
    # print(len(data["feats"]),len(data["feats"][0]),len(data["feats"][0][0]),"111")

    return graph


def collate_fn(batch):
    batch = from_numpy(batch)
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch

def ref_copy(data):
    if isinstance(data, list):
        return [ref_copy(x) for x in data]
    if isinstance(data, dict):
        d = dict()
        for key in data:
            d[key] = ref_copy(data[key])
        return d
    return data

    #nbr是Single Scale的前驱(Pre)和后继(Suc)关系  num_nodes是Graph中Node的总数量   num_scales是Multi Scale的总层数，代码中默认取6
def dilated_nbrs(nbr, num_nodes, num_scales):
    data = np.ones(len(nbr['u']), np.bool)
    csr = sparse.csr_matrix((data, (nbr['u'], nbr['v'])), shape=(num_nodes, num_nodes))

    mat = csr
    nbrs = []
    for i in range(1, num_scales):
        mat = mat * mat

        nbr = dict()
        coo = mat.tocoo()
        nbr['u'] = coo.row.astype(np.int64)
        nbr['v'] = coo.col.astype(np.int64)
        nbrs.append(nbr)
    return nbrs


def dilated_nbrs2(nbr, num_nodes, scales):
    data = np.ones(len(nbr['u']), np.bool)
    csr = sparse.csr_matrix((data, (nbr['u'], nbr['v'])), shape=(num_nodes, num_nodes))

    mat = csr
    nbrs = []
    for i in range(1, max(scales)):
        mat = mat * csr

        if i + 1 in scales:
            nbr = dict()
            coo = mat.tocoo()
            nbr['u'] = coo.row.astype(np.int64)
            nbr['v'] = coo.col.astype(np.int64)
            nbrs.append(nbr)
    return nbrs

 
def collate_fn(batch):
    batch = from_numpy(batch)
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch


def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data


def cat(batch):
    if torch.is_tensor(batch[0]):
        batch = [x.unsqueeze(0) for x in batch]
        return_batch = torch.cat(batch, 0)
    elif isinstance(batch[0], list) or isinstance(batch[0], tuple):
        batch = zip(*batch)
        return_batch = [cat(x) for x in batch]
    elif isinstance(batch[0], dict):
        return_batch = dict()
        for key in batch[0].keys():
            return_batch[key] = cat([x[key] for x in batch])
    else:
        return_batch = batch
    return return_batch
