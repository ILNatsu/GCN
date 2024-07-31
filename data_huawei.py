# Copyright (c) 2020 Uber Technologies, Inc.
# Please check LICENSE for more detail

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import sparse
import pickle5
import io
import copy
from skimage.transform import rotate
from read_pkl import get_samples
import torch.multiprocessing

import os
#os.environ['PYDEVD_WARN_SLOW_RESOLE_TIMEUT'] = '10'

endpoint = 'http://10.233.58.195:8080'  # 'http://7.183.83.207:80'
os.environ['S3_ENDPOINT'] = endpoint
os.environ['S3_USE_HTTPS'] = '0'
os.environ['ACCESS_KEY_ID'] = 'y00612510'
os.environ['SECRET_ACCESS_KEY'] = 'c7ebc922600e41a4b79cbf8665e85923'
os.environ['MOX_SILENT_MODE'] = '1'

import moxing as mox

mox.file.set_auth(ak='y00612510', sk='c7ebc922600e41a4b79cbf8665e85923', server='http://10.233.58.195:8080', ssl_verify=False, retry=20)
# from tf_general_vis import get_samples
mox.file.set_auth(retry=20)
torch.multiprocessing.set_sharing_strategy('file_system')


class HuaweiDataset(Dataset):
    def __init__(self, split, config, train=True):
        self.config = config
        self.train = train
        if train:
            sample_file = 'ICA_CITY_ica_city_merge_baseline_lyj_v3_1_3XX.txt'  # 训练集
            #sample_file = 'sampling_ICA_CITY_ica_city_merge_baseline_lyj_v3_1_3XX.txt' #采样
            #sample_file = 'LaneGCN/LaneGCN/ICA_CITY_prediction_test_v2_2_3XX_step_5.txt' #采样2
            self.split = get_samples(
                sample_file, "/", sample_ratio=0.01, public_cloud=False)
            self.split.sort()
        else:
            sample_file = 'test_ICA_CITY_ica_city_baseline_lyj_new3_3XX.txt'  # 测试集
            self.split = get_samples(
                sample_file, "/", sample_ratio=0.1, public_cloud=False)
            self.split.sort()
        # before
        # config['preprocess'] = False
        # if 'preprocess' in config and config['preprocess']:
        #    if train:
        #       self.split = np.load(
        #            self.config['preprocess_train'], allow_pickle=True)
            # print(len(split))
            # self.split = np.load(self.config['preprocess_train_single_small_curve'], allow_pickle=True)
        #    else:
        #        self.split = np.load(
        #            self.config['preprocess_test'], allow_pickle=True)
        # else:

            # sample_file = 'ncasd_rcapi_train_data_pd_v1_1_sample_keep_13344030.txt'  # 训练集
        #    sample_file = 'ncase_train_data_v6_11_sample_merge_24773050.txt'  # 训练集
            # sample_file = 'NCA_CITY_rc_api_v2_test_ncacity_0302_3xx_3XX.txt'  #测试集
            # sample_file = 'NCA_CITY_ncasd_pd_v1_1_debug_nca_city_curvature20230529_3xx.txt'  #测试集
            # sample_file = 'test_data_0823_straight_lane.txt'  #测试集
            self.split = get_samples(
                sample_file, "/", sample_ratio=0.01, public_cloud=False)  # 大曲率
            self.split.sort()

    def __getitem__(self, idx):
        # if 'preprocess' in self.config and self.config['preprocess']:
        #     data = self.split[idx]
        #     if self.train and self.config['rot_aug']:
        #         new_data = dict()
        #         for key in ['orig', 'gt_preds', 'has_preds']:
        #             if key in data:
        #                 new_data[key] = ref_copy(data[key])

        #         dt = np.random.rand() * self.config['rot_size']  # np.pi * 2.0
        #         theta = data['theta'] + dt
        #         new_data['theta'] = theta
        #         new_data['rot'] = np.asarray([
        #             [np.cos(theta), -np.sin(theta)],
        #             [np.sin(theta), np.cos(theta)]], np.float32)

        #         rot = np.asarray([
        #             [np.cos(-dt), -np.sin(-dt)],
        #             [np.sin(-dt), np.cos(-dt)]], np.float32)
        #         new_data['feats'] = data['feats'].copy()
        #         new_data['feats'][:, :, :2] = np.matmul(
        #             new_data['feats'][:, :, :2], rot)
        #         new_data['ctrs'] = np.matmul(data['ctrs'], rot)

        #         graph = dict()
        #         # for key in ['num_nodes', 'lane_idcs', 'left_pairs', 'right_pairs', 'left', 'right']:
        #         for key in ['num_nodes']:
        #             graph[key] = ref_copy(data['graph'][key])
        #         graph['ctrs'] = np.matmul(data['graph']['ctrs'], rot)
        #         graph['feats'] = np.matmul(data['graph']['feats'], rot)
        #         new_data['graph'] = graph
        #         data = new_data
        #     else:
        #         new_data = dict()
        #         for key in ['orig', 'gt_preds', 'has_preds', 'theta', 'rot', 'feats', 'ctrs', 'graph']:
        #             if key in data:
        #                 new_data[key] = ref_copy(data[key])
        #         data = new_data

        #     if 'raster' in self.config and self.config['raster']:
        #         data.pop('graph')
        #         x_min, x_max, y_min, y_max = self.config['pred_range']
        #         cx, cy = data['orig']

        #         region = [cx + x_min, cx + x_max, cy + y_min, cy + y_max]
        #         raster = self.map_query.query(
        #             region, data['theta'], data['city'])

        #         data['raster'] = raster
        #     return data

        sample_path = self.split[idx]


        with io.BytesIO(mox.file.read(sample_path, binary=True)) as fb1:
            df = pickle5.load(fb1)
        # print(df.keys())
        # pred_id = 1

        data = self.get_obj_feats(df)
        data['idx'] = idx
        data['graph'] = self.get_lane_graph(df, data)
        return data
        ###
        # sample_path = self.split[idx]
        # with io.BytesIO(mox.file.read(sample_path, binary=True)) as fb1:
        #         df = pickle5.load(fb1)
        # # pred_id = 1
        # data = self.get_obj_feats(df)
        # data['idx'] = idx
        # data['graph'] = self.get_lane_graph(df,data)
        # return data

    def __len__(self):
        print(len(self.split))
        return len(self.split)

    # 获取障碍物数据特征
    def get_obj_feats(self, df):
        object_feat = df['object_feat'].values[0].copy().astype(np.float32)
        label = df['full_label'].values[0].copy().astype(np.float32)
        if object_feat.shape[0] <= 1:
            pred_id = 0
        else:
            pred_id = 1
        # pred_id = 0
        orig = object_feat[pred_id][-1,
                                    :2].copy().astype(np.float32)  # agent最后一秒数据
        theta = object_feat[pred_id][-1, 2]  # agent theta
        label_mask = df['label_mask'].values[0]
        object_mask = df['object_mask'].values[0]
        rot = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]], np.float32)

        feats, ctrs, gt_preds, has_preds = [], [], [], []
        object_num = object_feat.shape[0]
        # 添加feat_pred
        feat_pred = np.zeros((20, 3), np.float32)
        feat_pred[:, :2] = np.matmul(
            rot, (object_feat[pred_id][11:51:2, :2]-orig.reshape(-1, 2)).T).T
        feat_pred[1:, :2] -= feat_pred[:-1, :2]  # 做增量
        feat_pred[:, 2] = object_mask[pred_id][11:51:2]
        feats.append(feat_pred)
        ctrs.append(feat_pred[-1, :2].copy())

        # feats.append(np.matmul(rot,(object_feat[pred_id][:, :2].copy()-orig.reshape(-1, 2)).T).T)

        for index in range(object_num):
            if index != pred_id:
                feat = np.zeros((20, 3), np.float32)
                feat[:, :2] = np.matmul(
                    rot, (object_feat[index][11:51:2, :2]-orig.reshape(-1, 2)).T).T
                feat[1:, :2] -= feat[:-1, :2]  # 做增量
                feat[:, 2] = object_mask[index][11:51:2]
                feats.append(feat)
                ctrs.append(feat[-1, :2].copy())

        gt_preds.append(label[pred_id][0:15, :2]-orig)
        has_preds.append(label_mask[pred_id][0:15])
        for index_pred in range(object_num):
            if index_pred != pred_id:
                gt_preds.append(label[index][0:15, :2]-orig)
                has_preds.append(label_mask[index][0:15])

        feats = np.asarray(feats, np.float32)
        ctrs = np.asarray(ctrs, np.float32)
        gt_preds = np.asarray(gt_preds, np.float32)
        has_preds = np.asarray(has_preds, np.bool_)

        data = dict()
        data['feats'] = feats  # （n,20,3）矩阵，历史轨迹
        # print(len(data['feats']),len(data['feats'][0]),len(data['feats'][0][0]))
        data['ctrs'] = ctrs  #
        data['orig'] = orig  # agent第2s轨迹
        data['theta'] = theta  # 两帧之间车辆位置转角
        data['rot'] = rot  # 两帧之间角度矩阵
        data['gt_preds'] = gt_preds  # 预测轨迹真值
        data['has_preds'] = has_preds  # bool类型
        # print(feats,ctrs,orig,rot,gt_preds,has_preds)
        return data

    def get_lane_graph(self, df, data):
        """Get a rectangle area defined by pred_range."""  # 获取指定范围的所有高精地图lane
        x_min, x_max, y_min, y_max = self.config['pred_range']
        # radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        # get_lane_ids_in_xy_bbox：获取xy平面中具有曼哈顿距离搜索半径的所有车道ID
        # lane_ids = self.am.get_lane_ids_in_xy_bbox(data['orig'][0], data['orig'][1], data['city'], radius)
        # lane_ids = copy.deepcopy(lane_ids)

        # 坐标系调整。将高精地图Lane的坐标系原点移动、旋转到与Agent一致。
        staticRG_feat = df['staticRG_feat'].values[0]
        staticRG_mask = df['staticRG_mask'].values[0]

        centerlines = []
        lanes = dict()
        lane_ids = []
        lane_num = staticRG_feat.shape[0]
        for i in range(lane_num):
            lane_ids.append(i)
        # print(lane_ids)
        for lane_id in lane_ids:
            # print(lane_id)
            centerline = staticRG_feat[lane_id][:, :2]
            lane = []
            lane = np.matmul(data['rot'], (centerline -
                             data['orig'].reshape(-1, 2)).T).T
            # x, y = centerline[:, 0], centerline[:, 1]
            centerlines.append(lane)
        # print(len(centerlines),lanes_id)
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
        graph['num_nodes'] = num_nodes  # 节点数
        graph['feats'] = np.concatenate(feats, 0)

        return graph


# class ArgoTestDataset(ArgoDataset):
#     def __init__(self, config, train=False):

#         self.config = config
#         self.train = train

#         if 'preprocess' in config and config['preprocess']:
#             if train:
#                 self.split = np.load(self.config['preprocess_train'], allow_pickle=True)
#             else:
#                 # self.split = np.load(self.config['preprocess_val'], allow_pickle=True)
#                 self.split = np.load(self.config['preprocess_test'], allow_pickle=True)
#         else:
#             sample_file = 'ncasd_rcapi_train_data_pd_v1_1_sample_keep_13344030.txt'  #训练集
#             # sample_file = 'NCA_CITY_rc_api_v2_test_ncacity_0302_3xx_3XX.txt'  #测试集
#             self.split = get_samples(sample_file, "/", sample_ratio=0.1, public_cloud=False)
#             self.split.sort()


#     def __getitem__(self, idx):
#         if 'preprocess' in self.config and self.config['preprocess']:
#             data = self.split[idx]
#             # data['argo_id'] = int(self.avl.seq_list[idx].name[:-4]) #160547

#             if self.train and self.config['rot_aug']:
#                 new_data = dict()
#                 for key in ['orig', 'gt_preds', 'has_preds']:
#                     if key in data:
#                         new_data[key] = ref_copy(data[key])

#                 dt = np.random.rand() * self.config['rot_size']#np.pi * 2.0
#                 theta = data['theta'] + dt
#                 new_data['theta'] = theta
#                 new_data['rot'] = np.asarray([
#                     [np.cos(theta), -np.sin(theta)],
#                     [np.sin(theta), np.cos(theta)]], np.float32)

#                 rot = np.asarray([
#                     [np.cos(-dt), -np.sin(-dt)],
#                     [np.sin(-dt), np.cos(-dt)]], np.float32)
#                 new_data['feats'] = data['feats'].copy()
#                 new_data['feats'][:, :, :2] = np.matmul(new_data['feats'][:, :, :2], rot)
#                 new_data['ctrs'] = np.matmul(data['ctrs'], rot)

#                 graph = dict()
#                 # for key in ['num_nodes', 'lane_idcs', 'left_pairs', 'right_pairs', 'left', 'right']:
#                 for key in ['num_nodes']:
#                     graph[key] = ref_copy(data['graph'][key])
#                 graph['ctrs'] = np.matmul(data['graph']['ctrs'], rot)
#                 graph['feats'] = np.matmul(data['graph']['feats'], rot)
#                 new_data['graph'] = graph
#                 data = new_data
#             else:
#                 new_data = dict()
#                 for key in ['orig', 'gt_preds', 'has_preds', 'theta', 'rot', 'feats', 'ctrs', 'graph']:
#                     if key in data:
#                         new_data[key] = ref_copy(data[key])
#                 data = new_data

#             if 'raster' in self.config and self.config['raster']:
#                 data.pop('graph')
#                 x_min, x_max, y_min, y_max = self.config['pred_range']
#                 cx, cy = data['orig']

#                 region = [cx + x_min, cx + x_max, cy + y_min, cy + y_max]
#                 raster = self.map_query.query(region, data['theta'], data['city'])

#                 data['raster'] = raster
#             return data

#         sample_path = self.split[idx]
#         with io.BytesIO(mox.file.read(sample_path, binary=True)) as fb1:
#                 df = pickle5.load(fb1)
#         # print(df.keys())
#         # pred_id = 1
#         data = self.get_obj_feats(df)
#         data['idx'] = idx

#         if 'raster' in self.config and self.config['raster']:
#             x_min, x_max, y_min, y_max = self.config['pred_range']
#             cx, cy = data['orig']

#             region = [cx + x_min, cx + x_max, cy + y_min, cy + y_max]
#             raster = self.map_query.query(region, data['theta'], data['city'])

#             data['raster'] = raster
#             return data
#         data['graph'] = self.get_lane_graph(df,data)
#         # data['graph'] = self.get_lane_graph(data)
#         return data

#     def __len__(self):
#         print(len(self.split))
#         return len(self.split)

class MapQuery(object):
    # TODO: DELETE HERE No used
    """[Deprecated] Query rasterized map for a given region"""

    def __init__(self, scale, autoclip=True):
        """
        scale: one meter -> num of `scale` voxels 
        """
        super(MapQuery, self).__init__()
        assert scale in (1, 2, 4, 8)
        self.scale = scale
        root_dir = '/mnt/yyz_data_1/users/ming.liang/argo/tmp/map_npy/'
        mia_map = np.load(f"{root_dir}/mia_{scale}.npy")
        pit_map = np.load(f"{root_dir}/pit_{scale}.npy")
        self.autoclip = autoclip
        self.map = dict(
            MIA=mia_map,
            PIT=pit_map
        )
        self.OFFSET = dict(
            MIA=np.array([502, -545]),
            PIT=np.array([-642, 211]),
        )
        self.SHAPE = dict(
            MIA=(3674, 1482),
            PIT=(3043, 4259)
        )

    def query(self, region, theta=0, city='MIA'):
        """
        region: [x0,x1,y0,y1]
        city: 'MIA' or 'PIT'
        theta: rotation of counter-clockwise, angel/degree likd 90,180
        return map_mask: 2D array of shape (x1-x0)*scale, (y1-y0)*scale
        """
        region = [int(x) for x in region]

        map_data = self.map[city]
        offset = self.OFFSET[city]
        shape = self.SHAPE[city]
        x0, x1, y0, y1 = region
        x0, x1 = x0+offset[0], x1+offset[0]
        y0, y1 = y0+offset[1], y1+offset[1]
        x0, x1, y0, y1 = [round(_*self.scale) for _ in [x0, x1, y0, y1]]
        # extend the crop region to 2x -- for rotation
        H, W = y1-y0, x1-x0
        x0 -= int(round(W/2))
        y0 -= int(round(H/2))
        x1 += int(round(W/2))
        y1 += int(round(H/2))
        results = np.zeros([H*2, W*2])
        # padding of crop -- for outlier
        xstart, ystart = 0, 0
        if self.autoclip:
            if x0 < 0:
                xstart = -x0
                x0 = 0
            if y0 < 0:
                ystart = -y0
                y0 = 0
            x1 = min(x1, shape[1]*self.scale-1)
            y1 = min(y1, shape[0]*self.scale-1)
        map_mask = map_data[y0:y1, x0:x1]
        _H, _W = map_mask.shape
        results[ystart:ystart+_H, xstart:xstart+_W] = map_mask
        results = results[::-1]  # flip to cartesian
        # rotate and remove margin
        rot_map = rotate(results, theta, center=None,
                         order=0)  # center None->map center
        H, W = results.shape
        outputH, outputW = round(H/2), round(W/2)
        startH, startW = round(H//4), round(W//4)
        crop_map = rot_map[startH:startH+outputH, startW:startW+outputW]
        return crop_map


def ref_copy(data):
    if isinstance(data, list):
        return [ref_copy(x) for x in data]
    if isinstance(data, dict):
        d = dict()
        for key in data:
            d[key] = ref_copy(data[key])
        return d
    return data

    # nbr是Single Scale的前驱(Pre)和后继(Suc)关系  num_nodes是Graph中Node的总数量   num_scales是Multi Scale的总层数，代码中默认取6


def dilated_nbrs(nbr, num_nodes, num_scales):
    data = np.ones(len(nbr['u']), np.bool)
    csr = sparse.csr_matrix(
        (data, (nbr['u'], nbr['v'])), shape=(num_nodes, num_nodes))

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
    csr = sparse.csr_matrix(
        (data, (nbr['u'], nbr['v'])), shape=(num_nodes, num_nodes))

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

