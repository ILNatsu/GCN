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
from attack_functions import Combination
from scipy.interpolate import CubicSpline
import math

torch.multiprocessing.set_sharing_strategy('file_system')

default_params = {"smooth-turn": {"attack_power": 0, "pow": 3, "border": 5},
                "double-turn": {"attack_power": 0, "pow": 3, "l": 10, "border": 5},
                "ripple-road": {"attack_power": 0, "l": 60, "border": 5}}
attack_functions = ["smooth-turn", "double-turn", "ripple-road"]

###增强参数设置
# params = copy.deepcopy(default_params)
# attack_params = params
# attack_function = Combination(params)

# params["smooth-turn"]["attack_power"] = 0
# params["double-turn"]["attack_power"] = 0
# params["ripple-road"]["attack_power"] = 0




correct_speed = True
scale_factor = 1

class HuaweiDataset(Dataset):
    def __init__(self,split,config, train=True):
        self.config = config
        self.train = train

        # config['preprocess'] = False
        if 'preprocess' in config and config['preprocess']:
            if train:
                self.split = np.load(self.config['preprocess_train'], allow_pickle=True)
            else:
                self.split = np.load(self.config['preprocess_test'], allow_pickle=True)
        else:
            # sample_file = 'ncasd_rcapi_train_data_pd_v1_1_sample_keep_13344030.txt'  #训练集
            # sample_file = 'NCA_CITY_rc_api_v2_test_ncacity_0302_3xx_3XX.txt'  #测试集
            # sample_file = 'NCA_CITY_ncasd_pd_v1_1_debug_nca_city_curvature20230529_3xx.txt'  #测试集
            sample_file = 'big_curve_train_data_0829_03.txt'  #少车、直道
            self.split = get_samples(sample_file, "/", sample_ratio=1.0, public_cloud=False) #大曲率            
            self.split.sort()


    def __getitem__(self, idx):
        if 'preprocess' in self.config and self.config['preprocess']:
            data = self.split[idx]
            if self.train and self.config['rot_aug']:
                new_data = dict()
                for key in ['orig', 'gt_preds', 'has_preds']:
                    if key in data:
                        new_data[key] = ref_copy(data[key])

                dt = np.random.rand() * self.config['rot_size']#np.pi * 2.0
                theta = data['theta'] + dt
                new_data['theta'] = theta
                new_data['rot'] = np.asarray([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]], np.float32)

                rot = np.asarray([
                    [np.cos(-dt), -np.sin(-dt)],
                    [np.sin(-dt), np.cos(-dt)]], np.float32)
                new_data['feats'] = data['feats'].copy()
                new_data['feats'][:, :, :2] = np.matmul(new_data['feats'][:, :, :2], rot)
                new_data['ctrs'] = np.matmul(data['ctrs'], rot)

                graph = dict()
                # for key in ['num_nodes', 'lane_idcs', 'left_pairs', 'right_pairs', 'left', 'right']:
                for key in ['num_nodes']:
                    graph[key] = ref_copy(data['graph'][key])
                graph['ctrs'] = np.matmul(data['graph']['ctrs'], rot)
                graph['feats'] = np.matmul(data['graph']['feats'], rot)
                new_data['graph'] = graph
                data = new_data
            else:
                new_data = dict()
                for key in ['orig', 'gt_preds', 'has_preds', 'theta', 'rot', 'feats', 'ctrs', 'graph']:
                    if key in data:
                        new_data[key] = ref_copy(data[key])
                data = new_data
           
            if 'raster' in self.config and self.config['raster']:
                data.pop('graph')
                x_min, x_max, y_min, y_max = self.config['pred_range']
                cx, cy = data['orig']
                
                region = [cx + x_min, cx + x_max, cy + y_min, cy + y_max]
                raster = self.map_query.query(region, data['theta'], data['city'])

                data['raster'] = raster
            return data

        # sample_path = self.split[idx]
        id = math.floor(idx/2)
        sample_path = self.split[id]
        with io.BytesIO(mox.file.read(sample_path, binary=True)) as fb1:
                df = pickle5.load(fb1)


        params = copy.deepcopy(default_params)
        attrack_powder = 0
        if idx % 57 < 19:
            attrack_powder = round(idx % 57 - 9,2)
            params["smooth-turn"]["attack_power"] = attrack_powder
            params["ripple-road"]["attack_power"] = 0
            params["double-turn"]["attack_power"] = 0
        elif idx % 57 < 38:
            attrack_powder = round(idx % 57 -28,2)
            params["smooth-turn"]["attack_power"] = 0
            params["double-turn"]["attack_power"] = attrack_powder
            params["ripple-road"]["attack_power"] = 0
        elif idx % 57 <57:
            attrack_powder = round(idx % 57 - 47,2)
            params["smooth-turn"]["attack_power"] = 0
            params["double-turn"]["attack_power"] = 0
            params["ripple-road"]["attack_power"] = attrack_powder
        else:
            params["smooth-turn"]["attack_power"] = 0
            params["double-turn"]["attack_power"] = 0
            params["ripple-road"]["attack_power"] = 0
        # if idx % 32 < 32:
        #     params["smooth-turn"]["attack_power"] = 0
        #     params["double-turn"]["attack_power"] = 0
        #     params["ripple-road"]["attack_power"] = round((idx % 32 - 16)/2,2)
        # else:
        #     params["ripple-road"]["attack_power"] = 0
        # print(params["smooth-turn"]["attack_power"], params["double-turn"]["attack_power"],params["ripple-road"]["attack_power"])
        global attack_params,attack_function
        attack_params = params
        attack_function = Combination(params)
        data = self.get_obj_feats(df)
        data['idx'] = idx
        data['graph'] = self.get_lane_graph(df,data)
        return data
        ###
    
    def __len__(self):
        print(len(self.split))
        return len(self.split)

    #获取障碍物数据特征
    def get_obj_feats(self,df):
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
        if object_num > 1:
            object_num = 1
        #添加feat_pred
        feat_pred = np.zeros((20, 3), np.float32)
        feat_pred[:,:2] = np.matmul(rot,(object_feat[pred_id][11:51:2,:2]-orig.reshape(-1, 2)).T).T
        # feat_pred[1:, :2] -= feat_pred[:-1, :2] #做增量
        feat_pred[:,2] = object_mask[pred_id][11:51:2]
        feats.append(feat_pred)
        ctrs.append(feat_pred[-1,:2].copy())

        gt_preds.append(label[pred_id][0:15,:2]-orig)
        has_preds.append(label_mask[pred_id][0:15])
        # feats.append(np.matmul(rot,(object_feat[pred_id][:, :2].copy()-orig.reshape(-1, 2)).T).T)
        for index in range(object_num):
            if index != pred_id and object_feat[index][-1,0]<=0:  #筛选在自车后方车辆
                feat = np.zeros((20, 3), np.float32)       
                feat[:,:2] = np.matmul(rot,(object_feat[index][11:51:2,:2]-orig.reshape(-1, 2)).T).T
                # feat[1:, :2] -= feat[:-1, :2] #做增量
                feat[:,2] = object_mask[index][11:51:2]
                feats.append(feat)
                ctrs.append(feat[-1,:2].copy())
                gt_preds.append(label[index][0:15,:2]-orig)
                has_preds.append(label_mask[index][0:15])
# # and object_feat[index][-1,0]<0 and object_feat[index][-1,0]>-20

#         for index_pred in range(object_num):
#             if index_pred != pred_id and object_feat[index][-1,0]>-10:


        feats = np.asarray(feats, np.float32)
        ctrs = np.asarray(ctrs, np.float32)
        gt_preds = np.asarray(gt_preds, np.float32)
        has_preds = np.asarray(has_preds, np.bool_)
        

        #数据增强
        for i, feat in enumerate(feats):
            feat[:, :2] = self.apply_transform_function(feat[:, :2])  #获取变换后的特征（feats表示历史轨迹）
            if i == 0 and correct_speed:  # apply speed correction on the history of agent if the correct_speed flag is on
                feat[:, :2] = self.correct_history(feat[:, :2])
            feat[1:, :2] -= feat[:-1, :2]

        ctrs = self.apply_transform_function(ctrs)

        for i, gt_pred in enumerate(gt_preds):
            # first rotate ground truth like it is done in data for the prediction
            gt_pred[:, :] = np.matmul(rot, (gt_pred - orig.reshape(-1, 2)).T).T
            # skew the ground truths
            gt_pred[:, :] = self.apply_transform_function(gt_pred)
            if i == 0:  # applies gt correction given the scale_factor calculated in history correction. Note that if
                # the speed correction is off, scale_factor is equal to 1 so this won't change the gt
                gt_pred[:, :] = self.correct_gt(gt_pred, scale_factor)
            # undo the rotation to bring the ground truth back to the world coordinates
            gt_pred[:, :] = np.matmul(rot, gt_pred.T).T + orig.reshape(-1, 2)


        data = dict()
        data['feats'] = feats  #（n,20,3）矩阵，历史轨迹
        data['ctrs'] = ctrs  #
        data['orig'] = orig  #agent第2s轨迹
        data['theta'] = theta  #两帧之间车辆位置转角
        data['rot'] = rot #两帧之间角度矩阵
        data['gt_preds'] = gt_preds #预测轨迹真值
        data['has_preds'] = has_preds  #bool类型 
        return data

    def get_lane_graph(self,df,data):
        """Get a rectangle area defined by pred_range.""" #获取指定范围的所有高精地图lane
        x_min, x_max, y_min, y_max = self.config['pred_range']
        #坐标系调整。将高精地图Lane的坐标系原点移动、旋转到与Agent一致。
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
            centerline = staticRG_feat[lane_id][:,:2]
            lane = []
            lane = np.matmul(data['rot'], (centerline - data['orig'].reshape(-1, 2)).T).T
            lane = self.apply_transform_function(lane)  #数据增强 
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
        graph['num_nodes'] = num_nodes  #节点数
        graph['feats'] = np.concatenate(feats, 0)
        return graph


 #矫正曲线？
    def correct_gt(self, gt, scale_factor):
        """
        inputs gt points and a scale factor and scales gt points according to that scale_factor in a way that
        the output points rely on the same curve as the input points. In other words, this function only moves the
        gt points on their curve so that their distances are multiplied by the scale_factor.
        :param gt: a list of points as ground truth
        :param scale_factor: the constant we want the distance between gt points to be multiplied by
        :return:
        """
        gt_speeds = np.sqrt(((gt[1:] - gt[:-1]) ** 2).sum(1))
        return self.get_points_on_curve(gt, gt_speeds * scale_factor)

    def get_points_on_curve(self, curve_points, dists):
        """
        inputs a set of points on a curve and outputs points on that curve having distances according to dists
        :param curve_points: a list of points on a curve
        :param dists: a list of numbers indicating distances between output points
        :return: a list of points on the given curve having distances according to dists
        """
        t = np.arange(len(curve_points))
        csx = CubicSpline(t, curve_points[:, 0])
        csy = CubicSpline(t, curve_points[:, 1])
        points = [curve_points[0]]
        cur_time = 0
        for d in dists:
            dx, dy = csx(cur_time, 1), csy(cur_time, 1)
            delta_t = d / np.sqrt(dx ** 2 + dy ** 2)
            new_time = cur_time + delta_t

            left, right = cur_time, len(curve_points)
            mx_dist = np.sqrt((csx(right) - csx(cur_time)) ** 2 + (csy(right) - csy(cur_time)) ** 2)
            while mx_dist < d:
                right *= 2
                mx_dist = np.sqrt((csx(right) - csx(cur_time)) ** 2 + (csy(right) - csy(cur_time)) ** 2)

            while True:
                dist = np.sqrt((csx(new_time) - csx(cur_time)) ** 2 + (csy(new_time) - csy(cur_time)) ** 2)
                if np.abs(dist - d) <= 0.01:
                    break
                if dist > d:
                    right = new_time
                elif dist < d:
                    left = new_time
                new_time = (left + right) / 2

            cur_time = new_time
            points.append(np.array([csx(cur_time), csy(cur_time)]))
        points = np.array(points)
        return points
 #对历史轨迹做一个限速处理
    def correct_history(self, history):  # inputs history points in the agents coordinate system and corrects its speed   
        # calculating the minimum r of the attacking turn
        border1 = attack_params["smooth-turn"]["border"]
        border2 = attack_params["double-turn"]["border"]
        border3 = attack_params["ripple-road"]["border"]
        l_range = min(border1, border2, border3)
        r_range = max(border1 + 10, border2 + 20 + attack_params["double-turn"]["l"],
                      border3 + attack_params["ripple-road"]["l"])

        search_points = np.linspace(l_range, r_range, 100)
        search_point_rs = self.calc_radius(search_points)
        min_r = search_point_rs.min()
        g = 9.8
        miu_s = 0.7
        max_speed = np.sqrt(miu_s * g * min_r)
        self.current_speed = np.sqrt(((history[-1] - history[-2]) ** 2).sum()) * 10
        if self.current_speed <= max_speed:
            return history
        scale_factor = max_speed / self.current_speed
        return history * scale_factor
        

    def calc_curvature(self, x):
        """
        given any set of points x, outputs the curvature of the attack_function on those points
        """
        # print(params["smooth-turn"]["attack_power"])
        numerator = attack_function.f_zegond(x)
        denominator = (1 + attack_function.f_prime(x) ** 2) ** 1.5
        return numerator / denominator

    def calc_radius(self, x):
        """
        given any set of points x, outputs the radius of a circle fitting to the attack_function on those points
        """
        curv = self.calc_curvature(x)
        ret = np.zeros_like(x)
        ret[curv == 0] = 1000_000_000_000  # inf
        ret[curv != 0] = 1 / np.abs(curv[curv != 0])
        return ret

    def apply_transform_function(self, points):
        """
        Applies attack_function on the input points
        :param points: np array of points that we want to apply the transformation on
        :return: transformed points
        """
        points = points.copy()
        points[:, 1] += attack_function.f(points[:, 0])
        return points


class MapQuery(object):
    #TODO: DELETE HERE No used
    """[Deprecated] Query rasterized map for a given region"""
    def __init__(self, scale, autoclip=True):
        """
        scale: one meter -> num of `scale` voxels 
        """
        super(MapQuery, self).__init__()
        assert scale in (1,2,4,8)
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
                MIA=np.array([502,-545]),
                PIT=np.array([-642,211]),
            )
        self.SHAPE=dict(
                MIA=(3674, 1482),
                PIT= (3043, 4259)
            )
    def query(self,region,theta=0,city='MIA'):
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
        x0,x1,y0,y1 = region
        x0,x1 = x0+offset[0],x1+offset[0]
        y0,y1 = y0+offset[1],y1+offset[1]
        x0,x1,y0,y1 = [round(_*self.scale) for _ in [x0,x1,y0,y1]]
        # extend the crop region to 2x -- for rotation
        H,W = y1-y0,x1-x0
        x0 -= int(round(W/2))
        y0 -= int(round(H/2))
        x1 += int(round(W/2))
        y1 += int(round(H/2))
        results = np.zeros([H*2,W*2])
        # padding of crop -- for outlier
        xstart,ystart=0,0
        if self.autoclip:
            if x0<0:
                xstart = -x0 
                x0 = 0
            if y0<0:
                ystart = -y0 
                y0 = 0
            x1 = min(x1,shape[1]*self.scale-1)
            y1 = min(y1,shape[0]*self.scale-1)
        map_mask = map_data[y0:y1,x0:x1]
        _H,_W = map_mask.shape
        results[ystart:ystart+_H, xstart:xstart+_W]=map_mask
        results = results[::-1] # flip to cartesian
        # rotate and remove margin
        rot_map = rotate(results,theta,center=None,order=0) # center None->map center
        H,W = results.shape
        outputH,outputW = round(H/2),round(W/2)
        startH,startW = round(H//4),round(W//4)
        crop_map = rot_map[startH:startH+outputH,startW:startW+outputW]
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
