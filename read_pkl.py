import os
import io
import argparse
import pickle5
import numpy as np

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw vectorized BEV figure.')
    parser.add_argument('--sample_file', metavar='PATH', type=str, help='directory to pickle files.')
    args = parser.parse_args()
    sample_file = 'ncasd_rcapi_train_data_pd_v1_1_sample_keep_13344030.txt'
    
    # sample_files = get_samples(args.sample_file, "/", sample_ratio=0.001, public_cloud=False)
    sample_files = get_samples(sample_file, "/", sample_ratio=0.001, public_cloud=False)
    for i, sample_file in enumerate(sample_files):
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
        # position_x, position_y = object_feat[0][-1, :2]
        # agt_ts = np.sort(np.unique(df["time_stamp"].values[0])) #时间戳序列
        agt_ts = df["time_stamp"].values[0]
        print(agt_ts)
        print(vel_ids)
        idx = 2

        object_feat = df['object_feat'].values[0].copy().astype(np.float32)
        label = df['full_label'].values[0].copy().astype(np.float32)
        orig = object_feat[idx][-1,:2].copy().astype(np.float32) #agent最后一秒数据
        theta = object_feat[idx][-1,2]  #agent theta
        label_mask = df['label_mask'].values[0]
        object_mask = df['object_mask'].values[0]
    
        rot = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]], np.float32) 
        
        feats, ctrs, gt_preds, has_preds = [], [], [], []
        # feats.append(np.matmul(rot,(object_feat[idx][-1, :2].copy()-orig.reshape(-1, 2)).T).T)
        object_num = object_feat.shape[0]
        for index in range(object_num):
            feat = np.zeros((51, 3), np.float32)
            if index == idx:
                continue
            feat[:,:2] = np.matmul(rot,(object_feat[index][:,:2]-orig.reshape(-1, 2)).T).T
            # print(len(object_mask[index]))
            for i in range(len(object_mask[index])):
                if(object_mask[index][i]):
                    feat[i,2] = 1.0
            feats.append(feat)
            ctrs.append(feat[-1,:2].copy())

        gt_preds.append(label[idx][:,:2]-orig)
        has_preds.append(label_mask[idx])
        for index_pred in range(object_num):
            if index_pred == idx:
                continue
            gt_preds.append(label[index][:,:2]-orig)
            has_preds.append(label_mask[index])
        # print(gt_preds,has_preds)

        feats = np.asarray(feats, np.float32)
        ctrs = np.asarray(ctrs, np.float32)
        gt_preds = np.asarray(gt_preds, np.float32)
        has_preds = np.asarray(has_preds, np.bool_)

        # print(object_mask[index],feats)
        centerlines = []
        # lanes_is_left = staticRG_feat[:,:,3]
        # lanes_is_right = staticRG_feat[:,:,4]
        lane_num = staticRG_feat.shape[0]
        for index in range(lane_num):
            centerline = staticRG_feat[index][:,:2]
            lane = []
            lane = np.matmul(rot, (centerline - orig.reshape(-1, 2)).T).T
            x, y = centerline[:, 0], centerline[:, 1]
            # if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
            #     continue
            # else:
            centerlines.append(lane)
        # print(centerline)

        ctrs, feats, turn, control, intersect = [], [], [], [], []
        for lane_id in range(lane_num):
            centerline = centerlines[lane_id]
            ctrln = centerline
            num_segs = len(ctrln) - 1
            ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32))
            feats.append(np.asarray(ctrln[1:] - ctrln[:-1], np.float32))
            print(ctrln,num_segs)


        


        
