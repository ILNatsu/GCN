import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from config import get_test_result
import torch
import pandas as pd
from config import get_argo_train_preprocessed,get_argo_train_preprocessed2
# from tf_general_vis import draw_background,draw_object_state,draw_fut_traj

def pred_metrics(preds, gt_preds):
    # assert has_preds.all()
    preds = np.asarray(preds, np.float32)
    gt_preds = np.asarray(gt_preds, np.float32)

    """batch_size x num_mods x num_preds"""
    err = []
    distances = np.zeros(preds[0].shape[0])
    # distances_end = np.zeros(preds[0].shape[0])
    det_x = (preds[0][:,0]-gt_preds[:,0])
    det_y = (preds[0][:,1]-gt_preds[:,1])
    det_x_end = (preds[0][-1,0]-gt_preds[-1,0])
    det_y_end = (preds[0][-1,0]-gt_preds[-1,0])
    for i in range(len(det_x)):
        distances[i] = np.sqrt((det_x[i]**2+det_y[i]**2))
    distances_end = np.sqrt(det_x_end**2+det_y_end**2)

    ade = np.mean(distances)
    fde = distances_end

    return ade,fde

output_dir = '/root/workspace/zj/LaneGCN/results/change_curve_plot_0829_01/'
output_dir2 = '/root/workspace/zj/LaneGCN/results/change_curve_pred_test_0822_orign/'
test_data = torch.load('/root/workspace/zj/LaneGCN/results/lanegcn/results_test_0823_test_01.pkl')
datasets = np.load(get_argo_train_preprocessed(), allow_pickle=True)
datasets2 = np.load(get_argo_train_preprocessed2(), allow_pickle=True)


#数据合并
# with open(get_argo_train_preprocessed(), 'rb') as f:
#     data1 = pickle.load(f)
# with open(get_argo_train_preprocessed2(), 'rb') as f:
#     data2 = pickle.load(f)
# # merged_data = {**data1, **data2}
# merged_data = data1+data2
# with open('merged_file_0829_01.p', 'wb') as f:
#     pickle.dump(merged_data, f)


# for idx in range(0,170000,1000):
for idx in range(300):
    # idx  = idx*1000
    data = datasets[idx]
    print(len(data["feats"]))
    for index in range(len(data["feats"])):
        # if index 
        plt.plot(data["feats"][index][0],data["feats"][index][1],'r--',label='历史轨迹')
        plt.plot(data["gt_preds"][index][:,0],data["gt_preds"][index][:,1],'b--',label='真值')
        # plt.plot(data2["feats"][index][0],data2["feats"][index][1],'g--',label='预测值')
        # plt.plot(data2["gt_preds"][index][:,0],data2["gt_preds"][index][:,1],'y--',label='真值')
        plt.plot(data["graph"]["centerlines"][0][:,0],data["graph"]["centerlines"][0][:,1],'g--',label='道路中心线')
        plt.xlabel('x')#设置x,y轴标记
        plt.ylabel('y')
        # plt.xlim(-50,50)
        plt.ylim(-20,20)
        plt.gca().set_aspect('equal')
        plt.savefig(os.path.join(output_dir, str(idx)+"_"+str(index) + ".png"))
        plt.cla()



# 绘制真值和预测值
# for index in range(256*8):
    # print(len(test_data["feats"]))
    # print(len(test_data["feats"]))
    # for i in range(len(test_data["preds"][index])):
    # plt.plot(test_data["preds"][index][0][:,0],test_data["preds"][index][0][:,1],'r--',label='预测值')
    # plt.plot(test_data["gts"][index][:,0],test_data["gts"][index][:,1],'b--',label='真值')
    # plt.plot(test_data["graph"]["centerlines"][0][:,0],test_data["graph"]["centerlines"][0][:,1],'g--',label='道路中心线')
    # plt.plot(test_data["feats"][index][0][0],test_data["feats"][index][0][1],'g--',label='历史轨迹')

    # plt.xlabel('x')#设置x,y轴标记
    # plt.ylabel('y')
    # plt.xlim(-50,50)
    # plt.ylim(-20,20)
    # plt.title('Plotting two curves in the same plot')#设置图像标题
    # plt.title('预测值','真值')
    #保存图像
    # plt.savefig('fig3.png')
    # print(index)
    # plt.gca().set_aspect('equal')
    # plt.savefig(os.path.join(output_dir2, str(index) + ".png"))
    # plt.cla()
# mean_ade = []
# mean_fde = []
# for index in range(7925):
#     preds = test_data["preds"][index]
#     gt_preds = test_data["gts"][index]
#     ade,fde = pred_metrics(preds, gt_preds)
#     # print(ade)
#     mean_ade.append(ade)
#     mean_fde.append(fde)
# print(" mean_ade %2.4f, mean_fde %2.4f" % (np.mean(mean_ade), np.mean(mean_fde)))





