
config.py：预处理、测试数据、模型参数等位置信息
data.py：为原始Argoverse数据处理
data_huawei.py：为华为实车数据预处理
data_huawei_attrack.py：华为实车数据增强预处理（对道路进行不同程度弯曲预处理）
largecurvature_search_huawei.py：数据挖掘（弯道挖掘、直道挖掘、筛选历史轨迹满足增强模型中的限速、判断ego车辆是否在道路上、ego车辆轨迹是否为直线等）
preprocess_data.py：数据预处理（训练、测试、验证集）
tf_general_vis.py：可视化（道路中心线、车辆、预测轨迹、真值等）
result_plot.py：可视化（对弯曲后的数据进行可视化，包含道路中心线、数据增强后的未来轨迹）
test.py：模型测试
train.py：模型训练
lanegcn.py：模型结构
attack_functions.py：数据增强函数（三种：弯曲、双弯道、ripple）


# GCN# LaneGCN: Learning Lane Graph Representations for Motion Forecasting


 [Paper](https://arxiv.org/pdf/2007.13732) | [Slides](http://www.cs.toronto.edu/~byang/slides/LaneGCN.pdf)  | [Project Page]() | [**ECCV 2020 Oral** Video](https://yun.sfo2.digitaloceanspaces.com/public/lanegcn/video.mp4)

Ming Liang, Bin Yang, Rui Hu, Yun Chen, Renjie Liao, Song Feng, Raquel Urtasun


**Rank 1st** in [Argoverse Motion Forecasting Competition](https://evalai.cloudcv.org/web/challenges/challenge-page/454/leaderboard/1279)


![img](misc/arch.png)


Table of Contents
=================
  * [Install Dependancy](#install-dependancy)
  * [Prepare Data](#prepare-data-argoverse-motion-forecasting)
  * [Training](#training)
  * [Testing](#testing)
  * [Licence](#licence)
  * [Citation](#citation)



## Install Dependancy
You need to install following packages in order to run the code:
- [PyTorch>=1.3.1](https://pytorch.org/)
- [Argoverse API](https://github.com/argoai/argoverse-api#installation)


1. Following is an example of create environment **from scratch** with anaconda, you can use pip as well:
```sh
conda create --name lanegcn python=3.7
conda activate lanegcn
conda install pytorch==1.5.1 torchvision cudatoolkit=10.2 -c pytorch # pytorch=1.5.1 when the code is release

# install argoverse api
pip install  git+https://github.com/argoai/argoverse-api.git

# install others dependancy
pip install scikit-image IPython tqdm ipdb
```

2. \[Optional but Recommended\] Install [Horovod](https://github.com/horovod/horovod#install) and `mpi4py` for distributed training. Horovod is more efficient than `nn.DataParallel` for mulit-gpu training and easier to use than `nn.DistributedDataParallel`. Before install horovod, make sure you have openmpi installed (`sudo apt-get install -y openmpi-bin`).
```sh
pip install mpi4py

# install horovod with GPU support, this may take a while
HOROVOD_GPU_OPERATIONS=NCCL pip install horovod==0.19.4

# if you have only SINGLE GPU, install for code-compatibility
pip install horovod
```
if you have any issues regarding horovod, please refer to [horovod github](https://github.com/horovod/horovod)

## Prepare Data: Argoverse Motion Forecasting
You could check the scripts, and download the processed data instead of running it for hours.
```sh
bash get_data.sh
```

## Training


### [Recommended] Training with Horovod-multigpus


```sh
# single node with 4 gpus
horovodrun -np 4 -H localhost:4 python /path/to/train.py -m lanegcn

# 2 nodes, each with 4 gpus
horovodrun -np 8 -H serverA:4,serverB:4 python /path/to/train.py -m lanegcn
``` 

It takes 8 hours to train the model in 4 GPUS (RTX 5000) with horovod.

We also supply [training log](misc/train_log.txt) for you to debug.

### [Recommended] Training/Debug with Horovod in single gpu 
```sh
python train.py -m lanegcn
```


## Testing
You can download pretrained model from [here](http://yun.sfo2.digitaloceanspaces.com/public/lanegcn/36.000.ckpt) 
### Inference test set for submission
```
python test.py -m lanegcn --weight=/absolute/path/to/36.000.ckpt --split=test
```
### Inference validation set for metrics
```
python test.py -m lanegcn --weight=36.000.ckpt --split=val
```

**Qualitative results**

Labels(Red) Prediction (Green) Other agents(Blue)





<p>
<img src="misc/5304.gif" width = "30.333%"  align="left" />
<img src="misc/25035.gif" width = "30.333%" align="center"  />
 <img src="misc/19406.gif" width = "30.333%" align="right"   />
</p>

------

**Quantitative results**
![img](misc/res_quan.png)

## Licence
check [LICENSE](LICENSE)

## Citation
If you use our source code, please consider citing the following:
```bibtex
@InProceedings{liang2020learning,
  title={Learning lane graph representations for motion forecasting},
  author={Liang, Ming and Yang, Bin and Hu, Rui and Chen, Yun and Liao, Renjie and Feng, Song and Urtasun, Raquel},
  booktitle = {ECCV},
  year={2020}
}
```

If you have any questions regarding the code, please open an issue and [@chenyuntc](https://github.com/chenyuntc).
