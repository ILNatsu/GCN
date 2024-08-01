您好，看来您需要一些帮助来创作一部小说。根据您提供的信息，我将为您生成一个小说的大纲和一些关键情节的描述。请注意，这些内容仅作为创作启发，您可以根据自己的想法进一步发展和完善。

小说大纲
标题：《宫墙柳影》

背景设定：

故事发生在一个古代帝国，柳乐可是宫中的小太监，面容姣好，身材精瘦。
付瑞孔是京城外富商之子，英俊威武，有着不为人知的癖好。
嬴姬是大皇子，文武双全，与付瑞孔关系密切。
主要情节：

邂逅：付瑞孔在宫外偶遇柳乐可，被其美貌所吸引。
追求：付瑞孔开始用各种方式接近柳乐可，试图赢得其芳心。
调教：付瑞孔发现柳乐可的特殊体质，开始对其进行秘密调教。
情感纠葛：柳乐可逐渐对付瑞孔产生依赖，内心挣扎与情感纠葛。
社会压力：两人的关系受到外界的质疑和压力，包括皇宫和富商家族。
内心成长：柳乐可开始认识到自己对多个男人的迷恋，内心世界发生变化。
危机与决断：付瑞孔和柳乐可面临重大危机，必须做出决断。
胜利与和解：两人克服困难，最终确认彼此的感情，得到社会的接受。
结局：柳乐可和付瑞孔的爱情得到了周围人的祝福，两人幸福地生活在一起。

关键情节描述
邂逅：
在一个阳光明媚的午后，付瑞孔在宫外的市集上偶然看到了柳乐可。柳乐可的美貌和与众不同的气质立刻吸引了他的目光，他决定要接近这个小太监。

追求：
付瑞孔利用自己的身份和资源，开始频繁地出现在柳乐可的生活中。他送给柳乐可珍贵的礼物，邀请他参加各种宴会，逐渐赢得了柳乐可的信任。

调教：
付瑞孔发现柳乐可对某些事物有着特别的反应，他开始秘密地对柳乐可进行调教，希望柳乐可能够完全依赖于他。

情感纠葛：
随着时间的推移，柳乐可发现自己对付瑞孔的感情越来越深，但同时他也感到了内心的挣扎和矛盾，因为他知道自己的身份和付瑞孔的世界有着不可逾越的鸿沟。

社会压力：
两人的关系开始受到外界的关注，皇宫中的其他太监和宫女开始议论，富商家族也对这段关系表示担忧。

内心成长：
在付瑞孔的引导下，柳乐可开始探索自己内心的欲望，他发现自己对多个男人的迷恋，并开始接受这种感觉。

危机与决断：
当两人的关系被大皇子嬴姬发现后，他们面临了前所未有的危机。付瑞孔必须在家族荣誉和柳乐可之间做出选择。

胜利与和解：
经过一系列的努力和斗争，付瑞孔和柳乐可最终克服了所有的障碍。他们的关系得到了周围人的理解，两人决定勇敢地面对自己的生活。

写作建议
在描写情感和床戏时，注意把握尺度，既要满足读者的期待，也要遵守社会道德和法律规定。
利用细节描写来增强故事的真实感，让读者能够更好地沉浸在故事中。
塑造角色时，要深入挖掘他们的性格特点和内心世界，使角色更加立体和有说服力。
希望这些建议能够帮助您创作出一部精彩的小说。如果您需要更详细的分析或者有其他问题，请随时告诉我。
config.py：预处理、测试数据、模型参数等位置信息;
data.py：为原始Argoverse数据处理;
data_huawei.py：为华为实车数据预处理;
data_huawei_attrack.py：华为实车数据增强预处理（对道路进行不同程度弯曲预处理）;
largecurvature_search_huawei.py：数据挖掘（弯道挖掘、直道挖掘、筛选历史轨迹满足增强模型中的限速、判断ego车辆是否在道路上、ego车辆轨迹是否为直线等）;
preprocess_data.py：数据预处理（训练、测试、验证集）;
tf_general_vis.py：可视化（道路中心线、车辆、预测轨迹、真值等）;
result_plot.py：可视化（对弯曲后的数据进行可视化，包含道路中心线、数据增强后的未来轨迹）;
test.py：模型测试;
train.py：模型训练;
lanegcn.py：模型结构;
attack_functions.py：数据增强函数（三种：弯曲、双弯道、ripple）;


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
