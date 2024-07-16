# ---------------------------------------------------------------------------
# Learning Lane Graph Representations for Motion Forecasting
#
# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Ming Liang, Yun Chen
# ---------------------------------------------------------------------------

import argparse
import os
os.umask(0)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import pickle
import sys
from importlib import import_module
import time
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from config import get_ckpt_path,get_ckpt_path2
import numpy as np
from data import ArgoTestDataset
from utils import Logger, load_pretrain


root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)


# define parser
parser = argparse.ArgumentParser(description="Argoverse Motion Forecasting in Pytorch")
parser.add_argument(
    "-m", "--model", default="angle90", type=str, metavar="MODEL", help="model name"
)
parser.add_argument("--eval", action="store_true", default=True)
parser.add_argument(
    "--split", type=str, default="val", help='data split, "val" or "test"'
)
parser.add_argument(
    "--weight", default="", type=str, metavar="WEIGHT", help="checkpoint path"
)


def main():
    # Import all settings for experiment.
    args = parser.parse_args()
    model = import_module(args.model)
    config, Dataset, collate_fn, net, loss, post_process, opt = model.get_model()

    # load pretrain model
    # ckpt_path = args.weight
    # if not os.path.isabs(ckpt_path):
    #     ckpt_path = os.path.join(config["save_dir"], ckpt_path)
    # ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    # load_pretrain(net, ckpt["state_dict"])
    ckpt_path = get_ckpt_path()  ## load pretrain model 36.00ckpt
    ckpt_path = os.path.join(root_path, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    load_pretrain(net, ckpt["state_dict"])
    net.eval()

    # Data loader for evaluation
    dataset = Dataset(args.split, config, train=False)
    # dataset = Dataset(args.split, config, train=True)
    data_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=True,
    )
    epoch = config["epoch"]
    # begin inference
    preds = {}
    gts = {}
    feats = {}
    graph = {}
    centerlines = {}
    start_time = time.time()
    metrics = dict()
    display_iters = int(
        config["test_display_iters"] / (config["batch_size"])
    )
    num_batches = len(data_loader)
    epoch_per_batch = 1.0 / num_batches
    for ii, data in tqdm(enumerate(data_loader)):
        print(data.keys())
        with torch.no_grad():
            epoch += epoch_per_batch
            output = net(data)
            data = dict(data) 
            # print(data.keys())
            # agent_pred = output["reg"][0][0:1].detach().cpu().numpy().squeeze()[0]
            results = [x[0:1].detach().cpu().numpy() for x in output["reg"]]

            loss_out = loss(output, data)
            post_out = post_process(output, data)
            post_process.append(metrics, loss_out, post_out)
            num_iters = int(np.round(epoch * num_batches))

            if num_iters % display_iters == 0:
                dt = time.time() - start_time
                if True:
                    post_process.display(metrics, dt, epoch, None)
                start_time = time.time()
                metrics = dict()
        # centerlines[ii] = data["centerlines"]
        feats[ii] = data["feats"]
        graph[ii] = data["graph"]

        
        for i, pred_traj in enumerate(results):
            preds[i] = pred_traj.squeeze()
            gts[i] = data['gt_preds'][i][0] if 'gt_preds' in data else None
            # gts[i] = data['gt_preds'][i] if 'gt_preds' in data else None
            # data_orig[i] = data["orig"][i]

    # save for further visualization
    res = dict(
        preds = preds,
        gts = gts,
        feats = feats,
        graph = graph,
        centerlines = centerlines
    )
    # torch.save(res,f"{config['save_dir']}/results_test_0824_big_curve_01.pkl")
    # torch.save(res,f"{config['save_dir']}/results_test_0824_straight_01.pkl")
    torch.save(res,f"{config['save_dir']}/results_test_0824_test_01.pkl")
    
    # evaluate or submit
    # if args.split == "val":
    #     # for val set: compute metric
    #     from argoverse.evaluation.eval_forecasting import (
    #         compute_forecasting_metrics,
    #     )
    #     # Max #guesses (K): 6
    #     _ = compute_forecasting_metrics(preds, gts, cities, 6, 30, 2)
    #     # Max #guesses (K): 1
    #     _ = compute_forecasting_metrics(preds, gts, cities, 1, 30, 2)
    # else:
    #     # for test set: save as h5 for submission in evaluation server
    #     from argoverse.evaluation.competition_util import generate_forecasting_h5
    #     generate_forecasting_h5(preds, f"{config['save_dir']}/submit.h5")  # this might take awhile
    # import ipdb;ipdb.set_trace()


if __name__ == "__main__":
    main()
