# Copyright (c) 2020 Uber Technologies, Inc.
# See the License for the specific language governing permissions and
# limitations under the License.

from utils import Logger, load_pretrain
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Sampler, DataLoader
import torch
from tqdm import tqdm
from config import get_ckpt_path2
from numbers import Number
from importlib import import_module
import shutil
import time
import sys
import random
import numpy as np
import argparse
import os
import logging
# 有几块GPU写多少
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:100'

os.umask(0)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)


parser = argparse.ArgumentParser(description="Fuse Detection in Pytorch")
parser.add_argument("--local_rank", type=int, default=1,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("-m", "--model", default="lanegcn",
                    type=str, metavar="MODEL", help="model name")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--resume", default="", type=str,
                    metavar="RESUME", help="checkpoint path")
parser.add_argument("--weight", default="", type=str,
                    metavar="WEIGHT", help="checkpoint path")
parser.add_argument('--sample_file', metavar='PATH',
                    type=str, help='list to pickle files.')
parser.add_argument('--root_dir', metavar='PATH', type=str,
                    help='directory to pickle files.')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def main():
    logger.info("Starting training")
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Import all settings for experiment.
    args = parser.parse_args()
    model = import_module(args.model)
    config, Dataset, collate_fn, net, loss, post_process, opt = model.get_model()

    # if args.resume or args.weight:
    #     ckpt_path = args.resume or args.weight
    #     if not os.path.isabs(ckpt_path):
    #         ckpt_path = os.path.join(config["save_dir"], ckpt_path)
    #     ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    #     load_pretrain(net, ckpt["state_dict"])
    #     if args.resume:
    #         config["epoch"] = ckpt["epoch"]
    #         opt.load_state_dict(ckpt["opt_state"])
    ckpt_path = get_ckpt_path2()  # load pretrain model 36.00ckpt
    ckpt_path = os.path.join(root_path, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    load_pretrain(net, ckpt["state_dict"])

    # Create log and copy all code
    save_dir = config["save_dir"]
    # save_dir = config["save_dir2"]
    log = os.path.join(save_dir, "log")
    logger.info(f"Log will be saved to {log}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #sys.stdout = Logger(log)
    src_dirs = [root_path]
    dst_dirs = [os.path.join(save_dir, "files")]
    for src_dir, dst_dir in zip(src_dirs, dst_dirs):
        files = [f for f in os.listdir(src_dir) if f.endswith(".py")]
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for f in files:
            shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))
        logger.info(f"Log will be saved to {log}")                     
    # Data loader for training
    dataset = Dataset(config["train_split"], config, train=True)

    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        # sampler=train_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    epoch = config["epoch"]
    remaining_epochs = int(np.ceil(config["num_epochs"] - epoch))
    for i in range(remaining_epochs):
        train(epoch + i, config, train_loader, net, loss, post_process, opt)
        
        logger.info(f"Completed epoch {epoch + i}")

    logger.info("Training completed")

# def worker_init_fn(pid):
#     pass


def worker_init_fn(pid):
    np_seed = int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)


def train(epoch, config, train_loader, net, loss, post_process, opt):
    # train_loader.sampler.set_epoch(int(epoch))
    net.train()
    
    num_batches = len(train_loader)
    epoch_per_batch = 1.0 / num_batches
    save_iters = int(np.ceil(config["save_freq"] * num_batches))
    display_iters = int(
        config["display_iters"] / (config["batch_size"])
    )

    start_time = time.time()
    metrics = dict()
    progress_bar = tqdm(total=num_batches, desc=f"Epoch {epoch}")
    #for i, data in tqdm(enumerate(train_loader)):
    for i, data in enumerate(train_loader):
        epoch += epoch_per_batch
        data = dict(data)

        output = net(data)
        loss_out = loss(output, data)
        post_out = post_process(output, data)
        post_process.append(metrics, loss_out, post_out)
        post_process.append(metrics, loss_out, post_out)

        opt.zero_grad()
        loss_out["loss"].backward()
        lr = opt.step(epoch)

        num_iters = int(np.round(epoch * num_batches))
        if num_iters % save_iters == 0 or epoch >= config["num_epochs"]:
            save_ckpt(net, opt, config["save_dir"], epoch)

        if num_iters % display_iters == 0:
            dt = time.time() - start_time
            if True:
                post_process.display(metrics, dt, epoch, lr)
            start_time = time.time()
            metrics = dict()

        progress_bar.update(1)
    progress_bar.close()

    if epoch >= config["num_epochs"]:
        return


def val(config, data_loader, net, loss, post_process, epoch):
    net.eval()

    start_time = time.time()
    metrics = dict()
    for i, data in enumerate(data_loader):
        data = dict(data)
        with torch.no_grad():
            output = net(data)
            loss_out = loss(output, data)
            post_out = post_process(output, data)
            post_process.append(metrics, loss_out, post_out)

    dt = time.time() - start_time
    metrics = sync(metrics)
    if hvd.rank() == 0:
        post_process.display(metrics, dt, epoch)
    net.train()


def save_ckpt(net, opt, save_dir, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    save_name = "%3.3f.ckpt" % epoch
    torch.save(
        {"epoch": epoch, "state_dict": state_dict,
            "opt_state": opt.opt.state_dict()},
        os.path.join(save_dir, save_name),
    )


def sync(data):
    data_list = comm.allgather(data)
    data = dict()
    for key in data_list[0]:
        if isinstance(data_list[0][key], list):
            data[key] = []
        else:
            data[key] = 0
        for i in range(len(data_list)):
            data[key] += data_list[i][key]
    return data


if __name__ == "__main__":
    main()

