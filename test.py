import argparse, os, sys, datetime, glob, logging
from contextlib import contextmanager
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torchvision.utils as vutils

from omegaconf import OmegaConf  # 从文件、命令行等加载配置，统一处理
# DDP
from torch.utils import data
import torch.distributed as dist
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

from src.models.proposed_model import ProposedModel
from src.util import set_logger_format, get_dataloader, load_from_ckpt, TokenDecoder
from src.metric.metric import measure

logger = logging.getLogger(__name__)

def worker_init_fn_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def conditional_autocast(condition):
    if condition:
        with autocast():
            yield
    else:
        yield

# the only trainer
def predict(args):
    """
    Input: {
        "config": "xx/xx.yaml",
        "restart_train": bool,
        "local_rank": int,
        "n_gpu": int,
        "distributed": bool
    }

    """
    device_id = 'cuda:' + str(args.local_rank)
    logger.info(f"Using GPU: {device_id} for testing")
    worker_init_fn_seed(100)

    torch.cuda.set_device(args.local_rank)
    if args.distributed:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        dist.barrier()

    if args.local_rank != 0:
        logging.disable(logging.WARNING)  # only show log on rank 0

    config = OmegaConf.load(args.config)
    decoding_params = config.model.decoding_params
    decoding_schedule = getattr(config.test_params, "schedule_step", None)
    temp_step = f"{decoding_schedule.temperature:+}" if decoding_schedule and decoding_schedule.temperature else ''
    k_step = f"{decoding_schedule.top_k:+}" if decoding_schedule and decoding_schedule.top_k else ''
    p_step = f"{decoding_schedule.top_p:+}" if decoding_schedule and decoding_schedule.top_p else ''
    step = decoding_params.step if args.task == "offline" else decoding_params.online_step

    # region 1. make dir, check model
    model_dir = os.path.join(config.paths.ckpt_root, config.paths.experiment_name)
    log_dir = os.path.join(model_dir, "test_log")
    result_dir = os.path.join(model_dir, "test_result")
    save_dir = os.path.join(model_dir, "models")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    assert os.path.exists(save_dir)
    if args.local_rank == 0:
        logger.warning(f'checkpoint dir {model_dir}')
    # endregion 1. make dir, check model

    # region 2.1 init model
    model = ProposedModel(config.model.transformer_config,
                          config.model.visual_decoder_config,
                          use_scheduler=False,
                          device=device_id,
                          **decoding_params)
    # endregion 2.1 init model

    # region 2.3 load model for testing
    iter_name = f"iter_{args.load_iter}" if args.load_iter != -1 else 'last'
    ckpt_path = os.path.join(save_dir, f"{iter_name}.ckpt")
    assert os.path.exists(ckpt_path), f"checkpoint {ckpt_path} not exists"
    load_from_ckpt(save_dir, device_id, model, iter_name=iter_name)
    # endregion 2.3 load model for testing

    # region 4. set distributed parallel
    model.cuda()
    model.eval()
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device_id],
            # output_device='cuda:0',
            # broadcast_buffers=True,
        )
    model.set_task(args.task)
    # endregion 4. set distributed parallel

    # region 5. dataloader
    if args.dataset == "example":
        from src.data.ExampleDataset import ExampleDataset
        dataset = ExampleDataset(750)
    else:
        from src.data.TestDataset import TestDataset
        dataset = TestDataset(os.path.join(config.paths.data_root,
                                           config.paths.val_set if args.dataset == "val" else config.paths.test_set))

    batch_size = config.test_params.batch_size
    test_loader = get_dataloader(dataset,
                                 batch_size=batch_size,
                                 is_train=False,
                                 n_workers=1,
                                 n_tasks=args.n_gpu,
                                 rank=args.local_rank,
                                 is_distributed=args.distributed)
    # endregion 5. load data

    # region 6. prepare testing loop
    log_interval = config.train_params.log_interval

    if args.local_rank == 0:
        # 只有 rank 为 0 的进程才会显示进度条
        test_loader = tqdm(test_loader, dynamic_ncols=True)

    if args.distributed:
        dist.barrier()  # 分布式进程同步，所有进程都到这里了才运行后续代码
    if args.local_rank == 0:
        logger.warning(f'\n\tstart predict {args.task} {args.dataset} '
                       f'with step {decoding_params.step if args.task=="offline" else decoding_params.online_step}, '
                       f'temp {decoding_params.temperature}{temp_step},  top-k {decoding_params.top_k}{k_step}, '
                       f'top-p {decoding_params.top_p}{p_step}, '
                       f'{" temp_decay" if "temp_decay" in decoding_params else ""}')
    # endregion 6. prepare testing loop

    # 7. start testing
    total_l_AU_pred = []  # list of (batch_size, 10, T)
    total_l_exp_pred = []  # list of (batch_size, 10, T)
    total_l_VA_pred = []  # list of (batch_size, 10, T)
    prediction_probability = np.zeros((len(dataset), 10))
    with conditional_autocast(args.half_precision):
        for i_iter, batch in enumerate(test_loader):
            # reset the decoding params for each sample
            model.top_k = decoding_params.top_k
            model.top_p = decoding_params.top_p
            model.temperature = decoding_params.temperature

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()

            i_l_AU_pred, i_l_exp_pred, i_l_VA_pred = [], [], []  # list of (batch_size, 1, T)
            for j in range(10):
                logger.debug(f"iter {i_iter}, j {j}")
                if args.accumulate_probs:
                    l_AU_pred, l_exp_pred, l_VA_pred, accum_prob = model.predict(batch, return_prob=True)  # (batch_size, T)
                    prediction_probability[i_iter * batch_size: (i_iter + 1) * batch_size, j] = accum_prob # (batch_size,)
                else:
                    l_AU_pred, l_exp_pred, l_VA_pred = model.predict(batch)  # (batch_size, T)
                i_l_AU_pred.append(l_AU_pred.cpu().unsqueeze(1))  # (batch_size, 1, T)
                i_l_exp_pred.append(l_exp_pred.cpu().unsqueeze(1))
                i_l_VA_pred.append(l_VA_pred.cpu().unsqueeze(1))

                if decoding_schedule is not None:
                    model.top_k = int(model.top_k + decoding_schedule.top_k)
                    model.top_p += decoding_schedule.top_p
                    model.temperature += decoding_schedule.temperature

            total_l_AU_pred.append(torch.cat(i_l_AU_pred, dim=1))  # (batch_size, 10, T)
            total_l_exp_pred.append(torch.cat(i_l_exp_pred, dim=1))
            total_l_VA_pred.append(torch.cat(i_l_VA_pred, dim=1))
            # if i_iter > 2:
            #     logger.warning(total_l_AU_pred[0].shape)
            #     break

    total_l_AU_pred = torch.cat(total_l_AU_pred, dim=0)  # (n_samples, 10, T)
    total_l_exp_pred = torch.cat(total_l_exp_pred, dim=0)
    total_l_VA_pred = torch.cat(total_l_VA_pred, dim=0)

    total_pred = np.stack([total_l_AU_pred.numpy(),
                           total_l_exp_pred.numpy(),
                           total_l_VA_pred.numpy()], axis=0)

    logger.info(f"total_pred.shape {total_pred.shape}")      # (3, n_samples, 10, T)
    npy_path = os.path.join(result_dir, f'{args.dataset}_{config.paths.experiment_name}_{args.load_iter}_{args.task}_s{step}_'
                                        f't{decoding_params.temperature}{temp_step}_'
                                        f'k{decoding_params.top_k}{k_step}_'
                                        f'p{decoding_params.top_p}{p_step}_'
                                        f'{"decay_" if "temp_decay" in decoding_params else ""}tokens_pred.npy')
    logger.warning(f"save to {npy_path}")
    np.save(npy_path, total_pred)

    if args.accumulate_probs:
        np.save(npy_path.replace("tokens_pred", "probs"), prediction_probability)

def cal_metrics(args):
    config = OmegaConf.load(args.config)
    decoding_params = config.model.decoding_params
    decoding_schedule = getattr(config.test_params, "schedule_step", None)
    temp_step = f"{decoding_schedule.temperature:+}" if decoding_schedule and decoding_schedule.temperature else ''
    k_step = f"{decoding_schedule.top_k:+}" if decoding_schedule and decoding_schedule.top_k else ''
    p_step = f"{decoding_schedule.top_p:+}" if decoding_schedule and decoding_schedule.top_p else ''
    step = decoding_params.step if args.task == "offline" else decoding_params.online_step

    model_dir = os.path.join(config.paths.ckpt_root, config.paths.experiment_name)
    log_dir = os.path.join(model_dir, "test_log")
    result_dir = os.path.join(model_dir, "test_result")
    npy_path = os.path.join(result_dir, f'{args.dataset}_{config.paths.experiment_name}_{args.load_iter}_{args.task}_s{step}_'
                                        f't{decoding_params.temperature}{temp_step}_'
                                        f'k{decoding_params.top_k}{k_step}_'
                                        f'p{decoding_params.top_p}{p_step}_'
                                        f'{"decay_" if "temp_decay" in decoding_params else ""}tokens_pred.npy')
    logger.warning(f"load tokens_pred from {npy_path}")

    assert os.path.exists(npy_path), f"npy_path {npy_path} not exists"
    total_pred = np.load(npy_path)  # (3, n_samples, 10, T)

    decoder = TokenDecoder(
        config.test_params.AU_csv,
        config.test_params.exp_pkl,
        config.test_params.VA_csv,
    )

    total_pred = decoder.decode_npy(total_pred)  # (n_samples, 10, T, 25)
    if args.save_pred:
        decoded_name = os.path.basename(npy_path).replace("tokens_pred", "pred")
        np.save(os.path.join('./results', decoded_name), total_pred)
    total_pred = torch.from_numpy(total_pred)  # (n_samples, 10, T, 25)
    s_gt = torch.from_numpy(np.load(config.test_params.s_gt_path))
    l_gt = torch.from_numpy(np.load(config.test_params.l_gt_path))
    logger.info(f"total_pred.shape {total_pred.shape}, s_gt.shape {s_gt.shape}, l_gt.shape {l_gt.shape}")
    appro_matrix = np.load(os.path.join(config.paths.data_root, config.paths.val_appro))
    FRC, FRD, FRDvs, FRVar, smse, TLCC = measure(total_pred, l_gt, s_gt, appro_matrix)

    logger.warning(f"FRC {FRC:.6f}, FRDist {FRD:.6f}, FRDiv {smse:.6f}, "
                   f"FRVar {FRVar:.6f}, FRDvs {FRDvs:.6f}, FRSyn {TLCC:.6f}")
    with open(os.path.join(log_dir, f'{args.dataset}_{config.paths.experiment_name}_{args.load_iter}_{args.task}_s{step}_'
                                    f't{decoding_params.temperature}{temp_step}_'
                                    f'k{decoding_params.top_k}{k_step}_'
                                    f'p{decoding_params.top_p}{p_step}_'
                                    f'{"decay_" if "temp_decay" in decoding_params else ""}metrics.txt'), 'w') as f:
        f.write(f"FRC {FRC:.6f}, FRDist {FRD:.6f}, FRDiv {smse:.6f}, "
                f"FRVar {FRVar:.6f}, FRDvs {FRDvs:.6f}, FRSyn {TLCC:.6f}")


if __name__ == '__main__':
    # region cmd args, only specify the config file and distribution info
    parser = argparse.ArgumentParser(description="testing codes")
    parser.add_argument("--config", type=str, default="./configs/test-5s-randL-gp-0.yaml",
                        help="experiments configs")
    parser.add_argument("-p", "--predict", action="store_true", help="run prediction")
    parser.add_argument("-m", "--measure", action="store_true", help="run measure")
    parser.add_argument("-l", "--log_level", type=int, default=1,
                        help="0~4, 0: debug, 1: info, 2: warning, 3: error, 4: critical")
    parser.add_argument("-i", "--load_iter", type=int, default=-1,
                        help="which iter to load for evaluation")
    parser.add_argument("-t", "--task", type=str, default="offline",
                        help="which task to eval, 'online', 'offline'")
    parser.add_argument("-d", "--dataset", type=str, default="val",
                        help="which dataset to eval, 'val', 'test'")
    parser.add_argument("-hf", "--half_precision", action="store_true", help="inference with FP16")
    parser.add_argument("-s", "--save_pred", action="store_true", help="save decoded prediction")
    parser.add_argument("-ap", "--accumulate_probs", action="store_true", help="save probability of predictions")
    # endregion cmd args, only specify the config file and distribution info

    # set the args
    args = parser.parse_args()
    args.local_rank = int(os.environ["RANK"])

    log_levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
    set_logger_format(level=log_levels[args.log_level])

    logger.info(f"WORLD_SIZE: {os.environ['WORLD_SIZE']}, local_rank: {args.local_rank}")
    args.n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = args.n_gpu > 1

    # endregion cmd args

    if args.predict:
        predict(args)
    if args.measure:
        cal_metrics(args)