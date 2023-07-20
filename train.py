import argparse, os, sys, datetime, glob, logging
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
from torch.nn.utils import clip_grad_norm_
import torchvision.utils as vutils
from torch.cuda.amp import autocast, GradScaler

from omegaconf import OmegaConf     # 从文件、命令行等加载配置，统一处理
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
from src.util import set_logger_format, get_dataloader, get_start_iter, load_from_ckpt, save_model

logger = logging.getLogger(__name__)

def sample_data(loader:torch.utils.data.DataLoader, distributed):
    epoch = 0
    while True:
        if distributed:
            loader.sampler.set_epoch(epoch) # if sampler is shuffling, this synchronizes them across processes
            epoch += 1
        for batch in loader:
            yield batch

def worker_init_fn_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# the only trainer
def train(args):
    """
    Input: {
        "config": "xx/xx.yaml",
        "restart_train": bool,
        "n_data_workers": int,
        "local_rank": int,
        "n_gpu": int,
        "distributed": bool
    }

    """
    device_id = 'cuda:'+str(args.local_rank)
    logger.info(f"Using GPU: {device_id} for training")
    worker_init_fn_seed(100)

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    synchronize()

    if args.local_rank != 0:
        logging.disable(logging.FATAL)  # only show log on rank 0

    config = OmegaConf.load(args.config)
    train_mode = config.train_params.mode

    max_iter = config.train_params.max_iter     # 代替 epoch 数
    n_grad_accumulation = config.train_params.total_batch_size // config.train_params.batch_size // args.n_gpu
    config.model.scheduler_config.params.warm_up_steps //= n_grad_accumulation
    config.model.scheduler_config.params.max_decay_steps //= n_grad_accumulation

    # region 1. make dir
    model_dir = os.path.join(config.paths.ckpt_root, config.paths.experiment_name)
    log_dir = os.path.join(model_dir, "log", str(args.local_rank))
    save_dir = os.path.join(model_dir, "models")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    if get_rank() == 0:
        logger.warning(f'checkpoint dir {model_dir}')
    # endregion 1. make dir

    # region 2.1 init model
    start_iter = get_start_iter(None if args.restart_train else save_dir) if args.continue_iter == -1 else args.continue_iter + 1
    model = ProposedModel(config.model.transformer_config,
                          config.model.visual_decoder_config,
                          lr=config.model.base_learning_rate,
                          use_scheduler=config.model.use_scheduler,
                          scheduler_config=config.model.scheduler_config,
                          device=device_id,
                          **config.model.decoding_params)
    # endregion 2.1 init model

    # region 2.2 init optimizer, scheduler, and scaler
    base_lr = config.model.base_learning_rate
    bs = config.train_params.batch_size
    model.learning_rate = config.train_params.total_batch_size * base_lr
    if get_rank() == 0:
        logger.info(
            "Setting learning rate to {:.2e} = {} * (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                model.learning_rate, args.n_gpu, bs, base_lr))

    optimizer, scheduler = model.configure_optimizers(start_iter-2)
    scaler = GradScaler()   # for mixed precision training
    # endregion 2.2 init optimizer, scheduler, and scaler

    # region 2.3 load model, optimizer, scheduler, and scaler from checkpoint
    iter_name = f"iter_{args.continue_iter}" if args.continue_iter != -1 else 'last'
    ckpt_path = os.path.join(save_dir, f"{iter_name}.ckpt")
    if os.path.exists(ckpt_path) and not args.restart_train:
        logger.warning(f"continue training from {ckpt_path}")
        load_from_ckpt(save_dir, device_id, model, optimizer, scheduler, scaler, iter_name)
    elif train_mode == 'finetune':
        pretrain_dir = os.path.join(config.paths.ckpt_root, config.paths.pretrain_name, "models")
        ckpt_path = os.path.join(pretrain_dir, "last.ckpt")
        logger.warning(f"load pretrained weights from {ckpt_path}")
        load_from_ckpt(pretrain_dir, device_id, model, optimizer, scaler=scaler, iter_name="last")  # use new scheduler
    else:
        logger.warning(f"start new training")
        ckpt_path = None
    # endregion 2.3 load model, optimizer, scheduler, and scaler from checkpoint


    # region 4. set distributed parallel
    model.cuda()
    model.train()
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device_id],
        # output_device='cuda:0',
        # broadcast_buffers=True,
    )
    module = model.module
    # endregion 4. set distributed parallel

    # region 5. dataloader
    if train_mode == "pretrain" or train_mode == "finetune":
        from src.data.PretrainDataset import PretrainDataset
        dataset = PretrainDataset(os.path.join(config.paths.data_root, config.paths.training_set),
                                  os.path.join(config.paths.data_root, config.paths.train_appro),
                                  max_iter, shuffle_listener=config.train_params.shuffle_listener)
    elif train_mode == "example":
        from src.data.ExampleDataset import ExampleDataset
        dataset = ExampleDataset(750)
    else:
        raise ValueError("the dataset must be pretrain or finetune")

    if get_rank() == 0:
        logger.info(f"init dataloader with batch_size = {bs}")
    train_loader = get_dataloader(dataset,
                                  batch_size=bs,
                                  n_workers=args.n_data_workers,
                                  is_train=True,
                                  n_tasks=args.n_gpu,
                                  rank=args.local_rank,
                                  is_distributed=args.distributed)
    # endregion 5. load data

    # region 6. prepare training loop
    summary = SummaryWriter(
        log_dir=log_dir,
        filename_suffix=f"_continue-{args.local_rank}" if ckpt_path else f"_new-{args.local_rank}"
    )

    # make save_interval an integral multiple of n_grad_accumulation, for saving scaler after scaler.update()
    save_interval = config.train_params.save_interval // n_grad_accumulation * n_grad_accumulation
    max_norm = config.train_params.max_norm     # for gradient clip
    log_interval = config.train_params.log_interval

    pbar = range(start_iter, max_iter)
    if get_rank() == 0:
        # 只有 rank 为 0 的进程才会显示进度条
        pbar = tqdm(pbar, total=max_iter, initial=start_iter,
                    dynamic_ncols=True)

    if train_mode == "finetune":
        model.module.set_task(train_mode)

    train_loader = sample_data(train_loader, args.distributed)
    dist.barrier()  # 分布式进程同步，所有进程都到这里了才运行后续代码
    if get_rank() == 0:
        logger.warning(f'\n\tstart training from iter {start_iter}')
    # endregion 6. prepare training loop

    # 7. start training
    last_loss, last_loss_name = None, 'mrm' if train_mode == "finetune" else 'crm'
    optimizer.zero_grad()
    for i_iter in pbar:
        should_optimizer_step = i_iter % n_grad_accumulation == 0
        dataset.set_iter(i_iter)
        batch = next(train_loader)

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda()

        loss_dict = {}
        # no_sync() for gradient accumulation; autocast() for mixed precision training
        with model.no_sync(), autocast():
            for name, loss in model(batch, train_slm=config.train_params.train_slm):
                logger.debug(f"{name} {float(loss)}, rank {args.local_rank}")
                loss_dict[name] = float(loss)
                if name == last_loss_name and should_optimizer_step:
                    last_loss = loss    # last loss should backward outside the model.no_sync()
                else:
                    scaler.scale(loss).backward()  # gradient accumulation

        if should_optimizer_step:
            scaler.scale(last_loss).backward()  # trigger DistributedDataParallel to sync
            scale_before = scaler.get_scale()

            if max_norm:    # if not 0, do gradient clipping
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)  # step() once after several backward() is "gradient accumulation"
            scaler.update()
            if scale_before <= scaler.get_scale():
                scheduler.step()
            optimizer.zero_grad()
            last_loss = None

        # update logger
        for name, value in loss_dict.items():
            summary.add_scalar(tag='loss_'+name, scalar_value=value, global_step=i_iter)

        if get_rank() == 0:
            pbar.set_description(f"sum loss: {sum(loss_dict.values())};")    # 进度条显示 loss

            if i_iter % save_interval == 0:
                save_model(save_dir, i_iter, module, optimizer, scheduler, scaler)
                logger.warning(f"\n\tsave model at iter {i_iter}")

        if i_iter % log_interval == 1:
            if get_rank() == 0:
                logger.info(f"\niter: {i_iter}, lr: {scheduler.get_last_lr()} loss: {', '.join(['%s:%.4e'%(k, v) for k, v in loss_dict.items()])}")
                dist.barrier()
            else:
                dist.barrier()


if __name__ == '__main__':
    # region cmd args, only specify the config file and distribution info
    parser = argparse.ArgumentParser(description="training codes")
    parser.add_argument("--config", type=str, default="./configs/pretrain-5step-sch.yaml",
                        help="experiments configs")
    parser.add_argument("-n", "--n_data_workers", type=int, default=4,
                        help="n_workers for dataloader per GPU")
    parser.add_argument("-r", "--restart_train", action="store_true",
                        help="do not load checkpoint")
    parser.add_argument("-l", "--log_level", type=int, default=1,
                        help="0~4, 0: debug, 1: info, 2: warning, 3: error, 4: critical")
    parser.add_argument("-c", "--continue_iter", type=int, default=-1,
                        help="which iter to load for continue training")
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

    train(args)