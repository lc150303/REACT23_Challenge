import hashlib
import requests
from tqdm import tqdm
import torch
from torch.utils import data
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import logging
from glob import glob
import argparse

from omegaconf import OmegaConf

import pandas as pd
import ast
import pickle

import os
import logging
import ffmpeg
import numpy as np

logger = logging.getLogger(__name__)


def ffmpeg_video_write(data, video_path, fps=1):
    """Video writer based on FFMPEG.
    Args:
    data: A `np.array` with the shape of [seq_len, height, width, 3]
    video_path: A video file.
    fps: Use specific fps for video writing. (optional)
    """
    assert len(data.shape) == 4, f'input shape is not valid! Got {data.shape}!'
    _, height, width, _ = data.shape
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    writer = (
        ffmpeg
        .input('pipe:', framerate=fps, format='rawvideo',
               pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .output(video_path, pix_fmt='yuv420p')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    for frame in data:
        writer.stdin.write(frame.astype(np.uint8).tobytes())
    writer.stdin.close()


URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1",
    "re_first_stage": "https://heibox.uni-heidelberg.de/f/6db3e2beebe34c1e8d06/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "geofree/lpips/vgg.pth",
    "re_first_stage": "geofree/re_first_stage/last.ckpt",
}
CACHE = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
CKPT_MAP = dict((k, os.path.join(CACHE, CKPT_MAP[k])) for k in CKPT_MAP)

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a",
    "re_first_stage": "b8b999aba6b618757329c1f114e9f5a5"
}


#   credit to https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    blue = "\x1b[36;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s %(filename)s:%(lineno)d - %(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: blue + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def set_logger_format(level):
    """
    set global logger format
    """
    h = logging.StreamHandler()
    h.setFormatter(CustomFormatter())
    logging.basicConfig(
        level=level,
        handlers=[h]
    )


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_local_path(name, root=None, check=False):
    path = CKPT_MAP[name]
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        assert name in URL_MAP, name
        print("Downloading {} from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


# borrow from https://github.com/LeeSinLiang/microGPT
def top_k_top_p_filter(logits, top_k: int = 0, top_p: float = 1.0):
    """
    :param logits: (B, T, vocab_size)
    :param top_k: int, 0 means no filtering
    :param top_p: float, 1.0 means no filtering
    :return: logits (B, T, vocab_size)

    Usage:
        #>>> logits = torch.rand(2, 3, 10)  # (B, T, vocab_size)
        #>>> print(logits)
        #>>> print(top_k_top_p_filter(logits, top_k=0, top_p=0.5))
    """
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[..., [-1]]] = float('-inf')
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # both shapes are (B, T, vocab_size)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p  # zeros at the beginning of each row, followed by 1s
        filter[..., 1:] = filter[...,
                          :-1].clone()  # shift right by 1 since filter includes the first index that exceeds top_p
        filter[..., 0] = 0  # zero means valid, at least one index is valid

        # move 0s and 1s in filter to their original place according to sorted_indices
        indices_to_remove = filter.scatter(-1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')  # where filter is 1, logits will be set to -inf
    return logits


def mask_lower_n(probs: torch.Tensor, n_mask: int):
    """
    :param probs: (B, 3*T)
    :return: (B, T, 3)

    0 means masked
    """
    sorted_probs, _ = torch.sort(probs, dim=-1)
    cut_off_value = sorted_probs[:, n_mask - 1]  # (B, )
    mask = probs > cut_off_value.unsqueeze(-1)  # (B, 3*T) > (B, 1) == (B, 3*T)
    mask = mask.unsqueeze(-1)
    T = mask.shape[1] // 3
    return torch.cat((
        mask[:, :T], mask[:, T:2 * T], mask[:, 2 * T:]
    ), dim=-1)


def get_dataloader(dataset, batch_size, n_workers, is_train,
                   n_tasks=1, rank=0, is_distributed=False, drop_last=False):
    if is_distributed:
        sampler = data.DistributedSampler(dataset, num_replicas=n_tasks, rank=rank, shuffle=False)
    else:
        sampler = data.SequentialSampler(dataset)

    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        sampler=sampler,
        drop_last=drop_last,
    )


def get_start_iter(save_dir):
    if save_dir is None:
        return 1
    ckpts = glob(os.path.join(save_dir, "iter_*.ckpt"))
    iters = [int(os.path.basename(ckpt).split(".")[0].split("_")[-1]) for ckpt in ckpts]
    return max(iters + [0]) + 1


def save_model(save_dir, i_iter, model, optimizer, scheduler=None, scaler=None):
    # model
    torch.save(model.state_dict(), os.path.join(save_dir, f"iter_{i_iter}.ckpt"))
    torch.save(model.state_dict(), os.path.join(save_dir, "last.ckpt"))
    # optimizer
    torch.save(optimizer.state_dict(), os.path.join(save_dir, f"opt_iter_{i_iter}.ckpt"))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "opt_last.ckpt"))
    # scheduler
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(save_dir, f"schlr_iter_{i_iter}.ckpt"))
        torch.save(scheduler.state_dict(), os.path.join(save_dir, "schlr_last.ckpt"))
    # scaler
    if scaler is not None:
        torch.save(scaler.state_dict(), os.path.join(save_dir, f"scaler_iter_{i_iter}.ckpt"))
        torch.save(scaler.state_dict(), os.path.join(save_dir, "scaler_last.ckpt"))


def load_from_ckpt(save_dir, device, model, optimizer=None, scheduler=None, scaler=None, iter_name='last'):
    """
    load model, optimizer, scheduler, and scaler from the same dir to the same device.
    scheduler and scaler are optional.
    """
    model_file = os.path.join(save_dir, f'{iter_name}.ckpt')
    sd = torch.load(model_file, map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info(f"Restored from {model_file} with {len(missing)} missing keys and {len(unexpected)} unexpected keys.")

    if optimizer is not None:
        # optimizer
        opt_file = os.path.join(save_dir, f'opt_{iter_name}.ckpt')
        optimizer.load_state_dict(torch.load(opt_file, map_location=device))

    schlr_file = os.path.join(save_dir, f'schlr_{iter_name}.ckpt')
    if os.path.exists(schlr_file) and scheduler is not None:
        scheduler.load_state_dict(torch.load(schlr_file, map_location=device))

    scaler_file = os.path.join(save_dir, f'scaler_{iter_name}.ckpt')
    if os.path.exists(scaler_file) and scaler is not None:
        scaler.load_state_dict(torch.load(scaler_file, map_location=device))


class VAIndex:
    def __init__(self, file_path):
        self.bins = np.linspace(-1, 1, 81)
        self.df = pd.read_csv(file_path)
        self.index_dict = dict(zip(self.df['index'], self.df['combinations'].apply(ast.literal_eval)))
        self.combinations_dict = dict(zip(self.df['combinations'].apply(ast.literal_eval), self.df['index']))

    def _discrete_va(self, valence, arousal):
        valence_bin = np.digitize(valence, self.bins) - 1
        arousal_bin = np.digitize(arousal, self.bins) - 1

        return (self.bins[valence_bin] + 0.025, self.bins[arousal_bin] + 0.025)

    def get_combinations(self, index):
        return self.index_dict[index]

    def get_index(self, combinations):
        try:
            index = self.combinations_dict[self._discrete_va(combinations[0], combinations[1])]
        except:
            index = 0

        return index


class AUIndex:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.index_dict = dict(zip(self.df['index'], self.df['combinations'].apply(ast.literal_eval)))
        self.combinations_dict = dict(zip(self.df['combinations'].apply(ast.literal_eval), self.df['index']))

    def get_combinations(self, index):
        return self.index_dict[index]

    def get_index(self, combinations):
        return self.combinations_dict[combinations]


class ExpIndex:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
            self.centers = self.model.cluster_centers_

    def get_index(self, features):
        index = self.model.predict(features)
        return index

    def get_combinations(self, index):
        combinations = self.centers[index]
        return tuple(combinations)


class TokenDecoder():
    def __init__(self, AU_csv, exp_pkl, VA_csv):
        self.AU_decoder = AUIndex(AU_csv)
        self.exp_decoder = ExpIndex(exp_pkl)
        self.VA_decoder = VAIndex(VA_csv)

    def decode(self, AU_index: int, exp_index: int, VA_index: int):
        """
        Input:
            3 ints
        output:
            (25, )
        """
        AU = self.AU_decoder.get_combinations(AU_index)
        exp = self.exp_decoder.get_combinations(exp_index)
        VA = self.VA_decoder.get_combinations(VA_index)
        logger.debug(f"AU: {type(AU)}, exp: {type(exp)}, VA: {type(VA)}")

        return np.concatenate([AU, VA, exp], axis=-1)

    def decode_npy(self, total_pred):
        """
        Input:
            (3, n_samples, 10, T)
        Output:
            (n_samples, 10, T, 25)
        """
        AU_index = total_pred[0]
        exp_index = total_pred[1]
        VA_index = total_pred[2]
        logger.debug(f"AU_index: {AU_index.shape}")
        logger.debug(f"exp_index: {exp_index.shape}")

        # (n_samples, 10, T) to (n_samples, 10, T, len_AU)
        AU = np.array(np.vectorize(self.AU_decoder.get_combinations)(AU_index))
        AU = AU.transpose(*list(range(1, len(AU.shape))), 0)
        logger.debug(f"AU: {AU.shape}")
        # (n_samples, 10, T) to (n_samples, 10, T, len_VA)
        VA = np.array(np.vectorize(self.VA_decoder.get_combinations)(VA_index))
        VA = VA.transpose(*list(range(1, len(VA.shape))), 0)
        logger.debug(f"VA: {VA.shape}")
        # (n_samples, 10, T) to (n_samples, 10, T, len_exp)
        exp = np.array(np.vectorize(self.exp_decoder.get_combinations)(exp_index))
        exp = exp.transpose(*list(range(1, len(exp.shape))), 0)
        logger.debug(f"exp: {exp.shape}")

        return np.concatenate([AU, VA, exp], axis=-1)


def save_decoded_npy(token_pred_npy: str, AU_csv: str, exp_pkl: str, VA_csv: str, save_dir: str):
    """
    :param token_pred_npy: "path_to/*_tokens_pred.npy"
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    decoder = TokenDecoder(AU_csv, exp_pkl, VA_csv)
    total_pred = np.load(token_pred_npy)
    logger.warning(f"token_pred: {total_pred.shape}")
    decoded_pred = decoder.decode_npy(total_pred)
    logger.warning(f"decoded_pred: {decoded_pred.shape}")
    np.save(os.path.join(save_dir,
                         os.path.basename(token_pred_npy).replace("tokens_pred", "pred")),
            decoded_pred)

def select_five_samples_to_visual(decoded_val_npy:str, save_dir:str):
    idx_to_save = {
        "NoXI/001_2016-03-17_Paris/Expert_video/21": 306,
        "NoXI/023_2016-04-25_Paris/Expert_video/25": 455,
        "NoXI/019_2016-04-20_Paris/Expert_video/13": 582,
        "UDIVA/animal/152153/FC1/3": 667,
        "RECOLA/group-2/P41/2": 831
    }

    decoded_val = np.load(decoded_val_npy)  # (n_samples, 10, T, 25)
    print('decoded_val.shape', decoded_val.shape)

    """
    out_dict = {
        name: (10, T, 25)
    }
    """
    out_dict = {}
    for name, idx in idx_to_save.items():
        out_dict[name] = decoded_val[idx]


    pickle.dump(out_dict,
                open(os.path.join(save_dir,
                                  os.path.basename(decoded_val_npy).replace("pred.npy", "5_samples.pkl")),
                     "wb"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="utils manual functions")
    parser.add_argument("-d", "--decode_file", type=str, default="", help="the file path of tokens_pred.npy to decode")
    parser.add_argument("-s", "--select_for_visual", type=str, default="", help="the file path of pred.npy to select")
    args = parser.parse_args()

    if args.decode_file:
        data_root = "data/Index/"
        save_decoded_npy(
            args.decode_file,
            os.path.join(data_root, "AU_index.csv"),
            os.path.join(data_root, "kmeans.pkl"),
            os.path.join(data_root, "VA_index.csv"),
            save_dir="results"
        )

    if args.select_for_visual:
        select_five_samples_to_visual(
            args.select_for_visual,
            save_dir="results"
        )

    # test top_k_top_p_filter
    # logits = torch.rand(2, 3, 7)
    # print('total\n', logits)
    # logits = top_k_top_p_filter(logits[:, -2:], top_k=4, top_p=0)
    # print(logits)
    # logits = F.softmax(logits, dim=-1)
    # print('after softmax\n', logits)
    # print(Categorical(logits).sample())

    # test TokenDecoder
    # decoder = TokenDecoder('/home/wjh/React2023/Index/AU_index.csv',
    #                        '/home/wjh/React2023/Index/kmeans.pkl',
    #                        '/home/wjh/React2023/Index/VA_index.csv')
    # result = decoder.decode(0, 0, 0)
    # print(result)
    # result = decoder.decode_npy(np.zeros((3, 4, 10, 5), dtype=int))
    # print(result.shape)
    # print(result[0, 0, 0])
