#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Author: Cong Liang
Last modified: 2023/6/5
"""
import os
from tqdm import tqdm
import random
import logging
import h5py
from math import ceil

import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms, utils

from src.util import set_logger_format

logger = logging.getLogger(__name__)


class PretrainDataset(data.Dataset):
    """
    Args:
        h5_file_path (string): Path to the h5 file with preprocessed data.
        max_iter (int): the maximum iteration number, used for deciding number of mask
        p_random_n_mask (float): the probability of use random number of mask

    Returns: {
        'sample_name': sample_name,     # string
        's_exp': s_exp,     # (T, 512)
        's_AU': s_AU,       # (T, 25088)
        's_VA': s_VA,       # (T, 1408)
        's_pose': s_pose,   # (T, 1408)
        's_MFCC': s_MFCC,   # (T, 26)
        's_GeMapfunc': s_GeMapfunc.unsqueeze(0),    # (1, 6373)
        's_GeMaplld': s_GeMaplld.reshape(1, -1),    # (1, 65*2997=194805)
        'is_face': is_face,    # (T,)

        'l_AU': l_AU,       # (T, )
        'l_exp': l_exp,     # (T, )
        'l_VA': l_VA,       # (T, )
        'l_mask': l_mask,   # (T, )

        'l_fake_AU': l_fake_AU,     # (T, )
        'l_fake_exp': l_fake_exp,   # (T, )
        'l_fake_VA': l_fake_VA,     # (T, )
    }
    """
    def __init__(self, h5_file_path, appro_npy,
                 max_iter=200001, p_random_n_mask=0.15, shuffle_listener=0):
        """
        Args:
            h5_file_path (string): Path to the h5 file with preprocessed data.
            max_iter (int): the maximum iteration number, used for deciding number of mask
            p_random_n_mask (float): the probability of use random number of mask
        """
        self.h5_path = h5_file_path
        self.dataset = None # h5py.File(self.h5_path, 'r', swmr=True)
        self.sample_names = None # self.dataset['s_name'][()]
        self.l_tokens = None # self.dataset['l_tokens'][()]

        appropriate_matrix = np.load(appro_npy)
        self.neg_samples = dict()
        self.pos_samples = dict()
        count_posit = 0
        for k, v in np.argwhere(appropriate_matrix == 1):   # get idx where value is 1
            count_posit += 1
            if k in self.pos_samples:
                self.pos_samples[k].append(v)
            else:
                self.pos_samples[k] = [v]
        logger.debug(f"count_posit: {count_posit}")
        all_idx_set = set(range(appropriate_matrix.shape[0]))
        for i in self.pos_samples.keys():
            self.neg_samples[i] = list(all_idx_set - set(self.pos_samples[i]))

        self.length = appropriate_matrix.shape[0]
        logger.warning(f"dataset size: {self.length}")

        self.iter = 1
        self.max_iter = max_iter
        self.p_random_n_mask = p_random_n_mask
        self.shuffle_listener = shuffle_listener
        self.refer_idx = 0

        # region prepare data buffer for better speed
        self._s_exp = np.empty((750, 512), dtype=np.float32)
        self._s_AU = np.empty((750, 25088), dtype=np.float32)
        self._s_VA = np.empty((750, 1408), dtype=np.float32)
        self._s_pose = np.empty((150, 1408), dtype=np.float32)
        self._s_MFCC = np.empty((750, 26), dtype=np.float32)
        self._s_GeMapfunc = np.empty((6373,), dtype=np.float32)
        self._s_GeMaplld = np.empty((194805,), dtype=np.float32)
        self._is_face = np.empty((750,), dtype=np.int32)
        # endregion prepare data buffer for better speed

        logger.info("use cosine mask schedule")

    def __len__(self):
        return self.length

    def set_iter(self, iter: int):
        """
        deprecated
        """
        self.iter = iter

    def rand_refer_idx(self):
        """
        update self.refer_idx by random
        """
        self.refer_idx = random.randint(0, self.length-1)

    def _norm(self, tensor):
        return (tensor - torch.mean(tensor)) / torch.std(tensor)

    def _cal_n_mask(self, sample_len):
        """
        calculate the number of mask, following MaskGIT https://arxiv.org/pdf/2202.04200.pdf

        the variable names r and y_r follow the MaskGIT paper.
        """
        r = random.uniform(0, 1)
        y_r = np.cos(r * np.pi / 2)
        n_mask = ceil(sample_len * y_r)
        return n_mask

    def get_neg_idx(self, idx):
        return random.choice(self.neg_samples[idx])

    def get_pos_idx(self, idx):
        return random.choice(self.pos_samples[idx])

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_path, 'r', swmr=True)
            self.l_tokens = self.dataset['l_tokens'][()]
            self.sample_names = self.dataset['s_name'][()]

        # shuffle data between each epoch while set sampler(shuffle=False).
        # this keep fetch data in order to speed up reading h5 file.
        # example:
        #   epoch 1: self.refer_idx == 0, thus fetch data from 0 to length-1
        #   epoch 2: self.refer_idx == k, thus fetch data from k to length-1, then 0 to k-1
        # idx = (idx + self.refer_idx) % self.length      # refer_idx different for each epoch

        try:
            sample_name = self.sample_names[idx]
            self.dataset["s_exp"].read_direct(self._s_exp, np.s_[idx])
            self.dataset["s_AU"].read_direct(self._s_AU, np.s_[idx])
            self.dataset["s_VA"].read_direct(self._s_VA, np.s_[idx])
            self.dataset["s_pose"].read_direct(self._s_pose, np.s_[idx])
            self.dataset["s_MFCC"].read_direct(self._s_MFCC, np.s_[idx])
            self.dataset["s_GeMapfunc"].read_direct(self._s_GeMapfunc, np.s_[idx])
            self.dataset["s_GeMaplld"].read_direct(self._s_GeMaplld, np.s_[idx])
            self.dataset["is_face"].read_direct(self._is_face, np.s_[idx])

            # l_tokens[idx] is already the opponent's response to speaker[idx]
            l_idx = idx
            if random.random() < self.shuffle_listener:
                l_idx = self.get_pos_idx(idx)
            l_AU = self.l_tokens[l_idx][0]
            l_exp = self.l_tokens[l_idx][1]
            l_VA = self.l_tokens[l_idx][2]

            # region fake listener
            fake_idx = self.get_neg_idx(idx)    # get a random negative sample, where speaker behaves differently
            l_fake_AU = self.l_tokens[fake_idx][0]
            l_fake_exp = self.l_tokens[fake_idx][1]
            l_fake_VA = self.l_tokens[fake_idx][2]
            # endregion fake listener

            # region to Tensor
            s_exp = torch.from_numpy(self._s_exp)
            s_AU = torch.from_numpy(self._s_AU)
            s_VA = torch.from_numpy(self._s_VA)
            s_pose = torch.from_numpy(self._s_pose)
            s_MFCC = torch.from_numpy(self._s_MFCC)
            s_GeMapfunc = torch.from_numpy(self._s_GeMapfunc).view(-1)
            s_GeMapfunc = torch.where(torch.isnan(s_GeMapfunc), torch.zeros_like(s_GeMapfunc), s_GeMapfunc)
            s_GeMaplld = torch.from_numpy(self._s_GeMaplld).view(-1)
            s_GeMaplld = torch.where(torch.isnan(s_GeMaplld), torch.zeros_like(s_GeMaplld), s_GeMaplld)
            is_face = torch.from_numpy(self._is_face).bool()

            l_AU = torch.from_numpy(l_AU)
            l_exp = torch.from_numpy(l_exp)
            l_VA = torch.from_numpy(l_VA)

            l_fake_AU = torch.from_numpy(l_fake_AU)
            l_fake_exp = torch.from_numpy(l_fake_exp)
            l_fake_VA = torch.from_numpy(l_fake_VA)
            # endregion to Tensor

            # region random sample mask of listener
            n_masked = self._cal_n_mask(l_AU.shape[0])
            l_mask = torch.ones(l_AU.shape[0], 3)
            for i in range(3):
                idx_to_mask = random.sample(range(l_AU.shape[0]), n_masked)
                l_mask[idx_to_mask, i] = 0     # 0 means masked
            logger.debug(f"iter: {self.iter}, n_masked: {n_masked}, l_mask: {l_mask[0]}")
            # endregion random sample mask of listener


        except Exception as e:
            logger.fatal(f"Error in loading {idx}th sample: {e}")
            raise e

        return {
            'sample_name': sample_name,     # string
            's_exp': s_exp,     # (T, 512)
            's_AU': s_AU,       # (T, 25088)
            's_VA': s_VA,       # (T, 1408)
            's_pose': s_pose.repeat_interleave(5, dim=0),   # (T, 1408)
            's_MFCC': s_MFCC,   # (T, 26)
            's_GeMAPfunc': self._norm(s_GeMapfunc).unsqueeze(0),    # (1, 6373)
            's_GeMAPlld': self._norm(s_GeMaplld).unsqueeze(0),      # (1, 65*2997=194805)
            'is_face': is_face,    # (T,)

            'l_AU': l_AU.long(),       # (T, )
            'l_exp': l_exp.long(),     # (T, )
            'l_VA': l_VA.long(),       # (T, )
            'l_mask': l_mask.bool(),   # (T, 3)

            'l_fake_AU': l_fake_AU,     # (T, )
            'l_fake_exp': l_fake_exp,   # (T, )
            'l_fake_VA': l_fake_VA,     # (T, )
        }


if __name__ == "__main__":
    # test dataset
    set_logger_format(logging.INFO)

    batch_size = 1
    dataloader = data.DataLoader(
        PretrainDataset('/data03/wjh/train_shuffled.h5',
                        '/data03/wjh/Approprirate_train_shuffled.npy'),
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        drop_last=False,
    )

    for batch in tqdm(dataloader):
        s_exp = batch['s_exp']
        assert s_exp.shape == (batch_size, 750, 512), f"s_exp shape: {s_exp.shape}"
        s_AU = batch['s_AU']
        assert s_AU.shape == (batch_size, 750, 25088), f"s_AU shape: {s_AU.shape}"
        s_VA = batch['s_VA']
        assert s_VA.shape == (batch_size, 750, 1408), f"s_VA shape: {s_VA.shape}"
        s_pose = batch['s_pose']
        assert s_pose.shape == (batch_size, 750, 1408), f"s_pose shape: {s_pose.shape}"
        s_MFCC = batch['s_MFCC']
        assert s_MFCC.shape == (batch_size, 750, 26), f"s_MFCC shape: {s_MFCC.shape}"
        s_GeMapfunc = batch['s_GeMAPfunc']
        assert s_GeMapfunc.shape == (batch_size, 1, 6373), f"s_GeMapfunc shape: {s_GeMapfunc.shape}"
        s_GeMaplld = batch['s_GeMAPlld']
        assert s_GeMaplld.shape == (batch_size, 1, 194805), f"s_GeMaplld shape: {s_GeMaplld.shape}"
        # print("s_GeMaplld")
        # print(torch.any(torch.isnan(s_GeMaplld)))
        # print("s_GeMapfunc")
        # print(torch.any(torch.isnan(s_GeMapfunc)))
        is_face = batch['is_face']
        assert is_face.shape == (batch_size, 750), f"is_face shape: {is_face.shape}"

        l_AU = batch['l_AU']
        assert l_AU.shape == (batch_size, 750), f"l_AU shape: {l_AU.shape}"
        l_exp = batch['l_exp']
        assert l_exp.shape == (batch_size, 750), f"l_exp shape: {l_exp.shape}"
        l_VA = batch['l_VA']
        assert l_VA.shape == (batch_size, 750), f"l_VA shape: {l_VA.shape}"
        # print("l_VA", l_VA[0][:20])
        l_mask = batch['l_mask']
        assert l_mask.shape == (batch_size, 750, 3), f"l_mask shape: {l_mask.shape}"

        l_fake_AU = batch['l_fake_AU']
        assert l_fake_AU.shape == (batch_size, 750), f"l_fake_AU shape: {l_fake_AU.shape}"
        # print("l_fake_AU", l_fake_AU[0][:20])
        l_fake_exp = batch['l_fake_exp']
        assert l_fake_exp.shape == (batch_size, 750), f"l_fake_exp shape: {l_fake_exp.shape}"
        l_fake_VA = batch['l_fake_VA']
        assert l_fake_VA.shape == (batch_size, 750), f"l_fake_VA shape: {l_fake_VA.shape}"
        # input(":")

