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

import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms, utils

from src.util import set_logger_format

logger = logging.getLogger(__name__)


class TestDataset(data.Dataset):
    """
    Args:
        h5_file_path (string): Path to the h5 file with preprocessed data.

    Returns: {
        'sample_name': sample_name,     # string
        's_exp': s_exp,     # (T, 512)
        's_AU': s_AU,       # (T, 25088)
        's_VA': s_VA,       # (T, 1408)
        's_pose': s_pose,   # (T, 6)
        's_MFCC': s_MFCC,   # (T, 26)
        's_GeMapfunc': s_GeMapfunc.unsqueeze(0),    # (1, 6373)
        's_GeMaplld': s_GeMaplld.reshape(1, -1),    # (1, 65*2997=194805)
        'is_face': is_face,    # (T,)
    }
    """
    def __init__(self, h5_file_path):
        """
        Args:
            h5_file_path (string): Path to the h5 file with preprocessed data.
            max_iter (int): the maximum iteration number, used for deciding number of mask
            p_random_n_mask (float): the probability of use random number of mask
        """
        self.h5_path = h5_file_path
        self.dataset = h5py.File(self.h5_path, 'r', swmr=True)
        self.sample_names = self.dataset['s_name'][()]

        self.length = len(self.sample_names)
        logger.warning(f"dataset size: {self.length}")

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

    def __len__(self):
        return self.length

    def _norm(self, tensor):
        return (tensor - torch.mean(tensor)) / torch.std(tensor)

    def __getitem__(self, idx):
        # if self.dataset is None:
        #     self.dataset = h5py.File(self.h5_path, 'r', swmr=True)
        #     self.l_tokens = self.dataset['l_tokens'][()]
        #     self.sample_names = self.dataset['s_name'][()]
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
            # endregion to Tensor

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
        }


if __name__ == "__main__":
    # test dataset
    set_logger_format(logging.INFO)

    batch_size = 4
    dataloader = data.DataLoader(
        TestDataset('/data03/wjh/val_sequential.h5'),
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        drop_last=False,
    )

    for batch in tqdm(dataloader):
        print('\n', batch['sample_name'])
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
        input("next:")

