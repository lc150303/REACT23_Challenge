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


class ValDataset(data.Dataset):
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
        's_pose': s_pose,   # (T, 6)
        's_MFCC': s_MFCC,   # (T, 26)
        's_GeMAPfunc': s_GeMapfunc.unsqueeze(0),    # (1, 6373)
        's_GeMAPlld': s_GeMaplld.reshape(1, -1),    # (1, 65*2997=194805)
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
    def __init__(self, h5_file_path, mapping_csv, appro_npy, **kwargs):
        """
        Args:
            h5_file_path (string): Path to the h5 file with preprocessed data.
            max_iter (int): the maximum iteration number, used for deciding number of mask
            p_random_n_mask (float): the probability of use random number of mask
        """
        self.h5_path = h5_file_path
        self.dataset = h5py.File(self.h5_path, 'r', swmr=True)

        self.idx_mapping = dict()
        self.oppo_mapping = dict()
        with open(mapping_csv, 'r') as fmap:
            fmap.readline()     # header
            for idx, line in enumerate(fmap.readlines()):
                h5_idx, _, _, opponent_h5_idx = line.strip().split(',')
                self.idx_mapping[idx] = h5_idx
                self.oppo_mapping[idx] = opponent_h5_idx

        appropriate_matrix = np.load(appro_npy)
        self.neg_samples = dict()
        count_neg = 0
        for k, v in np.argwhere(appropriate_matrix == 0):   # get idx where value is 0
            count_neg += 1
            if k in self.neg_samples:
                self.neg_samples[k].append(v)
            else:
                self.neg_samples[k] = [v]
        logger.debug(f"count_neg: {count_neg}")

        self.length = len(self.idx_mapping)
        logger.warning(f"dataset size: {self.length}")

        # region prepare data buffer for better speed
        self._s_exp = np.empty((750, 512), dtype=np.float32)
        self._s_AU = np.empty((750, 25088), dtype=np.float32)
        self._s_VA = np.empty((750, 1408), dtype=np.float32)
        self._s_pose = np.empty((150, 1408), dtype=np.float32)
        self._s_MFCC = np.empty((750, 26), dtype=np.float32)
        self._s_GeMAPfunc = np.empty((6373,), dtype=np.float32)
        self._s_GeMAPlld = np.empty((2997, 65), dtype=np.float32)
        self._is_face = np.empty((750,), dtype=np.int32)

        self._s_gt_AU = np.empty((750, 15), dtype=np.int32)
        self._s_gt_exp = np.empty((750, 8), dtype=np.float32)
        self._s_gt_VA = np.empty((750, 2), dtype=np.float32)

        self._l_gt_AU = np.empty((750, 15), dtype=np.int32)
        self._l_gt_exp = np.empty((750, 8), dtype=np.float32)
        self._l_gt_VA = np.empty((750, 2), dtype=np.float32)
        # endregion prepare data buffer for better speed

    def __len__(self):
        return self.length

    def _norm(self, tensor):
        return (tensor - torch.mean(tensor)) / torch.std(tensor)

    def __getitem__(self, idx):
        try:
            sample = self.dataset[self.idx_mapping[idx]]
            logger.debug(f"{idx}th sample, sample.keys: {sample.keys()}")
            sample_name = sample['sample_name'][()].decode('utf-8')
            sample['s_exp'].read_direct(self._s_exp)
            sample['s_AU'].read_direct(self._s_AU)
            sample['s_VA'].read_direct(self._s_VA)
            sample['s_pose'].read_direct(self._s_pose)
            sample['s_MFCC'].read_direct(self._s_MFCC)
            sample['s_GeMapfunc'].read_direct(self._s_GeMAPfunc)
            sample['s_GeMaplld'].read_direct(self._s_GeMAPlld, np.s_[:2997])
            sample['is_face'].read_direct(self._is_face)

            sample['l_AU'].read_direct(self._s_gt_AU)
            sample['l_exp'].read_direct(self._s_gt_exp)
            sample['l_VA'].read_direct(self._s_gt_VA)

            opponent = self.dataset[self.oppo_mapping[idx]]
            opponent['l_AU'].read_direct(self._l_gt_AU)
            opponent['l_exp'].read_direct(self._l_gt_exp)
            opponent['l_VA'].read_direct(self._l_gt_VA)
            # invalid_idx = sample['invalid_idx'][()]

            # region to Tensor
            s_exp = torch.from_numpy(self._s_exp)
            s_AU = torch.from_numpy(self._s_AU)
            s_VA = torch.from_numpy(self._s_VA)
            s_pose = torch.from_numpy(self._s_pose)
            s_MFCC = torch.from_numpy(self._s_MFCC)
            s_GeMapfunc = torch.from_numpy(self._s_GeMAPfunc).view(-1)
            s_GeMapfunc = torch.where(torch.isnan(s_GeMapfunc), torch.zeros_like(s_GeMapfunc), s_GeMapfunc)
            s_GeMaplld = torch.from_numpy(self._s_GeMAPlld).view(-1)
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

            's_gt_AU': self._s_gt_AU.copy(),     # (T, 15)
            's_gt_exp': self._s_gt_exp.copy(),   # (T, 8)
            's_gt_VA': self._s_gt_VA.copy(),     # (T, 2)

            'l_gt_AU': self._l_gt_AU.copy(),     # (T, 15)
            'l_gt_exp': self._l_gt_exp.copy(),   # (T, 8)
            'l_gt_VA': self._l_gt_VA.copy(),     # (T, 2)
        }


if __name__ == "__main__":
    # test dataset
    set_logger_format(logging.INFO)

    batch_size = 1
    dataloader = data.DataLoader(
        ValDataset('/data03/wjh/train_new.h5',
                   '/home/wjh/React2023/data_split/val_df_new.csv',
                   '/data03/wjh/Approprirate_facial_reaction_val.npy'),
        batch_size=batch_size,
        num_workers=1,
        shuffle=False,
        drop_last=False,
    )

    s_gt_list, l_gt_list = [], []
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
        is_face = batch['is_face']
        assert is_face.shape == (batch_size, 750), f"is_face shape: {is_face.shape}"

        s_gt_AU = batch['s_gt_AU']      # (B, T, 15)
        s_gt_exp = batch['s_gt_exp']
        s_gt_VA = batch['s_gt_VA']
        s_gt_list.append(np.concatenate([s_gt_AU, s_gt_VA, s_gt_exp], axis=-1))

        l_gt_AU = batch['l_gt_AU']      # (B, T, 15)
        l_gt_exp = batch['l_gt_exp']
        l_gt_VA = batch['l_gt_VA']
        l_gt_list.append(np.concatenate([l_gt_AU, l_gt_VA, l_gt_exp], axis=-1))

    np.save('/data03/wjh/s_gt.npy', np.concatenate(s_gt_list, axis=0))
    np.save('/data03/wjh/l_gt.npy', np.concatenate(l_gt_list, axis=0))