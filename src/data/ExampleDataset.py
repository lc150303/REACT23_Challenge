#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Author: Cong Liang
Last modified: 2023/6/5
"""
import os

import torch
import numpy as np
import torch.utils.data as data
from torchvision import transforms, utils

class ExampleDataset(data.Dataset):
    def __init__(self, max_n_frame=750):
        super().__init__()
        self.n_frame = max_n_frame

    def __len__(self):
        return 6000

    def set_iter(self, iter: int):
        pass

    def __getitem__(self, idx):
        """
        return data with the right shape and random value
        """
        s_exp = torch.rand(self.n_frame, 512)
        s_AU = torch.rand(self.n_frame, 25088)
        s_VA = torch.rand(self.n_frame, 1408)
        s_pose = torch.rand(self.n_frame//5, 1408)
        s_MFCC = torch.rand(self.n_frame, 26)
        s_GeMAPfunc = torch.rand(6373)
        s_GeMAPlld = torch.rand(2997, 65)
        is_face = torch.randint(0, 2, (self.n_frame,)).bool()

        l_AU = torch.randint(0, 5976, (self.n_frame, ))
        l_exp = torch.randint(0, 8000, (self.n_frame, ))
        l_VA = torch.randint(0, 2710, (self.n_frame, ))
        l_3DMM = torch.randint(0, 2000, (self.n_frame, ))
        l_mask = torch.randint(0, 2, (self.n_frame, 3)).bool()

        l_fake_AU = torch.randint(0, 5976, (self.n_frame, ))
        l_fake_exp = torch.randint(0, 8000, (self.n_frame, ))
        l_fake_VA = torch.randint(0, 2710, (self.n_frame, ))

        return {
            "s_exp": s_exp,
            "s_AU": s_AU,
            "s_VA": s_VA,
            "s_pose": s_pose,
            "s_MFCC": s_MFCC,
            "s_GeMAPfunc": s_GeMAPfunc.unsqueeze(0),
            's_GeMAPlld': s_GeMAPlld.reshape(1, -1),
            "is_face": is_face,

            "l_AU": l_AU,
            "l_exp": l_exp,
            "l_VA": l_VA,
            "l_3DMM": l_3DMM,
            "l_mask": l_mask,

            'l_fake_AU': l_fake_AU,     # (T, )
            'l_fake_exp': l_fake_exp,   # (T, )
            'l_fake_VA': l_fake_VA,     # (T, )
        }

if __name__ == "__main__":
    pass
