"""
Unified transformer for multi-modal facial reaction generation.

Credit to: https://github.com/xrenaa/Look-Outside-Room
Heavily modified for REACT23 challenge.
"""
import os.path

#
import torch
import time
import logging
from math import ceil

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim.lr_scheduler import LambdaLR
from src.lr_scheduler import LambdaWarmUpCosineScheduler

from src.models.base_transformer import BaseTransformer
from src.util import set_logger_format, top_k_top_p_filter, mask_lower_n

logger = logging.getLogger(__name__)


class ProposedModel(BaseTransformer):
    """
    :param temperature: float, bigger temperature means more random.
    :param top_k: int, if > 0, only sample from the top k most likely tokens.
    :param top_p: float, if < 0.0, only sample from the most likely tokens whose cumulative probability mass exceeds p.
    :param step: used only in online task. following MaskGIT https://arxiv.org/pdf/2202.04200.pdf

    Proposed model for REACT23 challenge.

    Here we deal with save/load model and decoding.
    """
    def __init__(self,
                 transformer_config,
                 visual_config,
                 use_scheduler=True,
                 scheduler_config=None,
                 lr=0.0625,
                 step=10,
                 top_k=0,
                 top_p=1.0,
                 temperature=1.0,
                 temp_decay='',
                 online_step=1,
                 online_max_len=750,
                 device="cuda"
                 ):

        super().__init__(**transformer_config.params)

        self.learning_rate = lr
        self.use_scheduler = use_scheduler
        if use_scheduler:
            assert scheduler_config is not None
            self.scheduler_config = scheduler_config

        # region decoding params
        self.step = step
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.temp_decay = temp_decay
        self.online_step = online_step
        self.online_max_len = online_max_len
        # endregion decoding params

        self.is_train = False
        self.task = "pretrain"
        self.device = device
        self.to(device)     # move to GPU


    def configure_optimizers(self, last_iter=-1):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.Parameter)
        logger.debug('configering weight decay')
        for mn, m in self.named_modules():
            # logger.debug("---------------  " + str(mn))
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                # logger.debug(fpn)
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif isinstance(m, blacklist_weight_modules) or pn.endswith('me'):
                    # weights of blacklist modules and modality embeddings will NOT be decayed
                    no_decay.add(fpn)
                elif isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed

        no_decay.add('s_no_face')
        no_decay.add('l_mask')
        no_decay.add('time_emb')
        no_decay.add('token_slm')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "initial_lr": self.learning_rate, "weight_decay": 0.01},
            # {"params": [param_dict[pn] for pn in sorted(list(param_dict.keys() - union_params))], "weight_decay": 0.0},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "initial_lr": self.learning_rate, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))

        if self.use_scheduler:
            logger.info("Setting up LambdaWarmUpCosineScheduler scheduler...")
            scheduler = LambdaWarmUpCosineScheduler(**self.scheduler_config.params)
            scheduler = LambdaLR(optimizer, lr_lambda=scheduler.schedule, last_epoch=last_iter)

            return optimizer, scheduler

        return optimizer

    def train(self, mode: bool = True):
        self.is_train = mode
        return super().train(mode)

    def set_task(self, task:str):
        """
        :param task: "pretrain" | "train-3DMM" | "finetune-online" | "finetune-offline" | "online" | "offline"
        """
        self.task = task
        logger.warning(f"Set task to {task}.")

    def forward(self, batch, mrm_sample_rate=0.2, train_slm=True):
        """
        when training, use yield to save GPU memory

        Input:
            dict(
                'sample_name': sample_name,     # string
                's_exp': s_exp,     # (T, 512)
                's_AU': s_AU,       # (T, 25088)
                's_VA': s_VA,       # (T, 1408)
                's_pose': s_pose,   # (T, 1408)
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
            )
        """
        if self.is_train:
            if self.task == "pretrain":
                # return super().forward(batch, mrm_sample_rate, train_slm=train_slm)
                if train_slm:
                    logger.debug('forward slm')
                    yield 'slm_pos', super().forward_slm(batch, True)
                    neg_batch = {
                        "s_exp": batch["s_exp"],
                        "s_AU": batch["s_AU"],
                        "s_VA": batch["s_VA"],
                        "s_pose": batch["s_pose"],
                        "s_MFCC": batch["s_MFCC"],
                        "is_face": batch["is_face"],

                        "l_AU": batch["l_fake_AU"],
                        "l_exp": batch["l_fake_exp"],
                        "l_VA": batch["l_fake_VA"],
                    }
                    yield 'slm_neg', super().forward_slm(neg_batch, False)
                logger.debug('forward mrm')
                _, mrm_loss = super().forward_mrm(batch, mrm_sample_rate)
                yield 'mrm', mrm_loss
                logger.debug('forward crm')
                _, crm_loss = super().forward_crm(batch)
                yield 'crm', crm_loss
            elif self.task == "finetune":
                if train_slm:
                    logger.debug('forward slm')
                    yield 'slm_pos', super().forward_slm(batch, True)
                    neg_batch = {
                        "s_exp": batch["s_exp"],
                        "s_AU": batch["s_AU"],
                        "s_VA": batch["s_VA"],
                        "s_pose": batch["s_pose"],
                        "s_MFCC": batch["s_MFCC"],
                        "is_face": batch["is_face"],

                        "l_AU": batch["l_fake_AU"],
                        "l_exp": batch["l_fake_exp"],
                        "l_VA": batch["l_fake_VA"],
                    }
                    yield 'slm_neg', super().forward_slm(neg_batch, False)
                logger.debug('forward mrm')
                _, mrm_loss = super().forward_mrm(batch, mrm_sample_rate)
                yield 'mrm', mrm_loss
        else:
            yield self.predict(batch)

    @torch.no_grad()
    def predict(self, batch, **kwargs):
        """

        Input:
            dict(
                'sample_name': (B, string)
                's_exp': (B, T, 512)
                's_AU': (B, T, 25088)
                's_VA': (B, T, 1408)
                's_pose': (B, T, 1408)
                's_MFCC': (B, T, 26)
                's_GeMAPfunc': (B, 1, 6373)
                's_GeMAPlld': (B, 1, 65*2997=194805)
                'is_face': (B, T)
            )
        Output:
            l_AU_pred, l_exp_pred, l_VA_pred     # (B, T)

        online prediction see https://arxiv.org/pdf/2202.04200.pdf and https://github.com/google-research/maskgit


        self.temperature float, bigger temperature means more random.
        self.top_k: int, if > 0, only sample from the top k most likely tokens.
        self.top_p: float, if < 0.0, only sample from the most likely tokens whose cumulative probability mass exceeds p.
        self.step: used only in online task. following MaskGIT https://arxiv.org/pdf/2202.04200.pdf
        """
        assert not self.is_train, "Should not run predict() in train mode."
        temperature, top_k, top_p = self.temperature, self.top_k, self.top_p

        B, T = batch['s_exp'].shape[:2]
        l_AU_pred = torch.zeros(B, T, dtype=torch.long, device=batch['s_exp'].device)
        l_exp_pred = torch.zeros(B, T, dtype=torch.long, device=batch['s_exp'].device)
        l_VA_pred = torch.zeros(B, T, dtype=torch.long, device=batch['s_exp'].device)
        accumulated_prob = np.zeros(B)      # indicates the probability of each predicted sequence

        # offline task
        if self.task == "offline":
            """ predict by cosine decay schedule """
            logger.debug("run predicting offline")
            l_mask = torch.zeros(B, T, 3, device=l_AU_pred.device, requires_grad=False).bool()
            for k in range(1, self.step+1):
                batch_k = {
                    "s_exp": batch["s_exp"],
                    "s_AU": batch["s_AU"],
                    "s_VA": batch["s_VA"],
                    "s_pose": batch["s_pose"],
                    "s_MFCC": batch["s_MFCC"],
                    "s_GeMAPfunc": batch["s_GeMAPfunc"],
                    "s_GeMAPlld": batch["s_GeMAPlld"],
                    "is_face": batch["is_face"],

                    "l_AU": l_AU_pred,
                    "l_exp": l_exp_pred,
                    "l_VA": l_VA_pred,
                    "l_mask": l_mask    # (B, T, 3)
                }
                AU_logits, exp_logits, VA_logits = super().predict(batch_k, is_causal=False)

                """
                1. apply temperature, top-k, top-p to control sampling
                2. sample 1 token for each position by Categorical, resulting in (B, T) 
                3. record the probability of the sampled tokens, resulting also in (B, T)
                4. mark the probs of input token (sampled in prev loops) as infinity, to avoid affecting mask_lower_n
                """
                # region get token and probs
                AU_logits = top_k_top_p_filter(AU_logits / temperature, top_k, top_p)
                AU_probs = F.softmax(AU_logits, dim=-1)  # (B, T, n_l_AU)
                AU_tokens = Categorical(AU_probs).sample()  # (B, T)
                selected_AU_prob = torch.squeeze(
                    torch.take_along_dim(AU_probs, torch.unsqueeze(AU_tokens, -1), -1),
                    dim=-1
                )  # (B, T)
                selected_AU_prob = selected_AU_prob.masked_fill(l_mask[..., 0], torch.inf)

                exp_logits = top_k_top_p_filter(exp_logits / temperature, top_k, top_p)
                exp_probs = F.softmax(exp_logits, dim=-1)  # (B, T, n_l_exp)
                exp_tokens = Categorical(exp_probs).sample()  # (B, T)
                selected_exp_prob = torch.squeeze(
                    torch.take_along_dim(exp_probs, torch.unsqueeze(exp_tokens, -1), -1),
                    dim=-1
                )  # (B, T)
                selected_exp_prob = selected_exp_prob.masked_fill(l_mask[..., 1], torch.inf)

                VA_logits = top_k_top_p_filter(VA_logits / temperature, top_k, top_p)
                VA_probs = F.softmax(VA_logits, dim=-1)  # (B, T, n_l_VA)
                VA_tokens = Categorical(VA_probs).sample()  # (B, T)
                selected_VA_prob = torch.squeeze(
                    torch.take_along_dim(VA_probs, torch.unsqueeze(VA_tokens, -1), -1),
                    dim=-1
                )  # (B, T)
                selected_VA_prob = selected_VA_prob.masked_fill(l_mask[..., 2], torch.inf)
                # endregion get token and probs

                n_mask = round(3 * T * np.cos(k/self.step * np.pi / 2))     # cosine decay schedule

                if n_mask:
                    # mask n tokens with lowest probs, considering AU, exp, and VA tokens together
                    total_probs = torch.cat((
                        selected_AU_prob, selected_exp_prob, selected_VA_prob
                    ), dim=-1)  # (B, 3*T)
                    new_mask = mask_lower_n(total_probs, n_mask)    # (B, T, 3), 0 means masked

                    # tokens newly added to l_*_pred are where l_mask == 0 but new_mask == 1
                    add_mask = new_mask & ~ l_mask.bool()   # (B, T, 3)
                else:
                    # all masked tokens should be predicted. the last loop.
                    add_mask = ~ l_mask.bool()

                l_AU_pred = torch.where(add_mask[..., 0], AU_tokens, l_AU_pred)
                l_exp_pred = torch.where(add_mask[..., 1], exp_tokens, l_exp_pred)
                l_VA_pred = torch.where(add_mask[..., 2], VA_tokens, l_VA_pred)

                l_mask = l_mask | add_mask  # mask for next loop

                # region accumulate prob
                # accumulate prob of the tokens newly added to l_*_pred
                add_mask = add_mask.float()  # (B, T, 3)
                add_mask[..., 0] = add_mask[..., 0] * selected_AU_prob
                add_mask[..., 1] = add_mask[..., 1] * selected_exp_prob
                add_mask[..., 2] = add_mask[..., 2] * selected_VA_prob
                add_mask = add_mask.masked_fill(add_mask == 0, 1)       # avoid log(0). log(1) = 0.
                add_prob = np.log(add_mask.cpu().numpy()).sum(axis=(-1, -2))    # (B, T, 3) -> (B,)
                accumulated_prob += add_prob
                # endregion accumulate prob

                # the temperature decays as the unfilled masks decrease
                if self.temp_decay == "linear":
                    temperature = self.temperature * (1 - k/self.step)
        # online task
        else:
            """ predict frame-by-frame """
            online_step, max_len = self.online_step, self.online_max_len
            logger.debug("run predicting online")
            for k in range(online_step, T+online_step, online_step):
                logger.debug(f"predicting {k}/{T}")
                l_mask = torch.ones(B, k, 3, device=l_AU_pred.device)
                l_mask[:, -online_step:] = 0   # (B, k, 3), may longer than T
                start_step = max(0, min(k, T)-max_len)
                logger.debug(f"start_step: {start_step}, k: {k}")
                batch_k = {
                    "s_exp": batch["s_exp"][:,start_step:k],     # at most length T
                    "s_AU": batch["s_AU"][:, start_step:k],
                    "s_VA": batch["s_VA"][:, start_step:k],
                    "s_pose": batch["s_pose"][:, start_step:k],
                    "s_MFCC": batch["s_MFCC"][:, start_step:k],
                    "is_face": batch["is_face"][:, start_step:k],

                    "l_AU": l_AU_pred[:, start_step:k],
                    "l_exp": l_exp_pred[:, start_step:k],
                    "l_VA": l_VA_pred[:, start_step:k],
                    "l_mask": l_mask[:, start_step:min(k, T)]     # cut-off if longer than T
                }
                AU_logits, exp_logits, VA_logits = super().predict(batch_k, is_causal=True)     # (B, max_len, n_vocab)

                # modify probability by temperature, top-k, top-p
                AU_logits = top_k_top_p_filter(AU_logits[:, k-start_step-online_step:]/temperature, top_k, top_p)  # (B, online_step, n_l_AU)
                exp_logits = top_k_top_p_filter(exp_logits[:, k-start_step-online_step:]/temperature, top_k, top_p)    # (B, online_step, n_l_exp)
                VA_logits = top_k_top_p_filter(VA_logits[:, k-start_step-online_step:]/temperature, top_k, top_p)      # (B, online_step, n_l_VA)
                logger.debug(f"AU_logits: {AU_logits.shape}, exp_logits: {exp_logits.shape}, VA_logits: {VA_logits.shape}")

                # region sample from the distribution
                AU_probs = F.softmax(AU_logits, dim=-1)         # (B, online_step, n_l_AU)
                AU_tokens = Categorical(AU_probs).sample()      # (B, online_step)
                l_AU_pred[:, k-online_step:k] = AU_tokens

                exp_probs = F.softmax(exp_logits, dim=-1)
                exp_tokens = Categorical(exp_probs).sample()    # (B, online_step)
                l_exp_pred[:, k-online_step:k] = exp_tokens

                VA_probs = F.softmax(VA_logits, dim=-1)
                VA_tokens = Categorical(VA_probs).sample()      # (B, online_step)
                l_VA_pred[:, k-online_step:k] = VA_tokens
                # endregion sample from the distribution

                # region accumulate prob
                # accumulate prob of the tokens newly added to l_*_pred
                add_prob = torch.cat([
                    torch.take_along_dim(AU_probs, AU_tokens.unsqueeze(-1), dim=-1),    # (B, online_step)
                    torch.take_along_dim(exp_probs, exp_tokens.unsqueeze(-1), dim=-1),
                    torch.take_along_dim(VA_probs, VA_tokens.unsqueeze(-1), dim=-1)
                ], dim=-1).log().cpu().numpy().sum(axis=(-1, -2))    # (B, online_step, 3) -> (B,)
                accumulated_prob += add_prob
                # endregion accumulate prob

        if "return_prob" in kwargs and kwargs["return_prob"]:
            return l_AU_pred, l_exp_pred, l_VA_pred, accumulated_prob
        return l_AU_pred, l_exp_pred, l_VA_pred


if __name__ == '__main__':
    """ test the initialization and forwarding """
    # region 1. load config and init model
    from omegaconf import OmegaConf
    config = OmegaConf.load('/home/liangcong/dataset/REACT23_code/configs/pretrain.yaml')

    set_logger_format(logging.DEBUG)
    device = 'cuda:1'

    model = ProposedModel(config.model.transformer_config,
                          None,
                          lr=config.model.base_learning_rate,
                          use_scheduler=config.model.use_scheduler,
                          scheduler_config=config.model.scheduler_config,
                          device=device,
                          **config.model.decoding_params)

    opt, sch = model.configure_optimizers()
    # endregion 1. load config and init model

    # region 2. load data
    from src.data.ExampleDataset import ExampleDataset
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        ExampleDataset(750),
        batch_size=1,
        shuffle=True,
        num_workers=4)
    # endregion 2. load data

    # region 3. train
    from torch.cuda.amp import autocast, GradScaler

    scaler = GradScaler()

    model.train()
    for batch in dataloader:
        for k, v in batch.items():
            batch[k] = v.to(device)

        opt.zero_grad()
        with autocast():
            # _, _, loss = model(batch)
            for _, loss in model(batch, train_slm=config.train_params.train_slm):
                print(loss)
                input('continue backward:')
                scaler.scale(loss).backward()  # sum(loss.values()).backward()
        scaler.step(opt)  # opt.step()
        scaler.update()

        input('continue next iter:')
    # endregion 3. train
