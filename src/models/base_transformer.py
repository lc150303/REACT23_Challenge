"""
credit to: https://github.com/karpathy/minGPT/
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a self-attention block and an 1-hidden-layer MLP block
    - the MLP blocks are shared for online and offline tasks
    - the online (bi-directional) and offline (causal) tasks use different self-attention blocks
    - all blocks feed into a central residual pathway similar to resnets
- the final linear layer of GPT is split for different modal prediction
"""

import math
import logging
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.util import set_logger_format

logger = logging.getLogger(__name__)

def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, block_size, dim_embed, **kwargs):
        self.block_size = block_size
        self.dim_embed = dim_embed
        for k, v in kwargs.items():
            setattr(self, k, v)


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.

    Input:
        x: shape(B, T, dim)
        dynamic_mask: shape(B, T, dim), the masked positions are 1, otherwise 0

    Output:
        shape(B, T, dim)
    """

    def __init__(self, config, causal_mask=False):
        super().__init__()
        assert config.dim_embed % config.n_head == 0, f"dim_embed is {config.dim_embed} but n_head is {config.n_head}."
        # key, query, value projections for all heads
        self.key = nn.Linear(config.dim_embed, config.dim_embed)
        self.query = nn.Linear(config.dim_embed, config.dim_embed)
        self.value = nn.Linear(config.dim_embed, config.dim_embed)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.c_proj = nn.Linear(config.dim_embed, config.dim_embed)

        if causal_mask:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            mask = torch.tril(torch.ones(config.block_size,
                                         config.block_size))

            n_tok_per_frame = config.n_s_token + config.n_l_token
            chunk_size = config.online_step * n_tok_per_frame
            step_mask = torch.zeros(config.block_size, config.block_size)
            # attend bi-directionally within the speaker tokens at each time step
            for i in range(0, config.block_size, chunk_size):
                step_mask[i:i+chunk_size, i:i+chunk_size] = 1
                # keep one-dimensional causal mask for listener tokens
                for j in range(i, i+chunk_size, n_tok_per_frame):
                    mask[j + config.n_s_token : j + n_tok_per_frame] = 0
                    mask[:, j + config.n_s_token : j + n_tok_per_frame] = 0
            mask = mask + step_mask
            self.register_buffer("causal_mask", (mask == 0).view(1, 1, config.block_size, config.block_size))
        else:
            self.causal_mask = None
        self.n_head = config.n_head

    def forward(self, x, dynamic_mask=None):
        """
        Input:
            x: shape(B, T, dim)
            dynamic_mask: shape(B, T, dim), the masked positions are 1, otherwise 0

        Output:
            shape(B, T, dim)
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (dim_embed)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, k_dim)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, q_dim)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, v_dim)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.causal_mask is not None:
            att = att.masked_fill(self.causal_mask[:, :, :T, :T], float('-inf'))
        if dynamic_mask is not None:
            att = att.masked_fill(dynamic_mask[:, :, :T, :T], float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.c_proj(y))
        return y


class CausalSelfAttention(SelfAttention):
    def __init__(self, config):
        super(CausalSelfAttention, self).__init__(config, causal_mask=True)


class BiDirectionalSelfAttention(SelfAttention):
    def __init__(self, config):
        super(BiDirectionalSelfAttention, self).__init__(config, causal_mask=False)


class Block(nn.Module):
    """ a shared-fast-forward split-attention Transformer block """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.dim_embed)  # LayerNorm No.1 before self-attention
        self.ln2 = nn.LayerNorm(config.dim_embed)  # LayerNorm No.2 before fast-forward

        # split attention layers
        self.bi_attn = BiDirectionalSelfAttention(config)
        self.cas_attn = CausalSelfAttention(config)

        # shared fast-forward layers
        self.mlp = nn.Sequential(OrderedDict([
            ('c_fc', nn.Linear(config.dim_embed, 4 * config.dim_embed)),
            ('gelu', nn.GELU()),
            ('c_proj', nn.Linear(4 * config.dim_embed, config.dim_embed)),
            ('dropout', nn.Dropout(config.resid_pdrop)),
        ]))

    def forward(self, x, causal):
        if causal:
            # GPT-like attention for the online task
            x = x + self.cas_attn(self.ln1(x))
        else:
            # BERT-like for the offline task
            x = x + self.bi_attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class BaseTransformer(nn.Module):
    """
    adapt from AdaptiveGPT: https://github.com/xrenaa/Look-Outside-Room
    here we manage all learnable parameters and training pass

    Input:
        dict(
            "s_exp": (batch_size, n_frame, dim_s_exp),
            "s_AU": (batch_size, n_frame, dim_s_AU),
            "s_VA": (batch_size, n_frame, dim_s_VA),
            "s_pose": (batch_size, n_frame, dim_s_pose),
            "s_MFCC": (batch_size, n_frame, dim_s_MFCC),
            "s_GeMAPfunc": (batch_size, 1, dim_s_GeMAPfunc),
            "s_GeMAPlld": (batch_size, 1, dim_s_GeMAPlld),
            "is_face": (batch_size, n_frame),

            "l_AU": (batch_size, n_frame),
            "l_exp": (batch_size, n_frame),
            "l_VA": (batch_size, n_frame),
            "l_mask": (batch_size, n_frame, 3)      # 0 means masked, 1 means unmasked
        )

    Output:
        (mrm_AU_logits, mrm_exp_logits, mrm_VA_logits),
        (crm_AU_logits, crm_exp_logits, crm_VA_logits),
        loss_dict

    """

    def __init__(self, max_n_frame, online_step, n_s_token=3, n_l_token=3,  # sequence length
                 dim_s_exp=512, dim_s_AU=25088, dim_s_VA=1408,  # speaker features
                 dim_s_pose=128, dim_s_MFCC=12, dim_s_GeMAPfunc=32, dim_s_GeMAPlld=32,
                 n_l_AU=5976, n_l_exp=8, n_l_VA=2710,           # listener features
                 n_layer=12, n_head=16, dim_embed=1024,         # model structure
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0.):  # drop out
        super().__init__()
        block_size = max_n_frame * (n_s_token + n_l_token) + 1  # 750 * 6 + 1
        config = GPTConfig(block_size=block_size, n_s_token=n_s_token, n_l_token=n_l_token,
                           online_step=online_step, n_l_AU=n_l_AU, n_l_exp=n_l_exp, n_l_VA=n_l_VA,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, dim_embed=dim_embed)
        # region input embedding stem
        self.s_exp_proj = nn.Linear(dim_s_exp, config.dim_embed//3)
        self.s_AU_proj = nn.Linear(dim_s_AU, config.dim_embed//3)
        self.s_VA_proj = nn.Linear(dim_s_VA, config.dim_embed - config.dim_embed//3*2)
        self.s_pose_proj = nn.Linear(dim_s_pose, config.dim_embed)
        self.s_MFCC_proj = nn.Linear(dim_s_MFCC, config.dim_embed)
        self.s_GeMAPfunc_proj = nn.Linear(dim_s_GeMAPfunc, config.dim_embed//2)
        self.s_GeMAPlld_proj = nn.Linear(dim_s_GeMAPlld, config.dim_embed//2)

        self.l_AU_emb = nn.Embedding(n_l_AU, config.dim_embed)
        self.l_exp_emb = nn.Embedding(n_l_exp, config.dim_embed)
        self.l_VA_emb = nn.Embedding(n_l_VA, config.dim_embed)
        # endregion input embedding stem

        # region modality and position embedding
        # Speaker-Listener-Matching token
        self.token_slm = nn.Parameter(torch.zeros(1, 1, config.dim_embed))
        # modality embedding to denote different modality
        self.s_face_me = nn.Parameter(torch.zeros(1, 1, config.dim_embed))
        self.s_pose_me = nn.Parameter(torch.zeros(1, 1, config.dim_embed))
        self.s_MFCC_me = nn.Parameter(torch.zeros(1, 1, config.dim_embed))
        self.s_GeMAPS_me = nn.Parameter(torch.zeros(1, 1, config.dim_embed))
        # to fill the face-less frame
        self.s_no_face = nn.Parameter(torch.zeros(1, 1, config.dim_embed))

        self.l_AU_me = nn.Parameter(torch.zeros(1, 1, config.dim_embed))
        self.l_exp_me = nn.Parameter(torch.zeros(1, 1, config.dim_embed))
        self.l_VA_me = nn.Parameter(torch.zeros(1, 1, config.dim_embed))
        # to denote masked reaction modeling
        self.l_mask = nn.Parameter(torch.zeros(1, 1, 1, config.dim_embed))

        # time (position) embedding to denote sequence order
        self.time_len = max_n_frame
        self.time_emb = nn.Parameter(data=get_sinusoid_encoding(n_position=block_size, d_hid=config.dim_embed),
                                     requires_grad=False)
        logger.debug(f"time_emb shape: {self.time_emb.shape}")  # (1, block_size, dim_embed)
        # endregion modality and position embedding

        # dropout
        self.drop_input = nn.Dropout(config.embd_pdrop)
        # transformer blocks
        self.blocks = nn.ModuleList()

        for _ in range(int(n_layer)):
            self.blocks.append(Block(config))

        # region output linear layers
        self.ln_f = nn.LayerNorm(config.dim_embed)  # LayerNorm final
        self.head_slm = nn.Linear(config.dim_embed, 2, bias=False)  # for Speaker-Listener-Matching
        self.head_AU = nn.Linear(config.dim_embed, n_l_AU, bias=False)
        self.head_VA = nn.Linear(config.dim_embed, n_l_VA, bias=False)
        self.head_exp = nn.Linear(config.dim_embed, n_l_exp, bias=False)
        # endregion output linear layers

        self.apply(self._init_weights)
        # follow https://github.com/karpathy/minGPT, who follows GPT-2
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))

        self.token_per_frame = (n_s_token + n_l_token)
        self.max_n_frame = max_n_frame
        self.block_size = config.block_size
        self.config = config
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward_embedding(self, batch):
        """
        Forward pass step 1 and 2:
        1. project input features to embedding space
        2. dropout, add modality embedding and reshape

        Input:
            dict(
                "s_exp": (batch_size, n_frame, dim_s_exp),
                "s_AU": (batch_size, n_frame, dim_s_AU),
                "s_VA": (batch_size, n_frame, dim_s_VA),
                "s_pose": (batch_size, n_frame, dim_s_pose),
                "s_MFCC": (batch_size, n_frame, dim_s_MFCC),
                "s_GeMAPS": (batch_size, 1, dim_s_GeMAPS),
                "is_face": (batch_size, n_frame),

                "l_AU": (batch_size, n_frame),
                "l_exp": (batch_size, n_frame),
                "l_VA": (batch_size, n_frame),
                "l_mask": (batch_size, n_frame, 3)
            )

        Output: (
            s_exp_emb_usq,
            s_AU_emb_usq,
            s_VA_emb_usq,
            s_pose_emb_usq,
            s_MFCC_emb_usq,

            l_AU_emb_usq,
            l_exp_emb_usq,
            l_VA_emb_usq
        )
        """

        # region 1. project inputs to token embeddings
        s_exp_emb = self.s_exp_proj(batch['s_exp'])
        s_AU_emb = self.s_AU_proj(batch['s_AU'])
        s_VA_emb = self.s_VA_proj(batch['s_VA'])
        s_face_emb = torch.cat((s_exp_emb, s_AU_emb, s_VA_emb), dim=-1)
        
        s_pose_emb = self.s_pose_proj(batch['s_pose'])
        s_MFCC_emb = self.s_MFCC_proj(batch['s_MFCC'])

        # fill face-less frames by a specific embedding.
        input_s_mask = batch['is_face'].unsqueeze(-1).bool()  # (B, T) -> (B, T, 1)
        s_no_face = self.s_no_face.to(s_face_emb.dtype)     # torch.cuda.amp failed to auto-cast type
        logger.debug(f"input_s_mask: {input_s_mask.dtype}, s_face_emb: {s_face_emb.dtype}, s_no_face: {s_no_face.dtype}")
        s_face_emb = torch.where(input_s_mask, s_face_emb, s_no_face)  # (B, T, dim)

        l_AU_emb = self.l_AU_emb(batch['l_AU'])
        l_exp_emb = self.l_exp_emb(batch['l_exp'])
        l_VA_emb = self.l_VA_emb(batch['l_VA'])
        # endregion 1. project inputs to token embeddings

        # region 2.1. dropout of inputs, add modality embeddings (xx_me)
        s_face_emb = self.drop_input(s_face_emb) + self.s_face_me
        s_pose_emb = self.drop_input(s_pose_emb) + self.s_pose_me
        s_MFCC_emb = self.drop_input(s_MFCC_emb) + self.s_MFCC_me

        l_AU_emb = self.drop_input(l_AU_emb) + self.l_AU_me
        l_exp_emb = self.drop_input(l_exp_emb) + self.l_exp_me
        l_VA_emb = self.drop_input(l_VA_emb) + self.l_VA_me
        # endregion 2.1. dropout of inputs, add modality embeddings (xx_me)

        # region 2.2. prepare shape for interleaving all the inputs into a sequence
        s_face_emb_usq = s_face_emb.unsqueeze(-2)  # (B, T, dim) -> (B, T, 1, dim)
        s_pose_emb_usq = s_pose_emb.unsqueeze(-2)
        s_MFCC_emb_usq = s_MFCC_emb.unsqueeze(-2)

        l_AU_emb_usq = l_AU_emb.unsqueeze(-2)
        l_exp_emb_usq = l_exp_emb.unsqueeze(-2)
        l_VA_emb_usq = l_VA_emb.unsqueeze(-2)
        # endregion 2.2. prepare shape for interleaving all the inputs into a sequence

        return (
            s_face_emb_usq, s_pose_emb_usq, s_MFCC_emb_usq,
            l_AU_emb_usq, l_exp_emb_usq, l_VA_emb_usq
        )

    def forward_slm(self, batch, positive=True):
        """
        3.1 Speaker-Listener Matching (SLM)

        Input:
            dict(
                "s_exp": (batch_size, n_frame, dim_s_exp),
                "s_AU": (batch_size, n_frame, dim_s_AU),
                "s_VA": (batch_size, n_frame, dim_s_VA),
                "s_pose": (batch_size, n_frame, dim_s_pose),
                "s_MFCC": (batch_size, n_frame, dim_s_MFCC),
                "s_GeMAPS": (batch_size, 1, dim_s_GeMAPS),
                "is_face": (batch_size, n_frame),

                "l_AU": (batch_size, n_frame),
                "l_exp": (batch_size, n_frame),
                "l_VA": (batch_size, n_frame),
                "l_mask": (batch_size, n_frame, 3)
            )

        Output:
            loss_slm
        """
        s_face_emb_usq, s_pose_emb_usq, s_MFCC_emb_usq, \
            l_AU_emb_usq, l_exp_emb_usq, l_VA_emb_usq = self.forward_embedding(batch)

        input_slm = torch.cat((
            s_face_emb_usq, s_pose_emb_usq, s_MFCC_emb_usq,
            l_AU_emb_usq, l_exp_emb_usq, l_VA_emb_usq
        ), dim=-2).reshape(s_face_emb_usq.shape[0], -1,
                           s_face_emb_usq.shape[-1])  # (B, n_frame*token_per_frame, emb_dim)
        # append self.token_slm to the sequence
        logger.debug(
            f"input_slm.shape: {input_slm.shape}, self.token_slm.shape: {self.token_slm.shape}")

        x = torch.cat((
            input_slm,
            self.token_slm.repeat(input_slm.shape[0], 1, 1)
        ), dim=-2)  # (B, n_frame*token_per_frame + 1, emb_dim)
        logger.debug(f'x.shape: {x.shape}')
        x = x + self.time_emb[:, :x.shape[1]]
        logger.debug(f"x_slm.shape: {x.shape} ")

        for block in self.blocks:
            x = block(x, causal=False)
        x = self.ln_f(x)  # final LayerNorm

        logits = self.head_slm(x[:, -1])
        logger.debug(f"slm_logits.shape: {logits.shape}")

        # positive pairs
        if positive:
            # interleave all the inputs into a sequence
            slm_labels = torch.ones(s_face_emb_usq.shape[0], dtype=torch.long, requires_grad=False).to(input_slm.device)

        # negative pairs.
        else:
            slm_labels = torch.zeros(s_face_emb_usq.shape[0], dtype=torch.long, requires_grad=False).to(input_slm.device)

        # SLM loss
        return F.cross_entropy(logits, slm_labels)

    def forward_mrm(self, batch, mrm_sample_rate=0.2):
        """
        3.2 Masked Response Modeling (MRM)

        Input:
            dict(
                "s_exp": (batch_size, n_frame, dim_s_exp),
                "s_AU": (batch_size, n_frame, dim_s_AU),
                "s_VA": (batch_size, n_frame, dim_s_VA),
                "s_pose": (batch_size, n_frame, dim_s_pose),
                "s_MFCC": (batch_size, n_frame, dim_s_MFCC),
                "s_GeMAPfunc": (batch_size, 1, dim_s_GeMAPfunc),
                "s_GeMAPlld": (batch_size, 1, dim_s_GeMAPlld),
                "is_face": (batch_size, n_frame),

                "l_AU": (batch_size, n_frame),
                "l_exp": (batch_size, n_frame),
                "l_VA": (batch_size, n_frame),
                "l_mask": (batch_size, n_frame, 3)      # 0 means masked, 1 means unmasked
            )

        Output:
            (mrm_AU_logits, mrm_exp_logits, mrm_VA_logits),
            loss_mrm
        """
        s_face_emb_usq, s_pose_emb_usq, s_MFCC_emb_usq, \
            l_AU_emb_usq, l_exp_emb_usq, l_VA_emb_usq = self.forward_embedding(batch)
        # logger.debug(f"GeMAPS input nan: {torch.isnan(batch['s_GeMAPfunc']).any()}, {torch.isnan(batch['s_GeMAPlld']).any()}")
        # logger.debug(f"GeMAPSfunc max: {torch.max(batch['s_GeMAPfunc'])}, min: {torch.min(batch['s_GeMAPfunc'])}")
        # logger.debug(f"GeMAPlld max: {torch.max(batch['s_GeMAPlld'])}, min: {torch.min(batch['s_GeMAPlld'])}")
        s_GeMAPfunc_emb = self.s_GeMAPfunc_proj(batch['s_GeMAPfunc'])   # (B, 1, emb_dim//2)
        s_GeMAPlld_emb = self.s_GeMAPlld_proj(batch['s_GeMAPlld'])      # (B, 1, emb_dim//2)
        # logger.debug(f"GeMAPS embed nan: {torch.isnan(s_GeMAPlld_emb).any()}, {torch.isnan(s_GeMAPfunc_emb).any()}")
        s_GeMAPS_emb = torch.cat((s_GeMAPfunc_emb,
                                  s_GeMAPlld_emb), dim=-1) + self.s_GeMAPS_me     # (B, 1, emb_dim)


        # 3.2 Masked Response Modeling (MRM)
        input_mrm_l = torch.cat((
            l_AU_emb_usq, l_exp_emb_usq, l_VA_emb_usq
        ), dim=-2)  # (B, n_frame, 3, emb_dim)
        logger.debug(f"input_mrm_l.shape {input_mrm_l.shape}")
        # prepare the embedding to fill the masked tokens
        l_mask_me = torch.cat((
            self.l_mask + self.l_AU_me.unsqueeze(-2),
            self.l_mask + self.l_exp_me.unsqueeze(-2),
            self.l_mask + self.l_VA_me.unsqueeze(-2)
        ), dim=-2)  # (B, n_frame, 3, emb_dim)
        logger.debug(f"l_mask_me.shape {l_mask_me.shape}")
        # prepare mask shape
        input_l_mask = batch['l_mask'].unsqueeze(-1).bool()  # (B, n_frame, 3) -> (B, n_frame, 3, 1)
        # randomly mask the listener response, and let the model predict the masked tokens
        input_mrm_l_masked = torch.where(input_l_mask, input_mrm_l, l_mask_me)  # (B, n_frame, 3, emb_dim)
        x = torch.cat((
            s_face_emb_usq, s_pose_emb_usq, s_MFCC_emb_usq,
            input_mrm_l_masked
        ), dim=-2).reshape(s_face_emb_usq.shape[0], -1, s_face_emb_usq.shape[-1])  # (B, n_frame*token_per_frame, emb_dim)
        logger.debug(f'x.shape: {x.shape}, input_mrm_l_masked.shape: {input_mrm_l_masked.shape}')
        # add GeMAPS feature
        x = torch.cat((
            x,              # (B, n_frame*token_per_frame, emb_dim)
            s_GeMAPS_emb    # (B, 1, emb_dim)
        ), dim=-2)  # (B, n_frame*token_per_frame + 1, emb_dim)
        logger.debug(f'with GeMAPS x.shape: {x.shape}')
        x = x + self.time_emb[:, :x.shape[1]]
        logger.debug(f"mrm_x before blocks:\n{x[:, :3]}")

        for block in self.blocks:
            x = block(x, causal=False)
        x = self.ln_f(x)  # final LayerNorm (B, n_frame*token_per_frame, emb_dim)

        logger.debug(f"mrm_x before head:\n{x[:, :6]}")
        mrm_AU_logits = self.head_AU(x[:, self.config.n_s_token::self.token_per_frame])  # (B, n_frame, emb_dim) -> (B, n_frame, n_l_AU)
        mrm_exp_logits = self.head_exp(x[:, self.config.n_s_token+1::self.token_per_frame])  # (B, n_frame, emb_dim) -> (B, n_frame, n_l_exp)
        mrm_VA_logits = self.head_VA(x[:, self.config.n_s_token+2::self.token_per_frame])  # (B, n_frame, emb_dim) -> (B, n_frame, n_l_VA)
        logger.debug(f"mrm_VA_logits = {mrm_VA_logits[:, :3]}, mrm_loss_mask.shape = {batch['l_VA'][0, :3]}")

        # only masked tokens contribute to the loss
        # mrm_loss_mask = ~ batch['l_mask']
        # tokens contributing to loss = masked tokens + randomly sampled tokens
        mrm_loss_mask = ~ batch['l_mask'] + (
                torch.rand(*batch['l_mask'].shape, requires_grad=False).to(x.device) < mrm_sample_rate
        )  # (B, n_frame, 4)

        # these transposes are required by cross_entropy
        raw_mrm_loss = torch.stack((
            F.cross_entropy(mrm_AU_logits.transpose(1, 2), batch['l_AU'], reduction='none'),
            F.cross_entropy(mrm_exp_logits.transpose(1, 2), batch['l_exp'], reduction='none'),
            F.cross_entropy(mrm_VA_logits.transpose(1, 2), batch['l_VA'], reduction='none')
        ), dim=-1)
        logger.debug(f"raw_mrm_loss.shape = {raw_mrm_loss.shape}, mrm_loss_mask.shape = {mrm_loss_mask.shape}")

        return (mrm_AU_logits, mrm_exp_logits, mrm_VA_logits), \
            (raw_mrm_loss * mrm_loss_mask).sum() / mrm_loss_mask.sum()  # average over contributing tokens

    def forward_crm(self, batch):
        """
        3.3 Causal Response Modeling (CRM)

        Input:
            dict(
                "s_exp": (batch_size, n_frame, dim_s_exp),
                "s_AU": (batch_size, n_frame, dim_s_AU),
                "s_VA": (batch_size, n_frame, dim_s_VA),
                "s_pose": (batch_size, n_frame, dim_s_pose),
                "s_MFCC": (batch_size, n_frame, dim_s_MFCC),
                "s_GeMAPS": (batch_size, 1, dim_s_GeMAPS),
                "is_face": (batch_size, n_frame),

                "l_AU": (batch_size, n_frame),
                "l_exp": (batch_size, n_frame),
                "l_VA": (batch_size, n_frame),
                "l_mask": (batch_size, n_frame, 3)
            )

        Output:
            (crm_AU_logits, crm_exp_logits, crm_VA_logits),
            loss_crm
        """

        s_face_emb_usq, s_pose_emb_usq, s_MFCC_emb_usq, \
            l_AU_emb_usq, l_exp_emb_usq, l_VA_emb_usq = self.forward_embedding(batch)

        # interleave tokens into one sequence
        x = torch.cat((
            s_face_emb_usq, s_pose_emb_usq, s_MFCC_emb_usq,
            l_AU_emb_usq, l_exp_emb_usq, l_VA_emb_usq
        ), dim=-2).reshape(s_face_emb_usq.shape[0], -1, s_face_emb_usq.shape[-1])  # (B, n_frame*token_per_frame, emb_dim)
        x = x + self.time_emb[:, :x.shape[1]]
        logger.debug(f"crm_x before blocks: \n{x[:, :6]}")

        for block in self.blocks:
            x = block(x, causal=True)
        x = self.ln_f(x)  # final LayerNorm (B, n_frame*token_per_frame, emb_dim)

        logger.debug(f"crm_x before head: \n{x[:, self.config.n_s_token+2::self.token_per_frame][:, :3]}")
        crm_AU_logits = self.head_AU(x[:, 0::self.token_per_frame])  # (B, n_frame, emb_dim) -> (B, n_frame, n_l_AU)
        crm_exp_logits = self.head_exp(x[:, 1::self.token_per_frame])  # (B, n_frame, emb_dim) -> (B, n_frame, n_l_exp)
        crm_VA_logits = self.head_VA(x[:, 2::self.token_per_frame])  # (B, n_frame, emb_dim) -> (B, n_frame, n_l_VA)
        logger.debug(f"crm_VA_logits = \n{crm_VA_logits[:, :3]}\n, crm_loss_mask.shape = {batch['l_VA'][0, :3]}")
        crm_loss = (F.cross_entropy(crm_AU_logits.transpose(1, 2), batch['l_AU']) +
                    F.cross_entropy(crm_exp_logits.transpose(1, 2), batch['l_exp']) +
                    F.cross_entropy(crm_VA_logits.transpose(1, 2), batch['l_VA'])) / self.config.n_l_token

        return (crm_AU_logits, crm_exp_logits, crm_VA_logits), crm_loss

    def forward(self, batch, mrm_sample_rate=0.2, train_slm=True):
        """
        Input:
            dict(
                "s_exp": (batch_size, n_frame, dim_s_exp),
                "s_AU": (batch_size, n_frame, dim_s_AU),
                "s_VA": (batch_size, n_frame, dim_s_VA),
                "s_pose": (batch_size, n_frame, dim_s_pose),
                "s_MFCC": (batch_size, n_frame, dim_s_MFCC),
                "s_GeMAPS": (batch_size, 1, dim_s_GeMAPS),
                "is_face": (batch_size, n_frame),

                "l_AU": (batch_size, n_frame),
                "l_exp": (batch_size, n_frame),
                "l_VA": (batch_size, n_frame),
                "l_mask": (batch_size, n_frame, 3)
            )

        Output:
            (mrm_AU_logits, mrm_exp_logits, mrm_VA_logits),
            (crm_AU_logits, crm_exp_logits, crm_VA_logits),
            loss_dict
        """
        assert batch['s_exp'].shape[1] <= self.max_n_frame, "Cannot forward, batch length > max length."

        loss_dict = dict()
        # ==================================================== SLM forward
        # region 3.1. Speaker-Listener-Matching loss (SLM)
        if train_slm:
            loss_dict['slm'] = self.forward_slm(batch, True) + self.forward_slm(batch, False)
        # endregion 3.1. Speaker-Listener-Matching loss (SLM)

        # ==================================================== MRM forward
        # region 3.2. Masked-Response-Modeling loss (MRM)
        (mrm_AU_logits, mrm_exp_logits, mrm_VA_logits), loss_dict['mrm'] = self.forward_mrm(batch, mrm_sample_rate)
        # endregion 3.2. Masked-Response-Modeling loss (MRM)

        # ==================================================== CRM forward
        # region 3.3. Causal-Response-Modeling loss (CRM)
        (crm_AU_logits, crm_exp_logits, crm_VA_logits), loss_dict['crm'] = self.forward_crm(batch)
        # endregion 3.3. Causal-Response-Modeling loss (CRM)

        return (
            (mrm_AU_logits, mrm_exp_logits, mrm_VA_logits),
            (crm_AU_logits, crm_exp_logits, crm_VA_logits),
            loss_dict
        )

    @torch.no_grad()
    def predict(self, batch, is_causal=True):
        """
        Input:
            dict(
                "s_exp": (batch_size, n_frame, dim_s_exp),
                "s_AU": (batch_size, n_frame, dim_s_AU),
                "s_VA": (batch_size, n_frame, dim_s_VA),
                "s_pose": (batch_size, n_frame, dim_s_pose),
                "s_MFCC": (batch_size, n_frame, dim_s_MFCC),
                "s_GeMAPS": (batch_size, 1, dim_s_GeMAPS),
                "is_face": (batch_size, n_frame),

                "l_AU": (batch_size, n_frame),      # can be zero tensor
                "l_exp": (batch_size, n_frame),     # can be zero tensor
                "l_VA": (batch_size, n_frame),      # can be zero tensor
                "l_mask": (batch_size, n_frame, 3)
            )

        Output:
            AU_logits, exp_logits, VA_logits    (B, T, n_vocab)
        """
        s_face_emb_usq, s_pose_emb_usq, s_MFCC_emb_usq, \
            l_AU_emb_usq, l_exp_emb_usq, l_VA_emb_usq = self.forward_embedding(batch)

        input_l = torch.cat((
            l_AU_emb_usq, l_exp_emb_usq, l_VA_emb_usq
        ), dim=-2)  # (B, n_frame, 3, emb_dim)
        logger.debug(f"input_l.shape (B, n_frame, 3, emb_dim) = {input_l.shape}")

        # region mask the listener tokens that should not be seen
        l_mask_me = torch.cat((
            self.l_mask + self.l_AU_me.unsqueeze(-2),
            self.l_mask + self.l_exp_me.unsqueeze(-2),
            self.l_mask + self.l_VA_me.unsqueeze(-2)
        ), dim=-2)  # (1, 1, 3, emb_dim)
        logger.debug(f"l_mask_me.shape (1, 1, 3, emb_dim) = {l_mask_me.shape}")

        input_l_mask = batch['l_mask'].unsqueeze(-1).bool()  # (B, n_frame, 3) -> (B, n_frame, 3, 1)
        input_l_masked = torch.where(input_l_mask, input_l, l_mask_me)  # (B, n_frame, 3, emb_dim)
        # endregion mask the listener tokens that should not be seen

        x = torch.cat((
            s_face_emb_usq, s_pose_emb_usq, s_MFCC_emb_usq,
            input_l_masked
        ), dim=-2).reshape(s_face_emb_usq.shape[0], -1,
                           s_face_emb_usq.shape[-1])  # (B, n_frame*token_per_frame, emb_dim)
        logger.debug(f'x.shape (B, n_frame*token_per_frame, emb_dim) = {x.shape}, input_l_masked.shape: {input_l_masked.shape}')

        if not is_causal:
            # add GeMAPS feature
            s_GeMAPfunc_emb = self.s_GeMAPfunc_proj(batch['s_GeMAPfunc'])  # (B, 1, emb_dim//2)
            s_GeMAPlld_emb = self.s_GeMAPlld_proj(batch['s_GeMAPlld'])  # (B, 1, emb_dim//2)
            s_GeMAPS_emb = torch.cat((s_GeMAPfunc_emb,
                                      s_GeMAPlld_emb), dim=-1) + self.s_GeMAPS_me  # (B, 1, emb_dim)

            x = torch.cat((
                x,              # (B, n_frame*token_per_frame, emb_dim)
                s_GeMAPS_emb    # (B, 1, emb_dim)
            ), dim=-2)  # (B, n_frame*token_per_frame + 1, emb_dim)

        x = x + self.time_emb[:, :x.shape[1]]
        for block in self.blocks:
            x = block(x, causal=is_causal)
        x = self.ln_f(x)  # final LayerNorm (B, n_frame*token_per_frame, emb_dim)

        if is_causal:
            AU_logits = self.head_AU(x[:, 0::self.token_per_frame])  # (B, n_frame, emb_dim) -> (B, n_frame, n_l_AU)
            exp_logits = self.head_exp(x[:, 1::self.token_per_frame])  # (B, n_frame, emb_dim) -> (B, n_frame, n_l_exp)
            VA_logits = self.head_VA(x[:, 2::self.token_per_frame])  # (B, n_frame, emb_dim) -> (B, n_frame, n_l_VA)
        else:
            AU_logits = self.head_AU(
                x[:, self.config.n_s_token::self.token_per_frame])  # (B, n_frame, emb_dim) -> (B, n_frame, n_l_AU)
            exp_logits = self.head_exp(
                x[:, self.config.n_s_token + 1::self.token_per_frame])  # (B, n_frame, emb_dim) -> (B, n_frame, n_l_exp)
            VA_logits = self.head_VA(
                x[:, self.config.n_s_token + 2::self.token_per_frame])  # (B, n_frame, emb_dim) -> (B, n_frame, n_l_VA)

        # (B, n_frame, n_vocab)
        return AU_logits, exp_logits, VA_logits








