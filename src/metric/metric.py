import logging

import torch
import numpy as np
from src.metric import compute_FRC_mp, compute_FRC, compute_FRD_mp
from src.metric import compute_FRD, compute_FRDvs, compute_FRVar
from src.metric import compute_s_mse, compute_TLCC, compute_TLCC_mp

logger = logging.getLogger(__name__)

@torch.no_grad()
def measure(l_pred_10, l_gt, s_gt, appro_matrix, n_threads=8, ):
    """
    asserting all inputs are on cpu

    :param l_pred_10: (N, 10, 750, 25)
    :param l_gt: (N, 750, 25)
    :param s_gt: (N, 750, 25)
    :param appro_matrix: (N, N)

    :return: FRC, FRD, FRDvs, FRVar, smse, TLCC
    """

    # If you have problems running function compute_TLCC_mp, please replace this function with function compute_TLCC
    logger.warning("start compute TLCC")
    TLCC = compute_TLCC_mp(l_pred_10, s_gt, p=n_threads)
    logger.info(f"TLCC: {TLCC}")

    # If you have problems running function compute_FRC_mp, please replace this function with function compute_FRC
    logger.warning("start compute FRC")
    FRC = compute_FRC_mp(l_pred_10, l_gt, appro_matrix, p=n_threads)
    logger.info(f"FRC: {FRC}")

    # If you have problems running function compute_FRD_mp, please replace this function with function compute_FRD
    logger.warning("start compute FRD")
    FRD = compute_FRD_mp(l_pred_10, l_gt, appro_matrix, p=n_threads)    # FRDist
    logger.info(f"FRD: {FRD}")

    FRDvs = compute_FRDvs(l_pred_10)
    FRVar = compute_FRVar(l_pred_10)
    smse = compute_s_mse(l_pred_10)     # FRDiv

    return FRC, FRD, FRDvs, FRVar, smse, TLCC

