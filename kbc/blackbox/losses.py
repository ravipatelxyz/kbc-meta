# -*- coding: utf-8 -*-

import torch
from torch import Tensor

from kbc.blackbox.base import DifferentiableRanker

import logging

logger = logging.getLogger(__name__)


class NegativeMRR:
    def __init__(self,
                 lmbda: float):
        self._lmbda = lmbda
        self._differentiable_ranker = DifferentiableRanker.apply

    @property
    def lmbda(self):
        return self._lmbda

    @lmbda.setter
    def lmbda(self,
              value: float):
        if value < 0.0:
            raise ValueError("Lmbda >= 0")
        self._lmbda = value

    def __call__(self,
                 input: Tensor,
                 target: Tensor) -> Tensor:
        ranks_2d = self._differentiable_ranker(input, self._lmbda)
        batch_size = ranks_2d.shape[0]
        ranks = ranks_2d[torch.arange(batch_size), target]

        # negative MRR
        mrr = torch.mean(1.0 / ranks)
        ret = 1.0 - mrr

        return ret
