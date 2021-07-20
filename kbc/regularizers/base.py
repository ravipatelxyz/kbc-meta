# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor

from typing import List

import logging

logger = logging.getLogger(__name__)


class Regularizer(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self,
                 factors: List[Tensor]):
        raise NotImplementedError


class F2(Regularizer):
    def __init__(self):
        super().__init__()

    def __call__(self,
                 factors: List[Tensor]):
        norm = sum(torch.sum(f ** 2) for f in factors)
        return norm / factors[0].shape[0]


class N3(Regularizer):
    def __init__(self):
        super().__init__()

    def __call__(self,
                 factors: List[Tensor]):
        norm = sum(torch.sum(torch.abs(f) ** 3) for f in factors)
        return norm / factors[0].shape[0]


class Entropy(Regularizer):
    def __init__(self, use_logits: bool = False) -> None:
        super().__init__()
        self.use_logits = use_logits

    def __call__(self,
                 factors: List[Tensor]):
        if self.use_logits is True:
            # Inputs are logits - turn them into probabilities
            factors = [torch.softmax(f, dim=1) for f in factors]
        res = sum(torch.sum(- torch.log(f) * f) for f in factors)
        return res / factors[0].shape[0]


def weighted_f2(lmbda_lhs, lmbda_rel, lmbda_rhs, factors):
    lhs, rel, rhs = factors
    l_reg = ((lmbda_lhs * (lhs ** 2)).sum() / lhs.shape[0]
             + (lmbda_rel * (rel ** 2)).sum() / rel.shape[0]
             + (lmbda_rhs * (rhs ** 2)).sum() /rhs.shape[0])
    # l_reg = l_reg / factors[0].shape[0]
    l_reg_raw = ((lhs.abs() ** 2).sum() / lhs.shape[0]
                 + (rel.abs() ** 2).sum() / rel.shape[0]
                 + (rhs.abs() ** 2).sum() /rhs.shape[0])
    # l_reg_raw = l_reg_raw / factors[0].shape[0]
    lmbda_avg = (lmbda_lhs.mean() + lmbda_rel.mean() + lmbda_rhs.mean()) / 3
    return l_reg, l_reg_raw, lmbda_avg


def weighted_n3(lmbda_lhs, lmbda_rel, lmbda_rhs, factors):
    lhs, rel, rhs = factors
    l_reg = ((lmbda_lhs * (lhs.abs() ** 3)).sum() / lhs.shape[0]
             + (lmbda_rel * (rel.abs() ** 3)).sum() / rel.shape[0]
             + (lmbda_rhs * (rhs.abs() ** 3)).sum() / rhs.shape[0])
    # l_reg = l_reg / factors[0].shape[0]
    l_reg_raw = ((lhs.abs() ** 3).sum() / lhs.shape[0]
                + (rel.abs() ** 3).sum() / rel.shape[0]
                + (rhs.abs() ** 3).sum() / rhs.shape[0])
    # l_reg_raw = l_reg_raw / factors[0].shape[0]
    lmbda_avg = (lmbda_lhs.mean() + lmbda_rel.mean() + lmbda_rhs.mean()) / 3
    return l_reg, l_reg_raw, lmbda_avg
