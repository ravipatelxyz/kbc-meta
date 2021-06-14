# -*- coding: utf-8 -*-

import torch
from torch import Tensor

from typing import Optional, Tuple

import logging

logger = logging.getLogger(__name__)


def rank(seq: Tensor) -> Tensor:
    return torch.argsort(torch.argsort(seq, dim=1, descending=True)) + 1


class DifferentiableRanker(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                sequence: Tensor,
                lambda_val: float) -> Tensor:
        ranks = rank(sequence).float()
        ctx.lambda_val = lambda_val
        ctx.save_for_backward(sequence, ranks)
        return ranks

    @staticmethod
    def backward(ctx,
                 grad_output: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        sequence, ranks = ctx.saved_tensors
        assert grad_output.shape == ranks.shape
        sequence_prime = sequence + ctx.lambda_val * grad_output
        ranks_prime = rank(sequence_prime)
        gradient = -(ranks - ranks_prime) / (ctx.lambda_val + 1e-8)
        return gradient, None
