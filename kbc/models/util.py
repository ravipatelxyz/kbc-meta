# -*- coding: utf-8 -*-

import math
import torch
from torch import Tensor

import logging

logger = logging.getLogger(__name__)


@torch.jit.script
def normalize_phases(p_emb):
    # normalize phases so that they lie in [-pi,pi]
    # first shift phases by pi
    out = p_emb + math.pi
    # compute the modulo (result then in [0,2*pi))
    out = torch.remainder(out, 2.0 * math.pi)
    # shift back
    out = out - math.pi
    return out


@torch.jit.script
def pairwise_sum(X, Y):
    """Compute pairwise sum of rows of X and Y.
    Returns tensor of shape len(X) x len(Y) x dim."""
    return X.unsqueeze(1) + Y


@torch.jit.script
def pairwise_diff(X, Y):
    """Compute pairwise difference of rows of X and Y.
    Returns tensor of shape len(X) x len(Y) x dim."""
    return X.unsqueeze(1) - Y


@torch.jit.script
def pairwise_hadamard(X, Y):
    """Compute pairwise Hadamard product of rows of X and Y.
    Returns tensor of shape len(X) x len(Y) x dim."""
    return X.unsqueeze(1) * Y


@torch.jit.script
def hadamard_complex(x_re, x_im, y_re, y_im):
    """Hadamard product for complex vectors"""
    result_re = x_re * y_re - x_im * y_im
    result_im = x_re * y_im + x_im * y_re
    return result_re, result_im


@torch.jit.script
def pairwise_hadamard_complex(x_re, x_im, y_re, y_im):
    """Pairwise Hadamard product for complex vectors"""
    result_re = pairwise_hadamard(x_re, y_re) - pairwise_hadamard(x_im, y_im)
    result_im = pairwise_hadamard(x_re, y_im) + pairwise_hadamard(x_im, y_re)
    return result_re, result_im


@torch.jit.script
def diff_complex(x_re, x_im, y_re, y_im):
    """Difference of complex vectors"""
    return x_re - y_re, x_im - y_im


@torch.jit.script
def pairwise_diff_complex(x_re, x_im, y_re, y_im):
    """Pairwise difference of complex vectors"""
    # print('X shape: {} Y shape: {}'.format(x_re, y_re))
    return pairwise_diff(x_re, y_re), pairwise_diff(x_im, y_im)


@torch.jit.script
def abs_complex(x_re: Tensor, x_im: Tensor):
    """Compute magnitude of given complex numbers"""
    x_re_im = torch.stack((x_re, x_im), dim=0)  # dim0: real, imaginary
    return torch.norm(x_re_im, p=2, dim=0)  # sqrt(real^2+imaginary^2)


@torch.jit.script
def norm_nonnegative(x: Tensor, dim: int, p: float):
    """Computes lp-norm along dim assuming that all inputs are non-negative."""
    if p == 1.0:
        # speed up things for this common case. We known that the inputs are
        # non-negative here.
        return torch.sum(x, dim=dim)
    else:
        return torch.norm(x, dim=dim, p=p)
