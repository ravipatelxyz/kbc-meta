#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import torch


def rank(seq):
    return torch.argsort(torch.argsort(seq, dim=1, descending=True)) + 1


# Source: https://github.com/martius-lab/blackbox-backprop/blob/master/blackbox_backprop/ranking.py
class DifferentiableRanker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sequence, lambda_val):
        ranks = rank(sequence).float()
        ctx.lambda_val = lambda_val
        ctx.save_for_backward(sequence, ranks)
        return ranks

    @staticmethod
    def backward(ctx, grad_output):
        sequence, ranks = ctx.saved_tensors
        assert grad_output.shape == ranks.shape
        sequence_prime = sequence + ctx.lambda_val * grad_output
        ranks_prime = rank(sequence_prime)
        gradient = -(ranks - ranks_prime) / (ctx.lambda_val + 1e-8)
        return gradient, None


def main(argv):

    score_values = [
        [0.5, 0.3, 0.7, 0.1, 0.2],
        [0.8, 0.3, 0.7, 0.1, 0.2],
        [0.1, 0.3, 0.7, 0.1, 0.2]
    ]

    score_tensor = torch.tensor(score_values, dtype=torch.float, requires_grad=True)

    ranking_function = DifferentiableRanker.apply
    ranks = ranking_function(score_tensor, 1)

    mrr = torch.mean(1.0 / ranks[:, 0])

    mrr.backward()

    print(score_tensor.grad)


if __name__ == '__main__':
    main(sys.argv[1:])
