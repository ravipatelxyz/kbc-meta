#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
import argparse
import sys
import os

import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import pandas as pd
import torch
from torch import nn, optim

import higher
import wandb

torch.set_num_threads(multiprocessing.cpu_count())


# %%

class ToyMeta2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor_a):
        return tensor_a


def parse_args(argv):
    parser = argparse.ArgumentParser('KBC Research', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # a,h,c starting vals
    parser.add_argument('--a', action='store', required=True, type=float)
    parser.add_argument('--h', action='store', required=True, type=float)
    parser.add_argument('--c', action='store', required=True, type=float)

    # inner loop
    parser.add_argument('--optimizer', '-oi', action='store', type=str, default='adam',
                        choices=['adam', 'adagrad', 'sgd'])
    parser.add_argument('--learning_rate', '-li', action='store', type=float, default=0.01)
    parser.add_argument('--inner_steps', '-is', action='store', type=int, default=100)

    # outer loop
    parser.add_argument('--optimizer_outer', '-oo', action='store', type=str, default='adam',
                        choices=['adagrad', 'adam', 'sgd'])
    parser.add_argument('--learning_rate_outer', '-lo', action='store', type=float, default=0.005)
    parser.add_argument('--outer_steps', '-os', action='store', type=int, default=100)
    parser.add_argument('--stopping_tol_outer', '-to', action='store', type=float, default=0.02)

    # other
    parser.add_argument('--save_figs', '-sf', action='store', type=str, default='False', choices=['True', 'False'])
    parser.add_argument('--use_wandb', '-wb', action='store', type=str, default='False', choices=['True', 'False'])

    return parser.parse_args(argv)


def main(argv):
    """
    Model: Single parameter a that it returns
    Inner optimisation: min_a ||a-h||, with a the single model parameter, h being a hyperparameter
    Outer optimisation: min_p ||a-c||, with c set to a fixed value
    """
    a = torch.tensor(args.a, requires_grad=True)
    h = torch.tensor(args.h, requires_grad=True)
    c = torch.tensor(args.c, requires_grad=False)
    inner_steps = args.inner_steps
    outer_steps = args.outer_steps
    learning_rate = args.learning_rate
    learning_rate_outer = args.learning_rate_outer
    optimizer_name = args.optimizer
    optimizer_outer_name = args.optimizer_outer
    save_figs = args.save_figs == 'True'
    use_wandb = args.use_wandb == 'True'
    stopping_tol_outer = args.stopping_tol_outer

    if use_wandb == True:
        wandb.init(entity="uclnlp", project="kbc_meta", group=f"toymeta2")
        wandb.config.update(args)

    device = torch.device('cpu')

    model = ToyMeta2().to(device)

    # for logging / plotting
    meta_losses = []
    a_vals = []
    h_vals = []
    c_vals = []
    converged = False
    convergence_outer_step = None
    convergence_total_steps = None

    h_graph = deepcopy(h)

    h_grads = []

    # def mult_grad(grad):
    #     return grad
    def store_h_grad(grad):
        h_grads.append(grad)

    # h_graph.register_hook(mult_grad)
    h_graph.register_hook(store_h_grad)

    optimizer_factory_outer = {
        'adagrad': lambda: optim.Adagrad([h_graph], lr=learning_rate_outer),
        'adam': lambda: optim.Adam([h_graph], lr=learning_rate_outer),
        'sgd': lambda: optim.SGD([h_graph], lr=learning_rate_outer)
    }

    optimizer_outer = optimizer_factory_outer[optimizer_outer_name]()

    for outer_step in range(outer_steps):

        a_graph = deepcopy(a)

        optimizer_factory = {
            'adagrad': lambda: optim.Adagrad([a_graph], lr=learning_rate),
            'adam': lambda: optim.Adam([a_graph], lr=learning_rate),
            'sgd': lambda: optim.SGD([a_graph], lr=learning_rate)
        }

        optimizer = optimizer_factory[optimizer_name]()

        diffopt = higher.get_diff_optim(optimizer, [a_graph], track_higher_grads=True)

        for inner_step in range(inner_steps):
            loss = torch.norm(model.forward(a_graph) - h_graph)
            a_graph = diffopt.step(loss, params=[a_graph])
            a_graph = a_graph[0]

        meta_loss = torch.norm(a_graph - c)
        if meta_loss < stopping_tol_outer and converged == False:
            convergence_outer_step = outer_step
            convergence_total_steps = outer_step * inner_steps
            converged = True

        # Metrics for logging
        meta_losses += [meta_loss.detach().clone().item()]
        a_vals += [a_graph.detach().clone().item()]
        h_vals += [h_graph.detach().clone().item()]
        c_vals += [c.detach().clone().item()]

        meta_loss.backward()
        if use_wandb == True:
            train_log = {'meta_loss': meta_losses[-1],
                         'a_value': a_vals[-1],
                         'h_value': h_vals[-1],
                         'c_value': c_vals[-1],
                         'outer_convergence_value': np.absolute(h_vals[-1] - c_vals[-1]),
                         'convergence_outer_step': convergence_outer_step,
                         'convergence_total_steps': convergence_total_steps,
                         'hypergradient_h': h_grads[-1]}
            wandb.log(train_log, step=outer_step)

        optimizer_outer.step()
        optimizer_outer.zero_grad()

    plt.figure(1)
    plt.plot(meta_losses)
    plt.plot(a_vals)
    plt.plot(h_vals)
    plt.plot(c_vals)
    plt.legend(["meta loss", "a", "h", "c"])
    plt.xlabel("Outer step")
    plt.ylabel("Value as specified in legend")
    plt.title(
        f"Starting values: a={args.a}, h={args.h}, c={args.c}\nInner steps: {inner_steps} | LR: {learning_rate} | Outer steps: {outer_steps} | OuterLR: {learning_rate_outer} \nOptim: {optimizer_name} | Converges step: {convergence_outer_step} | Converges total steps: {convergence_total_steps}")
    plt.tight_layout()
    if save_figs == True:
        filename = f"toymeta2_loss_a{args.a}_h{args.h}_c{args.c}_is{inner_steps}_lr{learning_rate}_os{outer_steps}_outlr{learning_rate_outer}_optim{optimizer_name}.png"
        if use_wandb == True:
            plt.savefig(os.path.join(wandb.run.dir, filename))
        else:
            plt.savefig(f"./toymeta2/plots/{filename}")

    # plt.show()

    df_grad = pd.DataFrame(h_grads)
    rolling_mean_grads = df_grad.rolling(window=40).mean()
    plt.figure(2)
    plt.plot(rolling_mean_grads)
    plt.title("Rolling 40-epoch mean gradient value")
    plt.xlabel("Outer step")
    plt.ylabel("Gradient of loss wrt hyperparameter h")
    plt.tight_layout()
    if save_figs == True:
        filename = f"toymeta2_grad_rollingavg_a{args.a}_h{args.h}_c{args.c}_is{inner_steps}_lr{learning_rate}_os{outer_steps}_outlr{learning_rate_outer}_optim{optimizer_name}.png"
        if use_wandb == True:
            plt.savefig(os.path.join(wandb.run.dir, filename))
        else:
            plt.savefig(f"./toymeta2/plots/{filename}")
    # plt.show()

    array_grads = np.array(h_grads)
    plt.figure(3)
    plt.hist(np.absolute(array_grads), bins=80)
    plt.title(
        f" Magnitude of hypergradient of loss wrt hyperparameter h \nMean: {np.mean(np.absolute(array_grads)):.10f} | SD: {np.std(h_grads):.10f} | Median {np.median(np.absolute(array_grads)):.10f}")
    plt.ylabel("Frequency")
    plt.xlabel("Hypergradient magnitude")
    plt.tight_layout()
    if save_figs == True:
        filename = f"toymeta2_grad_gradhist_a{args.a}_h{args.h}_c{args.c}_is{inner_steps}_lr{learning_rate}_os{outer_steps}_outlr{learning_rate_outer}_optim{optimizer_name}.png"
        if use_wandb == True:
            plt.savefig(os.path.join(wandb.run.dir, filename))
        else:
            plt.savefig(f"./toymeta2/plots/{filename}")
    # plt.show()

    if use_wandb == True:
        train_log.update({'hypergrad_mean_magnitude': np.mean(np.absolute(array_grads)),
                          'hypergrad_SD': np.std(h_grads),
                          'hypergrad_median_magnitude': np.median(np.absolute(array_grads))})
        wandb.run.summary.update(train_log)
    wandb.finish()


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
