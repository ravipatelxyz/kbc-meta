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
    parser.add_argument('--torch_seed', '-ts', action='store', type=int)
    parser.add_argument('--embedding_size', '-es', action='store', type=int)
    parser.add_argument('--a', action='store', type=float, default=0.53)
    parser.add_argument('--h', action='store', type=float, default=0.5)
    parser.add_argument('--c', action='store', type=float, default=0.8)

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

    # noise injection
    parser.add_argument('--hypergrad_noise_sd', '-hn', action='store', type=float, default=0)
    parser.add_argument('--inner_loss_noise_sd', '-in', action='store', type=float, default=0)

    # other
    parser.add_argument('--save_figs', '-sf', action='store', type=str, default='False', choices=['True', 'False'])
    parser.add_argument('--use_wandb', '-wb', action='store', type=str, default='False', choices=['True', 'False'])

    return parser.parse_args(argv)


def main(args):
    """
    Model: Single parameter a that it returns
    Inner optimisation: min_a ||a-h||, with a the single model parameter, h being a hyperparameter
    Outer optimisation: min_h ||a-c||, with c set to a fixed value
    """
    if 'embedding_size' in args:
        if 'torch_seed' in args:
            torch.manual_seed(args.torch_seed)
        a = torch.rand(args.embedding_size, requires_grad=True)
        h = torch.rand(args.embedding_size, requires_grad=True)
        c = torch.rand(args.embedding_size, requires_grad=False)
    else:
        a = torch.tensor(args.a, requires_grad=True)
        h = torch.tensor(args.h, requires_grad=True)
        c = torch.tensor(args.c, requires_grad=False)
    inner_steps = args.inner_steps
    outer_steps = args.outer_steps
    learning_rate = args.learning_rate
    learning_rate_outer = args.learning_rate_outer
    optimizer_name = args.optimizer
    optimizer_outer_name = args.optimizer_outer
    hypergrad_noise_sd = args.hypergrad_noise_sd
    inner_loss_noise_sd = args.inner_loss_noise_sd
    save_figs = args.save_figs == 'True'
    use_wandb = args.use_wandb == 'True'
    stopping_tol_outer = args.stopping_tol_outer

    # logging
    if use_wandb == True:
        wandb.init(entity="uclnlp", project="kbc_meta", group=f"toymeta2")
        wandb.config.update(args)

    device = torch.device('cpu')

    # initialise model
    model = ToyMeta2().to(device)

    # for logging / plotting
    meta_losses = []
    a_vals = []
    h_vals = []
    c_vals = []
    converged = False
    convergence_outer_step = None
    convergence_total_steps = None
    steps_in_convergence_tol = 0

    # Hyperparameter h setup
    h_graph = deepcopy(h)

    h_grads = []
    h_grads_noisy = []

    # backprop hooks
    def store_h_grad(grad):
        h_grads.append(torch.norm(grad))
    def add_noise_to_hypergrad(grad):
        shape = grad.shape
        noise = torch.normal(mean=torch.ones(shape), std=torch.ones(shape)*hypergrad_noise_sd)
        return grad*noise
    def store_h_grad_noisy(grad):
        h_grads_noisy.append(torch.norm(grad))

    h_graph.register_hook(store_h_grad)
    h_graph.register_hook(add_noise_to_hypergrad)
    h_graph.register_hook(store_h_grad_noisy)

    optimizer_factory_outer = {
        'adagrad': lambda: optim.Adagrad([h_graph], lr=learning_rate_outer),
        'adam': lambda: optim.Adam([h_graph], lr=learning_rate_outer),
        'sgd': lambda: optim.SGD([h_graph], lr=learning_rate_outer)
    }

    optimizer_outer = optimizer_factory_outer[optimizer_outer_name]()

    # outer loop min_h ||a-c||
    for outer_step in range(outer_steps):

        print(f"Outer step: {outer_step}")
        # Parameter a setup
        a_graph = deepcopy(a)

        optimizer_factory = {
            'adagrad': lambda: optim.Adagrad([a_graph], lr=learning_rate),
            'adam': lambda: optim.Adam([a_graph], lr=learning_rate),
            'sgd': lambda: optim.SGD([a_graph], lr=learning_rate)
        }

        optimizer = optimizer_factory[optimizer_name]()

        diffopt = higher.get_diff_optim(optimizer, [a_graph], track_higher_grads=True)

        # inner loop  min_a ||a-h||
        for inner_step in range(inner_steps):
            loss = (torch.norm(model.forward(a_graph) - h_graph) ** 2) * torch.normal(mean=torch.ones(1), std=torch.ones(1)*inner_loss_noise_sd)
            a_graph = diffopt.step(loss, params=[a_graph])
            a_graph = a_graph[0]

        # logging
        meta_loss = torch.norm(a_graph - c)**2
        if meta_loss < stopping_tol_outer:
            if converged == False:
                convergence_outer_step = outer_step
                convergence_total_steps = outer_step * inner_steps
                converged = True

            steps_in_convergence_tol += 1

        meta_losses += [meta_loss.detach().clone().item()]
        a_vals += [a_graph.detach().clone()]
        h_vals += [h_graph.detach().clone()]
        c_vals += [c.detach().clone()]

        # Backprop grad of outer loss wrt h
        meta_loss.backward()

        # logging
        if use_wandb == True:
            train_log = {'meta_loss': meta_losses[-1],
                         'a_value': a_vals[-1],
                         'h_value': h_vals[-1],
                         'c_value': c_vals[-1],
                         'outer_convergence_value': np.absolute(h_vals[-1] - c_vals[-1]),
                         'hypergradient_h': h_grads[-1]}
            wandb.log(train_log, step=outer_step)

        # Update h
        optimizer_outer.step()
        optimizer_outer.zero_grad()

    print(c_vals[-1])
    print(meta_losses[-1])
    print(a_vals[-1])
    print(h_vals[-1])

    plt.figure(1)
    plt.plot(meta_losses)
    # plt.plot(a_vals)
    # plt.plot(h_vals)
    # plt.plot(c_vals)
    # plt.legend(["meta loss", "a", "h", "c"])
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

    plt.show()

    df_grad = pd.DataFrame(h_grads)
    window = 1
    rolling_mean_grads = df_grad.rolling(window=window).mean()
    plt.figure(2)
    plt.plot(rolling_mean_grads)
    plt.plot(h_grads_noisy)
    plt.title(f"Rolling {window}-step mean gradient value")
    plt.xlabel("Outer step")
    plt.ylabel("L2 norm of gradient of loss wrt hyperparameter h")
    plt.tight_layout()
    if save_figs == True:
        filename = f"toymeta2_grad_rollingavg_a{args.a}_h{args.h}_c{args.c}_is{inner_steps}_lr{learning_rate}_os{outer_steps}_outlr{learning_rate_outer}_optim{optimizer_name}.png"
        if use_wandb == True:
            plt.savefig(os.path.join(wandb.run.dir, filename))
        else:
            plt.savefig(f"./toymeta2/plots/{filename}")
    plt.show()

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
    plt.show()

    if use_wandb == True:
        train_log.update({'convergence_nb_outersteps_within_tol': steps_in_convergence_tol,
                          'hypergrad_mean_magnitude': np.mean(np.absolute(array_grads)),
                          'hypergrad_SD': np.std(h_grads),
                          'hypergrad_normalizedSD': np.std(h_grads)/np.mean(np.absolute(array_grads)),
                          'hypergrad_median_magnitude': np.median(np.absolute(array_grads)),
                          'hypergrad_IQR': np.percentile(array_grads, 75) - np.percentile(array_grads, 25),
                          'hypergrad_normalizedIQR': (np.percentile(array_grads, 75) - np.percentile(array_grads, 25)) / np.median(np.absolute(array_grads))})
        if convergence_outer_step != None:
            train_log.update({'convergence_outer_step': convergence_outer_step,
                              'convergence_total_steps': convergence_total_steps,
                              'convergence_proportion_outersteps_sustained_within_tol': (steps_in_convergence_tol) / (outer_steps - convergence_outer_step),
                              'convergence_nb_outersteps_post_convergence': outer_steps - convergence_outer_step})

        wandb.run.summary.update(train_log)
        wandb.finish()


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
