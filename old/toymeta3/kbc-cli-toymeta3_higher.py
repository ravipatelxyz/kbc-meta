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

    def forward(self, tensor_a, tensor_b, tensor_c):
        score = torch.dot(tensor_a, tensor_b) + torch.dot(tensor_b, tensor_c)
        return score


def parse_args(argv):
    parser = argparse.ArgumentParser('KBC Research', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # a,b,c,h starting vals
    parser.add_argument('--a', action='store', required=True, type=float)
    parser.add_argument('--b', action='store', required=True, type=float)
    parser.add_argument('--c', action='store', required=True, type=float)
    parser.add_argument('--h', action='store', required=True, type=float)

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
    Model:
    Inner optimisation:
    Outer optimisation:
    """
    a = torch.tensor([args.a], requires_grad=True)
    b = torch.tensor([args.b], requires_grad=True)
    c = torch.tensor([args.c], requires_grad=True)
    h = torch.tensor([args.h], requires_grad=True)
    inner_steps = args.inner_steps
    outer_steps = args.outer_steps
    learning_rate = args.learning_rate
    learning_rate_outer = args.learning_rate_outer
    optimizer_name = args.optimizer
    optimizer_outer_name = args.optimizer_outer
    save_figs = args.save_figs == 'True'
    use_wandb = args.use_wandb == 'True'
    stopping_tol_outer = args.stopping_tol_outer

    # logging
    if use_wandb == True:
        wandb.init(entity="uclnlp", project="kbc_meta", group=f"toymeta3")
        wandb.config.update(args)

    device = torch.device('cpu')

    # initialise model
    model = ToyMeta2().to(device)

    # for logging / plotting
    meta_losses = []
    a_vals = []
    b_vals = []
    c_vals = []
    h_vals = []
    converged = False
    convergence_outer_step = None
    convergence_total_steps = None
    steps_in_convergence_tol = 0

    # Hyperparameter h setup
    h_graph = deepcopy(h)

    h_grads = []

    def store_h_grad(grad):
        h_grads.append(grad)

    h_graph.register_hook(store_h_grad)

    optimizer_factory_outer = {
        'adagrad': lambda: optim.Adagrad([h_graph], lr=learning_rate_outer),
        'adam': lambda: optim.Adam([h_graph], lr=learning_rate_outer),
        'sgd': lambda: optim.SGD([h_graph], lr=learning_rate_outer)
    }

    optimizer_outer = optimizer_factory_outer[optimizer_outer_name]()

    # outer loop min_h ||a-c||
    for outer_step in range(outer_steps):
        print(f"Outer step: {outer_step}")

        # Parameter setup
        a_graph = deepcopy(a)
        b_graph = deepcopy(b)
        c_graph = deepcopy(c)

        optimizer_factory = {
            'adagrad': lambda: optim.Adagrad([a_graph, b_graph, c_graph], lr=learning_rate),
            'adam': lambda: optim.Adam([a_graph, b_graph, c_graph], lr=learning_rate),
            'sgd': lambda: optim.SGD([a_graph, b_graph, c_graph], lr=learning_rate)
        }

        optimizer = optimizer_factory[optimizer_name]()

        diffopt = higher.get_diff_optim(optimizer, [a_graph, b_graph, c_graph], track_higher_grads=True)

        # inner loop  min_a ||a-h||
        for inner_step in range(inner_steps):
            loss = -model.forward(a_graph, b_graph, c_graph) + h_graph * (
                        torch.dot(a_graph, a_graph) + 2 * torch.dot(b_graph, b_graph) + torch.dot(c_graph,
                                                                                                  c_graph))
            a_graph, b_graph, c_graph = diffopt.step(loss, params=[a_graph, b_graph, c_graph])
            # params = [a_graph, b_graph, c_graph]

        # logging
        meta_loss = -torch.dot(a_graph, c_graph)
        if h_graph-0.5 < stopping_tol_outer:
            if converged == False:
                convergence_outer_step = outer_step
                convergence_total_steps = outer_step * inner_steps
                converged = True

            steps_in_convergence_tol += 1

        meta_losses += [meta_loss.detach().clone().item()]
        a_vals += [a_graph.detach().clone().item()]
        b_vals += [b_graph.detach().clone().item()]
        c_vals += [c_graph.detach().clone().item()]
        h_vals += [h_graph.detach().clone().item()]

        # Backprop grad of outer loss wrt h
        meta_loss.backward()

        # logging
        if use_wandb == True:
            train_log = {'meta_loss': meta_losses[-1],
                         'a_value': a_vals[-1],
                         'b_value': b_vals[-1],
                         'c_value': c_vals[-1],
                         'h_value': h_vals[-1],
                         'outer_convergence_value': np.absolute(h_vals[-1] - 0.5),
                         'hypergradient_h': h_grads[-1]}
            wandb.log(train_log, step=outer_step)

        # Update h
        optimizer_outer.step()
        optimizer_outer.zero_grad()

    plt.figure(1)
    # plt.plot(meta_losses)
    plt.plot(a_vals)
    plt.plot(b_vals)
    plt.plot(c_vals)
    plt.plot(h_vals)
    plt.plot([0.5 for i in range(outer_steps)])
    plt.legend(["a", "b", "c", "h", "h_target"])
    plt.xlabel("Outer step")
    plt.ylabel("Value as specified in legend")
    plt.title(
        f"Starting values: a={args.a}, b={args.b},  c={args.c}, h={args.h}\nInner steps: {inner_steps} | LR: {learning_rate} | Outer steps: {outer_steps} | OuterLR: {learning_rate_outer} \nOptim: {optimizer_name} | Converges step: {convergence_outer_step} | Converges total steps: {convergence_total_steps}")
    plt.tight_layout()
    if save_figs == True:
        filename = f"toymeta3_loss_a{args.a}_b{args.b}_c{args.c}_h{args.h}_is{inner_steps}_lr{learning_rate}_os{outer_steps}_outlr{learning_rate_outer}_optim{optimizer_name}.png"
        if use_wandb == True:
            plt.savefig(os.path.join(wandb.run.dir, filename))
        else:
            plt.savefig(f"./toymeta2/plots/{filename}")

    plt.show()

    df_grad = pd.DataFrame(h_grads)
    rolling_mean_grads = df_grad.rolling(window=10).mean()
    plt.figure(2)
    plt.plot(rolling_mean_grads)
    plt.title("Rolling 40-epoch mean gradient value")
    plt.xlabel("Outer step")
    plt.ylabel("Gradient of loss wrt hyperparameter h")
    plt.tight_layout()
    if save_figs == True:
        filename = f"toymeta3_grad_rollingavg_a{args.a}_b{args.b}_c{args.c}_h{args.h}_is{inner_steps}_lr{learning_rate}_os{outer_steps}_outlr{learning_rate_outer}_optim{optimizer_name}.png"
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
        filename = f"toymeta3_grad_gradhist_a{args.a}_b{args.b}_c{args.c}_h{args.h}_is{inner_steps}_lr{learning_rate}_os{outer_steps}_outlr{learning_rate_outer}_optim{optimizer_name}.png"
        if use_wandb == True:
            plt.savefig(os.path.join(wandb.run.dir, filename))
        else:
            plt.savefig(f"./toymeta2/plots/{filename}")
    plt.show()

    if use_wandb == True:
        train_log.update({'convergence_nb_outersteps_within_tol': steps_in_convergence_tol,
                          'hypergrad_mean_magnitude': np.mean(np.absolute(array_grads)),
                          'hypergrad_SD': np.std(h_grads),
                          'hypergrad_normalizedSD': np.std(h_grads) / np.mean(np.absolute(array_grads)),
                          'hypergrad_median_magnitude': np.median(np.absolute(array_grads)),
                          'hypergrad_IQR': np.percentile(array_grads, 75) - np.percentile(array_grads, 25),
                          'hypergrad_normalizedIQR': (np.percentile(array_grads, 75) - np.percentile(array_grads,
                                                                                                     25)) / np.median(
                              np.absolute(array_grads))})
        if convergence_outer_step != None:
            train_log.update({'convergence_outer_step': convergence_outer_step,
                              'convergence_total_steps': convergence_total_steps,
                              'convergence_proportion_outersteps_sustained_within_tol': (steps_in_convergence_tol) / (
                                          outer_steps - convergence_outer_step),
                              'convergence_nb_outersteps_post_convergence': outer_steps - convergence_outer_step})
        wandb.run.summary.update(train_log)
        wandb.finish()

    print({'params': [a_graph, b_graph, c_graph],
           'h_val': h_graph.item(),
           'convergence_nb_outersteps_within_tol': steps_in_convergence_tol,
           'hypergrad_mean_magnitude': np.mean(np.absolute(array_grads)),
           'hypergrad_SD': np.std(h_grads),
           'hypergrad_normalizedSD': np.std(h_grads) / np.mean(np.absolute(array_grads)),
           'hypergrad_median_magnitude': np.median(np.absolute(array_grads)),
           'hypergrad_IQR': np.percentile(array_grads, 75) - np.percentile(array_grads, 25),
           'hypergrad_normalizedIQR': (np.percentile(array_grads, 75) - np.percentile(array_grads, 25)) / np.median(
               np.absolute(array_grads))})

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
