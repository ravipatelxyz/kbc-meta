#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%

import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import torch
from torch import nn, optim

import higher
from kbc.util import set_seed

torch.set_num_threads(multiprocessing.cpu_count())


# %%

class ToyMeta2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor_a):
        return tensor_a

def main():
    """
    Model: Single parameter a that it returns
    Inner optimisation: min_a ||a-h||, with a the single model parameter, h being a hyperparameter
    Outer optimisation: min_p ||a-c||, with c set to a fixed value
    """
    optimizer_name = OPTIMIZER
    optimizer_outer_name = OPTIMIZER_OUTER
    inner_steps = INNER_STEPS
    outer_steps = OUTER_STEPS
    learning_rate = LEARNING_RATE
    learning_rate_outer = LEARNING_RATE_OUTER

    a = torch.tensor(A, requires_grad=True)
    h = torch.tensor(H, requires_grad=True)
    c = torch.tensor(C, requires_grad=False)

    device = torch.device('cpu')

    # a_lst = nn.ParameterList([a]).to(device)
    # p_lst = nn.ParameterList([p]).to(device)

    model = ToyMeta2().to(device)

    meta_losses = []
    a_vals = []
    h_vals = []
    c_vals = []

    h_graph = deepcopy(h)

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
            loss = torch.norm(model.forward(a_graph)-h_graph)
            a_graph = diffopt.step(loss, params=[a_graph])
            a_graph = a_graph[0]
            # print(a_graph)

        meta_loss = torch.norm(a_graph - c)

        # metrics for plotting
        meta_losses += [meta_loss.detach().clone().item()]
        a_vals += [a_graph.detach().clone().item()]
        h_vals += [h_graph.detach().clone().item()]
        c_vals += [c.detach().clone().item()]

        meta_loss.backward()
        optimizer_outer.step()
        optimizer_outer.zero_grad()

    print(a_graph)
    print(h_graph)
    print(c)
    plt.plot(meta_losses)
    plt.plot(a_vals)
    plt.plot(h_vals)
    plt.plot(c_vals)
    plt.legend(["meta loss","a","h","c"])
    plt.xlabel("Outer steps")
    plt.ylabel("Value as specified in legend")
    plt.title(f"Starting values: a={A}, h={H}, c={C}\nInner steps: {inner_steps} | LR: {learning_rate} | Outer steps: {outer_steps} | OuterLR: {learning_rate_outer} | Optim: {optimizer_name}")
    plt.show()
    plt.savefig(f"./plots/toymeta2_a{A}_h{H}_c{C}_is{inner_steps}_lr{learning_rate}_os{outer_steps}_outlr{learning_rate_outer}_optim{optimizer_name}.png")

    debug=0



if __name__ == '__main__':

    # Specify experimental parameters
    OUTER_STEPS = 2000
    INNER_STEPS = 80
    LEARNING_RATE = 0.01
    LEARNING_RATE_OUTER = 0.005
    OPTIMIZER = "adam"
    OPTIMIZER_OUTER = OPTIMIZER
    A = 0.45
    H = 0.48
    C = 0.71

    main()