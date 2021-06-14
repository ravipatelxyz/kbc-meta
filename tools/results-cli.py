#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def read_results(path: str) -> Tuple[float, str]:
    best_dev_mrr = None
    best_test_line = None

    with open(path) as f:
        for line in f.readlines():
            if 'dev results' in line:
                tokens = line.split()
                mrr_idx = tokens.index('MRR') + 1
                mrr_value = float(tokens[mrr_idx])

                if best_dev_mrr is None or best_dev_mrr < mrr_value:
                    best_dev_mrr = mrr_value
                    best_test_line = None

            if 'test results' in line:
                if best_test_line is None:
                    best_test_line = line[line.find('MRR'):].strip()

    return best_dev_mrr, best_test_line


def plot(path, df):
    ax = sns.lineplot(x='k',
                      y='mrr',
                      hue='c',
                      style='c',
                      data=df)

    ax.set_title(path)

    fig = ax.get_figure()
    fig.savefig(f'{path}')

    plt.clf()


def main(argv):
    parser = argparse.ArgumentParser('Show Results', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('paths', nargs='+', type=str, default=None)

    # e.g. k, c
    parser.add_argument('--hyperparameters', '-H', nargs='+', type=str, default=None)
    parser.add_argument('--plot', action='store', type=str, default=None)

    args = parser.parse_args(argv)

    paths = args.paths
    hyperparameters = args.hyperparameters
    plot_path = args.plot

    hyp_to_mrr = {}
    hyp_to_line = {}
    hyp_to_path = {}

    for path in paths:
        mrr, line = read_results(path)

        if mrr is not None:
            path_split = path.split('_')

            key = None
            for h in hyperparameters:
                for entry in path_split:
                    if entry.startswith(f'{h}='):
                        if key is None:
                            key = entry
                        else:
                            key += ' ' + entry

            if key not in hyp_to_mrr or mrr > hyp_to_mrr[key]:
                hyp_to_mrr[key] = mrr
                hyp_to_line[key] = line
                hyp_to_path[key] = path

    for k, v in hyp_to_line.items():
        print(k, hyp_to_path[k])
        print(k, v)

    if plot_path is not None:
        hyp_set = set(hyperparameters)
        other_hyp_set = hyp_set - {'c', 'k'}

        def key_to_dict(_key):
            res = dict()
            for entry in _key.split(' '):
                k, v = entry.split('=')
                res[k] = v
            return res

        hyp_to_lst = dict()
        for key, mrr in hyp_to_mrr.items():
            key_dict = key_to_dict(key)
            key_dict['mrr'] = hyp_to_mrr[key]

            for hyp in key_dict:
                if hyp not in hyp_to_lst:
                    hyp_to_lst[hyp] = []

            for k, v in key_dict.items():
                if k in {'k'}:
                    v = int(v)
                hyp_to_lst[k] += [v]

        df = pd.DataFrame(hyp_to_lst)

        print(df)

        plot(plot_path, df)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main(sys.argv[1:])
