#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import os.path

import sys
import logging


def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs if k not in {'d'}])


def to_cmd(c, _path=None):
    mask_str = '--mask' if c["mask"] else ''
    command = f'PYTHONPATH=. python3 ./bin/kbc-cli.py ' \
        f'--train data/{c["data"]}/train.tsv ' \
        f'--dev data/{c["data"]}/dev.tsv ' \
        f'--test data/{c["data"]}/test.tsv ' \
        f'-m {c["m"]} -k {c["k"]} -b {c["b"]} -e {c["e"]} ' \
        f'--F2 {c["f2"]} --N3 {c["n3"]} -l {c["lr"]} -I {c["i"]} -V {c["V"]} -o {c["o"]} -c {c["c"]} -q ' \
        f'--blackbox-lambda {c["lmbda"]} {mask_str}'
    return command


def to_logfile(c, path):
    outfile = "{}/base_beaker_v3.{}.log".format(path, summary(c).replace("/", "_"))
    return outfile


def main(argv):
    hyp_space = dict(
        data=['kinship', 'nations', 'umls'],
        m=['complex'],
        k=[100, 200, 500, 1000],
        b=[10, 50, 100, 500],
        e=[100],
        f2=[0],
        n3=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1],
        lr=[0.1, 0.01, 0.001],
        # i=['standard'],
        i=['standard', 'reciprocal'],
        lmbda=[0.1, 1, 5, 10, 50, 100],
        mask=[True, False],
        V=[3],
        o=['adagrad'],
        # c=['so', 'spo']
        c=['so']
    )

    configurations = list(cartesian_product(hyp_space))

    path = 'logs/base/base_beaker_v3'
    is_rc = False

    # Check that we are on the UCLCS cluster first
    if os.path.exists('/home/pminervi/'):
        is_rc = True
        # If the folder that will contain logs does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)

    command_lines = set()
    for cfg in configurations:
        logfile = to_logfile(cfg, path)

        completed = False
        if is_rc is True and os.path.isfile(logfile):
            with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                completed = 'Training finished' in content

        if not completed:
            command_line = '{} > {} 2>&1'.format(to_cmd(cfg), logfile)
            command_lines |= {command_line}

    # Sort command lines and remove duplicates
    sorted_command_lines = sorted(command_lines)

    import random
    rng = random.Random(0)
    rng.shuffle(sorted_command_lines)

    nb_jobs = len(sorted_command_lines)

    header = """#!/bin/bash -l

#$ -cwd
#$ -S /bin/bash
#$ -o $HOME/array.out
#$ -e $HOME/array.err
#$ -t 1-{}
#$ -l tmem=12G
#$ -l h_rt=01:00:00
#$ -l gpu=true

conda activate gpu

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd $HOME/workspace/kbc-lm

""".format(nb_jobs)

    print(header)

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        print('test $SGE_TASK_ID -eq {} && sleep 10 && {}'.format(job_id, command_line))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
