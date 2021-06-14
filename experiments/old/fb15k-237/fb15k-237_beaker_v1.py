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
    command = f'PYTHONPATH=. python3 kbc_meta/kbc-cli.py ' \
        f'--train kbc_meta/data/fb15k-237/train.tsv ' \
        f'--dev kbc_meta/data/fb15k-237/dev.tsv ' \
        f'--test kbc_meta/data/fb15k-237/test.tsv ' \
        f'-m {c["m"]} -k {c["k"]} -b {c["b"]} -e {c["e"]} ' \
        f'--F2 {c["f2"]} --N3 {c["n3"]} -l {c["lr"]} -I {c["i"]} -V {c["V"]} -o {c["o"]} -c {c["c"]} -q '
    return command


def to_logfile(c, path):
    outfile = "{}/fb15k-237_beaker_v1.{}.log".format(path, summary(c).replace("/", "_"))
    return outfile


def main(argv):
    hyp_space_1 = dict(
        m=['complex'],
        k=[100, 200, 500, 1000, 2000, 4000],
        b=[50, 100, 500],
        e=[100],
        f2=[1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1],
        n3=[0],
        lr=[0.1],
        # i=['standard', 'reciprocal'],
        i=['standard'],
        V=[1],
        o=['adagrad'],
        c=['so', 'spo']
    )

    hyp_space_2 = dict(
        m=['complex'],
        k=[100, 200, 500, 1000, 2000, 4000],
        b=[50, 100, 500],
        e=[100],
        f2=[0],
        n3=[1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1],
        lr=[0.1],
        # i=['standard', 'reciprocal'],
        i=['standard'],
        V=[1],
        o=['adagrad'],
        c=['so', 'spo']
    )

    configurations = list(cartesian_product(hyp_space_1)) + list(cartesian_product(hyp_space_2))

    path = 'kbc_meta/logs/fb15k-237/fb15k-237_complex_base_beaker_v1'
    is_rc = False

    # Check that we are on the UCLCS cluster first
    if os.path.exists('/home/ravpatel'):
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

        # # if you don't wish to save the models, use the following and comment out the below
        # if not completed:
        #     command_line = '{} > {} 2>&1'.format(to_cmd(cfg), logfile)
        #     command_lines |= {command_line}

        # if you wish to save models, use the following and comment out the above
        if not completed:
            command_line = '{} --save {}.pth > {} 2>&1'.format(to_cmd(cfg), logfile[:-4], logfile)
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
#$ -o $HOME/kbc_meta/logs/array.out
#$ -e $HOME/kbc_meta/logs/array.err
#$ -t 1-{}
#$ -l tmem=16G
#$ -l h_rt=48:00:00
#$ -l gpu=true
export LANG="en_US.utf8"
export LANGUAGE="en_US:en"
source $HOME/.activate_conda
export WANDB_API_KEY = 9903f079d28315317cc996b863de147f9135a2bf
cd $HOME/kbc_meta

""".format(nb_jobs)

    print(header)

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        print('test $SGE_TASK_ID -eq {} && sleep 30 && {}'.format(job_id, command_line))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
