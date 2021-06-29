#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from copy import deepcopy

import argparse

import multiprocessing

import higher
import numpy as np

import torch
from torch import nn, optim

from kbc.util import is_torch_tpu_available, set_seed

from kbc.training.data import Data
from kbc.training.batcher import Batcher

from kbc.models import DistMult, ComplEx, TransE

from kbc.regularizers import F2, N3
from kbc.evaluation import evaluate

from kbc.blackbox import NegativeMRR

import logging
import wandb

logger = logging.getLogger(os.path.basename(sys.argv[0]))
torch.set_num_threads(multiprocessing.cpu_count())


def metrics_to_str(metrics):
    def m(i: int) -> str:
        key = f"hits@{i}"
        return f'\tH@{i} {metrics[key]:.6f}' if key in metrics else ''

    return f'MRR {metrics["MRR"]:.6f}' + ''.join([m(i) for i in [1, 3, 5, 10, 20, 50, 100]])

def parse_args(argv):
    parser = argparse.ArgumentParser('KBC Research', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train', action='store', required=True, type=str)

    parser.add_argument('--dev', action='store', type=str, default=None)
    parser.add_argument('--test', action='store', type=str, default=None)

    # model params
    parser.add_argument('--model', '-m', action='store', type=str, default='distmult',
                        choices=['distmult', 'complex', 'transe'])

    parser.add_argument('--embedding_size', '-k', action='store', type=int, default=100)
    parser.add_argument('--batch_size', '-b', action='store', type=int, default=100)
    parser.add_argument('--eval_batch_size', '-B', action='store', type=int, default=None)

    # training params (inner loop)
    parser.add_argument('--epochs', '-e', action='store', type=int, default=100)
    parser.add_argument('--learning_rate', '-l', action='store', type=float, default=0.1)
    parser.add_argument('--optimizer', '-o', action='store', type=str, default='adagrad',
                        choices=['adagrad', 'adam', 'sgd'])
    parser.add_argument('--regularizer', '-re', type=str, choices=["F2", "N3"], default="F2")
    # parser.add_argument('--F2', action='store', type=float, default=None)
    # parser.add_argument('--N3', action='store', type=float, default=None)

    parser.add_argument('--corruption', '-c', action='store', type=str, default='so',
                        choices=['so', 'spo'])
    parser.add_argument('--seed', action='store', type=int, default=0)
    parser.add_argument('--validate_every', '-V', action='store', type=int, default=None)
    parser.add_argument('--input_type', '-I', action='store', type=str, default='standard',
                        choices=['standard', 'reciprocal'])

    parser.add_argument('--blackbox_lambda', action='store', type=float, default=None)

    # training params (outer loop)
    parser.add_argument('--optimizer_outer', '-oo', action='store', type=str, default='adam',
                        choices=['adagrad', 'adam', 'sgd'])
    parser.add_argument('--learning_rate_outer', '-lo', action='store', type=float, default=0.005)
    parser.add_argument('--outer_steps', '-os', action='store', type=int, default=100)
    parser.add_argument('--stopping_tol_outer', '-to', action='store', type=float, default=0.02)

    # other
    parser.add_argument('--load', action='store', type=str, default=None)
    parser.add_argument('--save', action='store', type=str, default=None)
    parser.add_argument('--quiet', '-q', action='store_true', default=False)
    parser.add_argument('--save_figs', '-sf', action='store', type=str, default='False', choices=['True', 'False'])
    parser.add_argument('--use_wandb', '-wb', action='store', type=str, default='False', choices=['True', 'False'])

    return parser.parse_args(argv)

def main(args):
    import pprint
    pprint.pprint(vars(args))

    train_path = args.train
    dev_path = args.dev
    test_path = args.test

    model_name = args.model
    optimizer_name = args.optimizer
    embedding_size = args.embedding_size
    batch_size = args.batch_size
    eval_batch_size = batch_size if args.eval_batch_size is None else args.eval_batch_size
    nb_epochs = args.epochs
    seed = args.seed
    learning_rate = args.learning_rate
    # F2_weight = args.F2
    # N3_weight = args.N3
    regularizer = args.regularizer
    corruption = args.corruption
    validate_every = args.validate_every
    input_type = args.input_type
    blackbox_lambda = args.blackbox_lambda

    optimizer_outer_name = args.optimizer_outer
    learning_rate_outer = args.learning_rate_outer
    outer_steps = args.outer_steps
    stopping_tol_outer = args.stopping_tol_outer

    load_path = args.load
    save_path = args.save
    save_figs = args.save_figs == 'True'
    use_wandb = args.use_wandb == 'True'

    is_quiet = args.quiet

    set_seed(seed)
    random_state = np.random.RandomState(seed)

    if use_wandb == True:
        wandb.init(entity="uclnlp", project="kbc_meta", group=f"base")
        wandb.config.update(args)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    if is_torch_tpu_available():
        import torch_xla.core.xla_model as xm
        torch.set_default_tensor_type('torch.FloatTensor')
        device = xm.xla_device()

    logger.info(f'Device: {device}')
    if use_wandb == True:
        wandb.config.update({'device': device})

    data = Data(train_path=train_path, dev_path=dev_path, test_path=test_path,
                test_i_path=None, test_ii_path=None, input_type=input_type)

    # the .dev_triples attributes here contain ID strings
    triples_name_pairs = [
        (data.dev_triples, 'dev'),
        (data.test_triples, 'test'),
    ]

    # When this dictionary is indexed by model name, the appropriate model class will be initialised
    model_factory = {
        'distmult': lambda: DistMult(),
        'complex': lambda: ComplEx(),
        'transe': lambda: TransE()
    }

    # Initialise correct model
    model = model_factory[model_name]().to(device)

    rank = embedding_size * 2 if model_name in {'complex'} else embedding_size
    init_size = 1e-3

    # nn.Embedding using to a lookup table of embeddings (i.e. you can index entity_embeddings to return given entities embedding)
    # Nice explanation found in Escachator's answer here: https://stackoverflow.com/questions/50747947/embedding-in-pytorch
    entity_embeddings = nn.Embedding(data.nb_entities, rank).to(device)
    predicate_embeddings = nn.Embedding(data.nb_predicates, rank).to(device)

    # Downscale the randomly initialised embeddings (initialised with N(0,1))
    # entity_embeddings.weight is a tensor of shape (num_embeddings, rank)
    entity_embeddings.weight.data *= init_size
    predicate_embeddings.weight.data *= init_size

    # Specify norm to use as regularizer
    if regularizer == "F2":
        F2_reg = F2()
    elif regularizer == "N3":
        N3_reg = N3()

    # regularizer_weights = nn.Embedding(1, 1).to(device)
    # reg_weight_graph = deepcopy(regularizer_weights.weight)
    reg_weight_graph = torch.tensor(0.2, requires_grad=True).to(device)
    print(reg_weight_graph)

    optimizer_factory_outer = {
        'adagrad': lambda: optim.Adagrad([reg_weight_graph], lr=learning_rate_outer),
        'adam': lambda: optim.Adam([reg_weight_graph], lr=learning_rate_outer),
        'sgd': lambda: optim.SGD([reg_weight_graph], lr=learning_rate_outer)
    }

    optimizer_outer = optimizer_factory_outer[optimizer_outer_name]()

    # Specify loss function (cross-entropy by default)
    if blackbox_lambda is None:
        loss_function = nn.CrossEntropyLoss(reduction='mean')
    else:
        loss_function = NegativeMRR(lmbda=blackbox_lambda)

    # Specify outer loss function (cross-entropy)

    outer_loss_function = nn.CrossEntropyLoss(reduction='mean')

    # Training loop

    last_logged_epoch = 0
    best_mrr = 0

    for outer_step in range(outer_steps):

        e_graph = deepcopy(entity_embeddings.weight)
        e_graph.to(device)
        p_graph = deepcopy(predicate_embeddings.weight)
        e_graph.to(device)

        optimizer_factory = {
            'adagrad': lambda: optim.Adagrad([e_graph, p_graph], lr=learning_rate),
            'adam': lambda: optim.Adam([e_graph, p_graph], lr=learning_rate),
            'sgd': lambda: optim.SGD([e_graph, p_graph], lr=learning_rate)
        }

        assert optimizer_name in optimizer_factory
        optimizer = optimizer_factory[optimizer_name]()

        diffopt = higher.get_diff_optim(optimizer, [e_graph, p_graph], track_higher_grads=True)

        for epoch_no in range(1, nb_epochs + 1):
            train_log = {}  # dictionary to store training metrics for uploading to wandb for each epoch
            batcher = Batcher(data.Xs, data.Xp, data.Xo, batch_size, 1, random_state)
            nb_batches = len(batcher.batches)

            epoch_loss_values = []  # to store loss for each batch in the epoch
            epoch_loss_nonreg_values = []

            for batch_no, (batch_start, batch_end) in enumerate(batcher.batches, 1):

                # Size [B] numpy arrays containing indices of each subject_entity, predicate, and object_entity in the batch
                xp_batch, xs_batch, xo_batch, xi_batch = batcher.get_batch(batch_start, batch_end)

                xs_batch = torch.tensor(xs_batch, dtype=torch.long, device=device)
                xp_batch = torch.tensor(xp_batch, dtype=torch.long, device=device)
                xo_batch = torch.tensor(xo_batch, dtype=torch.long, device=device)
                xi_batch = torch.tensor(xi_batch, dtype=torch.long, device=device)

                # Return embeddings for each s, p, o in the batch
                # This returns tensors of shape (batch_size, rank)
                xp_batch_emb = p_graph[xp_batch]
                xs_batch_emb = e_graph[xs_batch]
                xo_batch_emb = e_graph[xo_batch]

                loss = 0.0

                # If corruption="spo", then loss will be calculate based on predicting subjects, predicates, and objects
                # If corruption="sp", then loss will be calculate based on just predicting subjects and objects
                if 's' in corruption:
                    # shape of po_scores is (batch_size, Nb_preds in entire dataset)
                    po_scores = model.forward(xp_batch_emb, None, xo_batch_emb, entity_embeddings=e_graph, predicate_embeddings=p_graph)
                    loss += loss_function(po_scores, xs_batch)

                if 'o' in corruption:
                    # shape of sp_scores is (batch_size, Nb_entities in entire dataset)
                    sp_scores = model.forward(xp_batch_emb, xs_batch_emb, None, entity_embeddings=e_graph, predicate_embeddings=p_graph)
                    loss += loss_function(sp_scores, xo_batch)

                if 'p' in corruption:
                    # shape of so_scores is (batch_size, Nb_entities in entire dataset)
                    so_scores = model.forward(None, xs_batch_emb, xo_batch_emb, entity_embeddings=e_graph, predicate_embeddings=p_graph)
                    loss += loss_function(so_scores, xp_batch)

                loss_nonreg_value = loss.item()
                epoch_loss_nonreg_values += [loss_nonreg_value]

                factors = [model.factor(e) for e in [xp_batch_emb, xs_batch_emb, xo_batch_emb]]

                if regularizer == "F2":
                    reg_term = F2_reg(factors)
                    additional_loss = reg_weight_graph*reg_term
                    loss += additional_loss
                    # loss += reg_weight_graph * F2_reg(factors)

                if regularizer == "N3":
                    loss += reg_weight_graph * N3_reg(factors)

                e_graph, p_graph = diffopt.step(loss, params=[e_graph, p_graph])

                loss_value = loss.item()
                epoch_loss_values += [loss_value]

                if not is_quiet:
                    logger.info(f'Epoch {epoch_no}/{nb_epochs}\tBatch {batch_no}/{nb_batches}\tLoss {loss_value:.6f} ({loss_nonreg_value:.6f})')
                    # print(f'Epoch {epoch_no}/{nb_epochs}\tBatch {batch_no}/{nb_batches}\tLoss {loss_value:.6f} ({loss_nonreg_value:.6f})')

            loss_mean, loss_std = np.mean(epoch_loss_values), np.std(epoch_loss_values)
            loss_nonreg_mean, loss_nonreg_std = np.mean(epoch_loss_nonreg_values), np.std(epoch_loss_nonreg_values)

            # train_log['loss_mean'] = loss_mean
            # train_log['loss_std'] = loss_std
            # train_log['loss_nonreg_mean'] = loss_nonreg_mean
            # train_log['loss_nonreg_std'] = loss_nonreg_std
            # logger.info(f'Epoch {epoch_no}/{nb_epochs}\tLoss {loss_mean:.4f} ± {loss_std:.4f} ({loss_nonreg_mean:.4f} ± {loss_nonreg_std:.4f})')
            print(f'Epoch {epoch_no}/{nb_epochs}\tLoss {loss_mean:.4f} ± {loss_std:.4f} ({loss_nonreg_mean:.4f} ± {loss_nonreg_std:.4f})')

            # if validate_every is not None and epoch_no % validate_every == 0:
            #     for triples, name in [(t, n) for t, n in triples_name_pairs if len(t) > 0]:
            #         model.eval()
            #         metrics = evaluate(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
            #                            test_triples=triples, all_triples=data.all_triples,
            #                            entity_to_index=data.entity_to_idx, predicate_to_index=data.predicate_to_idx,
            #                            model=model, batch_size=eval_batch_size, device=device)
            #         logger.info(f'Epoch {epoch_no}/{nb_epochs}\t{name} results\t{metrics_to_str(metrics)}')
            #         metrics_new = {f'{name}_{k}': v for k, v in metrics.items()} # hack to get different keys for logging
            #         train_log.update(metrics_new)
            #         # print(f'Epoch {epoch_no}/{nb_epochs}\t{name} results\t{metrics_to_str(metrics)}')
            #     last_logged_epoch = epoch_no
            #
            #     if train_log['dev_MRR'] > best_mrr:
            #         best_mrr = train_log['dev_MRR']
            #         best_log=train_log
            #
            # if use_wandb == True:
            #     wandb.log(train_log, step=epoch_no, commit=True)



        # if last_logged_epoch != nb_epochs:
        #     eval_log = {}
        #     for triples, name in [(t, n) for t, n in triples_name_pairs if len(t) > 0]:
        #         model.eval()
        #         metrics = evaluate(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
        #                            test_triples=triples, all_triples=data.all_triples,
        #                            entity_to_index=data.entity_to_idx, predicate_to_index=data.predicate_to_idx,
        #                            model=model, batch_size=eval_batch_size, device=device)
        #         logger.info(f'Final \t{name} results\t{metrics_to_str(metrics)}')
        #
        #         metrics_new = {f'{name}_{k}': v for k, v in metrics.items()}  # hack to get different keys for logging
        #         eval_log.update(metrics_new)
        #         # print(f'Final \t{name} results\t{metrics_to_str(metrics)}')
        #
        #     if eval_log['dev_MRR'] > best_mrr:
        #         best_mrr = eval_log['dev_MRR']
        #         best_log=eval_log
        #
        #     if use_wandb == True:
        #         wandb.log(eval_log, step=nb_epochs, commit=True)
        #
        # if use_wandb == True:
        #     wandb.run.summary.update(best_log)
        #
        # if save_path is not None:
        #     torch.save(parameters_lst.state_dict(), save_path)

    if use_wandb == True:
        wandb.save(f"{save_path[:-4]}.log")
        wandb.save("kbc_meta/logs/array.err")
        wandb.save("kbc_meta/logs/array.out")
        wandb.finish()

    logger.info("Training finished")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = parse_args(sys.argv[1:])
    print(' '.join(sys.argv))
    main(args)
