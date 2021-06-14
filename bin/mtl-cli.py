#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import argparse

import multiprocessing
import numpy as np

import torch
from torch import nn, optim
from torch.nn import Parameter

from kbc.util import is_torch_tpu_available, set_seed

from kbc.training.data import Data
from kbc.training.batcher import Batcher
from kbc.training.masking import compute_masks

from kbc.models import DistMult, ComplEx

from kbc.regularizers import F2, N3

from kbc.blackbox import NegativeMRR

import higher

import logging


logger = logging.getLogger(os.path.basename(sys.argv[0]))
torch.set_num_threads(multiprocessing.cpu_count())


def metrics_to_str(metrics):
    def m(i: int) -> str:
        key = f"hits@{i}"
        return f'\tH@{i} {metrics[key]:.6f}' if key in metrics else ''

    return f'MRR {metrics["MRR"]:.6f}' + ''.join([m(i) for i in [1, 3, 5, 10, 20, 50, 100]])


def main(argv):
    parser = argparse.ArgumentParser('KBC Research', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train', action='store', required=True, type=str)

    parser.add_argument('--dev', action='store', type=str, default=None)
    parser.add_argument('--test', action='store', type=str, default=None)

    parser.add_argument('--test-i', action='store', type=str, default=None)
    parser.add_argument('--test-ii', action='store', type=str, default=None)

    # model params
    parser.add_argument('--model', '-m', action='store', type=str, default='distmult',
                        choices=['distmult', 'complex', 'transe'])

    parser.add_argument('--embedding-size', '-k', action='store', type=int, default=100)
    parser.add_argument('--batch-size', '-b', action='store', type=int, default=100)
    parser.add_argument('--eval-batch-size', '-B', action='store', type=int, default=None)

    # training params
    parser.add_argument('--epochs', '-e', action='store', type=int, default=100)
    parser.add_argument('--outer-epochs', '-E', action='store', type=int, default=100)

    parser.add_argument('--learning-rate', '-l', action='store', type=float, default=0.1)
    parser.add_argument('--outer-learning-rate', '-L', action='store', type=float, default=0.1)

    parser.add_argument('--optimizer', '-o', action='store', type=str, default='adagrad',
                        choices=['adagrad', 'adam', 'sgd'])
    parser.add_argument('--outer-optimizer', '-O', action='store', type=str, default='sgd',
                        choices=['adagrad', 'adam', 'sgd'])

    parser.add_argument('--corruption', '-c', action='store', type=str, default='so',
                        choices=['so', 'spo'])

    parser.add_argument('--seed', action='store', type=int, default=0)

    parser.add_argument('--input-type', '-I', action='store', type=str, default='standard',
                        choices=['standard', 'reciprocal'])

    parser.add_argument('--blackbox-lambda', action='store', type=float, default=None)
    parser.add_argument('--mask', action='store_true', default=False)

    parser.add_argument('--clip', '-C', action='store', type=float, default=None)
    parser.add_argument('--weight-decay', '-W', action='store', type=float, default=0.0)

    parser.add_argument('--load', action='store', type=str, default=None)
    parser.add_argument('--save', action='store', type=str, default=None)

    args = parser.parse_args(argv)

    import pprint
    pprint.pprint(vars(args))

    train_path = args.train

    dev_path = args.dev
    test_path = args.test

    test_i_path = args.test_i
    test_ii_path = args.test_ii

    model_name = args.model

    optimizer_name = args.optimizer
    outer_optimizer_name = args.outer_optimizer

    embedding_size = args.embedding_size

    batch_size = args.batch_size
    eval_batch_size = batch_size if args.eval_batch_size is None else args.eval_batch_size

    nb_epochs = args.epochs
    nb_outer_epochs = args.outer_epochs

    seed = args.seed

    learning_rate = args.learning_rate
    outer_learning_rate = args.outer_learning_rate

    corruption = args.corruption

    input_type = args.input_type

    blackbox_lambda = args.blackbox_lambda
    do_masking = args.mask

    clip = args.clip
    weight_decay = args.weight_decay

    load_path = args.load
    save_path = args.save

    set_seed(seed, is_deterministic=True)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    if is_torch_tpu_available():
        import torch_xla.core.xla_model as xm
        torch.set_default_tensor_type('torch.FloatTensor')
        device = xm.xla_device()

    logger.info(f'Device: {device}')

    data = Data(train_path=train_path, dev_path=dev_path, test_path=test_path,
                test_i_path=test_i_path, test_ii_path=test_ii_path, input_type=input_type)

    triples_name_pairs = [
        (data.dev_triples, 'dev'),
        (data.test_triples, 'test'),
        (data.test_i_triples, 'test-I'),
        (data.test_ii_triples, 'test-II'),
    ]

    rank = embedding_size * 2 if model_name in {'complex'} else embedding_size
    init_size = 1e-3

    entity_embeddings = nn.Embedding(data.nb_entities, rank, sparse=False).to(device)
    predicate_embeddings = nn.Embedding(data.nb_predicates, rank, sparse=False).to(device)

    entity_embeddings.weight.data *= init_size
    predicate_embeddings.weight.data *= init_size

    parameters_lst = nn.ModuleDict({
        'entities': entity_embeddings,
        'predicates': predicate_embeddings
    }).to(device)

    N3_weight = Parameter(torch.tensor(1e-4, device=device), requires_grad=True)
    N3_reg = N3() if N3_weight is not None else None

    e_weight = Parameter(torch.ones(data.nb_entities, device=device), requires_grad=True)
    p_weight = Parameter(torch.ones(data.nb_predicates, device=device), requires_grad=True)

    meta_parameters_lst = nn.ParameterDict({
        'n3': N3_weight,
        'ew': e_weight,
        'pw': p_weight
    }).to(device)

    if load_path is not None:
        parameters_lst.load_state_dict(torch.load(load_path))

    model_factory = {
        'distmult': lambda: DistMult(),
        'complex': lambda: ComplEx()
    }

    model = model_factory[model_name]().to(device)

    logger.info('Model state:')
    for param_tensor in parameters_lst.state_dict():
        logger.info(f'\t{param_tensor}\t{parameters_lst.state_dict()[param_tensor].size()}')

    optimizer_factory = {
        'adagrad': lambda params, lr, wd: optim.Adagrad(params, lr=lr, weight_decay=wd),
        'adam': lambda params, lr, wd: optim.Adam(params, lr=lr, weight_decay=wd),
        'sgd': lambda params, lr, wd: optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    }

    assert optimizer_name in optimizer_factory

    optimizer = optimizer_factory[optimizer_name](parameters_lst.parameters(), learning_rate, 0.0)
    meta_optimizer = optimizer_factory[outer_optimizer_name](meta_parameters_lst.parameters(), outer_learning_rate, weight_decay)

    loss_function = nn.CrossEntropyLoss(reduction='none')
    meta_loss_function = NegativeMRR(lmbda=blackbox_lambda)

    mask_sp = mask_po = None
    if do_masking:
        mask_sp, mask_po, _ = compute_masks(data.dev_triples, data.train_triples + data.dev_triples,
                                            data.entity_to_idx, data.predicate_to_idx)
        mask_sp = torch.tensor(mask_sp, dtype=torch.long, device=device)
        mask_po = torch.tensor(mask_po, dtype=torch.long, device=device)

    for outer_epoch in range(1, nb_outer_epochs + 1):
        e_tensor_lh = entity_embeddings.weight
        p_tensor_lh = predicate_embeddings.weight

        parameter_lst_lh = [e_tensor_lh, p_tensor_lh]

        diff_opt = higher.get_diff_optim(optimizer, parameter_lst_lh, device=device, track_higher_grads=True)

        random_state = np.random.RandomState(seed)

        for epoch_no in range(1, nb_epochs + 1):
            batcher = Batcher(data.Xs, data.Xp, data.Xo, batch_size, 1, random_state)

            for batch_no, (batch_start, batch_end) in enumerate(batcher.batches, 1):
                xp_batch, xs_batch, xo_batch, _ = batcher.get_batch(batch_start, batch_end)

                xs_batch = torch.tensor(xs_batch, dtype=torch.long, device=device)
                xp_batch = torch.tensor(xp_batch, dtype=torch.long, device=device)
                xo_batch = torch.tensor(xo_batch, dtype=torch.long, device=device)

                xs_batch_emb = e_tensor_lh[xs_batch, :]
                xp_batch_emb = p_tensor_lh[xp_batch, :]
                xo_batch_emb = e_tensor_lh[xo_batch, :]

                loss = 0.0

                s_loss_weights = e_weight[xs_batch]
                p_loss_weights = p_weight[xp_batch]
                o_loss_weights = e_weight[xo_batch]

                loss_weights = s_loss_weights * p_loss_weights * o_loss_weights

                if 's' in corruption:
                    po_scores = model.forward(xp_batch_emb, None, xo_batch_emb,
                                              entity_embeddings=e_tensor_lh, predicate_embeddings=p_tensor_lh)
                    loss = loss + torch.mean(loss_weights * loss_function(po_scores, xs_batch))

                if 'o' in corruption:
                    sp_scores = model.forward(xp_batch_emb, xs_batch_emb, None,
                                              entity_embeddings=e_tensor_lh, predicate_embeddings=p_tensor_lh)
                    loss = loss + torch.mean(loss_weights * loss_function(sp_scores, xo_batch))

                if 'p' in corruption:
                    so_scores = model.forward(None, xs_batch_emb, xo_batch_emb,
                                              entity_embeddings=e_tensor_lh, predicate_embeddings=p_tensor_lh)
                    loss = loss + torch.mean(loss_weights * loss_function(so_scores, xp_batch))

                factors = [model.factor(e, safe=True) for e in [xp_batch_emb, xs_batch_emb, xo_batch_emb]]

                if N3_weight is not None:
                    loss += N3_weight * N3_reg(factors)

                e_tensor_lh, p_tensor_lh = diff_opt.step(loss, params=parameter_lst_lh)
                parameter_lst_lh = [e_tensor_lh, p_tensor_lh]

        dev_xs_batch = torch.tensor(data.dev_Xs, dtype=torch.long, device=device)
        dev_xp_batch = torch.tensor(data.dev_Xp, dtype=torch.long, device=device)
        dev_xo_batch = torch.tensor(data.dev_Xo, dtype=torch.long, device=device)
        dev_xi_batch = torch.tensor(data.dev_Xi, dtype=torch.long, device=device)

        dev_xs_batch_emb = e_tensor_lh[dev_xs_batch, :]
        dev_xp_batch_emb = p_tensor_lh[dev_xp_batch, :]
        dev_xo_batch_emb = e_tensor_lh[dev_xo_batch, :]

        dev_loss = 0.0

        po_scores = model.forward(dev_xp_batch_emb, None, dev_xo_batch_emb,
                                  entity_embeddings=e_tensor_lh, predicate_embeddings=p_tensor_lh)
        if do_masking is True:
            po_scores = po_scores + mask_po[dev_xi_batch, :]
        dev_loss += meta_loss_function(po_scores, dev_xs_batch) / 2

        sp_scores = model.forward(dev_xp_batch_emb, dev_xs_batch_emb, None,
                                  entity_embeddings=e_tensor_lh, predicate_embeddings=p_tensor_lh)
        if do_masking is True:
            sp_scores = sp_scores + mask_sp[dev_xi_batch, :]
        dev_loss += meta_loss_function(sp_scores, dev_xo_batch) / 2

        print(f'{outer_epoch}\t{dev_loss:.5f}\t{N3_weight:.5f}')

        dev_loss.backward()

        if clip is not None:
            torch.nn.utils.clip_grad_norm_(meta_parameters_lst.parameters(), clip)

        meta_optimizer.step()
        meta_optimizer.zero_grad()

        if N3_weight is not None:
            N3_weight.data.clamp_(1e-6)

    if save_path is not None:
        torch.save(parameters_lst.state_dict(), save_path)

    logger.info("Training finished")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print(' '.join(sys.argv))
    main(sys.argv[1:])
