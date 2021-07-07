#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

import argparse

import multiprocessing
import numpy as np

import torch
from torch import nn, optim

from kbc.util import is_torch_tpu_available, set_seed

from kbc.training.data import Data
from kbc.training.batcher import Batcher
from kbc.training.masking import compute_masks

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

    parser.add_argument('--test-i', action='store', type=str, default=None)
    parser.add_argument('--test-ii', action='store', type=str, default=None)

    # model params
    parser.add_argument('--model', '-m', action='store', type=str, default='distmult',
                        choices=['distmult', 'complex', 'transe'])

    parser.add_argument('--embedding_size', '-k', action='store', type=int, default=100)
    parser.add_argument('--batch_size', '-b', action='store', type=int, default=100)
    parser.add_argument('--eval_batch_size', '-B', action='store', type=int, default=None)

    # training params
    parser.add_argument('--epochs', '-e', action='store', type=int, default=100)
    parser.add_argument('--learning_rate', '-l', action='store', type=float, default=0.1)

    parser.add_argument('--optimizer', '-o', action='store', type=str, default='adagrad',
                        choices=['adagrad', 'adam', 'sgd'])

    parser.add_argument('--F2', action='store', type=float, default=None)
    parser.add_argument('--N3', action='store', type=float, default=None)

    parser.add_argument('--corruption', '-c', action='store', type=str, default='so',
                        choices=['so', 'spo'])

    parser.add_argument('--seed', action='store', type=int, default=0)

    parser.add_argument('--validate_every', '-V', action='store', type=int, default=None)

    parser.add_argument('--input_type', '-I', action='store', type=str, default='standard',
                        choices=['standard', 'reciprocal'])

    parser.add_argument('--blackbox_lambda', action='store', type=float, default=None)
    parser.add_argument('--mask', action='store_true', default=False)

    parser.add_argument('--load', action='store', type=str, default=None)
    parser.add_argument('--save', action='store', type=str, default=None)
    parser.add_argument('--use_wandb', '-wb', action='store', type=str, default='False', choices=['True', 'False'])

    parser.add_argument('--quiet', '-q', action='store_true', default=False)

    return parser.parse_args(argv)

def main(args):
    import pprint
    pprint.pprint(vars(args))

    train_path = args.train

    dev_path = args.dev
    test_path = args.test

    test_i_path = args.test_i
    test_ii_path = args.test_ii

    model_name = args.model
    optimizer_name = args.optimizer

    embedding_size = args.embedding_size

    batch_size = args.batch_size
    eval_batch_size = batch_size if args.eval_batch_size is None else args.eval_batch_size

    nb_epochs = args.epochs
    seed = args.seed

    learning_rate = args.learning_rate

    F2_weight = args.F2
    N3_weight = args.N3

    corruption = args.corruption

    validate_every = args.validate_every
    input_type = args.input_type

    blackbox_lambda = args.blackbox_lambda
    do_masking = args.mask

    load_path = args.load
    save_path = args.save
    use_wandb = args.use_wandb == 'True'

    is_quiet = args.quiet

    set_seed(seed)
    random_state = np.random.RandomState(seed)

    if use_wandb:
        wandb.init(entity="uclnlp", project="kbc_meta", group=f"base")
        wandb.config.update(args)
    print(' '.join(sys.argv))

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    if is_torch_tpu_available():
        import torch_xla.core.xla_model as xm
        torch.set_default_tensor_type('torch.FloatTensor')
        device = xm.xla_device()

    logger.info(f'Device: {device}')
    if use_wandb:
        wandb.config.update({'device': device})

    data = Data(train_path=train_path, dev_path=dev_path, test_path=test_path,
                test_i_path=test_i_path, test_ii_path=test_ii_path, input_type=input_type)

    # the .dev_triples attributes here contain ID strings
    triples_name_pairs = [
        (data.train_triples, 'train'),
        (data.dev_triples, 'dev'),
        (data.test_triples, 'test'),
        (data.test_i_triples, 'test-I'),
        (data.test_ii_triples, 'test-II'),
    ]

    rank = embedding_size * 2 if model_name in {'complex'} else embedding_size
    init_size = 1e-3

    # nn.Embedding using to a lookup table of embeddings (i.e. you can index entity_embeddings to return given entities embedding)
    # Nice explanation found in Escachator's answer here: https://stackoverflow.com/questions/50747947/embedding-in-pytorch
    entity_embeddings = nn.Embedding(data.nb_entities, rank, sparse=False).to(device)
    predicate_embeddings = nn.Embedding(data.nb_predicates, rank, sparse=False).to(device)

    # Downscale the randomly initialised embeddings (initialised with N(0,1))
    entity_embeddings.weight.data *= init_size
    predicate_embeddings.weight.data *= init_size

    parameters_lst = nn.ModuleDict({
        'entities': entity_embeddings,
        'predicates': predicate_embeddings
    }).to(device)

    # Optionally load embedding values (e.g. pretrained values)
    if load_path is not None:
        parameters_lst.load_state_dict(torch.load(load_path))

    # emb.weight is a tensor of shape (num_embeddings, rank)
    entity_t = entity_embeddings.weight
    predicate_t = predicate_embeddings.weight

    # When this dictionary is indexed by model name, the appropriate model class will be initialised
    model_factory = {
        'distmult': lambda: DistMult(entity_embeddings=entity_t, predicate_embeddings=predicate_t),
        'complex': lambda: ComplEx(entity_embeddings=entity_t, predicate_embeddings=predicate_t),
        'transe': lambda: TransE(entity_embeddings=entity_t, predicate_embeddings=predicate_t)
    }

    # Initialise correct model
    model = model_factory[model_name]().to(device)

    # Log the entity and predicate tensor sizes
    logger.info('Model state:')
    for param_tensor in parameters_lst.state_dict():
        logger.info(f'\t{param_tensor}\t{parameters_lst.state_dict()[param_tensor].size()}')

    # When this dictionary is indexed by optimizer name, the appropriate optimizer class will be initialised
    optimizer_factory = {
        'adagrad': lambda: optim.Adagrad(parameters_lst.parameters(), lr=learning_rate),
        'adam': lambda: optim.Adam(parameters_lst.parameters(), lr=learning_rate),
        'sgd': lambda: optim.SGD(parameters_lst.parameters(), lr=learning_rate)
    }

    assert optimizer_name in optimizer_factory
    optimizer = optimizer_factory[optimizer_name]()

    # Specify loss function (cross-entropy by default)
    if blackbox_lambda is None:
        loss_function = nn.CrossEntropyLoss(reduction='mean')
    else:
        loss_function = NegativeMRR(lmbda=blackbox_lambda)

    # Specify norm to use as regularizer
    F2_reg = F2() if F2_weight is not None else None
    N3_reg = N3() if N3_weight is not None else None

    # Masking is False by default
    mask_sp = mask_po = mask_so = None
    if do_masking:
        mask_sp, mask_po, mask_so = compute_masks(data.train_triples, data.train_triples, data.entity_to_idx, data.predicate_to_idx)

        mask_sp = torch.tensor(mask_sp, dtype=torch.long, device=device, requires_grad=False)
        mask_po = torch.tensor(mask_po, dtype=torch.long, device=device, requires_grad=False)
        mask_so = torch.tensor(mask_so, dtype=torch.long, device=device, requires_grad=False)

    # Training loop

    last_logged_epoch = 0
    best_mrr = 0

    for epoch_no in range(1, nb_epochs + 1):
        train_log = {}  # dictionary to store training metrics for uploading to wandb for each epoch
        batcher = Batcher(data.Xs, data.Xp, data.Xo, batch_size, 1, random_state)
        nb_batches = len(batcher.batches)

        epoch_loss_values = []  # to store loss for each batch in the epoch
        epoch_loss_nonreg_values = []

        for batch_no, (batch_start, batch_end) in enumerate(batcher.batches, 1):
            model.train()  # model in training mode

            # Size [B] numpy arrays containing indices of each subject_entity, predicate, and object_entity in the batch
            xp_batch, xs_batch, xo_batch, xi_batch = batcher.get_batch(batch_start, batch_end)

            xs_batch = torch.tensor(xs_batch, dtype=torch.long, device=device)
            xp_batch = torch.tensor(xp_batch, dtype=torch.long, device=device)
            xo_batch = torch.tensor(xo_batch, dtype=torch.long, device=device)
            xi_batch = torch.tensor(xi_batch, dtype=torch.long, device=device)

            # Return embeddings for each s, p, o in the batch
            # This returns tensors of shape (batch_size, rank)
            xp_batch_emb = predicate_embeddings(xp_batch)
            xs_batch_emb = entity_embeddings(xs_batch)
            xo_batch_emb = entity_embeddings(xo_batch)

            loss = 0.0

            # If corruption="spo", then loss will be calculate based on predicting subjects, predicates, and objects
            # If corruption="sp", then loss will be calculate based on just predicting subjects and objects
            if 's' in corruption:
                # shape of po_scores is (batch_size, Nb_preds in entire dataset)
                po_scores = model.forward(xp_batch_emb, None, xo_batch_emb)
                if do_masking is True:
                    po_scores = po_scores + mask_po[xi_batch, :]

                loss += loss_function(po_scores, xs_batch)

            if 'o' in corruption:
                # shape of sp_scores is (batch_size, Nb_entities in entire dataset)
                sp_scores = model.forward(xp_batch_emb, xs_batch_emb, None)
                if do_masking is True:
                    sp_scores = sp_scores + mask_sp[xi_batch, :]

                loss += loss_function(sp_scores, xo_batch)

            if 'p' in corruption:
                # shape of so_scores is (batch_size, Nb_entities in entire dataset)
                so_scores = model.forward(None, xs_batch_emb, xo_batch_emb)
                if do_masking is True:
                    so_scores = so_scores + mask_so[xi_batch, :]

                loss += loss_function(so_scores, xp_batch)

            loss_nonreg_value = loss.item()
            epoch_loss_nonreg_values += [loss_nonreg_value]

            factors = [model.factor(e) for e in [xp_batch_emb, xs_batch_emb, xo_batch_emb]]

            if F2_weight is not None:
                loss += F2_weight * F2_reg(factors)

            if N3_weight is not None:
                loss += N3_weight * N3_reg(factors)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            loss_value = loss.item()
            epoch_loss_values += [loss_value]

            if not is_quiet:
                logger.info(f'Epoch {epoch_no}/{nb_epochs}\tBatch {batch_no}/{nb_batches}\tLoss {loss_value:.6f} ({loss_nonreg_value:.6f})')
                # print(f'Epoch {epoch_no}/{nb_epochs}\tBatch {batch_no}/{nb_batches}\tLoss {loss_value:.6f} ({loss_nonreg_value:.6f})')

        loss_mean, loss_std = np.mean(epoch_loss_values), np.std(epoch_loss_values)
        loss_nonreg_mean, loss_nonreg_std = np.mean(epoch_loss_nonreg_values), np.std(epoch_loss_nonreg_values)

        train_log['loss_mean'] = loss_mean
        train_log['loss_std'] = loss_std
        train_log['loss_nonreg_mean'] = loss_nonreg_mean
        train_log['loss_nonreg_std'] = loss_nonreg_std
        logger.info(f'Epoch {epoch_no}/{nb_epochs}\tLoss {loss_mean:.4f} ± {loss_std:.4f} ({loss_nonreg_mean:.4f} ± {loss_nonreg_std:.4f})')
        # print(f'Epoch {epoch_no}/{nb_epochs}\tLoss {loss_mean:.4f} ± {loss_std:.4f} ({loss_nonreg_mean:.4f} ± {loss_nonreg_std:.4f})')

        if validate_every is not None and epoch_no % validate_every == 0:
            for triples, name in [(t, n) for t, n in triples_name_pairs if len(t) > 0]:
                model.eval()
                metrics = evaluate(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                                   test_triples=triples, all_triples=data.all_triples,
                                   entity_to_index=data.entity_to_idx, predicate_to_index=data.predicate_to_idx,
                                   model=model, batch_size=eval_batch_size, device=device)
                logger.info(f'Epoch {epoch_no}/{nb_epochs}\t{name} results\t{metrics_to_str(metrics)}')
                metrics_new = {f'{name}_{k}': v for k, v in metrics.items()} # hack to get different keys for logging
                train_log.update(metrics_new)
                # print(f'Epoch {epoch_no}/{nb_epochs}\t{name} results\t{metrics_to_str(metrics)}')
            last_logged_epoch = epoch_no

            if train_log['dev_MRR'] > best_mrr:
                best_mrr = train_log['dev_MRR']
                best_log=train_log

        if use_wandb:
            wandb.log(train_log, step=epoch_no, commit=True)

    if last_logged_epoch != nb_epochs:
        eval_log = {}
        for triples, name in [(t, n) for t, n in triples_name_pairs if len(t) > 0]:
            model.eval()
            metrics = evaluate(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                               test_triples=triples, all_triples=data.all_triples,
                               entity_to_index=data.entity_to_idx, predicate_to_index=data.predicate_to_idx,
                               model=model, batch_size=eval_batch_size, device=device)
            logger.info(f'Final \t{name} results\t{metrics_to_str(metrics)}')

            metrics_new = {f'{name}_{k}': v for k, v in metrics.items()}  # hack to get different keys for logging
            eval_log.update(metrics_new)
            # print(f'Final \t{name} results\t{metrics_to_str(metrics)}')

        if eval_log['dev_MRR'] > best_mrr:
            best_mrr = eval_log['dev_MRR']
            best_log=eval_log

        if use_wandb:
            wandb.log(eval_log, step=nb_epochs, commit=True)

    if use_wandb:
        wandb.run.summary.update(best_log)

    if save_path is not None:
        torch.save(parameters_lst.state_dict(), save_path)

    if use_wandb:
        wandb.save(f"{save_path[:-4]}.log")
        wandb.save("kbc_meta/logs/array.err")
        wandb.save("kbc_meta/logs/array.out")
        wandb.finish()

    logger.info("Training finished")
    # print("Training finished")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = parse_args(sys.argv[1:])
    main(args)
