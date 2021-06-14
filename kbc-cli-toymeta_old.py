#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
import sys
import os

import multiprocessing
import numpy as np

import torch
from torch import nn, optim

from kbc.util import set_seed

from kbc.training.data import Data
from kbc.training.batcher import Batcher

from kbc.models import DistMult, ComplEx, TransE

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))
torch.set_num_threads(multiprocessing.cpu_count())

# %%

def p_target_regularizer(entity_embedding, reg_param):
    reg = torch.norm(entity_embedding-reg_param)

    return reg


# %%

def main():

    # args
    train_path = TRAIN_DIR
    dev_path = DEV_DIR
    test_path = TEST_DIR
    model_name = MODEL
    optimizer_name = OPTIMIZER
    optimizer_outer_name = OPTIMIZER_OUTER
    embedding_size = EMBEDDING_SIZE
    batch_size = BATCH_SIZE
    nb_epochs = EPOCHS
    outer_steps = OUTER_STEPS
    learning_rate = LEARNING_RATE
    learning_rate_outer = LEARNING_RATE_OUTER
    regularizer = REGULARIZER
    input_type = INPUT_TYPE
    is_quiet = QUIET
    seed = SEED
    set_seed(seed)
    random_state = np.random.RandomState(seed)

    device = torch.device('cpu')

    # Load toy dataset
    data = Data(train_path=train_path, dev_path=dev_path, test_path=test_path,
                test_i_path=None, test_ii_path=None, input_type=input_type)

    print(data.entity_to_idx)  # {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    # the .dev_triples attributes here contain ID strings
    triples_name_pairs = [
        (data.dev_triples, 'dev'),
        (data.test_triples, 'test'),
    ]

    rank = embedding_size * 2 if model_name in {'complex'} else embedding_size
    init_size = 1

    # nn.Embedding using to a lookup table of embeddings (i.e. you can index entity_embeddings to return given entities embedding)
    # Nice explanation found in Escachator's answer here: https://stackoverflow.com/questions/50747947/embedding-in-pytorch
    entity_embeddings = nn.Embedding(data.nb_entities, rank, sparse=True).to(device)
    predicate_embeddings = nn.Embedding(data.nb_predicates, rank, sparse=True).to(device)

    # Downscale the randomly initialised embeddings (initialised with N(0,1))
    entity_embeddings.weight.data *= init_size
    predicate_embeddings.weight.data *= init_size

    # Specify regularization term
    if regularizer == "p_target":
        reg_param = nn.Embedding(1, rank).to(device)
        reg_param.weight.data *= init_size
        print(f"STARTING reg param value, p: {reg_param.weight}")
        print(f"STARTING reg term: {p_target_regularizer(entity_embeddings.weight[1], reg_param.weight)}")

    parameters_lst = nn.ModuleDict({
        'entities': entity_embeddings,
        'predicates': predicate_embeddings
    }).to(device)

    parameters_lst_outer = nn.ModuleDict({
        'reg_param': reg_param
    }).to(device)

    # emb.weight is a tensor of shape (num_embeddings, rank)
    entity_t = entity_embeddings.weight
    predicate_t = predicate_embeddings.weight

    print(f"STARTING entity embeddings:\n {entity_t}")
    print(f"STARTING predicate embeddings:\n {predicate_t}")

    # When this dictionary is indexed by model name, the appropriate model class will be initialised
    model_factory = {
        'distmult': lambda: DistMult(entity_embeddings=entity_t, predicate_embeddings=predicate_t),
        'complex': lambda: ComplEx(entity_embeddings=entity_t, predicate_embeddings=predicate_t),
        'transe': lambda: TransE(entity_embeddings=entity_t, predicate_embeddings=predicate_t)
    }

    # Initialise correct model
    model = model_factory[model_name]().to(device)

    # When this dictionary is indexed by optimizer name, the appropriate optimizer class will be initialised
    optimizer_factory = {
        'adagrad': lambda: optim.Adagrad(parameters_lst.parameters(), lr=learning_rate),
        'adam': lambda: optim.Adam(parameters_lst.parameters(), lr=learning_rate),
        'sgd': lambda: optim.SGD(parameters_lst.parameters(), lr=learning_rate)
    }

    optimizer_factory_outer = {
        'adagrad': lambda: optim.Adagrad(parameters_lst_outer.parameters(), lr=learning_rate_outer),
        'adam': lambda: optim.Adam(parameters_lst_outer.parameters(), lr=learning_rate_outer),
        'sgd': lambda: optim.SGD(parameters_lst_outer.parameters(), lr=learning_rate_outer)
    }

    assert optimizer_name in optimizer_factory
    optimizer = optimizer_factory[optimizer_name]()
    optimizer_outer = optimizer_factory_outer[optimizer_outer_name]()

    # Specify loss function (cross-entropy by default)
    loss_function = nn.CrossEntropyLoss(reduction='mean')

    # outer loop
    for outer_step in range(outer_steps):
        # inner loop
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

                # Return embeddings for each s, p, o in the batch
                # This returns tensors of shape (batch_size, rank)
                xp_batch_emb = predicate_embeddings(xp_batch)
                xs_batch_emb = entity_embeddings(xs_batch)
                xo_batch_emb = entity_embeddings(xo_batch)

                loss = 0.0

                # "sp" corruption applied here (i.e. loss calculated based on model predications for subjects and objects)
                # shape of po_scores is (batch_size, Nb_entities in entire dataset)
                po_scores = model.forward(xp_batch_emb, None, xo_batch_emb)
                non_c_idx = [i for i in range(po_scores.shape[1]) if i != data.entity_to_idx['C']]
                xs_batch_c_removed = torch.where(xs_batch > data.entity_to_idx['C'], xs_batch-1, xs_batch)
                loss += loss_function(po_scores[:, non_c_idx], xs_batch_c_removed)  # train loss ignoring <A,r,C> terms

                # shape of sp_scores is (batch_size, Nb_entities in entire dataset)
                sp_scores = model.forward(xp_batch_emb, xs_batch_emb, None)
                xo_batch_c_removed = torch.where(xo_batch > data.entity_to_idx['C'], xo_batch - 1, xo_batch)
                loss += loss_function(sp_scores[:, non_c_idx], xo_batch_c_removed)  # train loss ignoring <A,r,C> terms

                # store loss
                loss_nonreg_value = loss.item()
                epoch_loss_nonreg_values += [loss_nonreg_value]

                # add on regularization term ||embedding(B)-reg_param||
                loss += p_target_regularizer(entity_embeddings(torch.tensor(data.entity_to_idx['B'])), reg_param.weight)

                # compute gradient for inner-loop (training backprop)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                loss_value = loss.item()
                epoch_loss_values += [loss_value]

                # print(f"Embedding B:\n {entity_embeddings.weight[1]}")
                # print(f"Regulariser param p:\n {reg_param.weight}")
                # print(f"reg term: {p_target_regularizer(entity_embeddings.weight[1], reg_param.weight)}")

                if not is_quiet:
                    # logger.info(f'Epoch {epoch_no}/{nb_epochs}\tBatch {batch_no}/{nb_batches}\tLoss {loss_value:.6f} ({loss_nonreg_value:.6f})')
                    print(f'Epoch {epoch_no}/{nb_epochs}\tBatch {batch_no}/{nb_batches}\tLoss {loss_value:.6f} ({loss_nonreg_value:.6f})')

            # loss_mean, loss_std = np.mean(epoch_loss_values), np.std(epoch_loss_values)
            # loss_nonreg_mean, loss_nonreg_std = np.mean(epoch_loss_nonreg_values), np.std(epoch_loss_nonreg_values)
            # train_log['entity_embeddings'] = entity_embeddings.weight
            # train_log['predicate_embeddings'] = predicate_embeddings.weight
            # train_log['reg_hyperparam_values'] = reg_param.weight
            # train_log['reg_term_norm_value'] = p_target_regularizer(entity_embeddings.weight[1], reg_param.weight)
            # train_log['loss_mean'] = loss_mean
            # train_log['loss_std'] = loss_std
            # train_log['loss_nonreg_mean'] = loss_nonreg_mean
            # train_log['loss_nonreg_std'] = loss_nonreg_std
            # logger.info(f'Epoch {epoch_no}/{nb_epochs}\tLoss {loss_mean:.4f} ± {loss_std:.4f} ({loss_nonreg_mean:.4f} ± {loss_nonreg_std:.4f})')
            # print(f'Epoch {epoch_no}/{nb_epochs}\tLoss {loss_mean:.4f} ± {loss_std:.4f} ({loss_nonreg_mean:.4f} ± {loss_nonreg_std:.4f})')


        # dev_batcher = Batcher(data.dev_Xs, data.dev_Xp, data.dev_Xo, 1, 1, random_state)
        # batch_meta_loss_values = [] # to store meta loss for each triple
        #
        # for batch_no, (batch_start, batch_end) in enumerate(dev_batcher.batches, 1):
        #
        #     # Size [B] numpy arrays containing indices of each subject_entity, predicate, and object_entity in the batch
        #     dev_xp_batch, dev_xs_batch, dev_xo_batch, dev_xi_batch = dev_batcher.get_batch(batch_start, batch_end)
        #
        #     dev_xs_batch = torch.tensor(dev_xs_batch, dtype=torch.long, device=device)
        #     dev_xp_batch = torch.tensor(dev_xp_batch, dtype=torch.long, device=device)
        #     dev_xo_batch = torch.tensor(dev_xo_batch, dtype=torch.long, device=device)
        #
        #     # Return embeddings for each s, p, o in the batch
        #     # This returns tensors of shape (batch_size, rank)
        #     dev_xp_batch_emb = predicate_embeddings(dev_xp_batch)
        #     dev_xs_batch_emb = entity_embeddings(dev_xs_batch)
        #     dev_xo_batch_emb = entity_embeddings(dev_xo_batch)
        #
        #     meta_loss = 0.0
        #
        #     # "sp" corruption applied here (i.e. loss calculated based on model predications for subjects and objects)
        #     # shape of po_scores is (batch_size, Nb_entities in entire dataset)
        #     po_scores = model.forward(dev_xp_batch_emb, None, dev_xo_batch_emb)
        #     non_b_idx = [i for i in range(po_scores.shape[1]) if i != data.entity_to_idx['B']]
        #     dev_xs_batch_b_removed = torch.where(dev_xs_batch > data.entity_to_idx['B'], dev_xs_batch - 1, xs_batch)
        #     meta_loss += loss_function(po_scores[:, non_b_idx],
        #                                dev_xs_batch_b_removed)
        #
        #     # shape of sp_scores is (batch_size, Nb_entities in entire dataset)
        #     sp_scores = model.forward(dev_xp_batch_emb, dev_xs_batch_emb, None)
        #     dev_xo_batch_b_removed = torch.where(dev_xo_batch > data.entity_to_idx['B'], dev_xo_batch - 1, xo_batch)
        #     meta_loss += loss_function(sp_scores[:, non_b_idx],
        #                                dev_xo_batch_b_removed)
        #
        #
        #     # compute gradient for outer-loop
        #     meta_loss.backward()
        #
        #     optimizer_outer.step()
        #     optimizer_outer.zero_grad()
        #
        #     # store loss
        #     batch_meta_loss_values += [meta_loss.item()]


    # logger.info("Training finished")
    print("\nTraining finished\n")

    print(f"FINAL entity embeddings: {entity_embeddings.weight}")
    print(f"FINAL predicate embeddings: {predicate_embeddings.weight}")
    print(f"FINAL reg param value, p: {reg_param.weight}")
    print(f"FINAL reg term: {p_target_regularizer(entity_embeddings.weight[1], reg_param.weight)}")

if __name__ == '__main__':

    # Specify experimental parameters
    TRAIN_DIR = "./data/toy/train.tsv"
    DEV_DIR = "./data/toy/dev.tsv"
    TEST_DIR = None  # "./data/toy/dev.tsv"
    MODEL = "distmult"
    EMBEDDING_SIZE = 50
    BATCH_SIZE = 2
    EPOCHS = 500
    OUTER_STEPS = 1
    LEARNING_RATE = 0.1
    LEARNING_RATE_OUTER = 0.001
    OPTIMIZER = "adagrad"
    OPTIMIZER_OUTER = "adagrad"
    REGULARIZER = "p_target"
    INPUT_TYPE = "standard"
    SEED= 5
    QUIET = True

    main()