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
    embedding_size = EMBEDDING_SIZE
    batch_size = BATCH_SIZE
    nb_epochs = EPOCHS
    range_values_regparam_grid = REG_PARAM_VALUES_RANGE
    outer_steps = OUTER_STEPS
    learning_rate = LEARNING_RATE
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

    rank = embedding_size * 2 if model_name in {'complex'} else embedding_size
    init_size = 1

    ent_embs = torch.normal(0, 1, (data.nb_entities, rank)).detach()
    pred_embs = torch.normal(0, 1, (data.nb_predicates, rank)).detach()
    print(f"\nSTARTING entity embeddings:\n {ent_embs}")
    print(f"STARTING predicate embeddings:\n {pred_embs}\n")

    # outer loop
    mean_meta_losses = []
    best_meta_loss = 999  # arbitrary large number to initialise
    best_reg_term = 999  # arbitrary large number to initialise

    for outer_step in range(outer_steps):

        # nn.Embedding using to a lookup table of embeddings (i.e. you can index entity_embeddings to return given entities embedding)
        # Nice explanation found in Escachator's answer here: https://stackoverflow.com/questions/50747947/embedding-in-pytorch
        entity_embeddings = nn.Embedding(data.nb_entities, rank, sparse=False).to(device)
        predicate_embeddings = nn.Embedding(data.nb_predicates, rank, sparse=False).to(device)

        # Set embeddings values the same in every outer loop so that each
        entity_embeddings.weight = nn.Parameter(ent_embs.detach().clone())
        predicate_embeddings.weight = nn.Parameter(pred_embs.detach().clone())

        # Downscale the randomly initialised embeddings (initialised with N(0,1))
        entity_embeddings.weight.data *= init_size
        predicate_embeddings.weight.data *= init_size

        parameters_lst = nn.ModuleDict({
            'entities': entity_embeddings,
            'predicates': predicate_embeddings
        }).to(device)

        # When this dictionary is indexed by model name, the appropriate model class will be initialised
        model_factory = {
            'distmult': lambda: DistMult(entity_embeddings=entity_embeddings.weight,
                                         predicate_embeddings=predicate_embeddings.weight),
            'complex': lambda: ComplEx(entity_embeddings=entity_embeddings.weight,
                                       predicate_embeddings=predicate_embeddings.weight),
            'transe': lambda: TransE(entity_embeddings=entity_embeddings.weight,
                                     predicate_embeddings=predicate_embeddings.weight)
        }

        # Initialise correct model
        model = model_factory[model_name]().to(device)

        # When this dictionary is indexed by optimizer name, the appropriate optimizer class will be initialised
        optimizer_factory = {
            'adagrad': lambda: optim.Adagrad(parameters_lst.parameters(), lr=learning_rate),
            'adam': lambda: optim.Adam(parameters_lst.parameters(), lr=learning_rate),
            'sgd': lambda: optim.SGD(parameters_lst.parameters(), lr=learning_rate)
        }

        assert optimizer_name in optimizer_factory
        optimizer = optimizer_factory[optimizer_name]()

        # Specify loss function (cross-entropy by default)
        loss_function = nn.CrossEntropyLoss(reduction='mean')

        # inner loop
        mean_losses = []
        reg_param = 2 * range_values_regparam_grid * (torch.rand(rank) - 0.5)
        print(f"Random reg param (p) value {outer_step}: {reg_param}")
        reg_param.requires_grad = False

        for epoch_no in range(1, nb_epochs + 1):
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
                xo_batch_c_removed = torch.where(xo_batch > data.entity_to_idx['C'], xo_batch-1, xo_batch)
                loss += loss_function(sp_scores[:, non_c_idx], xo_batch_c_removed)  # train loss ignoring <A,r,C> terms

                # store loss
                loss_nonreg_value = loss.item()
                epoch_loss_nonreg_values += [loss_nonreg_value]

                # add on regularization term ||embedding(B)-reg_param||
                reg_term = p_target_regularizer(entity_embeddings(torch.tensor(data.entity_to_idx['B'])), reg_param)
                loss += reg_term

                # compute gradient for inner-loop (training backprop)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                loss_value = loss.item()
                epoch_loss_values += [loss_value]

                if not is_quiet:
                    # logger.info(f'Epoch {epoch_no}/{nb_epochs}\tBatch {batch_no}/{nb_batches}\tLoss {loss_value:.6f} ({loss_nonreg_value:.6f})')
                    print(f'Epoch {epoch_no}/{nb_epochs}\tBatch {batch_no}/{nb_batches}\tLoss {loss_value:.6f} ({loss_nonreg_value:.6f})')

            loss_mean, loss_std = np.mean(epoch_loss_values), np.std(epoch_loss_values)
            mean_losses += [loss_mean]

        with torch.no_grad():
            model.eval()
            dev_batcher = Batcher(data.dev_Xs, data.dev_Xp, data.dev_Xo, 1, 1, random_state)
            batch_meta_loss_values = [] # to store meta loss for each triple

            for batch_no, (batch_start, batch_end) in enumerate(dev_batcher.batches, 1):

                # Size [B] numpy arrays containing indices of each subject_entity, predicate, and object_entity in the batch
                dev_xp_batch, dev_xs_batch, dev_xo_batch, dev_xi_batch = dev_batcher.get_batch(batch_start, batch_end)

                dev_xs_batch = torch.tensor(dev_xs_batch, dtype=torch.long, device=device)
                dev_xp_batch = torch.tensor(dev_xp_batch, dtype=torch.long, device=device)
                dev_xo_batch = torch.tensor(dev_xo_batch, dtype=torch.long, device=device)

                # Return embeddings for each s, p, o in the batch
                # This returns tensors of shape (batch_size, rank)
                dev_xp_batch_emb = predicate_embeddings(dev_xp_batch)
                dev_xs_batch_emb = entity_embeddings(dev_xs_batch)
                dev_xo_batch_emb = entity_embeddings(dev_xo_batch)

                meta_loss = 0.0

                # "sp" corruption applied here (i.e. loss calculated based on model predications for subjects and objects)
                # shape of po_scores is (batch_size, Nb_entities in entire dataset)
                dev_po_scores = model.forward(dev_xp_batch_emb, None, dev_xo_batch_emb)
                non_b_idx = [i for i in range(dev_po_scores.shape[1]) if i != data.entity_to_idx['B']]
                dev_xs_batch_b_removed = torch.where(dev_xs_batch > data.entity_to_idx['B'], dev_xs_batch - 1, dev_xs_batch)
                meta_loss += loss_function(dev_po_scores[:, non_b_idx], dev_xs_batch_b_removed)

                # shape of sp_scores is (batch_size, Nb_entities in entire dataset)
                dev_sp_scores = model.forward(dev_xp_batch_emb, dev_xs_batch_emb, None)
                dev_xo_batch_b_removed = torch.where(dev_xo_batch > data.entity_to_idx['B'], dev_xo_batch - 1, dev_xo_batch)
                meta_loss += loss_function(dev_sp_scores[:, non_b_idx], dev_xo_batch_b_removed)

                # store loss
                batch_meta_loss_values += [meta_loss.item()]

                if reg_term < best_reg_term:
                    best_reg_term = reg_term
                    best_reg_term_params = reg_param.detach()
                    best_reg_term_entity_embeddings = entity_embeddings.weight.detach().detach()
                    best_reg_term_pred_embeddings = predicate_embeddings.weight.detach().detach()
                    best_reg_term_loss = mean_losses[-1]

            meta_loss_mean, meta_loss_std = np.mean(batch_meta_loss_values), np.std(batch_meta_loss_values)

            # print("\n")
            # print(f"inner loop loss: {mean_losses[-1]}")
            # print(f"meta loss: {meta_loss.item()}")
            # # print(f"batch meta loss: {batch_meta_loss_values[-1]}")
            # # print(f"meta loss mean: {meta_loss_mean}")
            # print(f"reg param: {reg_param}")
            # print(f"reg term: {reg_term}")
            # print(f"entity embeddings: {entity_embeddings.weight}")

            if meta_loss_mean < best_meta_loss:
                best_meta_loss = meta_loss_mean
                best_reg_param = reg_param.clone()
                best_entity_embeddings = entity_embeddings.weight.detach().clone()
                best_pred_embeddings = predicate_embeddings.weight.detach().clone()
            mean_meta_losses += [meta_loss_mean]

    # logger.info("Training finished")
    print("\nTraining finished\n")

    print(f"Best meta loss: {best_meta_loss}")
    print(f"Corresponding reg param (p) value (based on meta-loss): {best_reg_param}")
    print(f"Corresponding entity embeddings (based on meta-loss): {best_entity_embeddings}")
    print(f"Corresponding predicate embeddings (based on meta-loss): {best_pred_embeddings}")

    print(f"\nBest reg term value (||emb(B)-p||): {best_reg_term}")
    print(f"Corresponding reg param (p) value (based on reg term value): {best_reg_term_params}")
    print(f"Corresponding entity embeddings (based on reg term value): {best_reg_term_entity_embeddings}")
    print(f"Corresponding predicate embeddings (based on meta-loss): {best_reg_term_pred_embeddings}")

if __name__ == '__main__':

    # Specify experimental parameters
    TRAIN_DIR = "./data/toy/train.tsv"
    DEV_DIR = "./data/toy/dev.tsv"
    TEST_DIR = None  # "./data/toy/dev.tsv"
    MODEL = "distmult"
    EMBEDDING_SIZE = 1
    BATCH_SIZE = 2
    EPOCHS = 500
    REG_PARAM_VALUES_RANGE = 3
    OUTER_STEPS = 50
    LEARNING_RATE = 0.05
    OPTIMIZER = "adagrad"
    INPUT_TYPE = "standard"
    SEED = 4
    QUIET = True

    main()