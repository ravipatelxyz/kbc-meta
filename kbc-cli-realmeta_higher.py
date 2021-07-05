#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from copy import deepcopy

import argparse
from typing import List, Tuple

import multiprocessing

import higher
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim, Tensor

from kbc.util import is_torch_tpu_available, set_seed, make_batches

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


def get_unreg_loss(xs_batch: Tensor,
                 xp_batch: Tensor,
                 xo_batch: Tensor,
                 xi_batch: Tensor,
                 corruption: str,
                 entity_embeddings: Tensor,
                 predicate_embeddings: Tensor,
                 model,
                 loss_function: nn.CrossEntropyLoss,
                 masks=None) -> Tuple[Tensor, List[Tensor]]:

    # Return embeddings for each s, p, o in the batch
    # This returns tensors of shape (batch_size, rank)
    xp_batch_emb = predicate_embeddings[xp_batch]
    xs_batch_emb = entity_embeddings[xs_batch]
    xo_batch_emb = entity_embeddings[xo_batch]

    loss = 0.0

    # If corruption="spo", then loss will be calculate based on predicting subjects, predicates, and objects
    # If corruption="sp", then loss will be calculate based on just predicting subjects and objects
    if 's' in corruption:
        # shape of po_scores is (batch_size, Nb_preds in entire dataset)
        po_scores = model.forward(xp_batch_emb, None, xo_batch_emb, entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings)
        if masks is not None:
            po_scores = po_scores + masks[0][xi_batch, :]
        loss += loss_function(po_scores, xs_batch)

    if 'o' in corruption:
        # shape of sp_scores is (batch_size, Nb_entities in entire dataset)
        sp_scores = model.forward(xp_batch_emb, xs_batch_emb, None, entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings)
        if masks is not None:
            sp_scores = sp_scores + masks[1][xi_batch, :]
        loss += loss_function(sp_scores, xo_batch)

    if 'p' in corruption:
        # shape of so_scores is (batch_size, Nb_entities in entire dataset)
        so_scores = model.forward(None, xs_batch_emb, xo_batch_emb, entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings)
        if masks is not None:
            so_scores = so_scores + masks[2][xi_batch, :]
        loss += loss_function(so_scores, xp_batch)

    factors = [model.factor(e) for e in [xp_batch_emb, xs_batch_emb, xo_batch_emb]]

    return loss, factors


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
    parser.add_argument('--stopping_tol_inner', '-ti', action='store', type=float, default=None)
    parser.add_argument('--optimizer', '-o', action='store', type=str, default='adagrad',
                        choices=['adagrad', 'adam', 'sgd'])
    parser.add_argument('--regularizer', '-re', type=str, choices=["F2", "N3"], default="F2")
    parser.add_argument('--regweight_init', '-rw', type=float, default=0.001)
    # parser.add_argument('--F2', action='store', type=float, default=None)
    # parser.add_argument('--N3', action='store', type=float, default=None)

    parser.add_argument('--corruption', '-c', action='store', type=str, default='so',
                        choices=['so', 'spo'])
    parser.add_argument('--seed', action='store', type=int, default=0)
    parser.add_argument('--validate_every', '-V', action='store', type=int, default=None)
    parser.add_argument('--input_type', '-I', action='store', type=str, default='standard',
                        choices=['standard', 'reciprocal'])
    parser.add_argument('--do_masking_outer_loss', '-om', action='store', type=str, default='False', choices=['True', 'False'])

    parser.add_argument('--blackbox_lambda', action='store', type=float, default=None)

    # training params (outer loop)
    parser.add_argument('--optimizer_outer', '-oo', action='store', type=str, default='adam',
                        choices=['adagrad', 'adam', 'sgd'])
    parser.add_argument('--learning_rate_outer', '-lo', action='store', type=float, default=0.005)
    parser.add_argument('--outer_steps', '-os', action='store', type=int, default=100)
    parser.add_argument('--stopping_tol_outer', '-to', action='store', type=float, default=None)

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
    stopping_tol_inner = args.stopping_tol_inner
    # F2_weight = args.F2
    # N3_weight = args.N3
    regularizer = args.regularizer
    regweight_init = args.regweight_init
    corruption = args.corruption
    validate_every = args.validate_every
    input_type = args.input_type
    blackbox_lambda = args.blackbox_lambda

    optimizer_outer_name = args.optimizer_outer
    learning_rate_outer = args.learning_rate_outer
    outer_steps = args.outer_steps
    stopping_tol_outer = args.stopping_tol_outer
    do_masking_outer_loss = args.do_masking_outer_loss == 'True'

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
        (data.train_triples, 'train'),
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
    reg_weight_graph = torch.tensor(regweight_init, requires_grad=True).to(device)

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

    loss_function_outer = nn.CrossEntropyLoss(reduction='mean')

    # Masking

    masks_outer = None
    if do_masking_outer_loss:
        mask_outer_sp, mask_outer_po, mask_outer_so = compute_masks(data.dev_triples,
                                                                    data.train_triples + data.dev_triples,
                                                                    data.entity_to_idx,
                                                                    data.predicate_to_idx)

        mask_outer_po = torch.tensor(mask_outer_po, dtype=torch.long, device=device)
        mask_outer_sp = torch.tensor(mask_outer_sp, dtype=torch.long, device=device)
        mask_outer_so = torch.tensor(mask_outer_so, dtype=torch.long, device=device)
        masks_outer = [mask_outer_po, mask_outer_sp, mask_outer_so]

    # Training loop

    best_loss_outer = np.inf

    losses_outer = []
    losses_inner_final_epoch = []
    e_vals = []
    p_vals = []
    reg_weight_vals = []

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

        losses_inner = []
        losses_inner_dev = []
        for epoch_no in range(1, nb_epochs + 1):
            train_log = {}  # dictionary to store training metrics for uploading to wandb for each epoch
            batcher = Batcher(data.Xs, data.Xp, data.Xo, batch_size, 1, random_state)

            epoch_loss_values = []  # to store loss for each batch in the epoch
            epoch_loss_nonreg_values = []

            for batch_no, (batch_start, batch_end) in enumerate(batcher.batches, 1):

                # Size [B] numpy arrays containing indices of each subject_entity, predicate, and object_entity in the batch
                xp_batch, xs_batch, xo_batch, xi_batch = batcher.get_batch(batch_start, batch_end)

                xs_batch = torch.tensor(xs_batch, dtype=torch.long, device=device)
                xp_batch = torch.tensor(xp_batch, dtype=torch.long, device=device)
                xo_batch = torch.tensor(xo_batch, dtype=torch.long, device=device)

                loss, factors = get_unreg_loss(xs_batch=xs_batch,
                                               xp_batch=xp_batch,
                                               xo_batch=xo_batch,
                                               xi_batch=xi_batch,
                                               corruption=corruption,
                                               entity_embeddings=e_graph,
                                               predicate_embeddings=p_graph,
                                               model=model,
                                               loss_function=loss_function)

                loss_nonreg_value = loss.item()
                epoch_loss_nonreg_values += [loss_nonreg_value]

                if regularizer == "F2":
                    loss += reg_weight_graph * F2_reg(factors)

                if regularizer == "N3":
                    loss += reg_weight_graph * N3_reg(factors)

                e_graph, p_graph = diffopt.step(loss, params=[e_graph, p_graph])

                loss_value = loss.item()
                epoch_loss_values += [loss_value]

            loss_mean, loss_std = np.mean(epoch_loss_values), np.std(epoch_loss_values)
            loss_nonreg_mean, loss_nonreg_std = np.mean(epoch_loss_nonreg_values), np.std(epoch_loss_nonreg_values)
            if not is_quiet:
                # logger.info(f'Epoch {epoch_no}/{nb_epochs}\tLoss {loss_mean:.4f} ± {loss_std:.4f} ({loss_nonreg_mean:.4f} ± {loss_nonreg_std:.4f})')
                print(f'Epoch {epoch_no}/{nb_epochs}\tLoss {loss_mean:.5f} ± {loss_std:.5f} ({loss_nonreg_mean:.5f} ± {loss_nonreg_std:.5f})')

            losses_inner += [loss_nonreg_mean]

            if stopping_tol_inner is not None:
                xs_dev = torch.tensor(data.dev_Xs, dtype=torch.long, device=device)
                xp_dev = torch.tensor(data.dev_Xp, dtype=torch.long, device=device)
                xo_dev = torch.tensor(data.dev_Xo, dtype=torch.long, device=device)
                xi_dev = data.dev_Xi

                loss_inner_dev, _ = get_unreg_loss(xs_batch=xs_dev,
                                               xp_batch=xp_dev,
                                               xo_batch=xo_dev,
                                               xi_batch=xi_dev,
                                               corruption=corruption,
                                               entity_embeddings=e_graph,
                                               predicate_embeddings=p_graph,
                                               model=model,
                                               loss_function=loss_function_outer,
                                               masks=masks_outer)
                losses_inner_dev += [loss_inner_dev.item()]

                if epoch_no > 20 and np.mean(losses_inner_dev[-10:-5]) - np.mean(losses_inner_dev[-5:]) < stopping_tol_inner:
                    print(f"Num inner steps: {epoch_no}")
                    break

        # plt.plot(losses_inner_dev)
        # plt.show()

            # if epoch_no > 20 and np.mean(losses_inner[-20:-10]) - np.mean(losses_inner[-10:]) < stopping_tol_inner:
            #     print(epoch_no)
            #     break

        if outer_step == 0 or outer_step == outer_steps-1:
            plt.figure()
            plt.plot(losses_inner)
            if stopping_tol_inner is not None:
                plt.plot(losses_inner_dev)
                plt.legend(["training loss", "dev loss"])
            plt.xlabel("Epoch (inner step)")
            plt.ylabel("Inner loss")
            plt.title("Inner loss by epoch for first inner loop")
            plt.show()

        losses_inner_final_epoch += [losses_inner[-1]]

        xs_dev = torch.tensor(data.dev_Xs, dtype=torch.long, device=device)
        xp_dev = torch.tensor(data.dev_Xp, dtype=torch.long, device=device)
        xo_dev = torch.tensor(data.dev_Xo, dtype=torch.long, device=device)
        xi_dev = data.dev_Xi

        loss_outer, _ = get_unreg_loss(xs_batch=xs_dev,
                                       xp_batch=xp_dev,
                                       xo_batch=xo_dev,
                                       xi_batch=xi_dev,
                                       corruption=corruption,
                                       entity_embeddings=e_graph,
                                       predicate_embeddings=p_graph,
                                       model=model,
                                       loss_function=loss_function_outer,
                                       masks=masks_outer)

        # store a copy of best embeddings
        if loss_outer < best_loss_outer:
            best_loss_outer = loss_outer
            best_reg_weight = reg_weight_graph.detach().clone()
            best_e_graph = e_graph.detach().clone()
            best_p_graph = p_graph.detach().clone()
            best_outer_step = outer_step

        # for plotting only
        losses_outer += [loss_outer.detach().clone().item()]
        e_vals += [torch.norm(e_graph.detach().clone())]
        p_vals += [torch.norm(p_graph.detach().clone())]
        reg_weight_vals += [reg_weight_graph.detach().clone()]

        print(f"meta loss: {loss_outer.item()}")
        print(f"reg param: {reg_weight_graph.item()}")

        loss_outer.backward()
        optimizer_outer.step()
        optimizer_outer.zero_grad()

        if use_wandb == True:
            wandb.run.summary.update(best_log)

        if save_path is not None:
            torch.save(parameters_lst.state_dict(), save_path)

    # eval_log = {}
    # Final outer step metrics
    print(f"Final \touter step: {outer_steps} \treg param: {reg_weight_vals[-1]} \touter loss {losses_outer[-1]}")
    for triples, name in [(t, n) for t, n in triples_name_pairs if len(t) > 0]:
        metrics = evaluate(entity_embeddings=e_graph, predicate_embeddings=p_graph,
                           test_triples=triples, all_triples=data.all_triples,
                           entity_to_index=data.entity_to_idx, predicate_to_index=data.predicate_to_idx,
                           model=model, batch_size=eval_batch_size, device=device)
        # logger.info(f'Final \t{name} results\t{metrics_to_str(metrics)}')

        # metrics_new = {f'{name}_{k}': v for k, v in metrics.items()}  # hack to get different keys for logging
        # eval_log.update(metrics_new)
        print(f'Final \t{name} results\t{metrics_to_str(metrics)}')

    # eval_log = {}
    # Best outer step metrics (i.e. step with lowest outer loss)
    print(f"Best \touter step: {best_outer_step+1} \treg param: {best_reg_weight} \touter loss {best_loss_outer}")
    for triples, name in [(t, n) for t, n in triples_name_pairs if len(t) > 0]:
        metrics = evaluate(entity_embeddings=best_e_graph, predicate_embeddings=best_p_graph,
                           test_triples=triples, all_triples=data.all_triples,
                           entity_to_index=data.entity_to_idx, predicate_to_index=data.predicate_to_idx,
                           model=model, batch_size=eval_batch_size, device=device)
        # logger.info(f'Best \t{name} results\t{metrics_to_str(metrics)}')

        # metrics_new = {f'{name}_{k}': v for k, v in metrics.items()}  # hack to get different keys for logging
        # eval_log.update(metrics_new)
        print(f'Best \t{name} results \t{metrics_to_str(metrics)}')

    if use_wandb == True:
        wandb.log(eval_log, step=nb_epochs, commit=True)

    if use_wandb == True:
        wandb.save(f"{save_path[:-4]}.log")
        wandb.save("kbc_meta/logs/array.err")
        wandb.save("kbc_meta/logs/array.out")
        wandb.finish()

    plt.figure(2)
    plt.plot(losses_inner_final_epoch)
    plt.xlabel("Outer step")
    plt.ylabel("Inner loss at final epoch of inner loop on training set")
    plt.title("Final epoch (inner step) loss by outer step")
    plt.show()

    plt.figure(3)
    plt.plot(losses_outer)
    plt.xlabel("Outer step")
    plt.ylabel("Outer loss on dev set")
    plt.title(f"Outer losses\nembedding size: {embedding_size} | batch_size: {batch_size} | reg: {regularizer} | reg weight init: {regweight_init}\nepochs: {nb_epochs} | innerOpt: {optimizer_name} | outerOpt: {optimizer_outer_name} | LR: {learning_rate} | outerLR: {learning_rate_outer}")
    plt.show()

    plt.figure(4)
    plt.plot(reg_weight_vals)
    plt.xlabel("Outer step")
    plt.ylabel("Regularisaton weight value")
    plt.title(f"Regularisation weight values (Start value: {regweight_init})")
    plt.show()

    plt.figure(5)
    plt.plot(e_vals)
    plt.plot(p_vals)
    plt.legend(["entities", "predicates"])
    plt.xlabel("Outer step")
    plt.ylabel("L2 norm of embeddings")
    plt.title("Entity and predicate embedding norms")
    plt.show()

    logger.info("Training finished")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = parse_args(sys.argv[1:])
    print(' '.join(sys.argv))
    main(args)
