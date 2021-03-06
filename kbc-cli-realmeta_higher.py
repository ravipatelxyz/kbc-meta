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

import logging
import wandb
import time

logger = logging.getLogger(os.path.basename(sys.argv[0]))
torch.set_num_threads(multiprocessing.cpu_count())
plt.rcParams['figure.dpi'] = 200

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

    # list of length 3, each element containing tensor of shape batch_size x rank
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

    parser.add_argument('--corruption', '-c', action='store', type=str, default='so',
                        choices=['so', 'spo'])
    parser.add_argument('--seed', action='store', type=int, default=0)
    parser.add_argument('--validate_all', '-V', action='store', type=str, default='False', choices=['True', 'False'])
    parser.add_argument('--input_type', '-I', action='store', type=str, default='standard',
                        choices=['standard', 'reciprocal'])
    parser.add_argument('--do_masking_dev_loss', '-md', action='store', type=str, default='False', choices=['True', 'False'])
    parser.add_argument('--do_masking_train_loss', '-mt', action='store', type=str, default='False',
                        choices=['True', 'False'])

    # training params (outer loop)
    parser.add_argument('--optimizer_outer', '-oo', action='store', type=str, default='adam',
                        choices=['adagrad', 'adam', 'sgd'])
    parser.add_argument('--learning_rate_outer', '-lo', action='store', type=float, default=0.005)
    parser.add_argument('--outer_steps', '-os', action='store', type=int, default=100)
    parser.add_argument('--stopping_tol_outer', '-to', action='store', type=float, default=None)
    parser.add_argument('--grad_clip_val_outer', '-gc', action='store', type=float, default=None)
    parser.add_argument('--regweight_rescaler', '-rr', action='store', type=float, default=1)
    parser.add_argument('--regweight_rescaler_tol', '-rt', action='store', type=float, default=1e-10)

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
    regularizer = args.regularizer
    regweight_init = args.regweight_init
    corruption = args.corruption
    validate_all = args.validate_all == 'True'
    input_type = args.input_type

    optimizer_outer_name = args.optimizer_outer
    learning_rate_outer = args.learning_rate_outer
    outer_steps = args.outer_steps
    stopping_tol_outer = args.stopping_tol_outer
    do_masking_dev_loss = args.do_masking_dev_loss == 'True'
    do_masking_train_loss = args.do_masking_train_loss == 'True'
    grad_clip_val_outer = args.grad_clip_val_outer
    regweight_rescaler = args.regweight_rescaler
    regweight_rescaler_tol = args.regweight_rescaler_tol

    load_path = args.load
    save_path = args.save
    save_figs = args.save_figs == 'True'
    use_wandb = args.use_wandb == 'True'

    timestr = time.strftime("%Y%m%d-%H%M%S")

    is_quiet = args.quiet

    set_seed(seed)
    random_state = np.random.RandomState(seed)

    if use_wandb:
        wandb.init(entity="uclnlp", project="kbc_meta", group=f"realmeta_single_nations")
        wandb.config.update(args)
    logger.info(' '.join(sys.argv))

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    if is_torch_tpu_available():
        import torch_xla.core.xla_model as xm
        torch.set_default_tensor_type('torch.FloatTensor')
        device = xm.xla_device()

    print(device)

    logger.info(f'Device: {device}')
    if use_wandb:
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
    reg_weight_graph = torch.tensor(np.log(regweight_init), requires_grad=True, device=device)

    optimizer_factory_outer = {
        'adagrad': lambda: optim.Adagrad([reg_weight_graph], lr=learning_rate_outer),
        'adam': lambda: optim.Adam([reg_weight_graph], lr=learning_rate_outer),
        'sgd': lambda: optim.SGD([reg_weight_graph], lr=learning_rate_outer)
    }

    optimizer_outer = optimizer_factory_outer[optimizer_outer_name]()

    if grad_clip_val_outer is not None:
        reg_weight_graph.register_hook(lambda grad: torch.clamp(grad, -grad_clip_val_outer, grad_clip_val_outer))

    # Specify loss function (cross-entropy by default), used for both inner and outer loops
    loss_function = nn.CrossEntropyLoss(reduction='mean')

    # Masking

    masks_train = None
    if do_masking_train_loss:
        mask_train_sp, mask_train_po, mask_train_so = compute_masks(data.train_triples,
                                                                    data.train_triples + data.dev_triples,
                                                                    data.entity_to_idx,
                                                                    data.predicate_to_idx)

        mask_train_po = torch.tensor(mask_train_po, dtype=torch.long, device=device, requires_grad=False)
        mask_train_sp = torch.tensor(mask_train_sp, dtype=torch.long, device=device, requires_grad=False)
        mask_train_so = torch.tensor(mask_train_so, dtype=torch.long, device=device, requires_grad=False)
        masks_train = [mask_train_po, mask_train_sp, mask_train_so]

    masks_dev = None
    if do_masking_dev_loss:
        mask_dev_sp, mask_dev_po, mask_dev_so = compute_masks(data.dev_triples,
                                                                    data.train_triples + data.dev_triples,
                                                                    data.entity_to_idx,
                                                                    data.predicate_to_idx)

        mask_dev_po = torch.tensor(mask_dev_po, dtype=torch.long, device=device, requires_grad=False)
        mask_dev_sp = torch.tensor(mask_dev_sp, dtype=torch.long, device=device, requires_grad=False)
        mask_dev_so = torch.tensor(mask_dev_so, dtype=torch.long, device=device, requires_grad=False)
        masks_dev = [mask_dev_po, mask_dev_sp, mask_dev_so]

    # Training loop

    best_loss_outer_dev = np.inf

    losses_outer_dev = []
    losses_outer_train = []
    e_vals = []
    p_vals = []
    reg_weight_vals = []
    gradients_outer = []

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

        losses_inner_train = []
        losses_inner_dev = []
        for epoch_no in range(1, nb_epochs + 1):
            batcher = Batcher(data.Xs, data.Xp, data.Xo, batch_size, 1, random_state)

            batch_losses_train_withreg = []  # to store loss for each batch in the epoch
            batch_losses_train_nonreg = []

            for batch_no, (batch_start, batch_end) in enumerate(batcher.batches, 1):

                # Size [B] numpy arrays containing indices of each subject_entity, predicate, and object_entity in the batch
                xp_batch, xs_batch, xo_batch, xi_batch = batcher.get_batch(batch_start, batch_end)

                xs_batch = torch.tensor(xs_batch, dtype=torch.long, device=device)
                xp_batch = torch.tensor(xp_batch, dtype=torch.long, device=device)
                xo_batch = torch.tensor(xo_batch, dtype=torch.long, device=device)

                batch_loss_train, factors = get_unreg_loss(xs_batch=xs_batch,
                                                           xp_batch=xp_batch,
                                                           xo_batch=xo_batch,
                                                           xi_batch=xi_batch,
                                                           corruption=corruption,
                                                           entity_embeddings=e_graph,
                                                           predicate_embeddings=p_graph,
                                                           model=model,
                                                           loss_function=loss_function,
                                                           masks=masks_train)

                batch_loss_train_nonreg = batch_loss_train.item()
                batch_losses_train_nonreg += [batch_loss_train_nonreg]

                if regularizer == "F2":
                    batch_loss_train += torch.exp(reg_weight_graph) * F2_reg(factors)

                if regularizer == "N3":
                    batch_loss_train += torch.exp(reg_weight_graph) * N3_reg(factors)

                e_graph, p_graph = diffopt.step(batch_loss_train, params=[e_graph, p_graph])

                batch_loss_train_withreg = batch_loss_train.item()
                batch_losses_train_withreg += [batch_loss_train_withreg]

            epoch_loss_train_nonreg_mean = np.mean(batch_losses_train_nonreg)
            losses_inner_train += [epoch_loss_train_nonreg_mean]

            if outer_step == 0 or outer_step == outer_steps - 1 or stopping_tol_inner is not None or validate_all:

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
                                                   loss_function=loss_function,
                                                   masks=masks_dev)
                losses_inner_dev += [loss_inner_dev.item()]

                completed_epochs = nb_epochs
                if stopping_tol_inner is not None and epoch_no > 20 and np.mean(losses_inner_dev[-10:-5]) - np.mean(
                        losses_inner_dev[-5:]) < stopping_tol_inner:
                    completed_epochs = epoch_no
                    if not is_quiet:
                        logger.info(f"Num inner steps: {epoch_no}")
                    break

        if outer_step == 0 or outer_step == outer_steps-1:
            plt.figure()
            plt.plot(losses_inner_train, 'k-')
            plt.plot(losses_inner_dev, 'k--')
            plt.legend(["training loss", "masked validation loss"])
            plt.xlabel("Epoch (inner step)")
            plt.ylabel("Inner loss")
            plt.title(f"Inner losses, for outer loop number {outer_step+1}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            if save_figs:
                filename = f"realmeta_nations_innerloss_outerstep{outer_step+1}_{timestr}.png"
                if use_wandb:
                    plt.savefig(os.path.join(wandb.run.dir, filename))
                else:
                    plt.savefig(f"./realmeta_nations/plots/{filename}")
            plt.show()

        losses_outer_train += [losses_inner_train[-1]]

        xs_dev = torch.tensor(data.dev_Xs, dtype=torch.long, device=device)
        xp_dev = torch.tensor(data.dev_Xp, dtype=torch.long, device=device)
        xo_dev = torch.tensor(data.dev_Xo, dtype=torch.long, device=device)
        xi_dev = data.dev_Xi

        loss_outer_dev, _ = get_unreg_loss(xs_batch=xs_dev,
                                       xp_batch=xp_dev,
                                       xo_batch=xo_dev,
                                       xi_batch=xi_dev,
                                       corruption=corruption,
                                       entity_embeddings=e_graph,
                                       predicate_embeddings=p_graph,
                                       model=model,
                                       loss_function=loss_function,
                                       masks=masks_dev)

        loss_outer_dev.backward()

        # for plotting only
        losses_outer_dev += [loss_outer_dev.detach().clone().item()]
        e_vals += [torch.norm(e_graph.detach().clone()).item()]
        p_vals += [torch.norm(p_graph.detach().clone()).item()]
        reg_weight_vals += [reg_weight_graph.detach().clone().item()]
        gradients_outer += [reg_weight_graph.grad.item()]
        if use_wandb:
            outer_log = {"train_loss_outer": losses_outer_train[-1],
                         "dev_loss_outer": losses_outer_dev[-1],
                         "completed_inner_steps": completed_epochs,
                         "L2_norm_entity_embeddings": e_vals[-1],
                         "L2_norm_predicate_embeddings": p_vals[-1],
                         "regularisation_weight": np.exp(reg_weight_vals[-1]),
                         "gradient_outer": gradients_outer[-1]
                         }
            wandb.log(outer_log, step=outer_step)

        if not is_quiet:
            logger.info(f"outer dev loss: {loss_outer_dev.item():.7f}, reg param: {np.exp(reg_weight_graph.item()):.7f} [{reg_weight_graph.item():.5f}], reg param gradient: {gradients_outer[-1]}")

        # store a copy of best embeddings
        if loss_outer_dev < best_loss_outer_dev:
            best_loss_outer_dev = loss_outer_dev
            best_reg_weight = reg_weight_graph.detach().clone().item()
            best_e_graph = e_graph.detach().clone()
            best_p_graph = p_graph.detach().clone()
            best_outer_step = outer_step
            best_losses_inner_train = losses_inner_train
            best_losses_inner_dev = losses_inner_dev
            if use_wandb:
                best_log = outer_log

        optimizer_outer.step()
        if torch.absolute(reg_weight_graph.grad) < regweight_rescaler_tol:
            reg_weight_graph.requires_grad = False
            reg_weight_graph *= regweight_rescaler
            reg_weight_graph.requires_grad = True
        optimizer_outer.zero_grad()

    metrics_log = {}
    # Final outer step metrics
    logger.info(f"Final \touter step: {outer_steps} \treg param: {np.exp(reg_weight_vals[-1])} [{reg_weight_vals[-1]}] \touter dev loss {losses_outer_dev[-1]}")
    for triples, name in [(t, n) for t, n in triples_name_pairs if len(t) > 0]:
        metrics_final = evaluate(entity_embeddings=e_graph.detach().clone(), predicate_embeddings=p_graph.detach().clone(),
                           test_triples=triples, all_triples=data.all_triples,
                           entity_to_index=data.entity_to_idx, predicate_to_index=data.predicate_to_idx,
                           model=model, batch_size=eval_batch_size, device=device)
        if use_wandb:
            metrics_final_new = {f'final_{name}_{k}': v for k, v in metrics_final.items()}  # hack to get different keys for logging
            metrics_final_new.update({f'final_{k}': v for k, v in outer_log.items()})
            metrics_log.update(metrics_final_new)
        logger.info(f'Final \t{name} results\t{metrics_to_str(metrics_final)}')

    # Best outer step metrics (i.e. step with lowest outer dev loss)
    logger.info(f"Best \touter step: {best_outer_step+1} \treg param: {np.exp(best_reg_weight)} [{best_reg_weight}] \touter dev loss {best_loss_outer_dev}")
    for triples, name in [(t, n) for t, n in triples_name_pairs if len(t) > 0]:
        metrics_best = evaluate(entity_embeddings=best_e_graph, predicate_embeddings=best_p_graph,
                           test_triples=triples, all_triples=data.all_triples,
                           entity_to_index=data.entity_to_idx, predicate_to_index=data.predicate_to_idx,
                           model=model, batch_size=eval_batch_size, device=device)
        if use_wandb:
            metrics_best_new = {f'best_{name}_{k}': v for k, v in metrics_best.items()}  # hack to get different keys for logging
            metrics_best_new.update({f'best_{k}': v for k, v in best_log.items()})
            metrics_log.update(metrics_best_new)
        logger.info(f'Best \t{name} results\t{metrics_to_str(metrics_best)}')

    if use_wandb:
        metrics_log.update({"starting_outer_dev_loss": losses_outer_dev[0],
                            "final_delta_outer_dev_loss": losses_outer_dev[-1] - losses_outer_dev[0],
                            "best_delta_outer_dev_loss": best_loss_outer_dev - losses_outer_dev[0]})
        wandb.run.summary.update(metrics_log)

    plt.figure()
    plt.plot(best_losses_inner_train, 'k-')
    plt.plot(best_losses_inner_dev, 'k--')
    plt.legend(["training loss", "masked validation loss"])
    plt.xlabel("Epoch (inner step)")
    plt.ylabel("Inner loss")
    plt.title(f"Inner losses, for outer loop number {best_outer_step + 1}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_figs:
        filename = f"realmeta_nations_innertrainloss_outerstep{best_outer_step + 1}_{timestr}.png"
        if use_wandb:
            plt.savefig(os.path.join(wandb.run.dir, filename))
        else:
            plt.savefig(f"./realmeta_nations/plots/{filename}")
    plt.show()

    plt.figure()
    plt.plot(losses_outer_train, 'k-')
    plt.plot(losses_outer_dev, 'k--')
    plt.legend(["training loss", "masked validation loss"])
    plt.xlabel("Outer step")
    plt.ylabel("Outer loss")
    plt.title(f"Outer losses", fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_figs:
        filename = f"realmeta_nations_outer_losses_{timestr}.png"
        if use_wandb:
            plt.savefig(os.path.join(wandb.run.dir, filename))
        else:
            plt.savefig(f"./realmeta_nations/plots/{filename}")
    plt.show()

    plt.figure()
    plt.plot(np.exp(reg_weight_vals), 'k-')
    plt.xlabel("Outer step")
    plt.ylabel("Regularisation weight value")
    plt.ylim([0, 0.045])
    plt.title(f"{regularizer} regularisation weight values", fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_figs:
        filename = f"realmeta_nations_reg_weights_{timestr}.png"
        if use_wandb:
            plt.savefig(os.path.join(wandb.run.dir, filename))
        else:
            plt.savefig(f"./realmeta_nations/plots/{filename}")
    plt.show()

    plt.figure()
    plt.plot(gradients_outer, 'k-')
    plt.xlabel("Outer step")
    plt.ylabel("Regularisation weight gradient")
    plt.title(f"{regularizer} regularisation weight gradients", fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_figs:
        filename = f"realmeta_nations_reg_weight_gradients_{timestr}.png"
        if use_wandb:
            plt.savefig(os.path.join(wandb.run.dir, filename))
        else:
            plt.savefig(f"./realmeta_nations/plots/{filename}")
    plt.show()

    plt.figure()
    plt.plot(e_vals, 'k-')
    plt.plot(p_vals, 'k--')
    plt.legend(["entities", "predicates"])
    plt.xlabel("Outer step")
    plt.ylabel("$||$embeddings$||_2$")
    plt.title("Entity and predicate embedding L2-norms", fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_figs:
        filename = f"realmeta_nations_entity_L2norms_{timestr}.png"
        if use_wandb:
            plt.savefig(os.path.join(wandb.run.dir, filename))
        else:
            plt.savefig(f"./realmeta_nations/plots/{filename}")
    plt.show()

    logger.info("Training finished")
    if use_wandb:
        # wandb.save("logs/array.err")
        # wandb.save("logs/array.out")
        wandb.finish()


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = parse_args(sys.argv[1:])
    main(args)
