# -*- coding: utf-8 -*-

import sys

import multiprocessing
import numpy as np

import torch
from torch import nn, optim, Tensor
from torch.nn import Parameter

from kbc.util import is_torch_tpu_available, set_seed

from kbc.training.data import Data
from kbc.training.batcher import Batcher
from kbc.training.masking import compute_masks

from kbc.regularizers import Regularizer

from kbc.models import BasePredictor, DistMult, ComplEx

from kbc.regularizers import F2, N3

from kbc.blackbox import NegativeMRR

from typing import Optional, List

import higher

import logging
import pytest

torch.set_num_threads(multiprocessing.cpu_count())
# torch.autograd.set_detect_anomaly(True)
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize, precision=4)

logger = logging.getLogger(__name__)


@pytest.mark.light
def test_meta_v1(blackbox_lambda: float, meta_learning_rate: float):
    train_path = 'data/nations/test.tsv'
    dev_path = 'data/nations/dev.tsv'
    test_path = 'data/nations/test.tsv'

    model_name = 'complex'
    optimizer_name = 'adagrad'

    embedding_size = 100

    batch_size = 100

    nb_epochs = 50
    seed = 0

    learning_rate = 0.1

    corruption = 'so'

    input_type = 'standard'

    do_masking = True

    set_seed(seed)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    if is_torch_tpu_available():
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        torch.set_default_tensor_type('torch.FloatTensor')

    print(f'Device: {device}')

    data = Data(train_path=train_path, dev_path=dev_path, test_path=test_path, input_type=input_type)

    rank = embedding_size * 2 if model_name in {'complex'} else embedding_size
    init_size = 1e-3

    entity_embeddings = nn.Embedding(data.nb_entities, rank, sparse=False).to(device)
    predicate_embeddings = nn.Embedding(data.nb_predicates, rank, sparse=False).to(device)

    class Ln(Regularizer):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(embedding_size, 1)

        def __call__(self,
                     factors: List[Tensor]):
            norm = sum(torch.sigmoid(self.hidden(f)) for f in factors)
            norm = torch.sum(norm)
            return norm / factors[0].shape[0]

    F2_weight = None # Parameter(torch.tensor(1e-4, device=device), requires_grad=True)
    N3_weight = Parameter(torch.tensor(1e-4, device=device), requires_grad=True)
    Ln_weight = Parameter(torch.tensor(1e-4, device=device), requires_grad=True)

    F2_reg = F2() if F2_weight is not None else None
    N3_reg = N3() if N3_weight is not None else None
    Ln_reg = Ln() if Ln_weight is not None else None

    entity_embeddings.weight.data *= init_size
    predicate_embeddings.weight.data *= init_size

    parameters_lst = nn.ModuleDict({
        'entities': entity_embeddings,
        'predicates': predicate_embeddings
    }).to(device)

    meta_parameters_lst = nn.ParameterDict({
        # 'f2': F2_weight,
        'n3': N3_weight,
        'ln': Ln_weight
    }).to(device)

    meta_parameters_2_lst = nn.ModuleDict({
        'ln': Ln_reg,
        # 'predicates': predicate_embeddings
    }).to(device)

    model = ComplEx().to(device)

    optimizer_factory = {
        'adagrad': lambda params, lr: optim.Adagrad(params, lr=lr),
        'adam': lambda params, lr: optim.Adam(params, lr=lr),
        'sgd': lambda params, lr: optim.SGD(params, lr=lr)
    }

    assert optimizer_name in optimizer_factory

    optimizer = optimizer_factory[optimizer_name](parameters_lst.parameters(), learning_rate)
    print(parameters_lst)

    # meta_optimizer = optimizer_factory[optimizer_name](meta_parameters_lst.parameters(), 0.05)
    meta_params = [p for p in meta_parameters_lst.parameters()] + [p for p in meta_parameters_2_lst.parameters()]

    # meta_optimizer = optim.SGD(meta_params, lr=meta_learning_rate)
    meta_optimizer = optim.Adagrad(meta_params, lr=meta_learning_rate)

    print(meta_parameters_lst)
    print(meta_parameters_2_lst)

    loss_function = nn.CrossEntropyLoss(reduction='mean')

    meta_loss_function = NegativeMRR(lmbda=blackbox_lambda)
    # meta_loss_function = nn.CrossEntropyLoss(reduction='mean')

    mask_sp = mask_po = None
    if do_masking:
        mask_sp, mask_po, _ = compute_masks(data.dev_triples, data.train_triples + data.dev_triples,
                                            data.entity_to_idx, data.predicate_to_idx)
        mask_sp = torch.tensor(mask_sp, dtype=torch.long, device=device)
        mask_po = torch.tensor(mask_po, dtype=torch.long, device=device)

    for outer_epoch in range(1000):
        e_tensor_lh = entity_embeddings.weight
        p_tensor_lh = predicate_embeddings.weight

        parameter_lst_lh = [e_tensor_lh, p_tensor_lh]
        # parameter_lst_lh = [e_tensor_lh]

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

                if 's' in corruption:
                    po_scores = model.forward(xp_batch_emb, None, xo_batch_emb,
                                              entity_embeddings=e_tensor_lh, predicate_embeddings=p_tensor_lh)
                    loss += loss_function(po_scores, xs_batch)

                if 'o' in corruption:
                    sp_scores = model.forward(xp_batch_emb, xs_batch_emb, None,
                                              entity_embeddings=e_tensor_lh, predicate_embeddings=p_tensor_lh)
                    loss += loss_function(sp_scores, xo_batch)

                if 'p' in corruption:
                    so_scores = model.forward(None, xs_batch_emb, xo_batch_emb,
                                              entity_embeddings=e_tensor_lh, predicate_embeddings=p_tensor_lh)
                    loss += loss_function(so_scores, xp_batch)

                factors = [model.factor(e) for e in [xp_batch_emb, xs_batch_emb, xo_batch_emb]]

                if F2_weight is not None:
                    loss += F2_weight * F2_reg(factors)

                if N3_weight is not None:
                    loss += N3_weight * N3_reg(factors)

                if Ln_weight is not None:
                    loss += Ln_weight * Ln_reg(factors)

                e_tensor_lh, p_tensor_lh = diff_opt.step(loss, params=parameter_lst_lh)
                # e_tensor_lh, = diff_opt.step(loss, params=parameter_lst_lh)

                parameter_lst_lh = [e_tensor_lh, p_tensor_lh]
                # parameter_lst_lh = [e_tensor_lh]

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

        dev_loss_value = dev_loss

        dev_loss.backward()

        torch.nn.utils.clip_grad_norm_(meta_params, 1.0)

        meta_optimizer.step()
        meta_optimizer.zero_grad()

        # Just in case
        # optimizer.zero_grad()

        if F2_weight is not None:
            F2_weight.data.clamp_(1e-6)

        if N3_weight is not None:
            N3_weight.data.clamp_(1e-6)

        if Ln_weight is not None:
            Ln_weight.data.clamp_(1e-6)

        print(f'{outer_epoch}\t{dev_loss_value}\t{F2_weight}\t{N3_weight}\t{Ln_weight}')


if __name__ == '__main__':
    # pytest.main([__file__])

    blackbox_lambda = 10.0
    meta_learning_rate = 0.001

    if len(sys.argv) > 1:
        blackbox_lambda = float(sys.argv[1])
        print(f'New blackbox_lambda: {blackbox_lambda}')

    if len(sys.argv) > 2:
        meta_learning_rate = float(sys.argv[2])
        print(f'New meta_learning_rate: {meta_learning_rate}')

    test_meta_v1(blackbox_lambda=blackbox_lambda, meta_learning_rate=meta_learning_rate)
