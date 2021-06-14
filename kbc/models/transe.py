# -*- coding: utf-8 -*-

import logging

import torch
from torch import Tensor

from kbc.models.base import BasePredictor

from typing import Optional

logger = logging.getLogger(__name__)


class TransE(BasePredictor):
    def __init__(self,
                 entity_embeddings: Tensor,
                 predicate_embeddings: Tensor) -> None:
        super().__init__()
        self.entity_embeddings = entity_embeddings
        self.predicate_embeddings = predicate_embeddings

        self.embedding_size = self.entity_embeddings.shape[1]

    def score(self,
              rel: Tensor,
              arg1: Tensor,
              arg2: Tensor,
              *args, **kwargs) -> Tensor:
        # [B, E]

        # [B] Tensor
        delta = arg1 + rel - arg2
        res = - torch.norm(delta, dim=1, p=2)

        # [B] Tensor
        return res

    def forward(self,
                rel: Optional[Tensor],
                arg1: Optional[Tensor],
                arg2: Optional[Tensor],
                *args, **kwargs) -> Tensor:

        ent_emb = self.entity_embeddings
        pred_emb = self.predicate_embeddings

        nb_entities = ent_emb.shape[0]
        nb_predicates = pred_emb.shape[0]

        emb_size = ent_emb.shape[1]

        assert ((1 if rel is None else 0) + (1 if arg1 is None else 0) + (1 if arg2 is None else 0)) == 1

        # [B] Tensor
        scores = None

        if rel is None:
            # proj = arg1 - arg2
            batch_size = arg1.shape[0]

            _arg1 = arg1.view(1, -1, emb_size).repeat(nb_predicates, 1, 1)
            _arg2 = arg2.view(1, -1, emb_size).repeat(nb_predicates, 1, 1)
            _rel = pred_emb.view(-1, 1, emb_size).repeat(1, batch_size, 1)

            delta = _arg1 + _rel - _arg2
            scores = - torch.norm(delta, dim=2, p=2).t()

        elif arg1 is None:
            # proj = rel - arg2
            batch_size = arg2.shape[0]

            _arg1 = ent_emb.view(-1, 1, emb_size).repeat(1, batch_size, 1)
            _arg2 = arg2.view(1, -1, emb_size).repeat(nb_entities, 1, 1)
            _rel = rel.view(1, -1, emb_size).repeat(nb_entities, 1, 1)

            delta = _arg1 + _rel - _arg2
            scores = - torch.norm(delta, dim=2, p=2).t()

        elif arg2 is None:
            # proj = arg1 + rel
            batch_size = arg1.shape[0]

            _arg1 = arg1.view(1, -1, emb_size).repeat(nb_entities, 1, 1)
            _arg2 = ent_emb.view(-1, 1, emb_size).repeat(1, batch_size, 1)
            _rel = rel.view(1, -1, emb_size).repeat(nb_entities, 1, 1)

            delta = _arg1 + _rel - _arg2
            scores = - torch.norm(delta, dim=2, p=2).t()

        assert scores is not None

        return scores

    def factor(self,
               vec: Tensor) -> Tensor:
        return vec
