# -*- coding: utf-8 -*-

import torch
from torch import Tensor

from kbc.models.base import BasePredictor

from typing import Optional

import logging

logger = logging.getLogger(__name__)


class DistMult(BasePredictor):
    def __init__(self,
                 entity_embeddings: Optional[Tensor] = None,
                 predicate_embeddings: Optional[Tensor] = None) -> None:
        super().__init__()
        self.entity_embeddings = entity_embeddings
        self.predicate_embeddings = predicate_embeddings

    def score(self,
              rel: Tensor,  # [B, E], predicates batch embeddings tensor
              arg1: Tensor,  # [B, E], subjects batch embeddings tensor
              arg2: Tensor,  # [B, E], objects batch embeddings tensor
              *args, **kwargs) -> Tensor:
        # [B] Tensor, i.e. returns vector of the scores for each triplet in the batch
        res = torch.sum(rel * arg1 * arg2, 1)
        return res

    def forward(self,
                rel: Optional[Tensor],  # [B, E], predicates batch embeddings tensor
                arg1: Optional[Tensor],  # [B, E], subjects batch embeddings tensor
                arg2: Optional[Tensor],  # [B, E], objects batch embeddings tensor
                entity_embeddings: Optional[Tensor] = None,  # [Nb_entities, E]
                predicate_embeddings: Optional[Tensor] = None,  # [Nb_preds, E]
                *args, **kwargs) -> Tensor:

        # Tensors of embeddings of entities and predicates for entire dataset
        # shape (total_nb_entities, rank), i.e. [Nb_entities, E]
        ent_emb = self.entity_embeddings if self.entity_embeddings is not None else entity_embeddings
        # shape (total_nb_predicates, rank), i.e. [Nb_preds, E]
        pred_emb = self.predicate_embeddings if self.predicate_embeddings is not None else predicate_embeddings

        # Only one of rel, arg1, or arg2 can be None
        assert ((1 if rel is None else 0) + (1 if arg1 is None else 0) + (1 if arg2 is None else 0)) == 1

        scores = None
        # [B, Nb_preds] = [B, E] @ [E, Nb_preds]
        if rel is None:  # tensor of scores
            scores = (arg1 * arg2) @ pred_emb.t()
        # [B, Nb_entities] = [B, E] @ [E, Nb_entities]
        elif arg1 is None:  # predicting the subject
            scores = (rel * arg2) @ ent_emb.t()
        # [B, Nb_entities] = [B, E] @ [E, Nb_entities]
        elif arg2 is None:  # predicting the object
            scores = (rel * arg1) @ ent_emb.t()

        assert scores is not None

        return scores

    def factor(self,
               vec: Tensor) -> Tensor:
        return vec
