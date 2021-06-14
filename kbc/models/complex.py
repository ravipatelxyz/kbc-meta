# -*- coding: utf-8 -*-

import logging

import torch
from torch import Tensor

from kbc.models.base import BasePredictor

from typing import Optional

logger = logging.getLogger(__name__)


class ComplEx(BasePredictor):
    def __init__(self,
                 entity_embeddings: Optional[Tensor] = None,
                 predicate_embeddings: Optional[Tensor] = None) -> None:
        super().__init__()
        self.entity_embeddings = entity_embeddings
        self.predicate_embeddings = predicate_embeddings

        # self.embedding_size = self.entity_embeddings.shape[1] // 2

    def score(self,
              rel: Tensor,  # [B, 2E], predicates batch embeddings tensor
              arg1: Tensor,  # [B, 2E], subjects batch embeddings tensor
              arg2: Tensor,  # [B, 2E], objects batch embeddings tensor
              *args, **kwargs) -> Tensor:
        embedding_size = rel.shape[1] // 2

        # [B, E]
        rel_real, rel_img = rel[:, :embedding_size], rel[:, embedding_size:]
        arg1_real, arg1_img = arg1[:, :embedding_size], arg1[:, embedding_size:]
        arg2_real, arg2_img = arg2[:, :embedding_size], arg2[:, embedding_size:]

        # [B] Tensor
        res = torch.sum(rel_real * arg1_real * arg2_real +
                        rel_real * arg1_img * arg2_img +
                        rel_img * arg1_real * arg2_img -
                        rel_img * arg1_img * arg2_real, 1)

        # [B] Tensor, i.e. returns vector of the scores for each triplet in the batch
        return res

    def forward(self,
                rel: Optional[Tensor],  # [B, 2E], predicates batch embeddings tensor
                arg1: Optional[Tensor],  # [B, 2E], subjects batch embeddings tensor
                arg2: Optional[Tensor],  # [B, 2E], objects batch embeddings tensor
                entity_embeddings: Optional[Tensor] = None,  # [Nb_entities, 2E]
                predicate_embeddings: Optional[Tensor] = None,  # [Nb_preds, 2E]
                *args, **kwargs) -> Tensor:

        # Tensors of embeddings of all entities and predicates in the dataset
        # shape (total_nb_entities, rank), i.e. [Nb_entities, 2E]
        ent_emb = self.entity_embeddings if self.entity_embeddings is not None else entity_embeddings
        # shape (total_nb_predicates, rank), i.e. [Nb_preds, 2E]
        pred_emb = self.predicate_embeddings if self.predicate_embeddings is not None else predicate_embeddings

        # embedding size is rank/2 because for implementation purposes each embedding contains:
        # 1st half of embedding corresponding to real component of embedding
        # 2nd half of embedding corresponding to imaginary component of embedding
        embedding_size = ent_emb.shape[1] // 2

        # Only one of rel, arg1, or arg2 can be None
        assert ((1 if rel is None else 0) + (1 if arg1 is None else 0) + (1 if arg2 is None else 0)) == 1

        # Separating out the real and imaginary parts of the embedding (1st half = real, 2nd half = imaginary)
        ent_real, ent_img = ent_emb[:, :embedding_size], ent_emb[:, embedding_size:]
        pred_real, pred_img = pred_emb[:, :embedding_size], pred_emb[:, embedding_size:]


        scores = None
        if rel is None:  # predicting the predicate
            arg1_real, arg1_img = arg1[:, :embedding_size], arg1[:, embedding_size:]
            arg2_real, arg2_img = arg2[:, :embedding_size], arg2[:, embedding_size:]

            score1_so = (arg1_real * arg2_real + arg1_img * arg2_img) @ pred_real.t()
            score2_so = (arg1_real * arg2_img - arg1_img * arg2_real) @ pred_img.t()

            scores = score1_so + score2_so

        elif arg1 is None:  # predicting the subject
            rel_real, rel_img = rel[:, :embedding_size], rel[:, embedding_size:]
            arg2_real, arg2_img = arg2[:, :embedding_size], arg2[:, embedding_size:]

            # [B, N] = [B, E] @ [E, N]
            score1_po = (rel_real * arg2_real + rel_img * arg2_img) @ ent_real.t()
            score2_po = (rel_real * arg2_img - rel_img * arg2_real) @ ent_img.t()

            # [B, N]
            scores = score1_po + score2_po

        elif arg2 is None:  # predicting the object
            rel_real, rel_img = rel[:, :embedding_size], rel[:, embedding_size:]
            arg1_real, arg1_img = arg1[:, :embedding_size], arg1[:, embedding_size:]

            score1_sp = (rel_real * arg1_real - rel_img * arg1_img) @ ent_real.t()
            score2_sp = (rel_real * arg1_img + rel_img * arg1_real) @ ent_img.t()

            scores = score1_sp + score2_sp

        assert scores is not None

        return scores

    # factors used in calculation of regularisation terms
    def factor(self,
               vec: Tensor,
               safe: bool = False) -> Tensor:
        embedding_size = vec.shape[1] // 2

        vec_real, vec_img = vec[:, :embedding_size], vec[:, embedding_size:]
        factor = vec_real ** 2 + vec_img ** 2

        if safe is True:
            # factor = torch.max(factor, torch.tensor(1e-45, device=vec.device, requires_grad=False))
            # factor = factor + torch.tensor(1e-45, device=vec.device, requires_grad=False)
            pass

        sqrt_factor = torch.sqrt(factor)

        return sqrt_factor
