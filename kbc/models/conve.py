# -*- coding: utf-8 -*-

import torch
from torch import Tensor

from kbc.models.base import BasePredictor

from typing import Optional

import logging

logger = logging.getLogger(__name__)


class ConvE(BasePredictor):
    def __init__(self,
                 entity_embeddings: Tensor,
                 predicate_embeddings: Tensor,
                 embedding_size: int,
                 embedding_height: int,
                 embedding_width: int,
                 filter_size: int = 3,
                 stride: int = 1,
                 feature_map_dropout: float = 0.2,
                 projection_dropout: float = 0.3,
                 convolution_bias: bool = True) -> None:
        super().__init__()
        self.entity_embeddings = entity_embeddings
        self.predicate_embeddings = predicate_embeddings

        self.embedding_size = embedding_size
        self.embedding_height = embedding_height
        self.embedding_width = embedding_width

        self.filter_size = filter_size
        self.stride = stride

        self.feature_map_dropout = feature_map_dropout
        self.projection_dropout = projection_dropout
        self.convolution_bias = convolution_bias

        self.feature_map_dropout = torch.nn.Dropout2d(self.feature_map_dropout)
        self.projection_dropout = torch.nn.Dropout(self.projection_dropout)

        self.convolution = torch.nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(self.filter_size, self.filter_size),
            stride=self.stride,
            padding=0,
            bias=self.convolution_bias,
        )

        self.bn1 = torch.nn.BatchNorm2d(32, affine=False)
        self.bn2 = torch.nn.BatchNorm1d(self.embedding_size, affine=False)

        conv_output_height = (((self.embedding_height * 2) - self.filter_size) / self.stride) + 1
        conv_output_width = ((self.embedding_width - self.filter_size) / self.stride) + 1

        self.projection = torch.nn.Linear(32 * int(conv_output_height * conv_output_width), int(self.embedding_size))
        self.non_linear = torch.nn.ReLU()

    def vectorise(self,
                  rel: Tensor,
                  arg: Tensor) -> Tensor:
        batch_size = rel.shape[0]

        arg_2d = arg.view(-1, 1, self.embedding_height, self.embedding_width)
        rel_2d = rel.view(-1, 1, self.embedding_height, self.embedding_width)

        stacked_inputs = torch.cat([arg_2d, rel_2d], 2)

        out = self.convolution(stacked_inputs)
        out = self.bn1(out)
        out = self.non_linear(out)
        out = self.feature_map_dropout(out)
        out = out.view(batch_size, -1)
        out = self.projection(out)
        out = self.projection_dropout(out)
        out = self.bn2(out)
        out = self.non_linear(out)
        return out

    def score(self,
              rel: Tensor,
              arg1: Tensor,
              arg2: Tensor,
              *args, **kwargs) -> Tensor:
        # [B]
        batch_size = rel.shape[0]
        embedding_size = arg1.shape[1]

        rel_sp, rel_po = rel[:, :embedding_size], rel[:, embedding_size:]
        out = self.vectorise(rel_sp, arg1)

        out = (out * arg2).sum(-1)
        res = out.view(batch_size)
        return res

    def forward(self,
                rel: Optional[Tensor],
                arg1: Optional[Tensor],
                arg2: Optional[Tensor],
                *args, **kwargs) -> Tensor:
        # [N, E]
        ent_emb = self.entity_embeddings
        pred_emb = self.predicate_embeddings

        assert ((1 if rel is None else 0) + (1 if arg1 is None else 0) + (1 if arg2 is None else 0)) == 1

        if rel is None:
            raise ValueError("rel=None is unsupported")

        batch_size = rel.shape[0]
        embedding_size = ent_emb.shape[1]

        rel_sp, rel_po = rel[:, :embedding_size], rel[:, embedding_size:]

        # [B] Tensor
        scores = None

        # [B, N]
        if arg2 is None:
            # [B, N]
            out = self.vectorise(rel_sp, arg1)
            scores = torch.mm(out, ent_emb.t())
        elif arg1 is None:
            # [B, N]
            out = self.vectorise(rel_po, arg2)
            scores = torch.mm(out, ent_emb.t())

        assert scores is not None

        return scores

    def factor(self,
               vec: Tensor) -> Tensor:
        return vec
