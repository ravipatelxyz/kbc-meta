# -*- coding: utf-8 -*-

from torch import nn, Tensor

from abc import abstractmethod

from typing import Optional

import logging

logger = logging.getLogger(__name__)


class BasePredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def score(self,
              rel: Tensor,
              arg1: Tensor,
              arg2: Tensor,
              *args, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self,
                rel: Optional[Tensor],
                arg1: Optional[Tensor],
                arg2: Optional[Tensor],
                *args, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def factor(self,
               vec: Tensor) -> Tensor:
        raise NotImplementedError
