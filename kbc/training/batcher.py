# -*- coding: utf-8 -*-

import numpy as np

from kbc.util import make_batches
from kbc.training.data import Data

from typing import Tuple


class Batcher:
    def __init__(self,
                 Xs: np.ndarray,
                 Xp: np.ndarray,
                 Xo: np.ndarray,
                 batch_size: int,
                 nb_epochs: int,
                 random_state: np.random.RandomState) -> None:
        self.Xs = Xs
        self.Xp = Xp
        self.Xo = Xo

        self.Xi = np.arange(start=0, stop=self.Xs.shape[0], dtype=np.int32)

        assert np.allclose(Xs.shape, Xp.shape)
        assert np.allclose(Xs.shape, Xo.shape)

        self.nb_examples = Xs.shape[0]

        self.batch_size = batch_size
        self.random_state = random_state

        size = nb_epochs * self.nb_examples
        self.curriculum_Xs = np.zeros(size, dtype=np.int32)
        self.curriculum_Xp = np.zeros(size, dtype=np.int32)
        self.curriculum_Xo = np.zeros(size, dtype=np.int32)
        self.curriculum_Xi = np.zeros(size, dtype=np.int32)

        for epoch_no in range(nb_epochs):
            curriculum_order = self.random_state.permutation(self.nb_examples)
            start = epoch_no * self.nb_examples
            end = (epoch_no + 1) * self.nb_examples
            self.curriculum_Xs[start: end] = self.Xs[curriculum_order]
            self.curriculum_Xp[start: end] = self.Xp[curriculum_order]
            self.curriculum_Xo[start: end] = self.Xo[curriculum_order]
            self.curriculum_Xi[start: end] = self.Xi[curriculum_order]

        self.batches = make_batches(self.curriculum_Xs.shape[0], batch_size)
        self.nb_batches = len(self.batches)

    def get_batch(self, batch_start: int, batch_end: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Positive examples
        xs_batch = self.curriculum_Xs[batch_start:batch_end]
        xp_batch = self.curriculum_Xp[batch_start:batch_end]
        xo_batch = self.curriculum_Xo[batch_start:batch_end]
        xi_batch = self.curriculum_Xi[batch_start:batch_end]
        return xp_batch, xs_batch, xo_batch, xi_batch
