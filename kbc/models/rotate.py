# -*- coding: utf-8 -*-

import torch
from torch import Tensor

from kbc.models.base import BasePredictor
from kbc.models.util import normalize_phases, hadamard_complex, diff_complex, abs_complex, norm_nonnegative, pairwise_diff_complex

from typing import Optional

import logging

logger = logging.getLogger(__name__)


class RotatE(BasePredictor):
    def __init__(self,
                 entity_embeddings: Tensor,
                 predicate_embeddings: Tensor) -> None:
        super().__init__()
        self.entity_embeddings = entity_embeddings
        self.predicate_embeddings = predicate_embeddings

    def score(self,
              rel: Tensor,
              arg1: Tensor,
              arg2: Tensor,
              *args, **kwargs) -> Tensor:
        # [B]
        n = rel.shape[0]

        # determine real and imaginary part
        s_emb_re, s_emb_im = torch.chunk(arg1, 2, dim=1)
        o_emb_re, o_emb_im = torch.chunk(arg2, 2, dim=1)

        # normalize phases so that they lie in [-pi,pi]
        rel = normalize_phases(rel)

        # convert from radians to points on complex unix ball
        p_emb_re, p_emb_im = torch.cos(rel), torch.sin(rel)

        # compute the difference vector (s*p-t)
        sp_emb_re, sp_emb_im = hadamard_complex(s_emb_re, s_emb_im, p_emb_re, p_emb_im)

        diff_re, diff_im = diff_complex(sp_emb_re, sp_emb_im, o_emb_re, o_emb_im)

        # compute the absolute values for each (complex) element of the difference vector
        diff_abs = abs_complex(diff_re, diff_im)

        # now take the norm of the absolute values of the difference vector
        out = - norm_nonnegative(diff_abs, dim=1, p=1.0)

        return out.view(n, -1)

    def forward(self,
                rel: Optional[Tensor],
                arg1: Optional[Tensor],
                arg2: Optional[Tensor],
                *args, **kwargs) -> Tensor:
        # [N, E]
        ent_emb = self.entity_embeddings
        # pred_emb = self.predicate_embeddings

        assert ((1 if rel is None else 0) + (1 if arg1 is None else 0) + (1 if arg2 is None else 0)) == 1

        if rel is None:
            raise ValueError("rel=None is unsupported")

        emb = ent_emb
        emb_re, emb_im = torch.chunk(emb, 2, dim=1)

        rel = normalize_phases(rel)

        # convert from radians to points on complex unix ball
        p_emb_re, p_emb_im = torch.cos(rel), torch.sin(rel)

        # [B] Tensor
        scores = None

        # [B, N]
        if arg2 is None:
            # determine real and imaginary part
            s_emb_re, s_emb_im = torch.chunk(arg1, 2, dim=1)

            # as above, but pair each sp-pair with each object
            sp_emb_re, sp_emb_im = hadamard_complex(s_emb_re, s_emb_im, p_emb_re, p_emb_im)
            # compute scores with every possible object
            diff_re, diff_im = pairwise_diff_complex(sp_emb_re, sp_emb_im, emb_re, emb_im)

            diff_abs = abs_complex(diff_re, diff_im)

            # might need to change p!!
            scores = -norm_nonnegative(diff_abs, dim=2, p=1.0)

        elif arg1 is None:
            # determine real and imaginary part
            o_emb_re, o_emb_im = torch.chunk(arg2, 2, dim=1)

            # compute the complex conjugate (cc) of the relation vector and perform inverse rotation on tail.
            # This uses || s*p - o || = || s - cc(p)*o || for a rotation p.
            p_emb_im = -p_emb_im
            po_emb_re, po_emb_im = hadamard_complex(p_emb_re, p_emb_im, o_emb_re, o_emb_im)

            diff_re, diff_im = pairwise_diff_complex( po_emb_re, po_emb_im, emb_re, emb_im)

            diff_abs = abs_complex(diff_re, diff_im)

            # might need to change p - this should be specifying the norm if non-zero scores!
            scores = -norm_nonnegative(diff_abs, dim=2, p=1.0)

        assert scores is not None

        return scores

    def factor(self,
               embedding_vector: Tensor,
               mode='ent') -> Tensor:

        if mode == 'ent':
            vec_real, vec_img = torch.chunk(embedding_vector, 2, dim=1)

        elif mode == 'rel':
            embedding_vector = normalize_phases(embedding_vector)
            vec_real, vec_img = torch.cos(embedding_vector), torch.sin(embedding_vector)

        sq_factor = vec_real ** 2 + vec_img ** 2
        factors = torch.sqrt(sq_factor)
        return factors
