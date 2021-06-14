# -*- coding: utf-8 -*-

import numpy as np

import torch
from torch import nn

from kbc.models import DistMult, ComplEx, TransE, ConvE, RotatE

import pytest


@pytest.mark.light
def test_distmult_v1():
    nb_entities = 10
    nb_predicates = 5
    embedding_size = 10

    init_size = 1.0

    rs = np.random.RandomState(0)

    for _ in range(128):
        with torch.no_grad():
            entity_embeddings = nn.Embedding(nb_entities, embedding_size, sparse=True)
            predicate_embeddings = nn.Embedding(nb_predicates, embedding_size, sparse=True)

            entity_embeddings.weight.data *= init_size
            predicate_embeddings.weight.data *= init_size

            model = DistMult(entity_embeddings.weight, predicate_embeddings.weight)

            xs = torch.tensor(rs.randint(nb_entities, size=32), dtype=torch.long)
            xp = torch.tensor(rs.randint(nb_predicates, size=32), dtype=torch.long)
            xo = torch.tensor(rs.randint(nb_entities, size=32), dtype=torch.long)

            xs_emb = entity_embeddings(xs)
            xp_emb = predicate_embeddings(xp)
            xo_emb = entity_embeddings(xo)

            scores_sp = model.forward(xp_emb, xs_emb, None)
            scores_so = model.forward(None, xs_emb, xo_emb)
            scores_po = model.forward(xp_emb, None, xo_emb)

            inf = model.score(xp_emb, xs_emb, xo_emb)

            inf = inf.cpu().numpy()

            scores_sp = scores_sp.cpu().numpy()
            scores_so = scores_so.cpu().numpy()
            scores_po = scores_po.cpu().numpy()

            for i in range(32):
                np.testing.assert_allclose(inf[i], scores_so[i, xp[i]], rtol=1e-3, atol=1e-3)
                np.testing.assert_allclose(inf[i], scores_sp[i, xo[i]], rtol=1e-3, atol=1e-3)
                np.testing.assert_allclose(inf[i], scores_po[i, xs[i]], rtol=1e-3, atol=1e-3)


@pytest.mark.light
def test_complex_v1():
    nb_entities = 10
    nb_predicates = 5
    embedding_size = 10

    init_size = 1.0

    rs = np.random.RandomState(0)

    for _ in range(128):
        with torch.no_grad():
            entity_embeddings = nn.Embedding(nb_entities, embedding_size, sparse=True)
            predicate_embeddings = nn.Embedding(nb_predicates, embedding_size, sparse=True)

            entity_embeddings.weight.data *= init_size
            predicate_embeddings.weight.data *= init_size

            model = ComplEx(entity_embeddings.weight, predicate_embeddings.weight)

            xs = torch.tensor(rs.randint(nb_entities, size=32), dtype=torch.long)
            xp = torch.tensor(rs.randint(nb_predicates, size=32), dtype=torch.long)
            xo = torch.tensor(rs.randint(nb_entities, size=32), dtype=torch.long)

            xs_emb = entity_embeddings(xs)
            xp_emb = predicate_embeddings(xp)
            xo_emb = entity_embeddings(xo)

            scores_sp = model.forward(xp_emb, xs_emb, None)
            scores_so = model.forward(None, xs_emb, xo_emb)
            scores_po = model.forward(xp_emb, None, xo_emb)

            inf = model.score(xp_emb, xs_emb, xo_emb)

            inf = inf.cpu().numpy()
            scores_sp = scores_sp.cpu().numpy()
            scores_so = scores_so.cpu().numpy()
            scores_po = scores_po.cpu().numpy()

            for i in range(32):
                np.testing.assert_allclose(inf[i], scores_so[i, xp[i]], rtol=1e-3, atol=1e-3)
                np.testing.assert_allclose(inf[i], scores_sp[i, xo[i]], rtol=1e-3, atol=1e-3)
                np.testing.assert_allclose(inf[i], scores_po[i, xs[i]], rtol=1e-3, atol=1e-3)


@pytest.mark.light
def test_transe_v1():
    nb_entities = 10
    nb_predicates = 5
    embedding_size = 4

    init_size = 1.0

    rs = np.random.RandomState(0)

    for _ in range(128):
        with torch.no_grad():
            entity_embeddings = nn.Embedding(nb_entities, embedding_size, sparse=True)
            predicate_embeddings = nn.Embedding(nb_predicates, embedding_size, sparse=True)

            entity_embeddings.weight.data *= init_size
            predicate_embeddings.weight.data *= init_size

            model = TransE(entity_embeddings.weight, predicate_embeddings.weight)

            xs = torch.tensor(rs.randint(nb_entities, size=32), dtype=torch.long)
            xp = torch.tensor(rs.randint(nb_predicates, size=32), dtype=torch.long)
            xo = torch.tensor(rs.randint(nb_entities, size=32), dtype=torch.long)

            xs_emb = entity_embeddings(xs)
            xp_emb = predicate_embeddings(xp)
            xo_emb = entity_embeddings(xo)

            scores_sp = model.forward(xp_emb, xs_emb, None)
            scores_so = model.forward(None, xs_emb, xo_emb)
            scores_po = model.forward(xp_emb, None, xo_emb)

            inf = model.score(xp_emb, xs_emb, xo_emb)

            inf = inf.cpu().numpy()

            scores_sp = scores_sp.cpu().numpy()
            scores_so = scores_so.cpu().numpy()
            scores_po = scores_po.cpu().numpy()

            for i in range(32):
                np.testing.assert_allclose(inf[i], scores_so[i, xp[i]], rtol=1e-3, atol=1e-3)
                np.testing.assert_allclose(inf[i], scores_sp[i, xo[i]], rtol=1e-3, atol=1e-3)
                np.testing.assert_allclose(inf[i], scores_po[i, xs[i]], rtol=1e-3, atol=1e-3)


@pytest.mark.light
def test_conve_v1():
    nb_entities = 10
    nb_predicates = 5
    embedding_size = 100

    init_size = 1.0

    rs = np.random.RandomState(0)

    for _ in range(128):
        with torch.no_grad():
            entity_embeddings = nn.Embedding(nb_entities, embedding_size, sparse=True)
            predicate_embeddings = nn.Embedding(nb_predicates, embedding_size * 2, sparse=True)

            entity_embeddings.weight.data *= init_size
            predicate_embeddings.weight.data *= init_size

            model = ConvE(entity_embeddings.weight,
                          predicate_embeddings.weight,
                          embedding_size=embedding_size,
                          embedding_height=10,
                          embedding_width=10)

            xs = torch.tensor(rs.randint(nb_entities, size=32), dtype=torch.long)
            xp = torch.tensor(rs.randint(nb_predicates, size=32), dtype=torch.long)
            xo = torch.tensor(rs.randint(nb_entities, size=32), dtype=torch.long)

            xs_emb = entity_embeddings(xs)
            xp_emb = predicate_embeddings(xp)
            xo_emb = entity_embeddings(xo)

            model.eval()

            scores_sp = model.forward(xp_emb, xs_emb, None)
            # scores_so = model.forward(None, xs_emb, xo_emb)
            scores_po = model.forward(xp_emb, None, xo_emb)

            inf = model.score(xp_emb, xs_emb, xo_emb)
            inf = inf.cpu().numpy()

            scores_sp = scores_sp.cpu().numpy()
            # scores_so = scores_so.cpu().numpy()
            scores_po = scores_po.cpu().numpy()

            for i in range(32):
                # np.testing.assert_allclose(inf[i], scores_so[i, xp[i]], rtol=1e-3, atol=1e-3)
                np.testing.assert_allclose(inf[i], scores_sp[i, xo[i]], rtol=1e-3, atol=1e-3)
                # np.testing.assert_allclose(inf[i], scores_po[i, xs[i]], rtol=1e-3, atol=1e-3)


@pytest.mark.light
def test_rotate_v1():
    nb_entities = 10
    nb_predicates = 5
    embedding_size = 4

    init_size = 1.0

    rs = np.random.RandomState(0)

    for _ in range(128):
        with torch.no_grad():
            entity_embeddings = nn.Embedding(nb_entities, embedding_size * 2, sparse=True)
            predicate_embeddings = nn.Embedding(nb_predicates, embedding_size, sparse=True)

            entity_embeddings.weight.data *= init_size
            predicate_embeddings.weight.data *= init_size

            model = RotatE(entity_embeddings.weight, predicate_embeddings.weight)

            xs = torch.tensor(rs.randint(nb_entities, size=32), dtype=torch.long)
            xp = torch.tensor(rs.randint(nb_predicates, size=32), dtype=torch.long)
            xo = torch.tensor(rs.randint(nb_entities, size=32), dtype=torch.long)

            xs_emb = entity_embeddings(xs)
            xp_emb = predicate_embeddings(xp)
            xo_emb = entity_embeddings(xo)

            scores_sp = model.forward(xp_emb, xs_emb, None)
            # scores_so = model.forward(None, xs_emb, xo_emb)
            scores_po = model.forward(xp_emb, None, xo_emb)

            inf = model.score(xp_emb, xs_emb, xo_emb)

            inf = inf.cpu().numpy()

            scores_sp = scores_sp.cpu().numpy()
            # scores_so = scores_so.cpu().numpy()
            scores_po = scores_po.cpu().numpy()

            for i in range(32):
                # np.testing.assert_allclose(inf[i], scores_so[i, xp[i]], rtol=1e-3, atol=1e-3)
                np.testing.assert_allclose(inf[i], scores_sp[i, xo[i]], rtol=1e-3, atol=1e-3)
                np.testing.assert_allclose(inf[i], scores_po[i, xs[i]], rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
    pytest.main([__file__])
