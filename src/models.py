# coding: utf-8

from __future__ import absolute_import
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

class MLP(chainer.Chain):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__(
            l1=L.Linear(in_dim, in_dim),
            l2=L.Linear(in_dim, in_dim),
            l3=L.Linear(in_dim, out_dim)
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = self.l3(h2)
        return h3


class Generator(chainer.Chain):
    def __init__(self, in_dim, ent_num, rel_num):
        super(Generator, self).__init__(
            ent_emb=L.EmbedID(ent_num, in_dim),
            rel_emb=L.EmbedID(rel_num, in_dim),
            mlp=MLP(in_dim * 2, in_dim),
        )

    def __call__(self, h, r):
        h_emb = self.ent_emb(h).reshape(h.shape[0], -1)
        r_emb = self.rel_emb(r).reshape(h.shape[0], -1)
        x = F.concat((h_emb, r_emb))
        return self.mlp(x)

    def embed_entity(self, e):
        return self.ent_emb(e).reshape(e.shape[0], -1)

    def embed_relation(self, r):
        return self.rel_emb(r).reshape(r.shape[0], -1)

    @staticmethod
    def create_generator(emb_sz, vocab_ent, vocab_rel):
        # embedding link starts from 0, but token id starts from 1,
        # thus make a spare embedding for dummy id 0
        g = Generator(emb_sz, len(vocab_ent) + 1, len(vocab_rel) + 1)
        return g

class Discriminator(chainer.Chain):
    def __init__(self, in_dim):
        super(Discriminator, self).__init__(
            # for h, r, and t
            l1=L.Linear(in_dim * 3, in_dim),
            l2=L.Linear(in_dim, in_dim),
            l3=L.Linear(in_dim, 1),
            # mlp=MLP(in_dim * 3, 1)
        )

    def __call__(self, h_emb, r_emb, t_emb):
        x = F.concat((h_emb, r_emb, t_emb))
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = self.l2(h2)
        return h3

class BilinearDiscriminator(chainer.Chain):
    def __init__(self, in_dim):
        super(BilinearDiscriminator, self).__init__(
            bl=L.Bilinear(in_dim * 3, in_dim * 3, 1)
        )

    def __call__(self, h_emb, r_emb, t_emb):
        x = F.concat((h_emb, r_emb, t_emb)) # batch * embedding
        return self.bl(x, x)




